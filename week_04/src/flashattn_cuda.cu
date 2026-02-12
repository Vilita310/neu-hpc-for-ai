#include "flashattn.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
    }
}

__device__ float row_reduce_max(float val, float* scratch, int row, int tx, int row_stride) {
    scratch[row * row_stride + tx] = val;
    __syncthreads();
    float out = -INFINITY;
    if (tx == 0) {
        for (int i = 0; i < row_stride; ++i) out = fmaxf(out, scratch[row * row_stride + i]);
        scratch[row * row_stride + 0] = out;
    }
    __syncthreads();
    return scratch[row * row_stride + 0];
}

__device__ float row_reduce_sum(float val, float* scratch, int row, int tx, int row_stride) {
    scratch[row * row_stride + tx] = val;
    __syncthreads();
    float out = 0.0f;
    if (tx == 0) {
        for (int i = 0; i < row_stride; ++i) out += scratch[row * row_stride + i];
        scratch[row * row_stride + 0] = out;
    }
    __syncthreads();
    return scratch[row * row_stride + 0];
}

__global__ void flashattn2_alg1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, int Br, int Bc)
{
    extern __shared__ float smem[];
    int qi_tile = (int)blockIdx.x;
    int qi0 = qi_tile * Br;

    int r  = (int)threadIdx.y;   // 0..Br-1
    int tx = (int)threadIdx.x;   // 0..Tx-1
    int Tx = (int)blockDim.x;
    int Ty = (int)blockDim.y;

    int br = min(Br, N - qi0);

    // Shared layout
    float* Qi   = smem;                  // Br*d
    float* Kj   = Qi   + Br * d;         // Bc*d
    float* Vj   = Kj   + Bc * d;         // Bc*d
    float* Oacc = Vj   + Bc * d;         // Br*d
    float* mi   = Oacc + Br * d;         // Br
    float* li   = mi   + Br;             // Br
    float* pij  = li   + Br;             // Br*Bc
    float* red  = pij  + Br * Bc;        // Br*Tx

    // Load Qi & init
    if (r < br) {
        for (int x = tx; x < d; x += Tx) {
            Qi[r * d + x] = Q[(qi0 + r) * d + x];
            Oacc[r * d + x] = 0.0f;
        }
        if (tx == 0) { mi[r] = -INFINITY; li[r] = 0.0f; }
    }
    __syncthreads();

    float scale = rsqrtf((float)d);

    // Loop over K/V tiles
    for (int kj0 = 0; kj0 < N; kj0 += Bc) {
        int bc = min(Bc, N - kj0);

        // Load K/V to shared
        int t = r * Tx + tx;
        int total = bc * d;
        for (int idx = t; idx < total; idx += Tx * Ty) {
            int c = idx / d;
            int x = idx % d;
            Kj[c * d + x] = K[(kj0 + c) * d + x];
            Vj[c * d + x] = V[(kj0 + c) * d + x];
        }
        __syncthreads();

        // (A) logits and max per row
        float local_max = -INFINITY;
        if (r < br) {
            for (int c = tx; c < bc; c += Tx) {
                float dot = 0.0f;
                for (int x = 0; x < d; ++x) dot += Qi[r * d + x] * Kj[c * d + x];
                float s = dot * scale;
                pij[r * Bc + c] = s;
                local_max = fmaxf(local_max, s);
            }
        }
        float mij = row_reduce_max(local_max, red, r, tx, Tx);

        // (B) pij and sum per row
        float local_sum = 0.0f;
        if (r < br) {
            for (int c = tx; c < bc; c += Tx) {
                float p = __expf(pij[r * Bc + c] - mij);
                pij[r * Bc + c] = p;
                local_sum += p;
            }
        }
        float lij = row_reduce_sum(local_sum, red, r, tx, Tx);

        // (C) online update
        float mi_old = (r < br) ? mi[r] : -INFINITY;
        float li_old = (r < br) ? li[r] : 0.0f;
        float mi_new = fmaxf(mi_old, mij);
        float alpha = __expf(mi_old - mi_new);
        float beta  = __expf(mij    - mi_new);
        float li_new = alpha * li_old + beta * lij;

        if (r < br) {
            for (int x = tx; x < d; x += Tx) {
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += pij[r * Bc + c] * Vj[c * d + x];
                }
                Oacc[r * d + x] = alpha * Oacc[r * d + x] + beta * pv;
            }
            if (tx == 0) { mi[r] = mi_new; li[r] = li_new; }
        }
        __syncthreads();
    }

    // write O = Oacc / li
    if (r < br) {
        float inv = 1.0f / li[r];
        for (int x = tx; x < d; x += Tx) {
            O[(qi0 + r) * d + x] = Oacc[r * d + x] * inv;
        }
    }
}

int flash_attention_alg1_cuda(const float* hQ, const float* hK, const float* hV,
                              float* hO, int N, int d, int Br, int Bc) {
    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    size_t bytes = (size_t)N * d * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&dQ, bytes); if (err) { checkCuda(err, "cudaMalloc dQ"); return (int)err; }
    err = cudaMalloc(&dK, bytes); if (err) { checkCuda(err, "cudaMalloc dK"); return (int)err; }
    err = cudaMalloc(&dV, bytes); if (err) { checkCuda(err, "cudaMalloc dV"); return (int)err; }
    err = cudaMalloc(&dO, bytes); if (err) { checkCuda(err, "cudaMalloc dO"); return (int)err; }

    err = cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice); if (err) { checkCuda(err, "H2D Q"); return (int)err; }
    err = cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice); if (err) { checkCuda(err, "H2D K"); return (int)err; }
    err = cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice); if (err) { checkCuda(err, "H2D V"); return (int)err; }

    int Tr = (N + Br - 1) / Br;

    // FIX: Tx * Br must be <= 1024 (max threads per block)
    const int int Tx = 1024 / Br; 
    Tx = max(1, min(Tx, 128)); // Br=64 → Tx=16
    dim3 block(Tx, Br, 1);
    dim3 grid(Tr, 1, 1);

    // Shared memory bytes
    size_t smem_bytes =
        (size_t)(Br*d) * sizeof(float) + // Qi
        (size_t)(Bc*d) * sizeof(float) + // Kj
        (size_t)(Bc*d) * sizeof(float) + // Vj
        (size_t)(Br*d) * sizeof(float) + // Oacc
        (size_t)(Br)   * sizeof(float) + // mi
        (size_t)(Br)   * sizeof(float) + // li
        (size_t)(Br*Bc)* sizeof(float) + // pij/logits
        (size_t)(Br*Tx)* sizeof(float);  // reduction scratch

    // Request extended shared memory if needed (for GPUs that support it)
    cudaFuncSetAttribute(flashattn2_alg1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    printf("Launch config: grid=(%d), block=(%d,%d), smem=%zu bytes\n",
           Tr, Tx, Br, smem_bytes);

    flashattn2_alg1_kernel<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N, d, Br, Bc);
    err = cudaGetLastError();
    if (err) { checkCuda(err, "kernel launch"); return (int)err; }
    err = cudaDeviceSynchronize();
    if (err) { checkCuda(err, "kernel sync"); return (int)err; }

    err = cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    if (err) { checkCuda(err, "D2H O"); return (int)err; }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}