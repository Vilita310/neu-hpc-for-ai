"""
run_flash_attention.py
----------------------
Deploy CuTe-based FlashAttention to Modal GPU.

Steps:
    pip install modal
    python3 -m modal setup          # authenticate once
    modal run run_flash_attention.py
"""

import modal

# ---------------------------------------------------------------------------
# 1. Define the remote environment (Docker image)
#    - CUDA 12 dev image  +  CUTLASS repo (CuTe headers live inside it)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "cmake")
    .run_commands(
        # CuTe headers are part of CUTLASS; clone it once at image-build time
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /cutlass"
    )
)

app = modal.App("cute-flash-attention", image=image)

# ---------------------------------------------------------------------------
# 2. The CUDA source embedded as a Python string
#    (keeps everything in one file – easy to submit / share)
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

#define CUDA_CHECK(x)                                                   \
    do {                                                                \
        cudaError_t e = (x);                                            \
        if (e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(e));         \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: FlashAttention Algorithm 1 with CuTe Layouts
//
//   Grid  = (num_heads, batch)
//   Block = Br threads  (one thread owns one query row in the tile)
//
//   CuTe usage:
//     - make_layout / make_shape / make_stride  →  describe tile memory layout
//     - make_tensor(smem_ptr, layout)           →  wrap shared mem as a Tensor
//     - tQ(row, col), tK(row, col), ...         →  coordinate-based access
// ---------------------------------------------------------------------------
template <int Br, int Bc, int kD>
__global__ void flash_attn_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int N, float scale)
{
    // ---- which (batch, head) does this CTA own? ----
    const int h  = blockIdx.x;
    const int b  = blockIdx.y;
    const int H  = gridDim.x;
    const int bh = (b * H + h) * N * kD;

    const float* Qg = Q + bh;
    const float* Kg = K + bh;
    const float* Vg = V + bh;
    float*       Og = O + bh;

    // ---- shared memory tiles ----
    __shared__ float smQ[Br * kD];
    __shared__ float smK[Bc * kD];
    __shared__ float smV[Bc * kD];
    __shared__ float smS[Br * Bc];

    // ---- CuTe Layouts (row-major: stride = (num_cols, 1)) ----
    //   layout(row, col) = row * num_cols + col * 1
    auto Ql = make_layout(make_shape (Int<Br>{}, Int<kD>{}),
                          make_stride(Int<kD>{}, Int<1>{}));
    auto Kl = make_layout(make_shape (Int<Bc>{}, Int<kD>{}),
                          make_stride(Int<kD>{}, Int<1>{}));
    auto Sl = make_layout(make_shape (Int<Br>{}, Int<Bc>{}),
                          make_stride(Int<Bc>{}, Int<1>{}));

    // ---- CuTe Tensors: Layout + shared-memory pointer ----
    auto tQ = make_tensor(make_smem_ptr(smQ), Ql);
    auto tK = make_tensor(make_smem_ptr(smK), Kl);
    auto tV = make_tensor(make_smem_ptr(smV), Kl);   // same layout as K
    auto tS = make_tensor(make_smem_ptr(smS), Sl);

    const int tid = threadIdx.x;
    const int Tr  = (N + Br - 1) / Br;
    const int Tc  = (N + Bc - 1) / Bc;

    // per-thread (= per-row) accumulators
    float Oa[kD];
    float mi, li;

    // =====================================================================
    // Outer loop over query tiles  (Algorithm 1, line 6)
    // =====================================================================
    for (int i = 0; i < Tr; ++i) {
        const int qi = i * Br + tid;   // global query row for this thread

        // --- init state ---
        mi = -FLT_MAX;  li = 0.f;
        for (int d = 0; d < kD; ++d) Oa[d] = 0.f;

        // --- load Q tile into shared mem via CuTe tensor tQ(tid, d) ---
        if (tid < Br) {
            for (int d = 0; d < kD; ++d)
                tQ(tid, d) = (qi < N) ? Qg[qi * kD + d] : 0.f;
        }
        __syncthreads();

        // =================================================================
        // Inner loop over key/value tiles  (Algorithm 1, line 8)
        // =================================================================
        for (int j = 0; j < Tc; ++j) {
            const int kvi = j * Bc + tid;  // global KV row for this thread

            if (tid < Bc) {
                for (int d = 0; d < kD; ++d) {
                    tK(tid, d) = (kvi < N) ? Kg[kvi * kD + d] : 0.f;
                    tV(tid, d) = (kvi < N) ? Vg[kvi * kD + d] : 0.f;
                }
            }
            __syncthreads();

            // S = scale * Q_tile @ K_tile^T   →  tS(row, col)
            if (tid < Br) {
                for (int c = 0; c < Bc; ++c) {
                    float dot = 0.f;
                    for (int d = 0; d < kD; ++d)
                        dot += tQ(tid, d) * tK(c, d);
                    tS(tid, c) = scale * dot;
                }
            }
            __syncthreads();

            // Online softmax update  (Algorithm 1, lines 10-14)
            if (tid < Br) {
                float mij = -FLT_MAX;
                for (int c = 0; c < Bc; ++c)
                    mij = fmaxf(mij, tS(tid, c));

                float lij = 0.f;
                for (int c = 0; c < Bc; ++c)
                    lij += expf(tS(tid, c) - mij);

                float mn = fmaxf(mi, mij);
                float ln = expf(mi - mn) * li + expf(mij - mn) * lij;

                for (int d = 0; d < kD; ++d) {
                    float pv = 0.f;
                    for (int c = 0; c < Bc; ++c)
                        pv += expf(tS(tid, c) - mn) * tV(c, d);
                    Oa[d] = (expf(mi - mn) * li * Oa[d] + pv) / ln;
                }
                mi = mn;  li = ln;
            }
            __syncthreads();
        }

        // Write output tile back to global memory
        if (tid < Br && qi < N)
            for (int d = 0; d < kD; ++d)
                Og[qi * kD + d] = Oa[d];

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// CPU reference  (naive O(N²) attention for correctness check)
// ---------------------------------------------------------------------------
void reference_attention(const float* Q, const float* K, const float* V,
                         float* O, int N, int d)
{
    float scale = 1.f / sqrtf((float)d);
    std::vector<float> S(N * N);

    for (int i = 0; i < N; ++i) {
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            float dot = 0;
            for (int dd = 0; dd < d; ++dd) dot += Q[i*d+dd] * K[j*d+dd];
            S[i*N+j] = dot * scale;
            m = std::max(m, S[i*N+j]);
        }
        float l = 0;
        for (int j = 0; j < N; ++j) { S[i*N+j] = expf(S[i*N+j]-m); l += S[i*N+j]; }
        for (int dd = 0; dd < d; ++dd) {
            float acc = 0;
            for (int j = 0; j < N; ++j) acc += S[i*N+j]/l * V[j*d+dd];
            O[i*d+dd] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// main: allocate, run, compare
// ---------------------------------------------------------------------------
int main() {
    constexpr int B = 1, H = 1, N = 128, d = 64;
    constexpr int Br = 32, Bc = 32;

    size_t sz = (size_t)B*H*N*d * sizeof(float);

    std::vector<float> hQ(B*H*N*d), hK(B*H*N*d), hV(B*H*N*d);
    std::vector<float> hO(B*H*N*d,0), hRef(B*H*N*d,0);

    srand(42);
    for (auto& v : hQ) v = (float)rand()/RAND_MAX * 0.2f - 0.1f;
    for (auto& v : hK) v = (float)rand()/RAND_MAX * 0.2f - 0.1f;
    for (auto& v : hV) v = (float)rand()/RAND_MAX * 0.2f - 0.1f;

    reference_attention(hQ.data(), hK.data(), hV.data(), hRef.data(), N, d);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sz));
    CUDA_CHECK(cudaMalloc(&dK, sz));
    CUDA_CHECK(cudaMalloc(&dV, sz));
    CUDA_CHECK(cudaMalloc(&dO, sz));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), sz, cudaMemcpyHostToDevice));

    float scale = 1.f / sqrtf((float)d);
    dim3 grid(H, B);
    dim3 block(Br);
    flash_attn_kernel<Br, Bc, d><<<grid, block>>>(dQ, dK, dV, dO, N, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hO.data(), dO, sz, cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < B*H*N*d; ++i)
        max_err = std::max(max_err, fabsf(hO[i]-hRef[i]));
    printf("Max error vs CPU reference: %.6f  %s\n",
           max_err, max_err < 1e-3f ? "[PASS]" : "[FAIL]");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return max_err < 1e-3f ? 0 : 1;
}
"""

# ---------------------------------------------------------------------------
# 3. Modal function: compile the CUDA source, then run the binary
# ---------------------------------------------------------------------------
@app.function(gpu="A10G", timeout=300)
def run_flash_attention():
    import subprocess, tempfile, os

    with tempfile.TemporaryDirectory() as td:
        src  = os.path.join(td, "fa.cu")
        exe  = os.path.join(td, "fa")

        with open(src, "w") as f:
            f.write(CUDA_SRC)

        compile_cmd = (
            f"nvcc -O2 -std=c++17 "
            f"-I/cutlass/include "
            f"--expt-relaxed-constexpr "
            f"-arch=sm_86 "           # A10G = sm_86; A100 = sm_80; H100 = sm_90
            f"{src} -o {exe}"
        )

        print("[compile]", compile_cmd)
        r = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("COMPILE ERROR:\n", r.stderr)
            return

        print("[run]", exe)
        r = subprocess.run(exe, capture_output=True, text=True, timeout=120)
        print(r.stdout)
        if r.stderr:
            print("stderr:", r.stderr)

# ---------------------------------------------------------------------------
# 4. Local entrypoint  (exactly like the Quickstart guide)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Submitting CuTe FlashAttention job to Modal …")
    run_flash_attention.remote()
