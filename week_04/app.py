import modal
import os
import subprocess
import time
from datetime import datetime

app = modal.App("flashattention-week4-final")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("cmake", "ninja-build", "build-essential")
)

CMAKELISTS = r"""
cmake_minimum_required(VERSION 3.18)
project(flashattn_week4 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(flashattn src/flashattn_cpu.cpp src/flashattn_cuda.cu)
target_include_directories(flashattn PUBLIC include)
set_target_properties(flashattn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(flashattn_test tests/test_main.cpp)
target_link_libraries(flashattn_test PRIVATE flashattn)
"""

FLASHATTN_H = r"""
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void naive_attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d);
void flash_attention_alg1_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d, int Br, int Bc);
int flash_attention_alg1_cuda(const float* hQ, const float* hK, const float* hV, float* hO, int N, int d, int Br, int Bc);
#ifdef __cplusplus
}
#endif
"""

FLASHATTN_CPU = r"""
#include "flashattn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static inline const float* row_ptr(const float* A, int row, int d) { return A + (size_t)row * d; }
static inline float* row_ptr(float* A, int row, int d) { return A + (size_t)row * d; }

void naive_attention_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
    const float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> logits((size_t)N);
    std::vector<float> probs((size_t)N);
    for (int i = 0; i < N; ++i) {
        const float* qi = row_ptr(Q, i, d);
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            const float* kj = row_ptr(K, j, d);
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) dot += qi[x] * kj[x];
            float s = dot * scale;
            logits[j] = s;
            m = std::max(m, s);
        }
        float l = 0.0f;
        for (int j = 0; j < N; ++j) {
            float p = std::exp(logits[j] - m);
            probs[j] = p;
            l += p;
        }
        float inv_l = 1.0f / l;
        float* oi = row_ptr(O, i, d);
        for (int x = 0; x < d; ++x) oi[x] = 0.0f;
        for (int j = 0; j < N; ++j) {
            const float* vj = row_ptr(V, j, d);
            float p = probs[j] * inv_l;
            for (int x = 0; x < d; ++x) oi[x] += p * vj[x];
        }
    }
}

void flash_attention_alg1_cpu(const float* Q, const float* K, const float* V,
                              float* O, int N, int d, int Br, int Bc) {
    const float scale = 1.0f / std::sqrt((float)d);
    for (int qi0 = 0; qi0 < N; qi0 += Br) {
        int br = std::min(Br, N - qi0);
        std::vector<float> mi((size_t)br, -std::numeric_limits<float>::infinity());
        std::vector<float> li((size_t)br, 0.0f);
        std::vector<float> Oacc((size_t)br * d, 0.0f);
        std::vector<float> s((size_t)Bc);
        std::vector<float> p((size_t)Bc);
        for (int kj0 = 0; kj0 < N; kj0 += Bc) {
            int bc = std::min(Bc, N - kj0);
            for (int r = 0; r < br; ++r) {
                const float* q = row_ptr(Q, qi0 + r, d);
                float mij = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < bc; ++c) {
                    const float* k = row_ptr(K, kj0 + c, d);
                    float dot = 0.0f;
                    for (int x = 0; x < d; ++x) dot += q[x] * k[x];
                    float val = dot * scale;
                    s[c] = val;
                    mij = std::max(mij, val);
                }
                float lij = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    float pc = std::exp(s[c] - mij);
                    p[c] = pc;
                    lij += pc;
                }
                float mi_old = mi[(size_t)r];
                float li_old = li[(size_t)r];
                float mi_new = std::max(mi_old, mij);
                float alpha = std::exp(mi_old - mi_new);
                float beta  = std::exp(mij    - mi_new);
                float li_new = alpha * li_old + beta * lij;
                std::vector<float> PV((size_t)d, 0.0f);
                for (int c = 0; c < bc; ++c) {
                    const float* v = row_ptr(V, kj0 + c, d);
                    float pc = p[c];
                    for (int x = 0; x < d; ++x) PV[(size_t)x] += pc * v[x];
                }
                float* oacc = &Oacc[(size_t)r * d];
                for (int x = 0; x < d; ++x) {
                    oacc[x] = alpha * oacc[x] + beta * PV[(size_t)x];
                }
                mi[(size_t)r] = mi_new;
                li[(size_t)r] = li_new;
            }
        }
        for (int r = 0; r < br; ++r) {
            float inv = 1.0f / li[(size_t)r];
            float* out = row_ptr(O, qi0 + r, d);
            const float* oacc = &Oacc[(size_t)r * d];
            for (int x = 0; x < d; ++x) out[x] = oacc[x] * inv;
        }
    }
}
"""

FLASHATTN_CUDA = r"""
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
    int r  = (int)threadIdx.y;
    int tx = (int)threadIdx.x;
    int Tx = (int)blockDim.x;
    int Ty = (int)blockDim.y;
    int br = min(Br, N - qi0);

    float* Qi   = smem;
    float* Kj   = Qi   + Br * d;
    float* Vj   = Kj   + Bc * d;
    float* Oacc = Vj   + Bc * d;
    float* mi   = Oacc + Br * d;
    float* li   = mi   + Br;
    float* pij  = li   + Br;
    float* red  = pij  + Br * Bc;

    if (r < br) {
        for (int x = tx; x < d; x += Tx) {
            Qi[r * d + x] = Q[(qi0 + r) * d + x];
            Oacc[r * d + x] = 0.0f;
        }
        if (tx == 0) { mi[r] = -INFINITY; li[r] = 0.0f; }
    }
    __syncthreads();

    float scale = rsqrtf((float)d);

    for (int kj0 = 0; kj0 < N; kj0 += Bc) {
        int bc = min(Bc, N - kj0);
        int t = r * Tx + tx;
        int total = bc * d;
        for (int idx = t; idx < total; idx += Tx * Ty) {
            int c = idx / d;
            int x = idx % d;
            Kj[c * d + x] = K[(kj0 + c) * d + x];
            Vj[c * d + x] = V[(kj0 + c) * d + x];
        }
        __syncthreads();

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

        float local_sum = 0.0f;
        if (r < br) {
            for (int c = tx; c < bc; c += Tx) {
                float p = __expf(pij[r * Bc + c] - mij);
                pij[r * Bc + c] = p;
                local_sum += p;
            }
        }
        float lij = row_reduce_sum(local_sum, red, r, tx, Tx);

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

    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    int Tr = (N + Br - 1) / Br;
    const int Tx = min(1024 / Br, 128);
    dim3 block(Tx, Br, 1);
    dim3 grid(Tr, 1, 1);

    size_t smem_bytes =
        (size_t)(Br*d) * sizeof(float) +
        (size_t)(Bc*d) * sizeof(float) +
        (size_t)(Bc*d) * sizeof(float) +
        (size_t)(Br*d) * sizeof(float) +
        (size_t)(Br)   * sizeof(float) +
        (size_t)(Br)   * sizeof(float) +
        (size_t)(Br*Bc)* sizeof(float) +
        (size_t)(Br*Tx)* sizeof(float);

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

    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
"""

TEST_MAIN = r"""
#include "flashattn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = (double)a[i] - (double)b[i];
        num += diff * diff;
        den += (double)b[i] * (double)b[i];
    }
    return (float)std::sqrt(num / (den + 1e-12));
}

static double ms_since(const std::chrono::high_resolution_clock::time_point& t0,
                       const std::chrono::high_resolution_clock::time_point& t1) {
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
}

int main(int argc, char** argv) {
    int N = 256, d = 64, Br = 64, Bc = 64, seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_i = [&]() -> int { if (i+1 >= argc) std::exit(1); return ++i; };
        if      (arg == "--N")    N  = std::atoi(argv[next_i()]);
        else if (arg == "--d")    d  = std::atoi(argv[next_i()]);
        else if (arg == "--Br")   Br = std::atoi(argv[next_i()]);
        else if (arg == "--Bc")   Bc = std::atoi(argv[next_i()]);
        else if (arg == "--seed") seed = std::atoi(argv[next_i()]);
    }

    printf("Config: N=%d d=%d Br=%d Bc=%d seed=%d\n", N, d, Br, Bc, seed);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t sz = (size_t)N * d;
    std::vector<float> Q(sz), K(sz), V(sz);
    for (size_t i = 0; i < sz; ++i) { Q[i] = dist(rng); K[i] = dist(rng); V[i] = dist(rng); }

    std::vector<float> O_naive(sz), O_cpu(sz), O_gpu(sz);

    auto t0 = std::chrono::high_resolution_clock::now();
    naive_attention_cpu(Q.data(), K.data(), V.data(), O_naive.data(), N, d);
    auto t1 = std::chrono::high_resolution_clock::now();
    double naive_ms = ms_since(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    flash_attention_alg1_cpu(Q.data(), K.data(), V.data(), O_cpu.data(), N, d, Br, Bc);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = ms_since(t0, t1);

    float cpu_max = max_abs_diff(O_cpu, O_naive);
    float cpu_rel = rel_l2(O_cpu, O_naive);
    printf("[CPU Flash vs Naive] max_abs=%.6g  rel_l2=%.6g  time_ms=%.3f\n", cpu_max, cpu_rel, cpu_ms);

    int rc = flash_attention_alg1_cuda(Q.data(), K.data(), V.data(), O_gpu.data(), N, d, Br, Bc);
    if (rc != 0) { printf("CUDA failed with code %d\n", rc); return 2; }

    t0 = std::chrono::high_resolution_clock::now();
    rc = flash_attention_alg1_cuda(Q.data(), K.data(), V.data(), O_gpu.data(), N, d, Br, Bc);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = ms_since(t0, t1);

    float gpu_max = max_abs_diff(O_gpu, O_naive);
    float gpu_rel = rel_l2(O_gpu, O_naive);
    printf("[CUDA Flash vs Naive] max_abs=%.6g  rel_l2=%.6g  time_ms=%.3f\n", gpu_max, gpu_rel, gpu_ms);

    const float tol = 5e-3f;
    bool pass = (cpu_max < tol && cpu_rel < tol && gpu_max < tol && gpu_rel < tol);
    printf("Naive_time_ms=%.3f\n", naive_ms);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
}
"""


def _write_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


@app.function(image=image, gpu="any", timeout=3600)
def build_run_and_report(N: int = 256, d: int = 64, Br: int = 64, Bc: int = 64, seed: int = 0):
    t_all0 = time.time()

    base = "/root/project"
    _write_text(f"{base}/CMakeLists.txt", CMAKELISTS)
    _write_text(f"{base}/include/flashattn.h", FLASHATTN_H)
    _write_text(f"{base}/src/flashattn_cpu.cpp", FLASHATTN_CPU)
    _write_text(f"{base}/src/flashattn_cuda.cu", FLASHATTN_CUDA)
    _write_text(f"{base}/tests/test_main.cpp", TEST_MAIN)

    build_dir = f"{base}/build"
    os.makedirs(build_dir, exist_ok=True)

    print("=== Configure (CMake+Ninja) ===")
    cfg = _run(["cmake", "..", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir)
    print(cfg.stdout)
    if cfg.stderr:
        print(cfg.stderr)

    print("=== Build ===")
    bld = _run(["cmake", "--build", ".", "-j"], cwd=build_dir)
    print(bld.stdout)
    if bld.stderr:
        print(bld.stderr)

    print("=== Run test ===")
    exe = os.path.join(build_dir, "flashattn_test")
    runp = subprocess.run(
        [exe, "--N", str(N), "--d", str(d), "--Br", str(Br), "--Bc", str(Bc), "--seed", str(seed)],
        cwd=build_dir, text=True, capture_output=True,
    )
    print(runp.stdout)
    if runp.stderr:
        print("STDERR:\n", runp.stderr)

    total_s = time.time() - t_all0

    report = f"""# Week 04 — FlashAttention (CPU + CUDA)

**Run timestamp:** {datetime.utcnow().isoformat()}Z
**Config:** N={N}, d={d}, Br={Br}, Bc={Bc}, seed={seed}
**Total pipeline time (build+run):** {total_s:.2f} s

## Build
- Base image: nvidia/cuda:12.3.2-devel-ubuntu22.04
- Toolchain: CMake + Ninja, C++17, CUDA

## Correctness & Runtime Output

```text
{runp.stdout.strip()}
```

## Notes
- CPU implementation: naive softmax attention + FlashAttention alg1 online update.
- CUDA kernel: tiled Q (Br rows) and KV (Bc cols) with shared memory; online softmax update (mi/li) per row.
- PASS condition: CPU Flash and CUDA Flash both close to naive within tolerance (tol=5e-3).
"""

    return {
        "returncode": runp.returncode,
        "stdout": runp.stdout,
        "stderr": runp.stderr,
        "report_md": report,
        "total_s": total_s,
    }


@app.local_entrypoint()
def main():
    res = build_run_and_report.remote()

    with open("report.md", "w") as f:
        f.write(res["report_md"])

    print("\n====================")
    print("Local file written: report.md")
    print("Remote return code:", res["returncode"])
    print("====================\n")
