# CUDA SGEMM Optimization Report (H100)

## 1. Experimental Environment

- Platform: Modal (course-supported cloud platform)
- GPU: NVIDIA H100
- CUDA Version: 12.3
- OS Image: nvidia/cuda:12.3.2-devel-ubuntu22.04
- Build System: CMake + Ninja
- Repository: https://github.com/siboehm/SGEMM_CUDA
- Compute Capability: sm_90

All experiments were conducted on NVIDIA H100 GPUs using the official SGEMM_CUDA implementation accompanying the worklog *“How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance”*.  
Matrix sizes were tested up to 4096 × 4096 × 4096, consistent with the original worklog.

---

## 2. GEMM FLOPs Derivation

We consider the general SGEMM formulation:

\[
C = \alpha AB + \beta C
\]

where:
- \(A \in \mathbb{R}^{M \times K}\)
- \(B \in \mathbb{R}^{K \times N}\)
- \(C \in \mathbb{R}^{M \times N}\)

### FLOPs Calculation

- Matrix multiplication \(AB\):  
  Each output element performs \(K\) fused multiply-add (FMA) operations.  
  Since 1 FMA = 2 FLOPs, total FLOPs:

\[
2MNK
\]

- Epilogue operations:
  - Multiply by \(\alpha\): \(MN\)
  - Multiply existing \(C\) by \(\beta\): \(MN\)
  - Addition: \(MN\)

\[
\text{Epilogue FLOPs} = 3MN
\]

### Total FLOPs

\[
\boxed{\text{Total FLOPs} = 2MNK + 3MN}
\]

For \(M = N = K = 4096\):

- Total FLOPs ≈ \(1.374 \times 10^{11}\)

---

## 3. Performance Results on H100

Performance is reported in GFLOPs/s for square matrices with \(M = N = K = 4096\).

| Kernel ID | Kernel Description | Performance (GFLOPs/s) | Relative to cuBLAS |
|----------:|-------------------|------------------------:|-------------------:|
| 0 | cuBLAS baseline | **50,989.4** | **100%** |
| 1 | Naive kernel | 498.7 | 0.98% |
| 2 | Global memory coalescing | 6,048.1 | 11.9% |
| 3 | Shared memory tiling | 9,189.8 | 18.0% |
| 4 | 1D block tiling | 17,152.5 | 33.6% |
| 5 | 2D block tiling | 25,860.5 | 50.7% |
| 6 | Vectorized memory access | 31,626.7 | 62.0% |
| 9 | Autotuned kernel | 31,264.4 | 61.3% |
| 10 | Warp-level tiling | **36,318.5** | **71.2%** |

---

## 4. Discussion

The naive SGEMM implementation exhibits extremely poor performance due to redundant global memory accesses and lack of data reuse, achieving less than 1% of cuBLAS performance.

Introducing global memory coalescing (Kernel 2) significantly improves performance by optimizing memory access patterns, resulting in more than a 10× speedup.

Shared memory tiling (Kernel 3) further improves arithmetic intensity by reducing repeated global memory reads, allowing better utilization of GPU compute resources.

Block-level tiling (Kernels 4 and 5) increases data reuse within thread blocks, substantially improving performance to over 50% of cuBLAS.

Vectorized memory access and autotuning (Kernels 6 and 9) improve instruction efficiency and memory throughput, further closing the gap with cuBLAS.

Finally, warp-level tiling (Kernel 10) exploits warp-synchronous execution and register-level data reuse, achieving over 71% of cuBLAS performance. This demonstrates that carefully designed hierarchical tiling strategies can approach vendor-optimized libraries even in custom CUDA kernels.

---

## 5. Conclusion

Through systematic optimization of memory access patterns, shared memory usage, and warp-level computation, SGEMM performance improves by more than two orders of magnitude compared to the naive implementation.  
The final optimized kernel achieves over 70% of cuBLAS performance on NVIDIA H100, validating the effectiveness of the optimization strategies discussed in the worklog.
