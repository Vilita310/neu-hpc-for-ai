# CUDA SGEMM Worklog Replication on H100

This project replicates the experiments and performance analysis described in the worklog:

**“How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance”**

using NVIDIA H100 GPUs on the course-supported Modal platform.

---

## 1. Platform and Environment

- Platform: Modal (course-supported cloud platform)
- GPU: NVIDIA H100
- CUDA Version: 12.3
- OS Image: nvidia/cuda:12.3.2-devel-ubuntu22.04
- Build Tools: CMake, Ninja
- Compute Capability: sm_90

All experiments were executed remotely on H100 GPUs provisioned by Modal.

---

## 2. Codebase

The implementation is based on the official repository provided with the worklog:

https://github.com/siboehm/SGEMM_CUDA

No functional changes were made to the kernel implementations.  
The only modification was updating the CUDA compute capability to match H100 hardware:

```cmake
set(CUDA_COMPUTE_CAPABILITY 90)
````

---

## 3. Build Instructions

```bash
git clone https://github.com/siboehm/SGEMM_CUDA.git
cd SGEMM_CUDA
mkdir -p build
cd build
cmake .. -G Ninja
cmake --build .
```

---

## 4. Running the Experiments

Each SGEMM kernel can be executed using the provided `sgemm` binary:

```bash
DEVICE=0 ./sgemm <kernel_id>
```

The following kernels were evaluated as part of this project:

| Kernel ID | Description              |
| --------: | ------------------------ |
|         0 | cuBLAS baseline          |
|         1 | Naive SGEMM              |
|         2 | Global memory coalescing |
|         3 | Shared memory tiling     |
|         4 | 1D block tiling          |
|         5 | 2D block tiling          |
|         6 | Vectorized memory access |
|         9 | Autotuned kernel         |
|        10 | Warp-level tiling        |

All kernels were tested with square matrices up to size 4096 × 4096 × 4096.

---

## 5. Results and Analysis

Detailed performance results, FLOPs derivation, and discussion of optimization effects are provided in:

**HW_Report.md**

This includes:

* Mathematical derivation of GEMM FLOPs
* Performance comparison across kernels
* Relative performance compared to cuBLAS
* Analysis of memory hierarchy and tiling strategies

---

## 6. Notes

* Performance measurements were collected directly from Modal execution logs.
* Nsight Compute profiling was not required to complete the assignment.
* While absolute performance values may differ across GPU architectures, the observed optimization trends are consistent with those reported in the original worklog.

---

## 7. References

* Siboehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
  [https://siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
* NVIDIA CUDA Programming Guide
* NVIDIA Hopper Architecture Documentation

```

