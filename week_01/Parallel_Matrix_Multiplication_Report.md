# Parallel Matrix Multiplication Using Pthreads

## 1. Introduction

This project explores shared-memory parallelism on multi-core CPUs using POSIX threads (pthreads).
Matrix multiplication is a fundamental operation in high-performance computing (HPC) and serves as a core workload in scientific computing and modern machine learning systems.

The objectives of this assignment are:

* To implement a correct single-threaded matrix multiplication in C
* To implement a multi-threaded version using pthreads
* To verify correctness across a wide range of matrix dimensions
* To evaluate performance speedup under different thread counts

---

## 2. Implementation Overview

### 2.1 Data Representation

Matrices are stored in row-major order using the following structure:

```c
typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;
```

This layout ensures contiguous memory access when iterating across rows, improving cache locality.

---

### 2.2 Single-threaded Baseline

The single-threaded implementation follows the standard triple-loop formulation:

[ C_{ij} = \sum_k A_{ik} \times B_{kj} ]

This version serves as the baseline reference (golden output) for validating the correctness of the parallel implementation.

---

### 2.3 Multi-threaded Design Using Pthreads

The parallel implementation partitions the output matrix by rows.

* Each thread computes a contiguous block of rows of matrix **C**
* Matrices **A** and **B** are read-only
* Each thread writes exclusively to its assigned rows in **C**

#### Race Condition Analysis

There are no race conditions because:

* No two threads write to the same memory locations
* All shared data (A and B) are read-only
* No synchronization primitives (locks or atomics) are required

This design ensures deterministic and thread-safe execution.

---

## 3. Correctness Testing

Correctness is verified by comparing the multi-threaded output against the single-threaded baseline.
All values are compared within a small floating-point tolerance.

### Test cases include:

* Small matrices: `1×1`, `1×5`
* Rectangular matrices: `2×1 × 1×3`
* Square matrices: `2×2`, `64×64`
* Large non-square matrices: `128×64 × 64×256`

All test cases passed for thread counts:

```
1, 2, 4, 8, 16, 32, 64, 128
```

---

## 4. Performance Evaluation

### Experimental Setup

* Matrix size: `1024 × 1024`
* Timing method: `clock_gettime(CLOCK_MONOTONIC)`
* Best of 3 runs recorded
* Execution platform: local multi-core CPU (shared-memory environment)

### Results

| Threads | Time (s) | Speedup |
| ------- | -------- | ------- |
| 1       | 0.1746   | 1.00×   |
| 4       | 0.0547   | 3.20×   |
| 16      | 0.0396   | 4.41×   |
| 32      | 0.0384   | 4.55×   |
| 64      | 0.0376   | 4.65×   |
| 128     | 0.0380   | 4.60×   |

Benchmark results are also automatically exported to `results.csv`.

---

## 5. Performance Analysis

Performance improves significantly when increasing the number of threads from 1 to 4.
However, speedup gradually saturates as the thread count increases.

This behavior is expected due to:

* Memory bandwidth limitations
* Cache contention among threads
* Thread scheduling and overhead

Beyond approximately 32 threads, the computation becomes memory-bound rather than compute-bound,
which limits further scalability on a CPU.

---

## 6. Scaling Laws and High-Performance Computing

The concept of scaling laws, as discussed in *Scaling Laws for Large Language Models*,
shows that model performance improves predictably with increased compute, data, and model size.

Large-scale matrix multiplications dominate the training and inference workloads of modern machine learning models.
As model sizes grow, high-performance computing becomes essential to keep training times feasible.

Parallelism across CPUs, GPUs, and distributed systems enables this scaling,
making efficient parallel computation a foundational requirement for advancing state-of-the-art AI systems.

---

## 7. Modal Platform Usage

The Modal platform was used to explore cloud-based compute resources and GPU execution models through the provided documentation examples.
The pthread-based matrix multiplication was implemented and benchmarked locally on a multi-core CPU,
which is the appropriate execution environment for shared-memory CPU parallelism.

---

## 8. Conclusion

This project demonstrates how pthread-based parallelism can significantly accelerate matrix multiplication while maintaining correctness.
The results highlight both the benefits and the practical limits of CPU-based parallel scaling,
reinforcing the importance of HPC techniques in modern computing and machine learning workloads.
