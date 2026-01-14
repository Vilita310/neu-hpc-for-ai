# Parallel Matrix Multiplication with Pthreads

This project implements and benchmarks single-threaded and multi-threaded matrix multiplication in C using POSIX threads (pthreads).

---

## Project Structure

```
matmul/
├── matmul.h        # Matrix structure and function declarations
├── matmul.c        # Single-threaded and pthread implementations
├── test.c          # Correctness tests (corner cases)
├── bench.c         # Performance benchmarking (CSV output)
├── Makefile        # Build rules
├── results.csv     # Benchmark results (auto-generated)
└── README.md
```

---

## Build Instructions

Compile the project using:

```bash
make
```

---

## Correctness Testing

Run the correctness tests:

```bash
./test
```

The test suite covers:

* Small matrices (e.g., 1×1)
* Rectangular matrices
* Square matrices
* Large matrices
* Multiple thread counts

All test cases must pass before benchmarking.

---

## Performance Benchmarking

Run the benchmark with:

```bash
./bench 1024 3
```

Arguments:

* `1024`: matrix dimension (N×N)
* `3`: number of iterations (best time recorded)

Benchmark results are printed to the console and automatically written to:

```
results.csv
```

---

## Parallel Design

* Row-based partitioning of the output matrix
* Each thread computes a disjoint set of rows
* No race conditions or synchronization overhead
* Shared-memory CPU parallelism using pthreads

---

## Modal Platform

The Modal platform was used to explore cloud-based execution and GPU programming through the provided examples.
The pthread-based implementation is executed locally on a multi-core CPU, which is the appropriate environment for shared-memory parallelism.

---

## Notes

* Speedup saturates at higher thread counts due to memory bandwidth and cache limitations
* Results may vary depending on CPU architecture and system load

---

## Author

Jing Cao
