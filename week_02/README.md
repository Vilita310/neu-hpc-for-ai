
# Week 02 — CUDA GEMM

This assignment implements a CUDA kernel for generalized matrix multiplication (GEMM):

C ← α · op(A) · op(B) + β · C

Supported operations:
- AB
- AᵀB
- ABᵀ
- AᵀBᵀ

The implementation:
- Uses shared memory tiling for better data locality
- Supports optional transpose of A and/or B
- Updates matrix C in place
- Does not rely on cuBLAS or cuDNN

## Files

- src/gemm.cu — CUDA implementation of GEMM

## Build & Run

From this directory:

nvcc src/gemm.cu -O3 -o gemm
./gemm

## Notes

All matrices are stored in row-major layout.
Shared memory tiling follows concepts from PMPP Chapter 5.
