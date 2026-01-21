
// gemm.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__device__ __forceinline__ float loadA(
    const float* A, int m, int k, bool transA, int row, int col) {
  if (!transA) return A[row * k + col];
  return A[col * k + row];
}

__device__ __forceinline__ float loadB(
    const float* B, int k, int n, bool transB, int row, int col) {
  if (!transB) return B[row * n + col];
  return B[col * n + row];
}

constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(
    float alpha,
    const float* A, bool transA,
    const float* B, bool transB,
    float beta,
    float* C,
    int m, int n, int k) {

  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;

  for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < m && aCol < k) ? loadA(A, m, k, transA, row, aCol) : 0.0f;

    Bs[threadIdx.y][threadIdx.x] =
        (bRow < k && col < n) ? loadB(B, k, n, transB, bRow, col) : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE; ++i) {
      acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = alpha * acc + beta * C[row * n + col];
  }
}

void gemm(
    float alpha,
    const float* dA, bool transposeA,
    const float* dB, bool transposeB,
    float beta,
    float* dC,
    int m, int n, int k) {

  dim3 threads(TILE, TILE);
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

  gemm_tiled_kernel<<<blocks, threads>>>(
      alpha, dA, transposeA, dB, transposeB, beta, dC, m, n, k);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
  printf("CUDA GEMM kernel compiled successfully.\\n");
  return 0;
}
