
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
  // Matrix dimensions: C = alpha * A * B + beta * C
  // A is m x k, B is k x n, C is m x n
  const int m = 32;
  const int n = 32;
  const int k = 32;
  
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const bool transposeA = false;
  const bool transposeB = false;

  // Allocate host memory
  std::vector<float> h_A(m * k);
  std::vector<float> h_B(k * n);
  std::vector<float> h_C(m * n);

  // Initialize matrices with test data
  for (int i = 0; i < m * k; ++i) {
    h_A[i] = static_cast<float>(i + 1);  // A: 1, 2, 3, ...
  }
  
  for (int i = 0; i < k * n; ++i) {
    h_B[i] = static_cast<float>(i + 1);  // B: 1, 2, 3, ...
  }

  // Initialize C to zeros
  std::fill(h_C.begin(), h_C.end(), 0.0f);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

  // Perform GEMM: C = alpha * A * B + beta * C
  printf("Running GEMM: C = %.1f * A * B + %.1f * C\n", alpha, beta);
  printf("Matrix dimensions: A[%d x %d], B[%d x %d], C[%d x %d]\n", m, k, k, n, m, n);
  
  gemm(alpha, d_A, transposeA, d_B, transposeB, beta, d_C, m, n, k);

  // Copy result back from device to host
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  // Print first few elements of result for verification
  printf("\nFirst 5x5 block of result matrix C:\n");
  for (int i = 0; i < std::min(5, m); ++i) {
    for (int j = 0; j < std::min(5, n); ++j) {
      printf("%8.1f ", h_C[i * n + j]);
    }
    printf("\n");
  }

  // Compute a simple verification: first element should be sum of first row of A * first col of B
  float expected_first = 0.0f;
  for (int i = 0; i < k; ++i) {
    expected_first += h_A[i] * h_B[i * n];
  }
  printf("\nVerification: C[0][0] = %.1f (expected: %.1f)\n", h_C[0], expected_first);

  // Free device memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  printf("\nGEMM demonstration completed successfully!\n");
  return 0;
}
