\
#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// CPU baseline: O = softmax(QK^T/sqrt(d)) V
void naive_attention_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d);

// CPU FlashAttention-2 Algorithm 1 (Section 3.1): tiled + online softmax
void flash_attention_alg1_cpu(const float* Q, const float* K, const float* V,
                              float* O, int N, int d, int Br, int Bc);

// CUDA FlashAttention-2 Algorithm 1 (Section 3.1)
// Returns 0 on success, non-zero on CUDA error.
int flash_attention_alg1_cuda(const float* hQ, const float* hK, const float* hV,
                              float* hO, int N, int d, int Br, int Bc);

#ifdef __cplusplus
}
#endif
