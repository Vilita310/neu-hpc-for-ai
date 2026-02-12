# Week 04 — FlashAttention-2: Algorithm 1 (CPU + CUDA)

Implementation of **Algorithm 1** from [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Section 3.1).

## Overview

Standard attention computes $O = \text{softmax}(QK^T / \sqrt{d}) \, V$ where $Q, K, V, O \in \mathbb{R}^{N \times d}$. This requires materializing the full $N \times N$ attention matrix, which is memory-intensive for large $N$.

FlashAttention avoids this by **tiling** the computation and using an **online softmax** trick to incrementally update the output without ever storing the full attention matrix. Each thread block processes one tile of $Q$ (size $B_r \times d$) and iterates over all tiles of $K, V$ (size $B_c \times d$), accumulating partial results in shared memory.

## Project Structure

```
week_04/
├── app.py                  # Modal deployment script (inline sources + build + run)
├── CMakeLists.txt
├── include/
│   └── flashattn.h         # C/CUDA function declarations
├── src/
│   ├── flashattn_cpu.cpp   # CPU naive attention + CPU FlashAttention Alg1
│   └── flashattn_cuda.cu   # CUDA FlashAttention Alg1 kernel
├── tests/
│   └── test_main.cpp       # Test harness with correctness checks
├── report.md               # Auto-generated run report
└── README.md
```

## Implementations

### 1. CPU Naive Attention (`naive_attention_cpu`)

Baseline implementation that materializes the full attention matrix: computes all $N^2$ dot products, applies row-wise softmax, then multiplies by $V$. Used as the ground truth for correctness validation.

### 2. CPU FlashAttention Alg1 (`flash_attention_alg1_cpu`)

Sequential C implementation of FlashAttention-2 Algorithm 1:
- Outer loop over Q tiles of size $B_r$
- Inner loop over K/V tiles of size $B_c$
- For each (Q tile, KV tile) pair:
  - Compute block logits $S_{ij} = Q_i K_j^T / \sqrt{d}$
  - Find block max $m_{ij}$ and compute $P_{ij} = \exp(S_{ij} - m_{ij})$
  - Online update: rescale running accumulator with $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ and new block contribution with $\beta = \exp(m_{ij} - m_{\text{new}})$
- Final normalization: $O_i = O_{\text{acc}} / l_i$

### 3. CUDA FlashAttention Alg1 (`flash_attention_alg1_cuda`)

Parallelized GPU implementation following the same algorithm:
- **Grid**: one thread block per Q tile ($T_r = \lceil N / B_r \rceil$ blocks)
- **Block**: `dim3(Tx, Br)` where $T_x \times B_r \leq 1024$
  - `threadIdx.y` (r) indexes the row within the Q tile
  - `threadIdx.x` (tx) parallelizes across K/V columns and output dimensions
- **Shared memory layout**: $Q_i$, $K_j$, $V_j$, $O_{\text{acc}}$, $m_i$, $l_i$, $P_{ij}$, reduction scratch
- **Row reductions** (max, sum) use shared memory scratch space with serial scan by thread 0

## Build & Run

### Prerequisites

- [Modal](https://modal.com/) account with GPU access
- Python 3.9+ with `modal` package installed

### Running on Modal

```bash
pip install modal
modal setup          # one-time authentication
modal run app.py     # build + run on remote GPU
```

This will compile the project inside a `nvidia/cuda:12.3.2-devel-ubuntu22.04` container, run the test, and generate `report.md` locally.

### Launch Configuration (N=256, d=64, Br=64, Bc=64)

| Parameter | Value |
|-----------|-------|
| Grid | 4 blocks |
| Block | (16, 64) = 1024 threads |
| Shared memory | 86,528 bytes |

## Results

```
Config: N=256 d=64 Br=64 Bc=64 seed=0
[CPU Flash vs Naive] max_abs=1.04308e-07  rel_l2=4.07452e-07  time_ms=3.220
[CUDA Flash vs Naive] max_abs=9.68575e-08  rel_l2=4.06583e-07  time_ms=1.727
Naive_time_ms=3.309
RESULT: PASS
```

Both CPU and CUDA FlashAttention implementations match the naive baseline within tolerance ($5 \times 10^{-3}$), with actual errors on the order of $10^{-7}$.

## References

1. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018
2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
3. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023