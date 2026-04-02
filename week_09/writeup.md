# Week 9 Writeup: Sequence Parallelism

## 1) What I implemented

I restructured a DeepSeek-style MoE into two execution paths:

- **Reference path** (`forward_reference`): token-by-token, expert-by-expert accumulation.
- **Grouped path** (`forward_grouped`): flatten selected `(token, expert)` pairs and compute with batched GEMM, then scatter-add to token outputs.

The grouped path is the software layout that maps naturally to ThunderKittens kernels (WMMA MMA loops + TMA-fed shared-memory tiles).

## 2) Blockwise Parallel Transformer (diagram explanation)

The blockwise diagrams show a nested-loop view:

- Outer loop iterates over **query blocks** `Q_i`.
- Inner loop streams over **key/value blocks** `(K_j, V_j)`.
- For one `Q_i`, we accumulate numerically stable partial softmax stats (`max`, `sumexp`, weighted value sum) across all `j`.
- Once attention output for `Q_i` is complete, we **immediately** apply FFN/MoE for that same block.

Why memory drops:

- Vanilla attention materializes full `[seq, d]` attention outputs for the whole sequence before FFN.
- Blockwise execution keeps only per-block temporary states in SRAM/registers, so peak activation memory scales with block size rather than full sequence length.

## 3) Correct blockwise computation definition

Given a query block `Q_i` and streamed KV blocks:

\[
S_{ij} = Q_i K_j^\top,\quad m_i = \max_j \max(S_{ij})
\]

\[
P_{ij} = \exp(S_{ij} - m_i),\quad l_i = \sum_j \sum P_{ij}
\]

\[
O_i = \frac{1}{l_i}\sum_j P_{ij}V_j
\]

This is equivalent to global softmax attention but computed by blockwise accumulation with stable running max/sum.

Additional reference for this definition:

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (online softmax and blockwise accumulation principle used by later blockwise/ring methods).

## 4) Ring Attention understanding

Ring Attention distributes long sequence context across devices:

- Each device owns one (or a shard of) query block(s).
- KV blocks circulate device-to-device in a ring.
- At each hop, a device computes attention partials for local queries with current KV block and overlaps:
  - **communication** (sending current KV onward, receiving next KV),
  - **computation** (attention + FFN/MoE update on current chunk).

After one full ring, each device has consumed all KV context, equivalent to global attention, without approximating attention and with high overlap.

## 5) ThunderKittens design decisions

For the target B200 path:

- **WMMA/tensor cores** perform tile MMA on bf16 operands.
- **TMA** handles global<->shared tile transfers asynchronously, reducing SM thread involvement in memory movement.
- **Grouped expert compute** is chosen so `(token, expert)` workloads can be packed into contiguous tile-friendly GEMMs.

In this repo:

- Python side already separates reference and grouped paths.
- `tk_moe_ext` hook is integrated via `src/tk_backend.py`.
- When extension is unavailable, code falls back to PyTorch matmul to keep tests reproducible.

## 6) Benchmark and correctness

- `benchmark.py` first checks reference-vs-grouped parity (`max_abs_err` threshold).
- Then reports median latencies and grouped speedup.
- Also reports whether `tk_moe_ext` was loaded.

Testcases:

- Router invariants: shape, valid expert IDs, gate probabilities sum to 1.
- Expert parity: extension/fallback output equals torch baseline.
- MoE parity: grouped path matches reference path.

## 7) What to demo live

1. `pytest -q tests`
2. `python benchmark.py`
3. Explain report fields and whether TK extension loaded
4. Walk through grouped path and why it maps to WMMA+TMA kernels
