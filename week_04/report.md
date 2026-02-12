# Week 04 — FlashAttention (CPU + CUDA)

**Run timestamp:** 2026-02-12T03:58:45.397907Z
**Config:** N=256, d=64, Br=64, Bc=64, seed=0
**Total pipeline time (build+run):** 9.20 s

## Build
- Base image: nvidia/cuda:12.3.2-devel-ubuntu22.04
- Toolchain: CMake + Ninja, C++17, CUDA

## Correctness & Runtime Output

```text
Config: N=256 d=64 Br=64 Bc=64 seed=0
[CPU Flash vs Naive] max_abs=1.04308e-07  rel_l2=4.07452e-07  time_ms=3.220
Launch config: grid=(4), block=(16,64), smem=86528 bytes
Launch config: grid=(4), block=(16,64), smem=86528 bytes
[CUDA Flash vs Naive] max_abs=9.68575e-08  rel_l2=4.06583e-07  time_ms=1.727
Naive_time_ms=3.309
RESULT: PASS
```

## Notes
- CPU implementation: naive softmax attention + FlashAttention alg1 online update.
- CUDA kernel: tiled Q (Br rows) and KV (Bc cols) with shared memory; online softmax update (mi/li) per row.
- PASS condition: CPU Flash and CUDA Flash both close to naive within tolerance (tol=5e-3).
