# Week 8: Multi-GPU DeepSeekV3 MoE with NCCL

Multi-GPU implementation of the DeepSeekV3 Mixture-of-Experts operator in CUDA using NCCL, with data parallelism and expert parallelism.

## How to Run

```bash
pip install modal
python3 -m modal setup       # one-time auth
modal run modal_moe_nccl.py
```

This will run on 2x A10G GPUs on Modal. The script does everything automatically:
1. Generates test cases using a PyTorch reference model
2. Compiles the CUDA + NCCL code
3. Runs correctness tests on both small and large configs
4. Runs a PyTorch benchmark for performance comparison

## Project Structure

Everything is in a single file `modal_moe_nccl.py` for easy deployment:

- **GENERATE_PY** — PyTorch reference implementation that generates test cases (weights, inputs, expected outputs) as JSON files
- **MOE_NCCL_CU** — The actual CUDA + NCCL implementation. This is the main deliverable. Written in C/CUDA with NCCL for inter-GPU communication.
- **BENCHMARK_PY** — PyTorch benchmark script for performance comparison
- **Modal wrapper** — Builds, runs, and reports everything on remote GPUs

## Parallelism Strategy

- **Data parallelism**: Input tokens are split evenly across GPUs. Each GPU processes `T/W` tokens (T = total tokens, W = number of GPUs).
- **Expert parallelism**: The routed experts are partitioned across GPUs. Each GPU owns `NR/W` experts (NR = total routed experts).
- **Replicated components**: The router and shared experts are copied to every GPU since they need to process all local tokens.

## Forward Pass (6 Phases)

The distributed forward follows the pipeline from lecture:

1. **Shared experts** — Each GPU runs shared experts on its local tokens. No communication needed.
2. **Routing** — Each GPU runs the replicated router on its local tokens to get top-K expert assignments and weights.
3. **Permutation (all-to-all)** — Tokens are sent to the GPU that owns the assigned expert. Uses NCCL send/recv to implement all-to-all.
4. **Expert computation** — Each GPU runs its local experts on the tokens it received.
5. **Un-permutation (all-to-all)** — Results are sent back to the originating GPU. Another round of NCCL send/recv.
6. **Scaling & combination** — Final output = shared expert output + weighted sum of routed expert outputs.

## Test Configs

| Config | H | IE | NR | K | NS | T | Purpose |
|--------|---|----|----|---|----|---|---------|
| small | 16 | 8 | 8 | 3 | 1 | 4 | Quick sanity check |
| large | 128 | 64 | 8 | 3 | 2 | 256 | Correctness + perf |
| xlarge | 512 | 256 | 16 | 4 | 2 | 1024 | PyTorch benchmark only |

## Results

<!-- TODO: replace with actual output after running -->

Correctness (2x A10G):
```
--- small ---
GPUs available: 2, using: 2
Config: H=16 IE=8 NR=8 NS=1 K=3 T=4 scale=2.50
  GPU 0: max_err=<FILL_IN>  time=<FILL_IN> ms
  GPU 1: max_err=<FILL_IN>  time=<FILL_IN> ms
PASS

--- large ---
GPUs available: 2, using: 2
Config: H=128 IE=64 NR=8 NS=2 K=3 T=256 scale=2.50
  GPU 0: max_err=<FILL_IN>  time=<FILL_IN> ms
  GPU 1: max_err=<FILL_IN>  time=<FILL_IN> ms
PASS
```

PyTorch benchmark:
```
  small   : <FILL_IN> ms  (T=4, H=16, NR=8)
  large   : <FILL_IN> ms  (T=256, H=128, NR=8)
  xlarge  : <FILL_IN> ms  (T=1024, H=512, NR=16)
```
