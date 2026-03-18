# Week 7: DeepSeekV3 MoE Operator

Pure C implementation of the DeepSeekV3 Mixture-of-Experts operator, with test cases generated from a PyTorch reference. Deployed and verified on a cloud GPU via Modal.

---

## Files

```
week_07/
├── run_moe.py      # Modal deployment script (contains all source inline)
└── README.md
```

`run_moe.py` embeds three things:
- `GENERATE_PY` — Python/PyTorch test case generator
- `MOE_C` — pure C MoE implementation
- Modal boilerplate to run the full pipeline on a remote GPU

---

## What it does

Three blocks are tested end-to-end:

| Block | What it computes |
|-------|-----------------|
| `DeepseekV3MLP` | Single FFN expert: `down(SiLU(gate(x)) * up(x))` |
| `DeepseekV3TopKRouter` | Softmax scores → top-K selection → normalize → scale |
| `DeepseekV3MoE` | Shared experts (always on) + routed experts (top-K) |

The pipeline runs in three steps:
1. Generate `test_cases.json` with PyTorch (reference outputs)
2. Compile the C implementation with `gcc`
3. Run the C binary against the JSON test cases and report pass/fail

---

## Requirements

```bash
pip install modal
python3 -m modal setup    # one-time browser login
```

---

## Running

```bash
modal run run_moe.py
```

First run builds the Docker image (~3 min for PyTorch download). Subsequent runs are cached.

---

## Results

Run on Modal A10G, CUDA 12.4.1, torch 2.10.0:

```
[Block 1] DeepseekV3MLP
  MLP                              PASS  (max_err=4.47e-08)

[Block 2] DeepseekV3TopKRouter
  Router                           PASS  (max_w_err=8.94e-08)

[Block 3] DeepseekV3MoE
  MoE                              PASS  (max_err=1.34e-07)

=== Results: 3 / 3 passed ===
```

Errors are in the float32 rounding range (~1e-7 to 1e-8), which is expected.

---

## Config

Small dimensions are used so test cases stay manageable:

| Parameter | This assignment | Real DeepSeekV3 |
|-----------|----------------|-----------------|
| `hidden_size` | 16 | 7168 |
| `moe_intermediate_size` | 8 | 2048 |
| `n_routed_experts` | 8 | 256 |
| `num_experts_per_tok` | 3 | 8 |
| `n_shared_experts` | 1 | 1 |

Weight seed and input seed are kept separate (`WEIGHT_SEED=42`, `INPUT_SEED=99`) so changing one doesn't affect the other. Generation is verified to be deterministic by running twice and comparing outputs.

---

## Note on router testing

`torch.topk` doesn't guarantee a consistent ordering of returned indices when scores are close. The test handles this by matching on expert ID rather than position — for each expert the C code selected, it finds the same ID in the reference and compares the weight value.
