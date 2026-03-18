# Week 7 Assignment: DeepSeekV3 MoE Operator in Pure C

## Overview

This week's assignment was to read the DeepSeekMoE paper, generate test cases for each block of the DeepSeekV3 MoE operator using HuggingFace Transformers as a reference, implement the full operator in pure C with no CUDA and no parallelism, and verify that the C implementation passes all generated test cases. Everything was deployed and run on a cloud GPU via Modal.

---

## 1. Reading the Paper

DeepSeekMoE (Dai et al., 2024) proposes two main improvements over conventional MoE architectures like GShard:

**Fine-grained expert segmentation** — instead of a small number of large experts, each expert FFN is split into $m$ smaller ones with $\frac{1}{m}$ the intermediate dimension. The number of activated experts is scaled up by the same factor $m$ to keep compute constant. This dramatically increases the number of possible expert combinations — going from $\binom{16}{2} = 120$ with standard top-2 routing to $\binom{64}{8} \approx 4.4 \times 10^9$ with fine-grained segmentation — which allows each expert to specialize more narrowly.

**Shared expert isolation** — $K_s$ experts are always activated regardless of routing, intended to capture common knowledge that would otherwise be redundantly learned by multiple routed experts. The number of activated routed experts is reduced by $K_s$ to keep compute the same.

The full MoE layer output (Equation 9 in the paper) is:

$$h_t^l = \sum_{i=1}^{K_s} \text{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t} \cdot \text{FFN}_i(u_t^l) + u_t^l$$

DeepSeekV3 uses 1 shared expert, 256 routed experts with top-8 selection, and a hidden dimension of 7168. For this assignment I used scaled-down versions (hidden=16, 8 routed experts, top-3) to keep test cases small.

---

## 2. Test Case Generation

The generator is in `generate_tests.py` (embedded inside `run_moe.py` for Modal deployment). It produces test cases for three blocks independently.

### Design decisions

**Separate seeds for weights and inputs.** Weight initialization uses `WEIGHT_SEED=42`, input generation uses `INPUT_SEED=99`. This way, if I change the network architecture (e.g., the number of experts), the inputs don't change, and vice versa — each seed controls exactly one thing.

**Determinism check.** The generator runs twice and asserts both outputs are identical before writing the JSON file. This catches any accidental non-determinism from PyTorch operations.

**Small fake weights.** The real DeepSeekV3 weights are hundreds of GB. The generator uses randomly initialized weights at a small scale, which is fine since the goal is just to verify mathematical correctness of the C implementation against the Python reference.

### Block 1: DeepseekV3MLP

A single FFN expert with SiLU gating:
```
output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
```
Weights saved: `gate_proj [IE×H]`, `up_proj [IE×H]`, `down_proj [H×IE]`.

### Block 2: DeepseekV3TopKRouter

Computes routing scores and selects top-K experts per token:
```
scores     = softmax( x @ W_router^T )         # [T, n_routed_experts]
topk_idx, topk_weight = TopK(scores, K)
topk_weight = normalize(topk_weight) × routed_scaling_factor
```
Saved: router weight matrix, input, topk_idx, topk_weight.

### Block 3: DeepseekV3MoE

Full MoE layer combining shared and routed experts. The output for each token is the sum of all shared expert outputs plus the weighted sum of the selected routed expert outputs.

---

## 3. Pure C Implementation

The C implementation (`moe.c`) has no external dependencies beyond `libc` and `libm`. It implements the same three blocks.

### MLP forward pass

```c
// output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
linear(gate, x, g, IE, H);
linear(up,   x, u, IE, H);
for (int i = 0; i < IE; i++)
    act[i] = silu(g[i]) * u[i];   // SiLU: x / (1 + exp(-x))
linear(down, act, out, H, IE);
```

### Router forward pass

For each token: compute softmax scores over all routed experts, select top-K by a simple selection sort (fine for small K), normalize the selected weights, then multiply by `routed_scaling_factor`.

One thing worth noting: the order in which `torch.topk` returns indices is not guaranteed to match the C selection sort order. So the test compares router outputs by matching on expert ID rather than position — for each expert ID that the C code selected, find the same ID in the reference output and compare the weight value.

### MoE forward pass

Loops over tokens, runs all shared experts unconditionally, calls the router to get top-K indices and weights, then loops over the K selected routed experts and accumulates their weighted outputs.

### JSON parsing

Rather than pulling in a JSON library, I wrote a minimal recursive parser for the subset of JSON the test cases use (nested float arrays and int arrays). It's ~80 lines of C and handles everything needed without any dependencies.

---

## 4. Running on Modal

The full pipeline runs in a single Modal function on an A10G GPU:

```python
@app.function(gpu="A10G", timeout=300)
def run_moe_tests():
    # Step 1: python gen.py  -> test_cases.json
    # Step 2: gcc -O2 moe.c -o moe_test -lm
    # Step 3: ./moe_test test_cases.json
```

The image is built from `nvidia/cuda:12.4.1-devel-ubuntu22.04` with `gcc` and PyTorch installed. Image build took ~198 seconds on first run (mostly downloading PyTorch); subsequent runs use the cached image.

---

## 5. Results

Run on Modal A10G (sm_86), CUDA 12.4.1, torch 2.10.0:

```
=======================================================
Step 1: Generating test cases (PyTorch)
=======================================================
Wrote 3 test cases -> /tmp/tmpxsadbaee/test_cases.json

=======================================================
Step 2: Compiling moe.c
=======================================================
[cmd] gcc -O2 -std=c11 /tmp/tmpxsadbaee/moe.c -o /tmp/tmpxsadbaee/moe_test -lm
Compilation successful.

=======================================================
Step 3: Running C tests
=======================================================
=== DeepSeekV3 MoE Operator Tests ===

[Block 1] DeepseekV3MLP
  MLP                              PASS  (max_err=4.47e-08)

[Block 2] DeepseekV3TopKRouter
  Router                           PASS  (max_w_err=8.94e-08)

[Block 3] DeepseekV3MoE
  MoE                              PASS  (max_err=1.34e-07)

=== Results: 3 / 3 passed ===
```

All three blocks pass with max absolute errors well below the tolerance of `1e-4`. The errors are in the range of single-precision floating point rounding (~1e-7 to 1e-8), which is expected when comparing C `float` arithmetic against PyTorch's float32.

---

## 6. Summary

| Item | Status |
|------|--------|
| DeepSeekMoE paper read | ✅ |
| Test case generator (3 blocks) | ✅ Deterministic, seed-separated |
| Pure C MoE implementation | ✅ No CUDA, no parallelism, no external libs |
| Block 1: MLP | ✅ max_err = 4.47e-08 |
| Block 2: Router | ✅ max_err = 8.94e-08 |
| Block 3: Full MoE | ✅ max_err = 1.34e-07 |
| Modal deployment | ✅ A10G, CUDA 12.4 |

The main challenge was the router test — since top-K doesn't guarantee a consistent ordering of returned indices, a naive element-wise comparison of `topk_idx` would fail even if the C code is correct. Matching by expert ID rather than position fixed this.
