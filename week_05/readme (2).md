# Week 5: FlashAttention with CuTe Layouts

Reimplementation of FlashAttention Algorithm 1 using NVIDIA's CuTe library, deployed on a cloud GPU via Modal.

---

## Files

```
week_05/
├── run_flash_attention.py   # Modal deployment script (contains CUDA source inline)
├── 00_layout.cpp            # CuTe layout basics demo
├── 01_layout.cpp            # Row-major layout + 2D print demo
└── README.md
```

---

## What's in the kernel

`run_flash_attention.py` embeds a complete CUDA kernel that implements FlashAttention Algorithm 1 with the following CuTe-specific features:

- **CuTe Layouts** for all four shared-memory tiles (Q, K, V, S), all row-major
- **Static `Int<N>{}`** dimensions so index math is resolved at compile time
- **`make_tensor(make_smem_ptr(...), layout)`** to access shared memory via 2D coordinates instead of flat offsets
- **Online softmax** — running `m` and `l` accumulators updated per KV tile, no full N×N matrix ever materialized

Grid is `(num_heads, batch)`, block is `Br` threads, one thread per query row.
Default config: `Br = Bc = 32`, `head_dim d = 64`, tested with `N = 128`.

---

## Requirements

```bash
pip install modal
python3 -m modal setup    # one-time browser login
```

No local CUDA install needed — everything compiles and runs remotely.

---

## Running

```bash
# Run FlashAttention on a cloud GPU (A10G)
modal run run_flash_attention.py
```

Expected output:
```
[compile] nvcc -O2 -std=c++17 -I/cutlass/include --expt-relaxed-constexpr -arch=sm_86 fa.cu -o fa
[run] ./fa
Max error vs CPU reference: 0.000000  [PASS]
```

The first run builds the Docker image (downloads CUTLASS, ~21s). Subsequent runs use the cached image and go straight to compilation.

---

## Layout Demos

`00_layout.cpp` and `01_layout.cpp` are standalone C++ files that demonstrate CuTe layout construction. They only need the CUTLASS headers to compile — no GPU required.

```bash
# If you have CUTLASS headers locally:
g++ -std=c++17 -I/path/to/cutlass/include 00_layout.cpp -o 00_layout && ./00_layout
g++ -std=c++17 -I/path/to/cutlass/include 01_layout.cpp -o 01_layout && ./01_layout
```

**`00_layout` output:**
```
_8:_1
8:1
(_2,_4):(_1,_2)
```

**`01_layout` output:**
```
(_2,_4):(_4,_1)
  0   1   2   3
  4   5   6   7
```

---

## GPU target

The compile flag `-arch=sm_86` targets the **A10G** (Ampere). Change this if Modal assigns a different GPU:

| GPU | arch flag |
|-----|-----------|
| A10G | `sm_86` |
| A100 | `sm_80` |
| H100 | `sm_90` |

---

## Correctness

A naive CPU reference (standard O(N²) attention) is included in the binary and compared against the GPU output. Max absolute error across all output elements was **0.000000** at float32 precision.
