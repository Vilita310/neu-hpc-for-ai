# Week 5 Assignment: FlashAttention Algorithm 1 with CuTe Layouts

## Overview

This week's assignment was to rewrite our FlashAttention Algorithm 1 kernel using NVIDIA's CuTe library. CuTe is part of the CUTLASS framework and gives you a cleaner way to express how data is laid out in memory — instead of computing raw byte offsets by hand, you define a Layout and let the library handle the indexing. The final kernel was compiled and run on a cloud GPU using Modal.

---

## 1. CuTe Layouts: The Core Idea

The central abstraction in CuTe is a `Layout`, which is just a pair of `(Shape, Stride)`. Given a multidimensional coordinate, it computes a 1-D memory index by taking the inner product of the coordinate with the stride vector:

$$\text{idx}(c_0, c_1, \ldots) = c_0 \cdot s_0 + c_1 \cdot s_1 + \cdots$$

So for a row-major 2×4 matrix, the layout is `(2,4):(4,1)`:
- Shape `(2,4)` means 2 rows, 4 columns
- Stride `(4,1)` means advancing one row skips 4 elements; advancing one column skips 1
- `layout(1, 2) = 1×4 + 2×1 = 6` — which is exactly where you'd expect element (1,2) to live in a row-major array

A `Tensor` wraps a layout together with an actual pointer, so you can access elements by coordinate without writing any index math yourself.

What makes this useful in practice is the `Int<N>{}` syntax for static values. When tile dimensions are compile-time constants, the compiler can resolve every index calculation at compile time and emit optimal PTX — no runtime overhead.

---

## 2. Implementation

### 2.1 Shared Memory Tiles

The kernel uses four shared memory tiles, all laid out row-major:

| Tile | Shape | Stride | What it holds |
|------|-------|--------|---------------|
| `tQ` | `(Br, d)` | `(d, 1)` | Query tile |
| `tK` | `(Bc, d)` | `(d, 1)` | Key tile |
| `tV` | `(Bc, d)` | `(d, 1)` | Value tile |
| `tS` | `(Br, Bc)` | `(Bc, 1)` | Attention scores |

Creating them looks like this:

```cpp
auto Q_layout = make_layout(make_shape (Int<Br>{}, Int<kD>{}),
                             make_stride(Int<kD>{}, Int<1>{}));
auto tQ = make_tensor(make_smem_ptr(smem_Q), Q_layout);
```

After this, `tQ(row, col)` compiles down to `smem_Q[row * kD + col]` — same as writing the index by hand, but the code reads like you're thinking in 2D coordinates rather than flat offsets.

### 2.2 Kernel Configuration

```
Grid:   (num_heads, batch)   — one CTA per (batch, head) pair
Block:  Br threads           — one thread per query row in the tile
Params: Br = 32, Bc = 32, head_dim d = 64
```

### 2.3 Algorithm 1, Step by Step

The structure follows Algorithm 1 from the FlashAttention paper directly:

```
outer loop over query tiles i = 0..Tr-1:
    init: m_i = -inf, l_i = 0, O_acc = 0

    load Q_i from global → smem via tQ(tid, d)

    inner loop over KV tiles j = 0..Tc-1:
        load K_j, V_j from global → smem via tK, tV

        // each thread computes its own row of S
        S[tid, c] = scale * dot(tQ[tid, :], tK[c, :])

        // online softmax update
        m_new = max(m_i, rowmax(S[tid, :]))
        l_new = exp(m_i - m_new)*l_i + exp(m_ij - m_new)*l_ij
        O_acc = [exp(m_i - m_new)*l_i*O_acc + exp(S - m_new)*V] / l_new

        m_i = m_new,  l_i = l_new

    write O_acc to global memory
```

The online softmax part is the trickiest piece — you have to rescale the running `O_acc` at each inner iteration so that the final result is correct without ever building the full N×N score matrix. That's what keeps FlashAttention memory-efficient.

The actual matrix multiply in the kernel uses CuTe tensor indexing rather than cuBLAS, keeping everything in shared memory:

```cpp
if (tid < Br) {
    for (int c = 0; c < Bc; ++c) {
        float dot = 0.f;
        for (int d = 0; d < kD; ++d)
            dot += tQ(tid, d) * tK(c, d);
        tS(tid, c) = scale * dot;
    }
}
```

---

## 3. Correctness Check

To verify the output, I wrote a naive CPU reference (standard O(N²) attention with regular softmax) and compared it against the GPU result element-by-element.

**Test setup:** B=1, H=1, N=128, d=64, random inputs in [-0.1, 0.1]

**Result:**
```
Max error vs CPU reference: 0.000000  [PASS]
```

Zero error — the online softmax accumulation matches the batch softmax exactly at float32 precision.

---

## 4. Modal Deployment

### Environment

The kernel needs CUTLASS headers (that's where CuTe lives), so the Modal image clones the CUTLASS repo at build time:

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04")
    .apt_install("git", "cmake")
    .run_commands("git clone --depth 1 https://github.com/NVIDIA/cutlass.git /cutlass")
)
```

Image build took about 21 seconds the first time; Modal caches it afterward so later runs skip straight to compilation.

The function itself writes the CUDA source to a temp file, compiles with `nvcc`, and runs the binary:

```python
@app.function(gpu="A10G", timeout=300)
def run_flash_attention():
    # nvcc -O2 -std=c++17 -I/cutlass/include --expt-relaxed-constexpr -arch=sm_86
    ...
```

`-arch=sm_86` targets the A10G (Ampere). If Modal assigns an A100 instead, you'd want `sm_80`; H100 would be `sm_90`.

### Run Output

```
Built image im-RJf6Yc1uXmLoJeo8OG4NwZ in 21.17s

[compile] nvcc -O2 -std=c++17 -I/cutlass/include \
          --expt-relaxed-constexpr -arch=sm_86 fa.cu -o fa
[run] ./fa
Max error vs CPU reference: 0.000000  [PASS]

✓ App completed.
```

---

## 5. Layout Demos (00 and 01)

**`00_layout.cpp`** shows how to construct basic layouts:

```cpp
Layout s8    = make_layout(Int<8>{});
Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
// prints: (_2,_4):(_1,_2)
```

The default stride is column-major — `(_1,_2)` means stride-1 along rows, stride-2 along columns.

**`01_layout.cpp`** overrides this with a row-major stride and prints the layout as a 2D grid:

```cpp
auto row_major = make_stride(Int<4>{}, Int<1>{});
Layout s2xs4  = make_layout(make_shape(Int<2>{}, Int<4>{}), row_major);
```

```
  0   1   2   3
  4   5   6   7
```

Consecutive columns are adjacent (`+1`), rows are separated by 4 — that's row-major, matching a C-style 2D array.

---

## 6. Summary

The main takeaway from this assignment is how CuTe's Layout abstraction separates the *logical* view of a tensor (rows and columns) from the *physical* memory layout (strides). This made the FlashAttention kernel noticeably easier to read compared to the raw-pointer version — `tQ(row, col)` is a lot clearer than `smem_Q[row * kD + col]` scattered throughout the inner loops, especially once you have four different tiles with different shapes.

| Item | Status |
|------|--------|
| CuTe layouts for all shared-memory tiles | ✅ |
| Static `Int<N>{}` dimensions | ✅ |
| FlashAttention Algorithm 1 with online softmax | ✅ |
| Correctness vs CPU reference | ✅ Max error = 0.000000 |
| Modal deployment on A10G | ✅ |
