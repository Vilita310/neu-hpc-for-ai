# Week 9: Sequence Parallelism - DeepSeekMoE + ThunderKittens

This repository contains a Week 9 submission scaffold for:

- DeepSeek-style top-k MoE routing
- Grouped expert compute path aligned with blockwise execution
- ThunderKittens extension hook (`tk_moe_ext`) for WMMA/TMA acceleration
- Performance comparison and correctness tests

## Project structure

- `src/moe_router.py`: top-k router and softmax gate weights
- `src/deepseek_moe.py`: reference MoE and grouped compute path
- `src/tk_backend.py`: ThunderKittens extension integration with fallback
- `src/tk_moe_kernel.cu`: CUDA kernel source placeholder for TK path
- `src/tk_moe_wrapper.cpp`: torch extension wrapper entrypoint
- `benchmark.py`: benchmark + parity check + speedup report
- `tests/`: router/expert/MoE parity tests
- `modal_app.py`: remote run target on B200
- `writeup.md`: theory and design discussion

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run benchmark locally

```bash
python benchmark.py
```

The script prints JSON with:

- `baseline_reference_median_ms`
- `grouped_median_ms`
- `grouped_speedup_vs_baseline`
- `tk_matmul_median_ms`
- `tk_extension_loaded`
- `max_abs_err_reference_vs_grouped`

Note:

- On CPU-only local machines, `tk_extension_loaded` can be `false` by design.
- On B200 Modal run, `REQUIRE_TK_EXTENSION=1` is enabled, so the run fails fast if TK extension is not loaded.

## Run tests

```bash
pytest -q tests
```

Covered testcases:

- Router shape and probability-sum invariants
- Expert matmul parity (TK extension or fallback path vs torch)
- MoE reference path parity vs grouped path

## Run on Modal (B200)

```bash
python run_modal_snapshot.py
```

This runs `modal run modal_app.py` from an immutable temporary snapshot of the project to avoid local file-change races during image build.
