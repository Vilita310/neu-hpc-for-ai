# nano-sglang (Week 10)

This repo is my Week 10 inference-systems assignment based on `nano-sglang`.
The model used here is `Qwen3-0.6B`.

## What I implemented

- `nano_sglang/kv_cache.py`
  - `update()`
  - `get()`
- `nano_sglang/engine.py`
  - `prefill()`
  - `generate()`
- `nano_sglang/scheduler.py`
  - `_decode_running()`
  - `step()`
  - `run_to_completion()`
- `benchmark.py`
  - simple throughput comparison for sequential vs scheduler mode
- `nano_sglang/block_manager.py` (stretch)
  - `allocate()`
  - `free()`

## Run tests

```bash
# local (no GPU required)
pytest tests/test_kv_cache.py -v

# full tests on GPU (Modal)
modal run modal_run.py::test
```

## Quick demo

```bash
# generate one sample
modal run modal_run.py::run

# benchmark throughput
python benchmark.py
```

## References

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [nano-vllm walkthrough](https://neutree.ai/blog/nano-vllm-part-1)
