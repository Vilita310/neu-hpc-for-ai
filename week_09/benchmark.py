import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from src.debug_log import log_debug
from src.deepseek_moe import DeepSeekMoE
from src.tk_backend import get_tk_extension_status, tk_forward_or_fallback, try_import_tk_extension

RUN_ID = os.getenv("DEBUG_RUN_ID", "pre-fix")


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    median_ms: float


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(fn, warmup: int = 10, iters: int = 40) -> BenchResult:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(timings, dtype=torch.float64)
    return BenchResult(name=fn.__name__, mean_ms=float(t.mean()), median_ms=float(t.median()))


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # region agent log
    log_debug(
        run_id=RUN_ID,
        hypothesis_id="H4",
        location="benchmark.py:main",
        message="benchmark environment",
        data={"device": device, "dtype": str(dtype)},
    )
    # endregion

    if device == "cuda":
        d_model, d_hidden, batch = 1024, 2048, 2048
        warmup, iters = 20, 80
    else:
        d_model, d_hidden, batch = 128, 256, 256
        warmup, iters = 3, 10

    model = DeepSeekMoE(d_model=d_model, d_hidden=d_hidden, num_experts=8, top_k=2).to(device=device, dtype=dtype)
    x = torch.randn(batch, d_model, device=device, dtype=dtype)
    a = torch.randn(d_model, d_model, device=device, dtype=dtype)
    b = torch.randn(d_model, d_model, device=device, dtype=dtype)

    def baseline_path():
        _sync_if_cuda(device)
        _ = model.forward_reference(x)
        _sync_if_cuda(device)

    def grouped_path():
        _sync_if_cuda(device)
        _ = model.forward_grouped(x)
        _sync_if_cuda(device)

    def tk_matmul_path():
        _sync_if_cuda(device)
        _ = tk_forward_or_fallback(a, b)
        _sync_if_cuda(device)

    ref = model.forward_reference(x)
    grp = model.forward_grouped(x)
    max_abs_err = float((ref - grp).abs().max().item())

    # region agent log
    log_debug(
        run_id=RUN_ID,
        hypothesis_id="H5",
        location="benchmark.py:main",
        message="reference parity check",
        data={"max_abs_err": max_abs_err},
    )
    # endregion

    if max_abs_err > 2e-2:
        raise RuntimeError(f"Parity check failed: max_abs_err={max_abs_err}")

    b1 = _bench(baseline_path, warmup=warmup, iters=iters)
    b2 = _bench(grouped_path, warmup=warmup, iters=iters)
    b3 = _bench(tk_matmul_path, warmup=warmup, iters=iters)
    speedup = b1.median_ms / b2.median_ms
    tk_status = get_tk_extension_status()
    tk_ext_loaded = bool(tk_status["loaded"])
    require_tk = os.getenv("REQUIRE_TK_EXTENSION", "0") == "1"

    if require_tk and device == "cuda" and not tk_ext_loaded:
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H10",
            location="benchmark.py:main",
            message="required TK extension missing on CUDA",
            data={"device": device, "tk_ext_loaded": tk_ext_loaded, "require_tk": require_tk},
        )
        # endregion
        raise RuntimeError("REQUIRE_TK_EXTENSION=1 but tk_moe_ext is not loaded on CUDA")

    report = {
        "baseline_reference_median_ms": b1.median_ms,
        "grouped_median_ms": b2.median_ms,
        "grouped_speedup_vs_baseline": speedup,
        "tk_matmul_median_ms": b3.median_ms,
        "tk_extension_loaded": tk_ext_loaded,
        "tk_extension_module": tk_status["module"],
        "tk_extension_build_label": tk_status["build_label"],
        "max_abs_err_reference_vs_grouped": max_abs_err,
        "note": "On B200 with compiled ThunderKittens extension, tk_matmul_path should represent WMMA/TMA acceleration.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
