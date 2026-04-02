from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from src.debug_log import log_debug

RUN_ID = os.getenv("DEBUG_RUN_ID", "pre-fix")
_TK_EXT = None
_TK_EXT_INITIALIZED = False
_TK_EXT_BUILD_LABEL = "uninitialized"
_TK_PATH_LOGGED = False


def _try_build_tk_extension():
    global _TK_EXT_BUILD_LABEL
    if not torch.cuda.is_available():
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H6",
            location="src/tk_backend.py:_try_build_tk_extension",
            message="skip build because CUDA unavailable",
            data={"cuda_available": False},
        )
        # endregion
        _TK_EXT_BUILD_LABEL = "skip-cuda-unavailable"
        return None

    if shutil.which("nvcc") is None:
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H7",
            location="src/tk_backend.py:_try_build_tk_extension",
            message="skip build because nvcc missing",
            data={"nvcc_found": False},
        )
        # endregion
        _TK_EXT_BUILD_LABEL = "skip-nvcc-missing"
        return None

    root = Path(__file__).resolve().parents[1]
    cpp_src = root / "src" / "tk_moe_wrapper.cpp"
    cu_src = root / "src" / "tk_moe_kernel.cu"
    try:
        nvcc_v = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        nvcc_v = ""
    dev_name = ""
    dev_cc = ()
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
            dev_cc = torch.cuda.get_device_capability(0)
        except Exception:
            dev_name = ""
            dev_cc = ()
    tk_home = os.getenv("THUNDERKITTENS_HOME", "")
    header_candidates = []
    if tk_home:
        home = Path(tk_home)
        header_candidates = [
            home / "kittens.cuh",
            home / "include" / "kittens.cuh",
            home / "src" / "kittens.cuh",
        ]
    tk_header = next((p for p in header_candidates if p.exists()), None)
    use_tk = tk_header is not None

    # region agent log
    log_debug(
        run_id=RUN_ID,
        hypothesis_id="H17",
        location="src/tk_backend.py:_try_build_tk_extension",
        message="build environment",
        data={
            "torch_cuda_version": str(torch.version.cuda),
            "device_name": dev_name,
            "device_capability": list(dev_cc) if dev_cc else [],
            "torch_cuda_arch_list": os.getenv("TORCH_CUDA_ARCH_LIST", ""),
            "nvcc_version_head": " | ".join(nvcc_v.splitlines()[:4])[:400],
        },
    )
    # endregion

    # region agent log
    log_debug(
        run_id=RUN_ID,
        hypothesis_id="H8",
        location="src/tk_backend.py:_try_build_tk_extension",
        message="build attempt configuration",
        data={
            "cpp_src_exists": cpp_src.exists(),
            "cu_src_exists": cu_src.exists(),
            "tk_home": tk_home,
            "tk_header_path": str(tk_header) if tk_header else "",
            "tk_header_exists": bool(tk_header),
            "use_tk": use_tk,
        },
    )
    # endregion

    if not use_tk:
        _TK_EXT_BUILD_LABEL = "skip-tk-header-missing"
        return None

    try:
        built = load(
            name="tk_moe_ext",
            sources=[str(cpp_src), str(cu_src)],
            with_cuda=True,
            extra_cuda_cflags=["-O3", "-DTK_AVAILABLE", "-std=c++20", "--extended-lambda"],
            extra_include_paths=[str(tk_header.parent)],
            verbose=False,
        )
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H9",
            location="src/tk_backend.py:_try_build_tk_extension",
            message="extension build success",
            data={"module": "tk_moe_ext", "label": "build with TK and CUDA"},
        )
        # endregion
        _TK_EXT_BUILD_LABEL = "build with TK and CUDA"
        return built
    except Exception as e:  # pragma: no cover - env dependent
        err_lines = str(e).splitlines()
        err_short = " | ".join(err_lines[:30])[:8000]
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H9",
            location="src/tk_backend.py:_try_build_tk_extension",
            message="extension build failed",
            data={
                "module": "tk_moe_ext",
                "label": "build with TK and CUDA",
                "error_type": type(e).__name__,
                "error": err_short,
                "error_repr": repr(e)[:1200],
            },
        )
        # endregion
        _TK_EXT_BUILD_LABEL = "failed-tk_moe_ext"
        return None


def try_import_tk_extension():
    global _TK_EXT, _TK_EXT_INITIALIZED, _TK_EXT_BUILD_LABEL
    if _TK_EXT_INITIALIZED:
        return _TK_EXT

    try:
        _TK_EXT = importlib.import_module("tk_moe_ext")
        _TK_EXT_INITIALIZED = True
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H3",
            location="src/tk_backend.py:try_import_tk_extension",
            message="TK extension import success",
            data={"module": "tk_moe_ext"},
        )
        # endregion
        _TK_EXT_BUILD_LABEL = "imported-prebuilt"
        return _TK_EXT
    except Exception as e:  # pragma: no cover - depends on local CUDA/TK env
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H3",
            location="src/tk_backend.py:try_import_tk_extension",
            message="TK extension import failed",
            data={"error_type": type(e).__name__, "error": str(e)},
        )
        # endregion
        _TK_EXT = _try_build_tk_extension()
        _TK_EXT_INITIALIZED = True
        return _TK_EXT


def get_tk_extension_status() -> dict[str, str | bool]:
    ext = try_import_tk_extension()
    return {
        "loaded": ext is not None,
        "module": getattr(ext, "__name__", "") if ext is not None else "",
        "build_label": _TK_EXT_BUILD_LABEL,
    }


def tk_forward_or_fallback(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _TK_PATH_LOGGED
    ext = try_import_tk_extension()
    if not _TK_PATH_LOGGED:
        # region agent log
        log_debug(
            run_id=RUN_ID,
            hypothesis_id="H16",
            location="src/tk_backend.py:tk_forward_or_fallback",
            message="selected execution path",
            data={
                "uses_extension": ext is not None and hasattr(ext, "forward"),
                "module": getattr(ext, "__name__", "") if ext is not None else "",
                "build_label": _TK_EXT_BUILD_LABEL,
            },
        )
        # endregion
        _TK_PATH_LOGGED = True
    if ext is not None and hasattr(ext, "forward"):
        return ext.forward(a, b)
    return torch.matmul(a, b)
