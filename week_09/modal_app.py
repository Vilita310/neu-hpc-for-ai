import modal
from pathlib import Path

app = modal.App("deepseek-moe-thunderkittens")
PROJECT_ROOT = Path(__file__).resolve().parent

SNAPSHOT_FILES = [
    "benchmark.py",
    "src/__init__.py",
    "src/debug_log.py",
    "src/deepseek_moe.py",
    "src/moe_reference.py",
    "src/moe_router.py",
    "src/tk_backend.py",
    "src/tk_moe_wrapper.cpp",
    "src/tk_moe_kernel.cu",
    "tests/test_router.py",
    "tests/test_expert.py",
    "tests/test_moe.py",
    "tests/conftest.py",
]
def build_snapshot() -> dict[str, str]:
    return {
        rel: (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for rel in SNAPSHOT_FILES
        if (PROJECT_ROOT / rel).exists()
    }

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "build-essential")
    .pip_install("torch", "numpy", "pytest", "ninja")
    .run_commands(
        "if [ ! -d /opt/thunderkittens ]; then git clone https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens; fi"
    )
)

@app.function(
    gpu="B200",
    image=image,
    timeout=3600,
)
def run(file_snapshot: dict[str, str], run_instance_id: str):
    import os
    import subprocess
    from pathlib import Path

    os.environ["THUNDERKITTENS_HOME"] = "/opt/thunderkittens"
    os.environ["REQUIRE_TK_EXTENSION"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
    os.environ["DEBUG_RUN_ID"] = run_instance_id
    os.environ["DEBUG_LOG_PATH"] = "/tmp/debug-week9.log"

    project_dir = Path("/root/week_09")
    for rel, content in file_snapshot.items():
        dst = project_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
    subprocess.run(["python", "benchmark.py"], check=True, cwd=str(project_dir))
    subprocess.run(["pytest", "-q", "tests"], check=True, cwd=str(project_dir))
    print(f"RUN_INSTANCE_ID: {run_instance_id}")
    print("RUN_STATUS: SUCCESS")


@app.local_entrypoint()
def main():
    import uuid

    snapshot = build_snapshot()
    run_instance_id = f"run-{uuid.uuid4().hex[:10]}"
    print(f"LOCAL_RUN_INSTANCE_ID: {run_instance_id}")
    run.remote(snapshot, run_instance_id)