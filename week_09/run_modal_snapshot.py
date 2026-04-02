from __future__ import annotations

import shutil
import subprocess
import tempfile
import json
import time
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDE_NAMES = {
    ".git",
    ".cursor",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
}


def _ignore_filter(_: str, names: list[str]) -> set[str]:
    ignored = {n for n in names if n in EXCLUDE_NAMES}
    ignored.update({n for n in names if n.endswith(".pyc")})
    return ignored


def main() -> None:
    log_path = PROJECT_ROOT / ".cursor" / "debug-52bd29.log"
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # region agent log
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "52bd29",
                    "id": f"log_{int(time.time() * 1000)}_snapshot_start",
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-fix",
                    "hypothesisId": "H21",
                    "location": "run_modal_snapshot.py:main",
                    "message": "snapshot runner started",
                    "data": {
                        "cwd": str(Path.cwd()),
                        "project_root": str(PROJECT_ROOT),
                        "script_sha256": script_sha,
                    },
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    # endregion

    with tempfile.TemporaryDirectory(prefix="week09_modal_snapshot_") as tmp:
        snapshot_root = Path(tmp) / "week_09"
        shutil.copytree(PROJECT_ROOT, snapshot_root, ignore=_ignore_filter)
        try:
            subprocess.run(
                ["modal", "run", "modal_app.py"],
                check=True,
                cwd=str(snapshot_root),
            )
        except subprocess.CalledProcessError as e:
            # region agent log
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "52bd29",
                            "id": f"log_{int(time.time() * 1000)}_snapshot_fail",
                            "timestamp": int(time.time() * 1000),
                            "runId": "post-fix",
                            "hypothesisId": "H23",
                            "location": "run_modal_snapshot.py:main",
                            "message": "snapshot runner failed",
                            "data": {
                                "returncode": e.returncode,
                            },
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            # endregion
            raise

    # region agent log
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "52bd29",
                    "id": f"log_{int(time.time() * 1000)}_snapshot_end",
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-fix",
                    "hypothesisId": "H22",
                    "location": "run_modal_snapshot.py:main",
                    "message": "snapshot runner finished",
                    "data": {
                        "script_sha256": script_sha,
                    },
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    # endregion


if __name__ == "__main__":
    main()
