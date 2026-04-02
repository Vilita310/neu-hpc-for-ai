from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CURSOR_DIR = PROJECT_ROOT / ".cursor"
DEBUG_LOG = CURSOR_DIR / "debug-52bd29.log"


def main() -> int:
    CURSOR_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log = CURSOR_DIR / f"modal-run-{stamp}.log"
    debug_copy = CURSOR_DIR / f"debug-52bd29-{stamp}.log"

    proc = subprocess.Popen(
        ["modal", "run", "modal_app.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    with run_log.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

    code = proc.wait()

    if DEBUG_LOG.exists():
        shutil.copy2(DEBUG_LOG, debug_copy)
        print(f"\nSaved debug log: {debug_copy}")
    else:
        print("\nDebug log file not found after run.")

    print(f"Saved run output: {run_log}")
    print(f"Exit code: {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
