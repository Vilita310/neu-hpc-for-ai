import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

LOG_PATH = Path(
    os.getenv(
        "DEBUG_LOG_PATH",
        "/Users/caojing/Desktop/INFO7375_high_performance_computing/week_09/.cursor/debug-52bd29.log",
    )
)
SESSION_ID = "52bd29"


def log_debug(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    # region agent log
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # endregion
