"""JSON-line process logs under logs/ (gitignored, private -- never
committed). One file per job, with full exception detail for
troubleshooting -- the UI only ever shows the short friendly message.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_job_logger(job_id: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(f"localscribe.job.{job_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Local-time prefix so files sort chronologically and the most
        # recent run is obvious at a glance; the short job_id suffix keeps
        # filenames unique without needing the full UUID.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{stamp}_{job_id[:8]}.jsonl"
        handler = logging.FileHandler(LOGS_DIR / filename, encoding="utf-8")
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
