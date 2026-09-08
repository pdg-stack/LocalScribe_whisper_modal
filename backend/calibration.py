"""Self-calibrating estimate store.

No generic public benchmark table for Whisper throughput per (model, GPU)
combination is fully reliable -- exact speed depends on hardware, driver,
batch size, and current load. The reliable fix is to record the *actual*
measured values from real completed jobs (which the pipeline already
times precisely) and prefer them going forward: estimates get more
accurate the more a given (model, device) combination is actually used,
and reflect this machine/account's real performance rather than someone
else's benchmark.

Two kinds of value are tracked, both as a running average keyed by
(model, device):
- "rtf" -- transcription real-time-factor (seconds of processing per
  second of audio), used to scale with file duration.
- "setup" -- one-time model load/install duration in seconds (roughly
  constant regardless of audio length).

Stored at project root as rtf_calibration.json (gitignored -- it's local
performance data, not something to commit; a different machine or Modal
account would see different real numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "rtf_calibration.json"

# Cap how much influence accumulated history has on the running average,
# so the estimate stays responsive to recent conditions (e.g. a driver
# update, a different Modal region) rather than being dragged down by
# many old samples.
MAX_SAMPLE_WEIGHT = 19


def _key(kind: str, model: str, device: str) -> str:
    return f"{kind}::{model}::{device}"


def _load() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get(kind: str, model: str, device: str) -> float | None:
    entry = _load().get(_key(kind, model, device))
    return entry["avg"] if entry else None


def _record(kind: str, model: str, device: str, value: float) -> None:
    if value <= 0:
        return
    data = _load()
    key = _key(kind, model, device)
    entry = data.get(key, {"avg": value, "samples": 0})
    weight = min(entry["samples"], MAX_SAMPLE_WEIGHT)
    entry["avg"] = (entry["avg"] * weight + value) / (weight + 1)
    entry["samples"] += 1
    data[key] = entry
    _save(data)


def get_rtf(model: str, device: str) -> float | None:
    return _get("rtf", model, device)


def record_rtf_sample(model: str, device: str, rtf: float) -> None:
    _record("rtf", model, device, rtf)


def get_setup_sec(model: str, device: str) -> float | None:
    return _get("setup", model, device)


def record_setup_sample(model: str, device: str, seconds: float) -> None:
    _record("setup", model, device, seconds)
