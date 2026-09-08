"""Per-file, per-phase time/cost estimates. Mirrors frontend/app.js's
Phase 1 client-side math (now the single source of truth server-side) --
used both by POST /api/preview and to weight the real pipeline's
progress bar.
"""

from __future__ import annotations

from backend.config import (
    GPU_OPTIONS,
    MODAL_DOWNLOAD_SEC,
    MODAL_SETUP_SEC,
    RTF_LOCAL_CPU,
    RTF_MODAL_GPU,
)

PHASE_ORDER_LOCAL = ["Audio Extraction", "Transcription", "Cleanup"]
PHASE_ORDER_MODAL = [
    "Audio Extraction",
    "Modal.com Setup & Model Install",
    "Transcription",
    "Download Transcript to Local",
    "Cleanup",
]


def _gpu_rate(gpu_id: str | None) -> float:
    gpu = next((g for g in GPU_OPTIONS if g["id"] == gpu_id), None)
    if gpu is None:
        raise ValueError(f"Unknown GPU type: {gpu_id}")
    return gpu["rate"]


def compute_steps(
    files: list[dict],
    model: str,
    execution: str,
    gpu_id: str | None,
    cleanup: bool,
) -> list[dict]:
    """files: [{"path", "type", "duration_sec"}, ...] (as sent by the frontend).
    Returns a flat, file-major ordered step list: each step is
    {"file": <the file dict>, "name": <phase name>, "sec": float, "cost": float}.
    """
    is_modal = execution == "modal"
    rtf_table = RTF_MODAL_GPU if is_modal else RTF_LOCAL_CPU
    rtf = rtf_table[model]
    rate = _gpu_rate(gpu_id) if is_modal else 0.0

    steps: list[dict] = []
    for f in files:
        if f["type"] == "video":
            steps.append({"file": f, "name": "Audio Extraction", "sec": (f["duration_sec"] / 60) * 2, "cost": 0.0})
        if is_modal:
            steps.append({
                "file": f, "name": "Modal.com Setup & Model Install",
                "sec": MODAL_SETUP_SEC, "cost": (MODAL_SETUP_SEC / 3600) * rate,
            })
        transcription_sec = f["duration_sec"] * rtf
        steps.append({
            "file": f, "name": "Transcription",
            "sec": transcription_sec, "cost": (transcription_sec / 3600) * rate if is_modal else 0.0,
        })
        if is_modal:
            steps.append({"file": f, "name": "Download Transcript to Local", "sec": MODAL_DOWNLOAD_SEC, "cost": 0.0})
        if cleanup and f["type"] == "video":
            steps.append({"file": f, "name": "Cleanup", "sec": 1.0, "cost": 0.0})
    return steps


def aggregate_phases(steps: list[dict], execution: str, cleanup: bool) -> list[dict]:
    order = PHASE_ORDER_MODAL if execution == "modal" else PHASE_ORDER_LOCAL
    totals = {name: {"sec": 0.0, "cost": 0.0} for name in order}
    for s in steps:
        totals[s["name"]]["sec"] += s["sec"]
        totals[s["name"]]["cost"] += s["cost"]
    return [
        {"name": name, "sec": totals[name]["sec"], "cost": totals[name]["cost"]}
        for name in order
        if name != "Cleanup" or cleanup
    ]
