"""Per-file, per-phase time/cost estimates. Mirrors frontend/app.js's
Phase 1 client-side math (now the single source of truth server-side) --
used both by POST /api/preview and to weight the real pipeline's
progress bar.
"""

from __future__ import annotations

from backend.calibration import get_rtf as get_calibrated_rtf
from backend.calibration import get_setup_sec as get_calibrated_setup_sec
from backend.config import (
    GPU_OPTIONS,
    GPU_SPEED_MULTIPLIER,
    LOCAL_MODEL_SETUP_SEC,
    MODAL_DOWNLOAD_SEC,
    MODAL_SETUP_SEC,
    RTF_LOCAL_CPU,
    RTF_MODAL_GPU,
)

PHASE_ORDER_LOCAL = ["Audio Extraction", "Whisper Model Setup", "Transcription", "Cleanup"]
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


def _estimate_rtf(model: str, execution: str, gpu_id: str | None) -> float:
    """Prefers real measured throughput (backend/calibration.py) once
    we've actually run this exact (model, device) combination before;
    otherwise falls back to the static approximate table -- no generic
    benchmark table is reliable enough on its own for this (see
    calibration.py's docstring)."""
    device = gpu_id if execution == "modal" else "cpu"
    calibrated = get_calibrated_rtf(model, device)
    if calibrated is not None:
        return calibrated

    if execution == "modal":
        multiplier = GPU_SPEED_MULTIPLIER.get(gpu_id, 1.0)
        return RTF_MODAL_GPU[model] / multiplier
    return RTF_LOCAL_CPU[model]


def _estimate_local_setup_sec(model: str) -> float:
    calibrated = get_calibrated_setup_sec(model, "cpu")
    return calibrated if calibrated is not None else LOCAL_MODEL_SETUP_SEC[model]


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
    rtf = _estimate_rtf(model, execution, gpu_id)
    rate = _gpu_rate(gpu_id) if is_modal else 0.0

    steps: list[dict] = []
    for i, f in enumerate(files):
        if f["type"] == "video":
            steps.append({"file": f, "name": "Audio Extraction", "sec": (f["duration_sec"] / 60) * 2, "cost": 0.0})
        if not is_modal and i == 0:
            # Local model load/caching happens once per job, not per file
            # (subsequent files reuse the already-loaded model), so this
            # only applies to the first file -- unlike Modal's per-file
            # setup, which is genuinely per-call since each transcription
            # spins up a fresh ephemeral container.
            steps.append({
                "file": f, "name": "Whisper Model Setup",
                "sec": _estimate_local_setup_sec(model), "cost": 0.0,
            })
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
