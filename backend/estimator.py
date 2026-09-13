"""Per-file, per-phase time/cost estimates. Mirrors frontend/app.js's
Phase 1 client-side math (now the single source of truth server-side) --
used both by POST /api/preview and to weight the real pipeline's
progress bar.
"""

from __future__ import annotations

from backend.calibration import get_rtf as get_calibrated_rtf
from backend.calibration import get_setup_sec as get_calibrated_setup_sec
from backend.calibration import get_upload_bytes_per_sec as get_calibrated_upload_bytes_per_sec
from backend.config import (
    GPU_OPTIONS,
    GPU_SPEED_MULTIPLIER,
    LOCAL_MODEL_SETUP_SEC,
    MAX_MODAL_UPLOAD_BYTES,
    MODAL_DOWNLOAD_SEC,
    MODAL_SETUP_SEC,
    MODAL_UPLOAD_BYTES_PER_SEC,
    RTF_LOCAL_CPU,
    RTF_MODAL_GPU,
    WAV_BYTES_PER_SEC,
)


class UploadSizeExceededError(ValueError):
    """Raised by compute_steps() when a Modal.com job's total estimated
    upload size would exceed MAX_MODAL_UPLOAD_BYTES -- caught in main.py
    and turned into a 400 so the frontend can show it as a Preview error,
    the same way any other Preview failure is already surfaced (Begin
    never appears without a successful Preview to begin with)."""

PHASE_ORDER_LOCAL = [
    "Audio Extraction",
    "Whisper Model Setup",
    "Transcription",
    "Cleanup - Release Whisper Model",
    "Cleanup - Intermediate Files",
]
PHASE_ORDER_MODAL = [
    "Audio Extraction",
    "Modal.com Setup & Model Install",
    "Upload audio files to Modal.com",
    "Transcription",
    "Download Transcript to Local",
    "Cleanup - Modal.com Teardown",
    "Cleanup - Intermediate Files",
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


def _estimate_upload_bytes_per_sec() -> float:
    """Prefers this machine's own real measured Modal.com upload
    throughput (recorded in pipeline.py after each Upload step
    completes) once at least one sample exists; otherwise falls back to
    the static conservative guess -- same self-correcting pattern as
    _estimate_rtf above."""
    calibrated = get_calibrated_upload_bytes_per_sec()
    return calibrated if calibrated is not None else MODAL_UPLOAD_BYTES_PER_SEC


def compute_steps(
    files: list[dict],
    model: str,
    execution: str,
    gpu_id: str | None,
    cleanup: bool,
) -> list[dict]:
    """files: [{"path", "type", "duration_sec"}, ...] (as sent by the frontend).
    Returns a flat, ordered step list -- organized *phase-wise*: every
    file's Audio Extraction happens before anyone starts Transcription,
    etc. -- matching the real pipeline's execution order (see
    pipeline.py's _group_by_pass). Each step is
    {"file": <the file dict, or None for job-level steps>, "name": <phase
    name>, "pass": <int, execution order>, "sec": float, "cost": float}.
    """
    is_modal = execution == "modal"
    rtf = _estimate_rtf(model, execution, gpu_id)
    rate = _gpu_rate(gpu_id) if is_modal else 0.0

    steps: list[dict] = []

    # Pass 0: Audio Extraction -- every video file, independent of the rest.
    for f in files:
        if f["type"] == "video":
            steps.append({"file": f, "name": "Audio Extraction", "pass": 0, "sec": (f["duration_sec"] / 60) * 2, "cost": 0.0})

    next_pass = 1
    # Pass 1: Model Setup -- once for the whole job, not per file. Local
    # loads the model into this process and reuses it for every file
    # (engine.py caches it); Modal now keeps one warm container + one
    # already-loaded model for the whole job too (see modal_app.py's
    # ModalTranscriber), so its cold start is likewise a single job-level
    # cost rather than something every file pays again.
    if is_modal:
        steps.append({
            "file": None, "name": "Modal.com Setup & Model Install", "pass": next_pass,
            "sec": MODAL_SETUP_SEC, "cost": (MODAL_SETUP_SEC / 3600) * rate,
        })
    else:
        steps.append({"file": None, "name": "Whisper Model Setup", "pass": next_pass, "sec": _estimate_local_setup_sec(model), "cost": 0.0})
    next_pass += 1

    # Pass 2 (Modal only): Upload audio files to Modal.com -- every file,
    # its own contiguous pass so every file's upload finishes before any
    # file's Transcription starts (matching the Transcription/Download
    # split below). Modal bills container time while the upload RPCs
    # run, so this carries a real (if rough -- upload speed varies
    # enormously by connection) cost estimate rather than $0.
    if is_modal:
        # Every file in the job sits fully uploaded on the same shared
        # container's /tmp at once before any of them are transcribed and
        # deleted (see MAX_MODAL_UPLOAD_BYTES's docstring in config.py) --
        # so the batch *total*, not any single file, is what has to stay
        # under Modal's per-container disk quota.
        total_upload_bytes = sum(f["duration_sec"] * WAV_BYTES_PER_SEC for f in files)
        if total_upload_bytes > MAX_MODAL_UPLOAD_BYTES:
            raise UploadSizeExceededError(
                f"Selected files would need to upload about "
                f"{total_upload_bytes / 1_000_000_000:.0f} GB to Modal.com, which exceeds the "
                f"{MAX_MODAL_UPLOAD_BYTES / 1_000_000_000:.0f} GB limit for a single job. "
                f"Select fewer files, or switch to Local execution."
            )

        upload_bytes_per_sec = _estimate_upload_bytes_per_sec()
        for f in files:
            upload_sec = (f["duration_sec"] * WAV_BYTES_PER_SEC) / upload_bytes_per_sec
            steps.append({
                "file": f, "name": "Upload audio files to Modal.com", "pass": next_pass,
                "sec": upload_sec, "cost": (upload_sec / 3600) * rate,
            })
        next_pass += 1

    # Pass 3: Transcription for every file -- kept as one contiguous pass
    # so every file's Transcription genuinely finishes, in the real
    # execution order, before any file's Download phase starts. When they
    # were interleaved per file, the two showed as running at once with
    # nothing to explain why.
    for f in files:
        transcription_sec = f["duration_sec"] * rtf
        steps.append({
            "file": f, "name": "Transcription", "pass": next_pass,
            "sec": transcription_sec, "cost": (transcription_sec / 3600) * rate if is_modal else 0.0,
        })
    next_pass += 1

    # Pass 4 (Modal only): Download Transcript to Local -- a formality by
    # this point (the transcript is already written to disk as part of
    # Transcription above); kept as its own pass purely for the time/cost
    # line item, and so it's only ever shown starting once every file's
    # Transcription has actually finished.
    if is_modal:
        for f in files:
            steps.append({"file": f, "name": "Download Transcript to Local", "pass": next_pass, "sec": MODAL_DOWNLOAD_SEC, "cost": 0.0})
        next_pass += 1

    # Pass N: Release Resources -- always runs, unlike the Cleanup pass
    # below, since it's not a user preference but releasing something
    # that's actively costing money (Modal) or holding memory (local).
    # Runs once the whole job is done and before file Cleanup, so a
    # container/model teardown failure can't leave WAVs undeleted behind
    # it -- pipeline.py catches and logs a teardown failure without
    # stopping the job from reaching Cleanup.
    if is_modal:
        steps.append({"file": None, "name": "Cleanup - Modal.com Teardown", "pass": next_pass, "sec": 2.0, "cost": 0.0})
    else:
        steps.append({"file": None, "name": "Cleanup - Release Whisper Model", "pass": next_pass, "sec": 1.0, "cost": 0.0})
    next_pass += 1

    # Pass N+1: delete the intermediate WAVs -- every video file, once all
    # transcriptions are done.
    if cleanup:
        for f in files:
            if f["type"] == "video":
                steps.append({"file": f, "name": "Cleanup - Intermediate Files", "pass": next_pass, "sec": 1.0, "cost": 0.0})

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
        if name != "Cleanup - Intermediate Files" or cleanup
    ]
