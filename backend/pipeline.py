"""Per-file pipeline: extract audio -> transcribe -> write outputs ->
optional cleanup. Runs entirely in a worker thread (via
asyncio.to_thread) so it doesn't block the event loop, and reports
progress/log events through the job's queue as it goes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from backend.estimator import aggregate_phases, compute_steps
from backend.ffmpeg_utils import extract_audio
from backend.jobs import Job
from backend.transcription import engine, writers


def _group_by_file(steps: list[dict]) -> list[dict]:
    groups: list[dict] = []
    current = None
    for s in steps:
        if current is None or current["file"] is not s["file"]:
            current = {"file": s["file"], "steps": []}
            groups.append(current)
        current["steps"].append(s)
    return groups


def _step_label(step: dict, model: str, execution: str) -> str:
    name = step["name"]
    path = step["file"]["path"]
    if name == "Audio Extraction":
        return f"Extracting audio: {path}"
    if name == "Modal.com Setup & Model Install":
        return f"Setting up Modal.com & installing {model} model: {path}"
    if name == "Transcription":
        return f"Transcribing ({execution}, {model}): {path}"
    if name == "Download Transcript to Local":
        return f"Downloading transcript: {path}"
    if name == "Cleanup":
        return f"Cleaning up intermediate audio: {path}"
    return f"{name}: {path}"


def _friendly_error(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _run_file_group(
    group: dict,
    folder_path: Path,
    model_name: str,
    formats: list[str],
    cleanup: bool,
    job: Job,
) -> tuple[bool, str | None]:
    """Runs one file's steps synchronously (in a worker thread). Returns
    (succeeded, error_message)."""
    file_info = group["file"]
    abs_path = folder_path / file_info["path"]
    wav_path = abs_path.with_suffix(".wav") if file_info["type"] == "video" else None
    audio_path = abs_path

    for step in group["steps"]:
        job.emit({"event": "log", "text": _step_label(step, model_name, "local")})
        try:
            if step["name"] == "Audio Extraction":
                extract_audio(abs_path, wav_path)
                audio_path = wav_path
            elif step["name"] == "Transcription":
                model = engine.get_model(model_name, device="cpu")
                group["_segments"] = engine.transcribe(model, audio_path)
                for fmt in formats:
                    out_path = abs_path.with_suffix(f".{fmt}")
                    writers.write_format(fmt, group["_segments"], out_path)
            elif step["name"] == "Cleanup":
                if wav_path is not None:
                    wav_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user as a log line
            message = _friendly_error(exc)
            job.emit({"event": "log", "text": f"Failed: {file_info['path']} — {message}", "fail": True})
            return False, message

        job.emit({"event": "step_done", "step": step["name"], "sec": step["sec"]})

    return True, None


async def run_job(
    job: Job,
    folder_path: str,
    files: list[dict],
    model: str,
    formats: list[str],
    execution: str,
    gpu: str | None,
    cleanup: bool,
) -> None:
    steps = compute_steps(files, model, execution, gpu, cleanup)
    total_estimated_sec = sum(s["sec"] for s in steps) or 1.0
    groups = _group_by_file(steps)
    folder = Path(folder_path)

    elapsed_estimated = 0.0
    succeeded = 0
    failed = 0
    started_at = time.perf_counter()

    for group in groups:
        if job.cancel_requested:
            job.emit({"event": "log", "text": "Cancelled — remaining files were not started."})
            break

        ok, _err = await asyncio.to_thread(
            _run_file_group, group, folder, model, formats, cleanup, job
        )
        for step in group["steps"]:
            elapsed_estimated += step["sec"]
        job.emit({"event": "progress", "percent": min(100, round(elapsed_estimated / total_estimated_sec * 100))})

        if ok:
            job.emit({"event": "log", "text": f"Done: {group['file']['path']}"})
            succeeded += 1
        else:
            failed += 1

    total_sec = time.perf_counter() - started_at
    skipped = len(groups) - succeeded - failed
    total_minutes = sum(f["duration_sec"] for f in files) / 60 or 1.0
    processed = (succeeded + failed) or 1

    job.emit({
        "event": "done",
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "cancelled": job.cancel_requested,
        "total_sec": total_sec,
        "total_cost": 0.0,  # local execution is always free
        "per_file_sec": total_sec / processed,
        "per_file_cost": 0.0,
        "per_minute_sec": total_sec / total_minutes,
        "per_minute_cost": 0.0,
    })
