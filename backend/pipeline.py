"""Phase-wise pipeline: every file goes through Audio Extraction together,
then Whisper Model Setup (local, once), then Transcription (bundling
Setup+Download per file for Modal, since that's one atomic RPC call),
then Cleanup -- rather than one file finishing all its phases before the
next file starts. Runs entirely in a worker thread per step (via
asyncio.to_thread) so it doesn't block the event loop, and reports
progress/log events through the job's queue as it goes.

A file that fails one phase is skipped (not retried) in every later
phase -- it doesn't stop the rest of the job, and its skipped steps still
count toward progress so the bar reaches 100% by the end.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from pathlib import Path

from backend import calibration
from backend.config import BEAM_SIZE, GPU_OPTIONS
from backend.estimator import aggregate_phases, compute_steps
from backend.ffmpeg_utils import FfmpegNotFoundError, extract_audio
from backend.jobs import Job
from backend.logging_utils import get_job_logger
from backend.transcription import engine, modal_app, writers


def _gpu_rate(gpu_id: str | None) -> float:
    gpu = next((g for g in GPU_OPTIONS if g["id"] == gpu_id), None)
    return gpu["rate"] if gpu else 0.0


def _group_by_pass(steps: list[dict]) -> list[dict]:
    groups: list[dict] = []
    current = None
    for s in steps:
        if current is None or current["pass"] != s["pass"]:
            current = {"pass": s["pass"], "steps": []}
            groups.append(current)
        current["steps"].append(s)
    return groups


def _step_label(step: dict, model: str, execution: str) -> str:
    name = step["name"]
    path = step["file"]["path"] if step["file"] else None
    if name == "Audio Extraction":
        return f"Extracting audio: {path}"
    if name == "Whisper Model Setup":
        return f"Loading {model} model (downloading if not already cached)"
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
    text = str(exc) or exc.__class__.__name__
    lower = text.lower()

    if isinstance(exc, FfmpegNotFoundError):
        return text  # already a clear, actionable message
    if isinstance(exc, FileNotFoundError) or "no such file" in lower:
        return "This file could not be found on disk -- it may have been moved or deleted."
    if isinstance(exc, PermissionError) or "permission denied" in lower:
        return "Permission denied reading this file -- check that it isn't open in another program."
    if isinstance(exc, MemoryError) or "out of memory" in lower or "cuda out of memory" in lower:
        return "Ran out of memory processing this file -- try a smaller Whisper model."
    if "auth" in lower and ("modal" in lower or "token" in lower or "unauthenticated" in lower):
        return "Modal.com rejected the provided credentials -- check the Token ID and Token Secret."
    if isinstance(exc, (ConnectionError, TimeoutError)) or "connection" in lower or "timed out" in lower:
        return "Could not reach Modal.com -- check your internet connection and try again."
    if "ffmpeg failed" in lower:
        return text  # already descriptive, from ffmpeg_utils.extract_audio
    return text


def _run_single_step(
    step: dict,
    folder_path: Path,
    model_name: str,
    formats: list[str],
    execution: str,
    gpu: str | None,
    modal_token_id: str | None,
    modal_token_secret: str | None,
    job: Job,
    logger: logging.Logger,
    file_state: dict,
) -> tuple[bool, str | None, float]:
    """Runs exactly one step (one file's one phase, or a job-level step
    when step["file"] is None) synchronously in a worker thread. Returns
    (succeeded, error_message, real_cost_usd)."""
    file_info = step["file"]
    label = _step_label(step, model_name, execution)
    job.emit({"event": "log", "text": label})
    job.emit({"event": "step_start", "step": step["name"]})
    logger.info(label)
    real_cost = 0.0

    try:
        if step["name"] == "Audio Extraction":
            abs_path = folder_path / file_info["path"]
            wav_path = abs_path.with_suffix(".wav")
            extract_audio(abs_path, wav_path)
            file_state[file_info["path"]]["wav_path"] = wav_path

        elif step["name"] == "Whisper Model Setup":
            started = time.perf_counter()
            engine.get_model(model_name, device="cpu")  # loads (or downloads) and caches
            calibration.record_setup_sample(model_name, "cpu", time.perf_counter() - started)

        elif step["name"] == "Modal.com Setup & Model Install":
            pass  # bundled into the Transcription step below (one atomic Modal RPC)

        elif step["name"] == "Transcription":
            abs_path = folder_path / file_info["path"]
            wav_path = file_state[file_info["path"]].get("wav_path")
            audio_path = wav_path if wav_path is not None else abs_path
            file_duration_sec = file_info["duration_sec"]

            if execution == "modal":
                started = time.perf_counter()
                raw = modal_app.transcribe_on_modal(
                    audio_path.read_bytes(), model_name, gpu, BEAM_SIZE,
                    modal_token_id, modal_token_secret,
                )
                # Cost is billed on the *full* RPC wall time (incl. cold
                # start); calibration uses inference_sec alone so a short
                # clip's cold-start overhead doesn't skew future
                # throughput estimates for this (model, GPU) pair.
                real_cost = ((time.perf_counter() - started) / 3600) * _gpu_rate(gpu)
                segments = [engine.Segment(**s) for s in raw["segments"]]
                if file_duration_sec > 0:
                    calibration.record_rtf_sample(model_name, gpu, raw["inference_sec"] / file_duration_sec)
            else:
                model = engine.get_model(model_name, device="cpu")  # already loaded by the Setup pass; cheap
                started = time.perf_counter()
                segments = engine.transcribe(model, audio_path)
                if file_duration_sec > 0:
                    calibration.record_rtf_sample(model_name, "cpu", (time.perf_counter() - started) / file_duration_sec)

            for fmt in formats:
                out_path = abs_path.with_suffix(f".{fmt}")
                writers.write_format(fmt, segments, out_path)

        elif step["name"] == "Download Transcript to Local":
            pass  # already covered by the Transcription step's bundled RPC

        elif step["name"] == "Cleanup":
            wav_path = file_state[file_info["path"]].get("wav_path")
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

    except Exception as exc:  # noqa: BLE001 -- surfaced to the user as a log line
        message = _friendly_error(exc)
        who = file_info["path"] if file_info else "job"
        job.emit({"event": "log", "text": f"Failed: {who} — {message}", "fail": True})
        job.emit({"event": "step_failed", "step": step["name"]})
        logger.exception("Failed during %s%s", step["name"], f" on {who}" if file_info else "")
        return False, message, real_cost

    job.emit({"event": "step_done", "step": step["name"], "sec": step["sec"]})
    return True, None, real_cost


async def run_job(
    job: Job,
    folder_path: str,
    files: list[dict],
    model: str,
    formats: list[str],
    execution: str,
    gpu: str | None,
    cleanup: bool,
    modal_token_id: str | None = None,
    modal_token_secret: str | None = None,
) -> None:
    logger = get_job_logger(job.id)
    logger.info(
        "Job %s starting: %d file(s), model=%s, execution=%s, gpu=%s, cleanup=%s",
        job.id, len(files), model, execution, gpu, cleanup,
    )

    steps = compute_steps(files, model, execution, gpu, cleanup)
    total_estimated_sec = sum(s["sec"] for s in steps) or 1.0
    passes = _group_by_pass(steps)
    # Cleanup is handled separately, as an unconditional final pass (see
    # below) -- it must still run for any file that got an intermediate
    # WAV extracted, even if a *later* phase failed for that file, or the
    # job was cancelled before reaching Cleanup normally. The point of
    # opting into cleanup is not leaving stray files behind; that
    # shouldn't depend on the rest of the job finishing cleanly.
    main_passes = [p for p in passes if p["steps"][0]["name"] != "Cleanup"]
    cleanup_pass = next((p for p in passes if p["steps"][0]["name"] == "Cleanup"), None)
    folder = Path(folder_path)

    file_state: dict[str, dict] = {f["path"]: {} for f in files}
    failed_files: dict[str, str] = {}  # path -> error message
    steps_expected_for_file = Counter(s["file"]["path"] for s in steps if s["file"] is not None)
    steps_done_for_file: Counter = Counter()

    elapsed_estimated = 0.0
    total_cost = 0.0
    started_at = time.perf_counter()
    cancelled = False

    def _emit_progress() -> None:
        job.emit({"event": "progress", "percent": min(100, round(elapsed_estimated / total_estimated_sec * 100))})

    for phase_pass in main_passes:
        if cancelled:
            break
        if job.cancel_requested:
            job.emit({"event": "log", "text": "Cancelled — remaining phases were not started."})
            logger.info("Job %s cancelled before pass %d", job.id, phase_pass["pass"])
            cancelled = True
            break

        for step in phase_pass["steps"]:
            file_info = step["file"]

            # A file that already failed an earlier phase is skipped in
            # every later phase -- still counted toward progress so the
            # bar reaches 100%, and toward the frontend's phase-icon
            # accounting via step_skipped, but never retried.
            if file_info is not None and file_info["path"] in failed_files:
                job.emit({"event": "step_skipped", "step": step["name"]})
                elapsed_estimated += step["sec"]
                _emit_progress()
                continue

            if job.cancel_requested:
                job.emit({"event": "log", "text": "Cancelled — remaining files/phases were not started."})
                logger.info("Job %s cancelled mid-pass %d", job.id, phase_pass["pass"])
                cancelled = True
                break

            ok, err, cost = await asyncio.to_thread(
                _run_single_step, step, folder, model, formats, execution, gpu,
                modal_token_id, modal_token_secret, job, logger, file_state,
            )
            total_cost += cost
            elapsed_estimated += step["sec"]
            _emit_progress()

            if ok:
                if file_info is not None:
                    steps_done_for_file[file_info["path"]] += 1
            elif file_info is not None:
                failed_files[file_info["path"]] = err

    # Unconditional cleanup pass: runs for every file that actually has a
    # recorded intermediate WAV, regardless of cancellation or failures
    # in later phases. A file with no WAV on record (extraction never
    # ran/succeeded, or it wasn't a video file) has nothing to clean up.
    if cleanup_pass is not None:
        for step in cleanup_pass["steps"]:
            file_info = step["file"]
            if file_state.get(file_info["path"], {}).get("wav_path") is None:
                job.emit({"event": "step_skipped", "step": step["name"]})
                elapsed_estimated += step["sec"]
                _emit_progress()
                continue

            ok, err, _cost = await asyncio.to_thread(
                _run_single_step, step, folder, model, formats, execution, gpu,
                modal_token_id, modal_token_secret, job, logger, file_state,
            )
            elapsed_estimated += step["sec"]
            _emit_progress()

            if ok:
                steps_done_for_file[file_info["path"]] += 1
            else:
                # Don't overwrite an earlier, more specific failure reason
                # for this file if it already had one.
                failed_files.setdefault(file_info["path"], err)

    total_sec = time.perf_counter() - started_at
    succeeded = sum(
        1 for f in files
        if f["path"] not in failed_files and steps_done_for_file[f["path"]] == steps_expected_for_file[f["path"]]
    )
    failed = len(failed_files)
    skipped = len(files) - succeeded - failed
    total_minutes = sum(f["duration_sec"] for f in files) / 60 or 1.0
    processed = (succeeded + failed) or 1

    logger.info(
        "Job %s finished: succeeded=%d failed=%d skipped=%d cancelled=%s total_sec=%.2f total_cost=%.4f",
        job.id, succeeded, failed, skipped, cancelled, total_sec, total_cost,
    )

    job.emit({
        "event": "done",
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "cancelled": cancelled,
        "total_sec": total_sec,
        "total_cost": total_cost,  # 0 for local execution; real Modal GPU-time cost otherwise
        "per_file_sec": total_sec / processed,
        "per_file_cost": total_cost / processed,
        "per_minute_sec": total_sec / total_minutes,
        "per_minute_cost": total_cost / total_minutes,
    })
