"""Phase-wise pipeline: every file goes through Audio Extraction together,
then Model Setup (once for the whole job -- Whisper Model Setup locally,
Modal.com Setup & Model Install remotely), then (Modal only) Upload to
Modal.com, then Transcription (+ Download per file for Modal -- a
formality by that point, the transcript is already written), then
Teardown/Release (once for the whole job), then Cleanup -- rather than
one file finishing all its phases before the next file starts. Runs
entirely in a worker thread per step (via asyncio.to_thread) so it
doesn't block the event loop, and reports progress/log events through
the job's queue as it goes.

A file that fails one phase is skipped (not retried) in every later
phase -- it doesn't stop the rest of the job, and its skipped steps still
count toward progress so the bar reaches 100% by the end.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

from backend import calibration
from backend.config import BEAM_SIZE, GPU_OPTIONS, MAX_MODAL_UPLOAD_BYTES
from backend.estimator import aggregate_phases, compute_steps
from backend.ffmpeg_utils import FfmpegNotFoundError, extract_audio
from backend.jobs import Job, StepCancelled
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
        return f"Setting up Modal.com & installing {model} model (once for the whole job)"
    if name == "Upload audio files to Modal.com":
        return f"Uploading to Modal.com: {path}"
    if name == "Transcription":
        return f"Transcribing ({execution}, {model}): {path}"
    if name == "Download Transcript to Local":
        return f"Downloading transcript: {path}"
    if name == "Cleanup - Modal.com Teardown":
        return "Tearing down the Modal.com container"
    if name == "Cleanup - Release Whisper Model":
        return "Releasing the Whisper model from memory"
    if name == "Cleanup - Intermediate Files":
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
    modal_transcriber: modal_app.ModalTranscriber | None,
    hf_token: str | None,
    job: Job,
    logger: logging.Logger,
    file_state: dict,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[str, str | None, float]:
    """Runs exactly one step (one file's one phase, or a job-level step
    when step["file"] is None) synchronously in a worker thread. Returns
    (outcome, error_message, real_cost_usd), where outcome is "ok",
    "failed", or "cancelled" -- a cancelled step is deliberately not
    "failed": it doesn't add the file to failed_files, since the file
    didn't error out, the user just stopped the job. should_cancel() is
    threaded into the actual blocking work (ffmpeg subprocess, Whisper
    segment loop, Modal RPC poll) below so a Cancel click interrupts
    whichever step is in flight immediately, rather than waiting for it
    to run to completion."""
    file_info = step["file"]
    label = _step_label(step, model_name, execution)
    job.emit({"event": "log", "text": label, "step": step["name"]})
    job.emit({"event": "step_start", "step": step["name"]})
    if progress_cb is not None and step["name"] in ("Audio Extraction", "Upload audio files to Modal.com", "Transcription"):
        # Shows "(0%)" the instant the step starts, for both local and
        # Modal execution, instead of leaving the suffix blank until the
        # first real segment/progress event arrives -- which, for Modal
        # specifically, can be a few seconds into the RPC (reading the
        # WAV off disk, uploading it, the remote call actually starting)
        # after the spinner already appeared. A bare spinner with no
        # number reads as "is this stuck?"; explicit 0% reads as "yes,
        # this really did just start."
        progress_cb(0.0)
    logger.info(label)
    real_cost = 0.0

    def should_cancel() -> bool:
        return job.cancel_requested

    try:
        if step["name"] == "Audio Extraction":
            abs_path = folder_path / file_info["path"]
            wav_path = abs_path.with_suffix(".wav")
            file_duration_sec = file_info["duration_sec"]

            def on_extraction_progress(processed_sec: float) -> None:
                if progress_cb is not None and file_duration_sec > 0:
                    progress_cb(min(1.0, processed_sec / file_duration_sec))

            try:
                extract_audio(abs_path, wav_path, should_cancel=should_cancel, on_progress=on_extraction_progress)
            except StepCancelled:
                # ffmpeg may have already written a partial file before
                # being killed -- record it so the unconditional Cleanup
                # pass still removes it rather than leaving it orphaned.
                if wav_path.exists():
                    file_state[file_info["path"]]["wav_path"] = wav_path
                raise
            file_state[file_info["path"]]["wav_path"] = wav_path

        elif step["name"] == "Whisper Model Setup":
            if hf_token:
                # Raises the Hugging Face Hub rate limit for this download;
                # huggingface_hub picks HF_TOKEN up from the environment
                # automatically. Setting it here (not globally at process
                # start) keeps it scoped to jobs that actually opted in.
                os.environ["HF_TOKEN"] = hf_token
            started = time.perf_counter()
            # acquire_model (not get_model) pins this model against the
            # matching release_model() in "Cleanup - Release Whisper
            # Model" below, so a concurrent second job using the same
            # model can't have this job's cleanup evict it first.
            engine.acquire_model(model_name, device="cpu")  # loads (or downloads) and caches
            calibration.record_setup_sample(model_name, "cpu", time.perf_counter() - started)

        elif step["name"] == "Modal.com Setup & Model Install":
            # A real, billable Modal call (container cold start + model
            # load) -- timed and credited the same way Transcription's
            # RPC time is, so the job's real total_cost isn't missing the
            # GPU-seconds this step actually spent (Preview already
            # estimates a non-zero cost for it via MODAL_SETUP_SEC).
            started = time.perf_counter()
            try:
                modal_transcriber.warm_up(should_cancel=should_cancel)
            except StepCancelled:
                # Cancelling still bills for GPU time actually used before
                # the container was terminated -- same reasoning as the
                # Transcription branch below.
                real_cost = ((time.perf_counter() - started) / 3600) * _gpu_rate(gpu)
                raise
            real_cost = ((time.perf_counter() - started) / 3600) * _gpu_rate(gpu)

        elif step["name"] == "Upload audio files to Modal.com":
            abs_path = folder_path / file_info["path"]
            wav_path = file_state[file_info["path"]].get("wav_path")
            audio_path = wav_path if wav_path is not None else abs_path
            upload_id = uuid.uuid4().hex
            audio_bytes = audio_path.read_bytes()

            def on_upload_progress(fraction: float) -> None:
                if progress_cb is not None:
                    progress_cb(fraction)

            started = time.perf_counter()
            try:
                modal_transcriber.upload(
                    audio_bytes, upload_id,
                    should_cancel=should_cancel, on_progress=on_upload_progress,
                )
            except StepCancelled:
                # Cancelling still bills for the container time already
                # spent receiving chunks -- same reasoning as every other
                # Modal step's cost tracking. Not recorded as a calibration
                # sample: a cancelled upload's elapsed time doesn't reflect
                # genuine full-file throughput.
                real_cost = ((time.perf_counter() - started) / 3600) * _gpu_rate(gpu)
                raise
            elapsed = time.perf_counter() - started
            real_cost = (elapsed / 3600) * _gpu_rate(gpu)
            if elapsed > 0:
                calibration.record_upload_sample(len(audio_bytes) / elapsed)
            # Handed to the Transcription step for this same file below --
            # no separate cleanup needed for the uploaded temp file itself:
            # the remote transcribe() call deletes it once done, and
            # anything left over is gone once the container is torn down.
            file_state[file_info["path"]]["modal_upload_id"] = upload_id

        elif step["name"] == "Transcription":
            abs_path = folder_path / file_info["path"]
            wav_path = file_state[file_info["path"]].get("wav_path")
            audio_path = wav_path if wav_path is not None else abs_path
            file_duration_sec = file_info["duration_sec"]

            def on_segment_end(end_sec: float) -> None:
                # Reports how far into the file transcription has reached
                # so far, as a 0..1 fraction -- this is what makes the "(NN%)"
                # label and the progress bar move *during* Transcription
                # instead of only jumping once the whole file is done, which
                # matters most here since it's normally the longest step.
                if progress_cb is not None and file_duration_sec > 0:
                    progress_cb(min(1.0, end_sec / file_duration_sec))

            if execution == "modal":
                upload_id = file_state[file_info["path"]]["modal_upload_id"]
                started = time.perf_counter()
                try:
                    raw = modal_transcriber.transcribe(
                        upload_id, BEAM_SIZE,
                        should_cancel=should_cancel, on_progress=on_segment_end,
                    )
                except StepCancelled:
                    # Cancelling still bills for GPU time actually used before
                    # the container was terminated -- credit it rather than
                    # silently losing track of real Modal spend.
                    real_cost = ((time.perf_counter() - started) / 3600) * _gpu_rate(gpu)
                    raise
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
                segments = engine.transcribe(model, audio_path, should_cancel=should_cancel, on_progress=on_segment_end)
                if file_duration_sec > 0:
                    calibration.record_rtf_sample(model_name, "cpu", (time.perf_counter() - started) / file_duration_sec)

            for fmt in formats:
                out_path = abs_path.with_suffix(f".{fmt}")
                writers.write_format(fmt, segments, out_path)

        elif step["name"] == "Download Transcript to Local":
            pass  # already covered by the Transcription step's bundled RPC

        elif step["name"] == "Cleanup - Modal.com Teardown":
            # Explicit, visible teardown of the job's warm container/session
            # -- a failure here is caught by the except Exception block
            # below like any other step (logged, shown as a failed phase)
            # without stopping the job from continuing on to file cleanup.
            modal_transcriber.__exit__(None, None, None)

        elif step["name"] == "Cleanup - Release Whisper Model":
            engine.release_model(model_name, device="cpu")

        elif step["name"] == "Cleanup - Intermediate Files":
            wav_path = file_state[file_info["path"]].get("wav_path")
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

    except StepCancelled:
        who = file_info["path"] if file_info else "job"
        job.emit({"event": "log", "text": f"Cancelled: {who}"})
        # Distinct from step_skipped (below and in run_job's main loop),
        # which means "an earlier phase already failed this file, so this
        # phase was never even attempted for it" -- harmless, doesn't
        # reflect on THIS phase. step_cancelled means this phase itself
        # was genuinely interrupted mid-flight -- the frontend needs that
        # distinction so a phase where one file succeeded and another was
        # cancelled shows as interrupted, not as a clean success.
        job.emit({"event": "step_cancelled", "step": step["name"]})
        logger.info("Cancelled during %s%s", step["name"], f" on {who}" if file_info else "")
        return "cancelled", None, real_cost

    except Exception as exc:  # noqa: BLE001 -- surfaced to the user as a log line
        message = _friendly_error(exc)
        who = file_info["path"] if file_info else "job"
        job.emit({"event": "log", "text": f"Failed: {who} — {message}", "fail": True})
        job.emit({"event": "step_failed", "step": step["name"]})
        logger.exception("Failed during %s%s", step["name"], f" on {who}" if file_info else "")
        return "failed", message, real_cost

    job.emit({"event": "step_done", "step": step["name"], "sec": step["sec"]})
    return "ok", None, real_cost


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
    hf_token: str | None = None,
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
    main_passes = [p for p in passes if p["steps"][0]["name"] != "Cleanup - Intermediate Files"]
    cleanup_pass = next((p for p in passes if p["steps"][0]["name"] == "Cleanup - Intermediate Files"), None)
    folder = Path(folder_path)

    file_state: dict[str, dict] = {f["path"]: {} for f in files}
    failed_files: dict[str, str] = {}  # path -> error message
    steps_expected_for_file = Counter(s["file"]["path"] for s in steps if s["file"] is not None)
    steps_done_for_file: Counter = Counter()

    elapsed_estimated = 0.0
    total_cost = 0.0
    started_at = time.perf_counter()
    cancelled = False

    # Audio Extraction, Upload audio files to Modal.com, and Transcription
    # all report real per-file progress. Two different audiences want two
    # different numbers out of that: the scrolling log line is about *this file*,
    # so it correctly resets to 0% at the start of each new file -- but
    # the Steps table has one row per phase for the *whole batch*, so its
    # "(NN%)" should read as one continuous 0->100% sweep across every
    # file in that phase, not repeatedly reset. batch_progress_done
    # tracks how many seconds of audio have already been fully accounted
    # for in each phase (advanced in the main loop below);
    # batch_progress_total is the fixed denominator for each (only video
    # files go through Audio Extraction; every file goes through Upload
    # and Transcription).
    batch_progress_total = {
        "Audio Extraction": sum(f["duration_sec"] for f in files if f["type"] == "video") or 1.0,
        "Upload audio files to Modal.com": sum(f["duration_sec"] for f in files) or 1.0,
        "Transcription": sum(f["duration_sec"] for f in files) or 1.0,
    }
    batch_progress_done = {name: 0.0 for name in batch_progress_total}

    def _emit_progress() -> None:
        job.emit({"event": "progress", "percent": min(100, round(elapsed_estimated / total_estimated_sec * 100))})

    def _make_progress_cb(step: dict) -> Callable[[float], None]:
        # Turns a 0..1 fraction of *this step's* estimated duration into a
        # finer-grained overall progress-bar update (instead of the bar
        # only moving once the whole, often very long, step completes).
        # Also emits two different percentages for display: `percent`
        # (this file only, for the log line) and `batch_percent` (blended
        # across the whole batch, for the Steps-table row). Throttled so
        # a fast model's segment-per-fraction-of-a-second callback rate
        # doesn't flood the SSE stream.
        last = {"t": 0.0, "pct": -1}

        def cb(fraction: float) -> None:
            now = time.perf_counter()
            pct = round(max(0.0, min(1.0, fraction)) * 100)
            if pct == last["pct"] or now - last["t"] < 0.5:
                return
            last["t"], last["pct"] = now, pct

            batch_pct = pct
            if step["name"] in batch_progress_total and step["file"] is not None:
                done = batch_progress_done[step["name"]]
                total = batch_progress_total[step["name"]]
                blended = (done + fraction * step["file"]["duration_sec"]) / total
                batch_pct = round(min(1.0, blended) * 100)

            job.emit({"event": "step_progress", "step": step["name"], "percent": pct, "batch_percent": batch_pct})
            partial = elapsed_estimated + step["sec"] * (pct / 100)
            job.emit({"event": "progress", "percent": min(100, round(partial / total_estimated_sec * 100))})

        return cb

    # One Modal App/container is opened for the *whole job*, not per file --
    # every file's Transcription step below reuses the same warm container
    # and already-loaded model via modal_transcriber.transcribe(), instead
    # of each file cold-starting its own ephemeral Modal session.
    modal_transcriber = None
    if execution == "modal":
        modal_transcriber = modal_app.ModalTranscriber(gpu, model, modal_token_id, modal_token_secret, hf_token)
        await asyncio.to_thread(modal_transcriber.__enter__)

    # Mirrors modal_transcriber's is_active-guarded safety net below, for
    # local execution: if the job is cancelled before its explicit
    # "Cleanup - Release Whisper Model" step gets a chance to run (e.g.
    # cancelled mid-Transcription), the model acquired in "Whisper Model
    # Setup" would otherwise never be released at all -- unlike Modal,
    # local has no single always-runs finally block already doing this,
    # so it's tracked explicitly instead.
    local_model_acquired = False
    local_model_released = False

    # The /api/preview-time check (estimator.py's UploadSizeExceededError)
    # is a fast, cheap pre-filter based on estimated duration*bitrate --
    # good enough to reject an obviously oversized batch before spending
    # any time on Setup or Extraction at all. But it's still an estimate:
    # a video's real extracted WAV size, or a non-WAV audio file's actual
    # size on disk, can differ from that estimate. This is the
    # authoritative check on *real* bytes, run once -- right after
    # Extraction has produced real WAVs (or immediately, if the batch is
    # audio-only and there's nothing to extract) and before the
    # expensive Upload/Transcription phases actually start spending
    # money.
    upload_size_checked = False

    try:
        for phase_pass in main_passes:
            if cancelled:
                break
            if job.cancel_requested:
                job.emit({"event": "log", "text": "Cancelled — remaining phases were not started."})
                logger.info("Job %s cancelled before pass %d", job.id, phase_pass["pass"])
                cancelled = True
                break

            if execution == "modal" and not upload_size_checked and phase_pass["steps"][0]["name"] != "Audio Extraction":
                upload_size_checked = True
                real_upload_bytes = 0
                for f in files:
                    if f["path"] in failed_files:
                        continue  # never reaches Upload, doesn't count
                    real_path = file_state.get(f["path"], {}).get("wav_path") or (folder / f["path"])
                    try:
                        real_upload_bytes += real_path.stat().st_size
                    except OSError:
                        pass  # best-effort -- a missing file fails its own step later anyway
                if real_upload_bytes > MAX_MODAL_UPLOAD_BYTES:
                    message = (
                        f"Selected files need to upload about {real_upload_bytes / 1_000_000_000:.0f} GB to "
                        f"Modal.com, which exceeds the {MAX_MODAL_UPLOAD_BYTES / 1_000_000_000:.0f} GB limit "
                        "for a single job. Cancelling -- select fewer files, or switch to Local execution."
                    )
                    job.emit({"event": "log", "text": message, "fail": True})
                    job.emit({"event": "fatal_error", "text": message})
                    logger.info("Job %s aborted: real upload size %d bytes exceeds limit", job.id, real_upload_bytes)
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
                    if step["name"] in batch_progress_done:
                        batch_progress_done[step["name"]] += file_info["duration_sec"]
                    _emit_progress()
                    continue

                if job.cancel_requested:
                    job.emit({"event": "log", "text": "Cancelled — remaining files/phases were not started."})
                    logger.info("Job %s cancelled mid-pass %d", job.id, phase_pass["pass"])
                    cancelled = True
                    break

                outcome, err, cost = await asyncio.to_thread(
                    _run_single_step, step, folder, model, formats, execution, gpu,
                    modal_transcriber, hf_token, job, logger, file_state,
                    _make_progress_cb(step),
                )
                total_cost += cost
                elapsed_estimated += step["sec"]
                if step["name"] in batch_progress_done and file_info is not None and outcome != "cancelled":
                    # Counts toward the blended batch-wide percent above
                    # whether this file's step succeeded or failed --
                    # either way the pipeline is past it, so it should no
                    # longer hold back the batch-wide percentage.
                    batch_progress_done[step["name"]] += file_info["duration_sec"]
                _emit_progress()

                if step["name"] == "Whisper Model Setup" and outcome == "ok":
                    local_model_acquired = True
                if step["name"] == "Cleanup - Release Whisper Model":
                    local_model_released = True

                if outcome == "ok":
                    if file_info is not None:
                        steps_done_for_file[file_info["path"]] += 1
                elif outcome == "failed":
                    if file_info is not None:
                        failed_files[file_info["path"]] = err
                else:  # "cancelled" -- the step noticed the cancel flag and stopped itself mid-flight
                    job.emit({"event": "log", "text": "Cancelled — remaining files/phases were not started."})
                    logger.info("Job %s cancelled mid-step during pass %d", job.id, phase_pass["pass"])
                    cancelled = True
                    break

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

                outcome, err, _cost = await asyncio.to_thread(
                    _run_single_step, step, folder, model, formats, execution, gpu,
                    modal_transcriber, hf_token, job, logger, file_state,
                )
                elapsed_estimated += step["sec"]
                _emit_progress()

                if outcome == "ok":
                    steps_done_for_file[file_info["path"]] += 1
                elif outcome == "failed":
                    # Don't overwrite an earlier, more specific failure reason
                    # for this file if it already had one.
                    failed_files.setdefault(file_info["path"], err)
    finally:
        if modal_transcriber is not None:
            # Only emit events if this is genuinely the one doing the
            # teardown work -- if the job reached its own explicit
            # "Cleanup - Modal.com Teardown" step normally, __exit__() is
            # already idempotent and this call is a real no-op, so it
            # must not also emit a second, redundant step_start/step_done
            # for that phase. But when the job is cancelled *before*
            # reaching that step, this silent safety net is the only
            # thing that actually tears the container down -- without
            # emitting its own events here, the frontend has no way to
            # know that happened and (via forceFinalizeIncompletePhases)
            # would show it as failed even though it just succeeded.
            was_active = modal_transcriber.is_active
            if was_active:
                job.emit({"event": "step_start", "step": "Cleanup - Modal.com Teardown"})
            try:
                await asyncio.to_thread(modal_transcriber.__exit__, None, None, None)
            except Exception as exc:  # noqa: BLE001 -- best-effort cleanup, never re-raised
                logger.exception("Job %s: Modal.com teardown failed during cleanup", job.id)
                if was_active:
                    job.emit({"event": "log", "text": f"Modal.com teardown failed: {_friendly_error(exc)}", "fail": True})
                    job.emit({"event": "step_failed", "step": "Cleanup - Modal.com Teardown"})
            else:
                if was_active:
                    job.emit({"event": "step_done", "step": "Cleanup - Modal.com Teardown", "sec": 0.0})

        # Same safety net as above, for local execution: only needed if
        # this job's Setup actually acquired the model (nothing to
        # release otherwise) and its own explicit "Cleanup - Release
        # Whisper Model" step never got a chance to run.
        if execution != "modal" and local_model_acquired and not local_model_released:
            job.emit({"event": "step_start", "step": "Cleanup - Release Whisper Model"})
            try:
                await asyncio.to_thread(engine.release_model, model, "cpu")
            except Exception as exc:  # noqa: BLE001 -- best-effort cleanup, never re-raised
                logger.exception("Job %s: releasing the Whisper model failed during cleanup", job.id)
                job.emit({"event": "log", "text": f"Releasing the Whisper model failed: {_friendly_error(exc)}", "fail": True})
                job.emit({"event": "step_failed", "step": "Cleanup - Release Whisper Model"})
            else:
                job.emit({"event": "step_done", "step": "Cleanup - Release Whisper Model", "sec": 0.0})

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


async def run_job_safe(job: Job, *args, **kwargs) -> None:
    """Wraps run_job with a last-resort guard for bugs in the pipeline
    itself (as opposed to a single file/step failing, which run_job
    already handles and reports per-file without this). Without this, a
    crash before the first "done" event (e.g. building the Modal
    App/Cls, or any other setup that runs before the per-step try/except
    blocks) leaves the job's SSE stream open with nothing more ever
    arriving -- the frontend just sits there spinning forever with no
    indication anything went wrong. This ensures the job always ends
    with a "done" event and a clear log line, however it fails."""
    try:
        await run_job(job, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        logger = get_job_logger(job.id)
        logger.exception("Job %s crashed unexpectedly", job.id)
        job.emit({"event": "log", "text": f"Job failed unexpectedly: {_friendly_error(exc)}", "fail": True})
        job.emit({
            "event": "done",
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "cancelled": False,
            "total_sec": 0.0,
            "total_cost": 0.0,
            "per_file_sec": 0.0,
            "per_file_cost": 0.0,
            "per_minute_sec": 0.0,
            "per_minute_cost": 0.0,
        })
