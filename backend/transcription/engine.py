"""Shared faster-whisper wrapper. One model instance is cached per
(model_name, device) pair and reused across files/jobs so it's only
loaded once -- loading is the expensive part, not switching audio files.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, NamedTuple

from faster_whisper import WhisperModel

from backend.config import BEAM_SIZE
from backend.jobs import StepCancelled

_model_cache: dict[tuple[str, str], WhisperModel] = {}
# Reference-counted so two concurrent local jobs sharing the same model
# (e.g. two browser tabs both transcribing with "small") don't have one
# job's Cleanup evict the model out from under the other, still-running
# job -- which would silently force an expensive reload on its next file
# instead of the "already loaded by Setup; cheap" reuse pipeline.py
# assumes. Guarded by a lock since acquire/release run in worker threads
# (via asyncio.to_thread) and could race across different jobs' threads.
_model_refcount: dict[tuple[str, str], int] = {}
_lock = threading.Lock()


class Segment(NamedTuple):
    start: float
    end: float
    text: str


def get_model(model_name: str, device: str = "cpu") -> WhisperModel:
    """Returns the cached model, creating it if needed, without touching
    the reference count -- used by Transcription, which relies on the
    job's own earlier acquire_model() (in its Setup step) to have already
    loaded and pinned it for the duration of this job."""
    key = (model_name, device)
    with _lock:
        if key not in _model_cache:
            compute_type = "int8" if device == "cpu" else "float16"
            _model_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        return _model_cache[key]


def acquire_model(model_name: str, device: str = "cpu") -> WhisperModel:
    """Like get_model(), but also pins this (model, device) pair for the
    calling job -- pair with exactly one release_model() call per job
    (its Setup and Cleanup steps, respectively) so the model is only
    actually evicted once no other running job still needs it."""
    key = (model_name, device)
    with _lock:
        if key not in _model_cache:
            compute_type = "int8" if device == "cpu" else "float16"
            _model_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        _model_refcount[key] = _model_refcount.get(key, 0) + 1
        return _model_cache[key]


def release_model(model_name: str, device: str = "cpu") -> None:
    """Drops this job's reference to the cached model; only actually
    evicts it (freeing its memory for GC) once no other running job's
    acquire_model() still needs it. There's no subprocess to kill for
    local execution (it runs in this process), so this is the local
    equivalent of Modal's container teardown."""
    key = (model_name, device)
    with _lock:
        remaining = _model_refcount.get(key, 0) - 1
        if remaining <= 0:
            _model_refcount.pop(key, None)
            _model_cache.pop(key, None)
        else:
            _model_refcount[key] = remaining


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[Segment]:
    """model.transcribe() returns a lazily-computed generator -- each
    segment is only decoded when we ask for the next one -- so checking
    should_cancel() between segments stops the job as soon as the segment
    in flight finishes, rather than waiting for the whole file. That's as
    fine-grained as it gets without running inference in a killable
    subprocess: a single segment's decode itself can't be interrupted
    mid-computation. on_progress (if given) is called with each segment's
    end timestamp (seconds into the audio) as segments are decoded, so the
    caller can report real progress through a step that can otherwise run
    for a very long time before anything else happens."""
    segments, _info = model.transcribe(str(audio_path), beam_size=BEAM_SIZE)
    result = []
    for s in segments:
        if should_cancel is not None and should_cancel():
            raise StepCancelled(f"Transcription cancelled: {audio_path.name}")
        result.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
        if on_progress is not None:
            on_progress(s.end)
    return result
