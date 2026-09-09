"""Shared faster-whisper wrapper. One model instance is cached per
(model_name, device) pair and reused across files/jobs so it's only
loaded once -- loading is the expensive part, not switching audio files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

from faster_whisper import WhisperModel

from backend.config import BEAM_SIZE
from backend.jobs import StepCancelled

_model_cache: dict[tuple[str, str], WhisperModel] = {}


class Segment(NamedTuple):
    start: float
    end: float
    text: str


def get_model(model_name: str, device: str = "cpu") -> WhisperModel:
    key = (model_name, device)
    if key not in _model_cache:
        compute_type = "int8" if device == "cpu" else "float16"
        _model_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model_cache[key]


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> list[Segment]:
    """model.transcribe() returns a lazily-computed generator -- each
    segment is only decoded when we ask for the next one -- so checking
    should_cancel() between segments stops the job as soon as the segment
    in flight finishes, rather than waiting for the whole file. That's as
    fine-grained as it gets without running inference in a killable
    subprocess: a single segment's decode itself can't be interrupted
    mid-computation."""
    segments, _info = model.transcribe(str(audio_path), beam_size=BEAM_SIZE)
    result = []
    for s in segments:
        if should_cancel is not None and should_cancel():
            raise StepCancelled(f"Transcription cancelled: {audio_path.name}")
        result.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
    return result
