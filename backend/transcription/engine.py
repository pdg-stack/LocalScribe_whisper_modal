"""Shared faster-whisper wrapper. One model instance is cached per
(model_name, device) pair and reused across files/jobs so it's only
loaded once -- loading is the expensive part, not switching audio files.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from faster_whisper import WhisperModel

from backend.config import BEAM_SIZE

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


def transcribe(model: WhisperModel, audio_path: Path) -> list[Segment]:
    segments, _info = model.transcribe(str(audio_path), beam_size=BEAM_SIZE)
    return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
