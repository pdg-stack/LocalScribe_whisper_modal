"""Format faster-whisper segments into txt/srt/vtt/json/tsv output files."""

from __future__ import annotations

import json
from pathlib import Path

from backend.transcription.engine import Segment


def _srt_timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_txt(segments: list[Segment], out_path: Path) -> None:
    out_path.write_text("\n".join(s.text for s in segments) + "\n", encoding="utf-8")


def write_srt(segments: list[Segment], out_path: Path) -> None:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(s.start)} --> {_srt_timestamp(s.end)}")
        lines.append(s.text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[Segment], out_path: Path) -> None:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_vtt_timestamp(s.start)} --> {_vtt_timestamp(s.end)}")
        lines.append(s.text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments: list[Segment], out_path: Path) -> None:
    data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tsv(segments: list[Segment], out_path: Path) -> None:
    lines = ["start\tend\ttext"]
    for s in segments:
        lines.append(f"{round(s.start * 1000)}\t{round(s.end * 1000)}\t{s.text}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


WRITERS = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
    "tsv": write_tsv,
}


def write_format(fmt: str, segments: list[Segment], out_path: Path) -> None:
    WRITERS[fmt](segments, out_path)
