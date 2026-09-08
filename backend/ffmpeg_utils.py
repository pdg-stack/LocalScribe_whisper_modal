"""Locate ffmpeg/ffprobe, probe media duration, extract audio."""

import json
import shutil
import subprocess
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it (e.g. `scoop install ffmpeg` "
            "or https://ffmpeg.org/download.html) and restart the server."
        )
    return path


def find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FfmpegNotFoundError(
            "ffprobe was not found on PATH. It ships alongside ffmpeg -- "
            "reinstall ffmpeg if only ffprobe is missing."
        )
    return path


def probe_duration_seconds(file_path: Path) -> float:
    """Return media duration in seconds via ffprobe, or 0.0 if unreadable."""
    ffprobe = find_ffprobe()
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0.0))
    except (subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        return 0.0


def extract_audio(source_path: Path, wav_path: Path) -> None:
    """Extract a 16kHz mono WAV from source_path via ffmpeg, overwriting wav_path."""
    ffmpeg = find_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg, "-y", "-i", str(source_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio from {source_path.name}: {result.stderr[-500:]}")
