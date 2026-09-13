"""Locate ffmpeg/ffprobe, probe media duration, extract audio."""

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from backend.jobs import StepCancelled


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


def _parse_out_time_seconds(value: str) -> float | None:
    # "-progress" reports elapsed output time as HH:MM:SS.ffffff -- parsed
    # directly rather than via the out_time_ms/out_time_us fields, whose
    # actual unit has varied across ffmpeg versions (a long-standing,
    # never-fixed quirk); the HH:MM:SS string is unambiguous.
    try:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None


def extract_audio(
    source_path: Path,
    wav_path: Path,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Extract a 16kHz mono WAV from source_path via ffmpeg, overwriting wav_path.

    Runs the process via Popen with -progress pipe:1, so ffmpeg reports
    machine-readable progress (out_time=HH:MM:SS.ffffff per block) instead
    of just running silently until it exits. Two dedicated reader threads
    drain stdout (progress) and stderr (diagnostics) concurrently --
    Windows pipes can't be polled with a timeout directly the way a
    socket can, so this is the portable way to read them incrementally
    while still checking should_cancel() on a regular cadence. Both pipes
    must be drained as they're produced, not just at the end: if only
    stdout were read while ffmpeg writes enough to stderr to fill its OS
    pipe buffer, ffmpeg would block on that write and the whole process
    would stall until the 3600s timeout, since a blocked ffmpeg never
    reaches EOF on stdout either."""
    ffmpeg = find_ffmpeg()
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-i", str(source_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-progress", "pipe:1", "-nostats",
            str(wav_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    progress_lines: "queue.Queue[str]" = queue.Queue()
    stderr_lines: list[str] = []

    def _read_progress() -> None:
        try:
            for line in proc.stdout:
                progress_lines.put(line)
        except Exception:
            pass

    def _read_stderr() -> None:
        try:
            for line in proc.stderr:
                stderr_lines.append(line)
        except Exception:
            pass

    reader = threading.Thread(target=_read_progress, daemon=True)
    stderr_reader = threading.Thread(target=_read_stderr, daemon=True)
    reader.start()
    stderr_reader.start()

    started = time.monotonic()
    finished = False
    while not finished:
        try:
            line = progress_lines.get(timeout=0.2).strip()
            if on_progress is not None and line.startswith("out_time="):
                seconds = _parse_out_time_seconds(line.split("=", 1)[1])
                if seconds is not None:
                    on_progress(seconds)
            elif line == "progress=end":
                finished = True
        except queue.Empty:
            if proc.poll() is not None:
                finished = True  # process exited -- no more progress lines coming

        if not finished:
            if should_cancel is not None and should_cancel():
                proc.kill()
                proc.wait()
                raise StepCancelled(f"Audio extraction cancelled: {source_path.name}") from None
            if time.monotonic() - started > 3600:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"ffmpeg timed out extracting audio from {source_path.name}")

    proc.wait()
    reader.join(timeout=1.0)
    stderr_reader.join(timeout=1.0)
    stderr = "".join(stderr_lines)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio from {source_path.name}: {stderr[-500:]}")
