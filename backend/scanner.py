"""Recursive folder walk + media classification + ffprobe-based metadata."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from backend.config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from backend.ffmpeg_utils import probe_duration_seconds

MediaType = Literal["video", "audio"]


class FolderNotFoundError(FileNotFoundError):
    pass


def classify(path: Path) -> MediaType | None:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return None


def _probe_one(root: Path, file_path: Path) -> dict:
    return {
        "path": str(file_path.relative_to(root)),
        "size_bytes": file_path.stat().st_size,
        "duration_sec": probe_duration_seconds(file_path),
    }


def _collect(root: Path, files: list[Path]) -> dict:
    by_type: dict[MediaType, list[Path]] = {"video": [], "audio": []}
    for f in files:
        media_type = classify(f)
        if media_type:
            by_type[media_type].append(f)

    result = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for media_type, type_files in by_type.items():
            probed = list(pool.map(lambda f: _probe_one(root, f), type_files))
            probed.sort(key=lambda entry: entry["path"].lower())
            result[media_type] = {"files": probed}
    return result


def scan_folder(folder_path: str) -> dict:
    root = Path(folder_path)
    if not root.exists():
        raise FolderNotFoundError(f"Folder does not exist: {folder_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    current_files = [root / name for name in os.listdir(root) if (root / name).is_file()]

    all_files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            all_files.append(Path(dirpath) / name)

    return {
        "folder": str(root),
        "scopes": {
            "current": _collect(root, current_files),
            "all": _collect(root, all_files),
        },
    }
