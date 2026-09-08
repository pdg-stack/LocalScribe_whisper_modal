"""FastAPI app: serves the frontend and the API used by it."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.ffmpeg_utils import FfmpegNotFoundError
from backend.models import PreferencesModel, ScanRequest
from backend.scanner import FolderNotFoundError, scan_folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PREFS_PATH = PROJECT_ROOT / "user_prefs.json"

app = FastAPI(title="LocalScribe_whisper_modal")


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    try:
        return scan_folder(req.folder_path)
    except FolderNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FfmpegNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preferences")
def get_preferences():
    if not PREFS_PATH.exists():
        return {}
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@app.post("/api/preferences")
def save_preferences(prefs: PreferencesModel):
    # Only ever the non-secret fields below -- credentials are never
    # accepted here and never touch disk.
    data = {k: v for k, v in prefs.model_dump().items() if v is not None}
    PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# Mounted last so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
