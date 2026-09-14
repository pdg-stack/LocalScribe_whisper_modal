"""FastAPI app: serves the frontend and the API used by it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.estimator import UploadSizeExceededError, aggregate_phases, compute_steps
from backend.ffmpeg_utils import FfmpegNotFoundError
from backend.jobs import create_job, get_job, request_cancel
from backend.models import PreferencesModel, PreviewRequest, ScanRequest, TranscribeRequest
from backend.pipeline import run_job_safe
from backend.scanner import FolderNotFoundError, scan_folder
from backend.system_info import get_local_resources

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PREFS_PATH = PROJECT_ROOT / "user_prefs.json"

app = FastAPI(title="LocalScribe_whisper_modal")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    # This app has no build step or cache-busting filenames -- editing
    # frontend/app.js takes effect immediately on disk, but browsers cache
    # static files by default, so an already-open tab (or even a plain
    # reload) can keep running a stale copy after an update. Since this
    # is a single-user local tool, unconditionally disabling caching costs
    # nothing and removes an entire class of "I fixed it but the browser
    # didn't notice" confusion.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


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
    # Includes modal_token_id/modal_token_secret at the user's request --
    # written in plaintext to PREFS_PATH (gitignored, local-only).
    # Merged into whatever's already on disk rather than overwriting the
    # whole file -- callers legitimately send partial updates (e.g. just
    # folder_history after a scan, separately from the options form's
    # full save), and a full overwrite would silently drop every field
    # that particular caller didn't happen to know about.
    #
    # exclude_unset=True (not "if v is not None") is what makes this
    # correct: it merges in only the fields this specific request body
    # actually included, whether their value is a real value or an
    # explicit null -- so a request that omits a field leaves it alone,
    # but a request that explicitly sends e.g. modal_token_id: null (the
    # frontend's way of saying "the field was cleared") actually clears
    # it, instead of the old value silently surviving forever because a
    # merge can never overwrite anything with a filtered-out None.
    existing = {}
    if PREFS_PATH.exists():
        try:
            existing = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(prefs.model_dump(exclude_unset=True))
    PREFS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing


@app.post("/api/browse-folder")
def browse_folder():
    # Opens a native OS folder-picker on the machine running the server --
    # only reasonable because this is a single-user, localhost-only tool
    # where the server and the browser are always the same machine.
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise HTTPException(status_code=501, detail="Native folder browser is not available on this system.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory()
    finally:
        root.destroy()
    return {"path": path or None}


@app.get("/api/local-resources")
def api_local_resources():
    return get_local_resources()


@app.post("/api/preview")
def api_preview(req: PreviewRequest):
    if req.execution == "modal" and not req.gpu:
        raise HTTPException(status_code=400, detail="GPU type is required for Modal.com execution.")
    files = [f.model_dump() for f in req.files]
    try:
        steps = compute_steps(files, req.model, req.execution, req.gpu, req.cleanup)
    except UploadSizeExceededError as e:
        raise HTTPException(status_code=400, detail=str(e))
    phases = aggregate_phases(steps, req.execution, req.cleanup)
    total_sec = sum(p["sec"] for p in phases)
    total_cost = sum(p["cost"] for p in phases)
    return {"phases": phases, "total": {"sec": total_sec, "cost": total_cost}}


@app.post("/api/transcribe")
async def api_transcribe(req: TranscribeRequest):
    if req.execution == "modal" and not (req.modal_token_id and req.modal_token_secret):
        raise HTTPException(status_code=400, detail="Modal Token ID and Token Secret are required for Modal.com execution.")
    files = [f.model_dump() for f in req.files]
    try:
        # Defense in depth -- Begin never appears without a successful
        # Preview first (which already runs this same check), but this
        # endpoint could in principle be called directly.
        compute_steps(files, req.model, req.execution, req.gpu, req.cleanup)
    except UploadSizeExceededError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = create_job()
    asyncio.create_task(run_job_safe(
        job, req.folder_path, files, req.model, req.formats, req.execution, req.gpu, req.cleanup,
        req.modal_token_id, req.modal_token_secret, req.hf_token,
    ))
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}/stream")
async def api_job_stream(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    async def event_source():
        while True:
            event = await job.queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("event") == "done":
                break

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    if not request_cancel(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return {"ok": True}


# Mounted last so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
