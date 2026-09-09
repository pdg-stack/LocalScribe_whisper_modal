"""FastAPI app: serves the frontend and the API used by it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.estimator import aggregate_phases, compute_steps
from backend.ffmpeg_utils import FfmpegNotFoundError
from backend.jobs import create_job, get_job, request_cancel
from backend.models import PreferencesModel, PreviewRequest, ScanRequest, TranscribeRequest
from backend.pipeline import run_job
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
    # Includes modal_token_id/modal_token_secret at the user's request --
    # written in plaintext to PREFS_PATH (gitignored, local-only).
    data = {k: v for k, v in prefs.model_dump().items() if v is not None}
    PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


@app.post("/api/preview")
def api_preview(req: PreviewRequest):
    if req.execution == "modal" and not req.gpu:
        raise HTTPException(status_code=400, detail="GPU type is required for Modal.com execution.")
    files = [f.model_dump() for f in req.files]
    steps = compute_steps(files, req.model, req.execution, req.gpu, req.cleanup)
    phases = aggregate_phases(steps, req.execution, req.cleanup)
    total_sec = sum(p["sec"] for p in phases)
    total_cost = sum(p["cost"] for p in phases)
    return {"phases": phases, "total": {"sec": total_sec, "cost": total_cost}}


@app.post("/api/transcribe")
async def api_transcribe(req: TranscribeRequest):
    if req.execution == "modal" and not (req.modal_token_id and req.modal_token_secret):
        raise HTTPException(status_code=400, detail="Modal Token ID and Token Secret are required for Modal.com execution.")
    job = create_job()
    files = [f.model_dump() for f in req.files]
    asyncio.create_task(run_job(
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
