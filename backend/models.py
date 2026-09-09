"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ScanRequest(BaseModel):
    folder_path: str


class PreferencesModel(BaseModel):
    model: str | None = None
    formats: list[str] | None = None
    execution: str | None = None
    gpu: str | None = None
    cleanup: bool | None = None
    # Persisted at the user's explicit request so they don't have to
    # re-enter them each time -- stored in plaintext in user_prefs.json
    # (gitignored, local-only, never transmitted anywhere but Modal).
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    # Optional: raises the Hugging Face Hub rate limit for downloading
    # Whisper model weights (unauthenticated requests are capped lower).
    # Never transmitted anywhere but Hugging Face itself, and only ever
    # set as an env var for the process/container doing the download.
    hf_token: str | None = None


class FileRef(BaseModel):
    path: str
    type: str  # "video" | "audio"
    duration_sec: float


class TranscribeRequest(BaseModel):
    folder_path: str
    files: list[FileRef]
    model: str
    formats: list[str]
    execution: str  # "local" | "modal"
    gpu: str | None = None
    cleanup: bool = False
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    hf_token: str | None = None


class PreviewRequest(TranscribeRequest):
    pass
