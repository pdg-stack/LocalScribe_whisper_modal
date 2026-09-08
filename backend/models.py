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


class PreviewRequest(TranscribeRequest):
    pass
