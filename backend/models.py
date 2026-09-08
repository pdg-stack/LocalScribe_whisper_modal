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
