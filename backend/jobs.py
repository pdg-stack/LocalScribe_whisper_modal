"""In-memory job registry: one asyncio.Queue of SSE events per job, plus
a cooperative cancel flag checked between pipeline steps/files.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field


class StepCancelled(Exception):
    """Raised from inside a running step (ffmpeg extraction, local Whisper
    inference, a Modal RPC) when it notices job.cancel_requested mid-flight
    and has stopped itself early, rather than running to completion."""


@dataclass
class Job:
    id: str
    queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    cancel_requested: bool = False
    finished: bool = False

    def emit(self, event: dict) -> None:
        self.queue.put_nowait(event)
        if event.get("event") == "done":
            self.finished = True


_jobs: dict[str, Job] = {}


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def request_cancel(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None:
        return False
    job.cancel_requested = True
    return True
