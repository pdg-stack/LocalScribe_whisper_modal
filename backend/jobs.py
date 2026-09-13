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
    # Captured in create_job() (called from the async route handler, so a
    # loop is always running then) -- emit() is called both from run_job
    # itself (on the loop) and, far more often, from the worker threads
    # asyncio.to_thread runs each pipeline step in (progress callbacks
    # alone fire roughly every 0.5s during Transcription/Audio
    # Extraction). asyncio.Queue.put_nowait() is documented as not
    # thread-safe -- it can wake a waiting getter via loop.call_soon(),
    # which itself must only ever be called from the loop's own thread.
    # Routing every emit() through call_soon_threadsafe() (safe to call
    # from any thread, including the loop's own) avoids that race
    # entirely, at the cost of a tiny bit of overhead for the handful of
    # calls that were already on the loop thread.
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def emit(self, event: dict) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)
        else:
            self.queue.put_nowait(event)
        if event.get("event") == "done":
            self.finished = True


_jobs: dict[str, Job] = {}


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    job._loop = asyncio.get_running_loop()
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
