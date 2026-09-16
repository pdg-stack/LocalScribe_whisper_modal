"""Keeps this (Windows) machine from going to sleep/hibernating while a
job is running, via the same Win32 SetThreadExecutionState API that
"caffeine"-style utilities use -- a job that runs for a long time (a
large local batch, or a slow Modal.com upload) can otherwise be cut off
mid-run by the OS's own idle-sleep timer (see modal_app.py's
ConflictError handling in pipeline.py for what that does to a Modal.com
job specifically).

This only overrides *automatic* idle-triggered sleep -- it does not, and
cannot, override a deliberately/manually triggered sleep (closing a
laptop's lid, the Start menu's Power > Sleep, a sleep hotkey): Windows
treats those as an explicit user action that takes priority over any
app's execution-state request, us included.
"""

from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> None:
    """Call once when a job starts. Safe to call on non-Windows/headless
    environments -- just logs and does nothing, since a job should never
    fail to *run* over a missing sleep-prevention nicety."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        logger.warning("Could not request sleep prevention for this job (not Windows, or the API is unavailable).")


def allow_sleep() -> None:
    """Call once when a job ends (success, failure, or cancellation) --
    restores normal idle-sleep behavior. Must be paired with every
    prevent_sleep() call, in a finally block, so a crashed job can't
    leave the machine permanently unable to sleep."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
