"""Real local hardware detection for the Options step's "Local resources
detected" line -- stdlib-only (no new dependency) since this only needs
to run on the same Windows machine as the server. Local execution always
runs on CPU (see transcription/engine.py -- device="cpu" is hardcoded,
faster-whisper's local path never uses a GPU even if one is present), so
this deliberately reports CPU/RAM only and says so, rather than detecting
and displaying a GPU that the app wouldn't actually use.
"""

from __future__ import annotations

import ctypes
import os
import platform


def _cpu_name() -> str:
    # The registry has a much friendlier string (e.g. "Intel(R) Core(TM)
    # i7-12700K CPU @ 3.60GHz") than platform.processor()'s raw value.
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if name and name.strip():
                return name.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _total_ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return None
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return None


def get_local_resources() -> dict:
    cpu_count = os.cpu_count() or 0
    ram_gb = _total_ram_gb()
    return {
        "cpu": f"{_cpu_name()} — {cpu_count} logical processors",
        "ram": f"{ram_gb:.0f} GB RAM" if ram_gb is not None else "Unknown RAM",
        # Not a detection result -- states a real constraint of this app
        # so the CPU/RAM numbers above aren't read as "and the GPU will
        # be used too".
        "note": "Local execution runs on CPU only (int8) -- select Modal.com to use a GPU.",
    }
