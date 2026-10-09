"""Windows named mutex so ASR / tray stay single-instance."""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

SERVER_MUTEX = "Global\\QwenASRServerSingleton"
TRAY_MUTEX = "Global\\QwenASRTraySingleton"

_ERROR_ALREADY_EXISTS = 183
_held: list[object] = []


def try_acquire(name: str) -> bool:
    """Return True if this process owns the mutex. Keep the handle for process life."""
    if sys.platform != "win32":
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CreateMutexW.restype = wintypes.HANDLE
    handle = k32.CreateMutexW(None, True, name)
    if not handle:
        return True
    _held.append(handle)
    return ctypes.get_last_error() != _ERROR_ALREADY_EXISTS
