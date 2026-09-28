"""Platform support for temporary, executable pytest compilation caches."""

import ctypes
import mmap
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryFile


def ram_cache_directory() -> Path | None:
    """Probe file-backed executable mappings, as used to load compiled libraries."""
    if sys.platform != "linux":
        return None
    ram = Path("/dev/shm")
    try:
        with TemporaryFile(dir=ram) as probe:
            probe.truncate(mmap.PAGESIZE)
            with mmap.mmap(
                probe.fileno(),
                mmap.PAGESIZE,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_EXEC,
            ):
                pass
    except OSError:
        return None
    return ram


def defer_windows_cleanup(root: str) -> None:
    """Start a helper that owns cleanup after this process unloads its DLLs."""
    helper = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), root, str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert helper.stdout is not None
    with helper.stdout:
        # The parent stays alive until the helper has opened its process handle,
        # avoiding a race with exit or PID reuse. No pytest imports in the helper.
        if helper.stdout.readline() != b"ready\n":
            helper.wait()
            raise RuntimeError("Could not start post-exit compilation cache cleanup")


def _wait_for_windows_parent(parent_pid: int) -> None:
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    synchronize, infinite = 0x00100000, 0xFFFFFFFF
    handle = kernel32.OpenProcess(synchronize, False, parent_pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sys.stdout.buffer.write(b"ready\n")
        sys.stdout.buffer.flush()
        if kernel32.WaitForSingleObject(handle, infinite) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    _wait_for_windows_parent(int(sys.argv[2]))
    shutil.rmtree(sys.argv[1])
