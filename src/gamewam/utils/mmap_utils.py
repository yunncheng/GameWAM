"""Utilities for memory-mapped file page-cache management."""

import ctypes
import ctypes.util

_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_POSIX_FADV_DONTNEED = 4


def fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    """Tell the kernel it may drop *offset..offset+length* from the page cache.

    After reading data from a memory-mapped file, call this to prevent the
    OS page cache from growing to the full file size.  Best-effort: failures
    are silently ignored.
    """
    if fd < 0 or length <= 0:
        return
    try:
        _LIBC.posix_fadvise(
            ctypes.c_int(fd),
            ctypes.c_longlong(offset),
            ctypes.c_longlong(length),
            ctypes.c_int(_POSIX_FADV_DONTNEED),
        )
    except Exception:
        pass  # best-effort
