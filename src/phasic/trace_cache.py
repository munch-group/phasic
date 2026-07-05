"""
Trace Cache Management Utilities

Provides Python-level tools for managing trace caching.
Caching happens both at C level and Python level.
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path


def get_cache_dir() -> Path:
    """Get path to trace cache directory.

    Returns
    -------
    Path
        Path to ``~/.phasic_cache/traces/``.
    """
    home = Path.home()
    cache_dir = home / ".phasic_cache" / "traces"
    return cache_dir


def clear_trace_cache() -> int:
    """
    Clear all cached elimination traces.

    Returns
    -------
    int
        Number of cache files removed.
    """
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        return 0

    count = 0
    for cache_file in cache_dir.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except OSError:
            pass

    return count


def get_trace_cache_stats() -> dict[str, object]:
    """
    Get statistics about the trace cache.

    Returns
    -------
    dict[str, object]
        Dictionary with cache statistics:
        - total_files: Number of cached traces
        - total_bytes: Total disk space used
        - total_mb: Total disk space used in megabytes
        - cache_dir: Path to cache directory
    """
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        return {
            "total_files": 0,
            "total_bytes": 0,
            "cache_dir": str(cache_dir)
        }

    total_files = 0
    total_bytes = 0

    for cache_file in cache_dir.glob("*.json"):
        total_files += 1
        try:
            total_bytes += cache_file.stat().st_size
        except OSError:
            pass

    return {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "cache_dir": str(cache_dir)
    }


def print_trace_cache_info() -> None:
    """
    Print formatted trace cache information.

    Displays statistics about cached elimination traces in a human-readable format.

    Examples
    --------
    >>> from phasic import print_trace_cache_info
    >>> print_trace_cache_info()
    Cache directory: /Users/you/.phasic_cache/traces
    Cached traces: 12
    Total size: 5.67 MB
    """
    stats = get_trace_cache_stats()

    print(f"Cache directory: {stats['cache_dir']}")

    if stats['total_files'] == 0:
        print("Status: No cached traces")
    else:
        print(f"Cached traces: {stats['total_files']}")
        print(f"Total size: {stats['total_mb']:.2f} MB")


def verify_cache_working() -> dict[str, object]:
    """
    Verify that trace cache is working correctly.

    Returns
    -------
    dict[str, object]
        Dictionary with cache status:
        - cache_dir: Path to cache directory
        - exists: Whether cache directory exists
        - writable: Whether we can write to cache
        - readable: Whether we can read from cache
        - test_passed: Whether test write/read succeeded
        - error: Error message if any test failed
        - disabled: Whether cache is disabled via environment variable

    Examples
    --------
    >>> from phasic.trace_cache import verify_cache_working
    >>> status = verify_cache_working()
    >>> if not status['test_passed']:
    ...     print(f"Cache not working: {status['error']}")
    """
    import tempfile
    import time

    cache_dir = get_cache_dir()

    status = {
        "cache_dir": str(cache_dir),
        "exists": False,
        "writable": False,
        "readable": False,
        "test_passed": False,
        "error": None,
        "disabled": os.environ.get('PHASIC_REWARD_COMPUTE_CACHE') != '1'
    }

    if status["disabled"]:
        status["error"] = (
            "Reward-compute cache disabled (default policy). "
            "Enable via phasic.configure(reward_compute_cache=True)."
        )
        return status

    # Check if directory exists
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            status["exists"] = True
        except Exception as e:
            status["error"] = f"Failed to create cache directory: {e}"
            return status
    else:
        status["exists"] = True

    # Check if writable
    try:
        test_file = cache_dir / f"test_{time.time()}.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        status["writable"] = True

        # Check if readable
        with open(test_file, 'r') as f:
            content = f.read()
        if content == "test":
            status["readable"] = True
            status["test_passed"] = True
        else:
            status["error"] = "Cache read returned incorrect data"

        # Cleanup
        test_file.unlink()

    except Exception as e:
        status["error"] = f"Cache read/write test failed: {e}"

    return status


