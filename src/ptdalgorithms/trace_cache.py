"""
Trace Cache Management Utilities

Provides Python-level tools for managing trace caching.
Caching happens both at C level and Python level.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional


def get_cache_dir() -> Path:
    """Get path to trace cache directory"""
    home = Path.home()
    cache_dir = home / ".ptdalgorithms_cache" / "traces"
    return cache_dir


def clear_trace_cache() -> int:
    """
    Clear all cached elimination traces

    Returns:
        Number of cache files removed
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


def get_trace_cache_stats() -> Dict[str, any]:
    """
    Get statistics about the trace cache

    Returns:
        Dictionary with cache statistics:
        - total_files: Number of cached traces
        - total_bytes: Total disk space used
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


def list_cached_traces() -> List[Dict[str, any]]:
    """
    List all cached elimination traces

    Returns:
        List of dictionaries with cache entry information:
        - hash: Content hash of the graph
        - size_bytes: File size in bytes
        - modified: Last modification time
    """
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        return []

    entries = []

    for cache_file in cache_dir.glob("*.json"):
        try:
            stat = cache_file.stat()
            hash_key = cache_file.stem  # Filename without .json

            # Try to get metadata from JSON
            metadata = {}
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    metadata['n_vertices'] = data.get('n_vertices')
                    metadata['param_length'] = data.get('param_length')
                    metadata['n_operations'] = len(data.get('operations', []))
            except (json.JSONDecodeError, OSError):
                pass

            entries.append({
                "hash": hash_key,
                "size_bytes": stat.st_size,
                "size_kb": stat.st_size / 1024,
                "modified": stat.st_mtime,
                **metadata
            })
        except OSError:
            pass

    # Sort by modification time (most recent first)
    entries.sort(key=lambda x: x['modified'], reverse=True)

    return entries


def remove_cached_trace(hash_key: str) -> bool:
    """
    Remove a specific cached trace by hash

    Args:
        hash_key: Content hash of the graph

    Returns:
        True if cache entry was removed, False otherwise
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{hash_key}.json"

    if not cache_file.exists():
        return False

    try:
        cache_file.unlink()
        return True
    except OSError:
        return False


def cleanup_old_traces(max_size_mb: float = 100.0, max_age_days: Optional[int] = None) -> int:
    """
    Clean up old or excess cached traces

    Args:
        max_size_mb: Maximum total cache size in MB
        max_age_days: Remove traces older than this many days (None = no age limit)

    Returns:
        Number of cache entries removed
    """
    import time

    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        return 0

    # Get all cache entries with metadata
    entries = []
    for cache_file in cache_dir.glob("*.json"):
        try:
            stat = cache_file.stat()
            entries.append({
                "path": cache_file,
                "size": stat.st_size,
                "mtime": stat.st_mtime
            })
        except OSError:
            pass

    removed = 0

    # Remove by age if specified
    if max_age_days is not None:
        cutoff_time = time.time() - (max_age_days * 86400)
        for entry in entries[:]:
            if entry["mtime"] < cutoff_time:
                try:
                    entry["path"].unlink()
                    entries.remove(entry)
                    removed += 1
                except OSError:
                    pass

    # Remove by size (LRU - remove oldest first)
    max_bytes = max_size_mb * 1024 * 1024
    total_bytes = sum(e["size"] for e in entries)

    if total_bytes > max_bytes:
        # Sort by modification time (oldest first)
        entries.sort(key=lambda x: x["mtime"])

        while total_bytes > max_bytes and entries:
            entry = entries.pop(0)
            try:
                entry["path"].unlink()
                total_bytes -= entry["size"]
                removed += 1
            except OSError:
                pass

    return removed


def save_trace_to_cache_python(graph, trace):
    """
    Save elimination trace to cache (Python-level)

    Args:
        graph: The graph that was eliminated
        trace: The recorded elimination trace
    """
    # Compute hash from graph
    graph_data = graph.serialize()

    # Create hash dict (structure only, no parameter values)
    hash_dict = {
        "state_length": graph_data.get("state_length"),
        "n_vertices": len(graph_data.get("vertices", [])),
        "edges": []
    }

    # Add edge structure (parameterized status and connectivity)
    for v in graph_data.get("vertices", []):
        for e in v.get("edges", []):
            hash_dict["edges"].append({
                "to": e.get("to_vertex_index"),
                "parameterized": e.get("parameterized", False),
                "state_length": len(e.get("edge_state", [])) if e.get("parameterized") else 0
            })

    # Compute SHA256 hash
    json_str = json.dumps(hash_dict, sort_keys=True)
    hash_hex = hashlib.sha256(json_str.encode()).hexdigest()

    # Save trace JSON
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{hash_hex}.json"

    # Use existing save_trace_json function from trace_elimination
    from .trace_elimination import save_trace_json
    save_trace_json(trace, str(cache_file))
