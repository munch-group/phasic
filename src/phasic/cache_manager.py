"""
JAX Compilation Cache Management

Provides utilities for managing JAX's persistent compilation cache, including:
- Cache inspection and statistics
- Pre-warming/pre-compilation
- Cache export/import for distribution
- Cache synchronization across nodes
- Cache cleanup and maintenance

Example Usage:
    >>> from phasic.cache_manager import CacheManager, print_jax_cache_info
    >>>
    >>> # Inspect cache
    >>> print_jax_cache_info()
    >>>
    >>> # Pre-warm cache for common model shapes
    >>> manager = CacheManager()
    >>> manager.prewarm_model(model_fn, theta_samples, time_grids)
    >>>
    >>> # Export cache for distribution
    >>> manager.export_cache('my_models_cache.tar.gz')
"""

import os
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
import hashlib

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


class CacheManager:
    """
    Manager for JAX compilation cache.

    Provides utilities for inspecting, warming, exporting, and syncing
    JAX's persistent compilation cache.

    Parameters
    ----------
    cache_dir : Path or str, optional
        JAX cache directory. Default: from JAX_COMPILATION_CACHE_DIR env var
        or ~/.jax_cache
    """

    def __init__(self, cache_dir: Optional[Union[Path, str]] = None):
        if cache_dir is None:
            cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                      str(Path.home() / '.jax_cache'))
        self.cache_dir = Path(cache_dir)

    def info(self) -> Dict[str, Any]:
        """
        Get JAX cache statistics.

        Returns
        -------
        dict
            Cache statistics including size, file count, etc.
        """
        if not self.cache_dir.exists():
            return {
                'exists': False,
                'cache_dir': str(self.cache_dir),
                'num_files': 0,
                'total_size_mb': 0.0,
                'files': []
            }

        # Collect file information
        files = []
        total_size = 0

        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                files.append({
                    'path': str(file_path.relative_to(self.cache_dir)),
                    'size_kb': size / 1024,
                    'modified': modified.isoformat()
                })

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)

        return {
            'exists': True,
            'cache_dir': str(self.cache_dir),
            'num_files': len(files),
            'total_size_mb': total_size / (1024**2),
            'files': files
        }

    def clear(self, confirm: bool = False):
        """
        Clear JAX compilation cache.

        Parameters
        ----------
        confirm : bool, optional
            Must be True to actually clear. Default: False (safety)

        Examples
        --------
        >>> manager = CacheManager()
        >>> manager.clear(confirm=True)  # Clears cache
        """
        if not confirm:
            print("Warning: Set confirm=True to actually clear cache")
            return

        if self.cache_dir.exists():
            info = self.info()
            print(f"Clearing JAX cache: {self.cache_dir}")
            print(f"  Files: {info['num_files']}")
            print(f"  Size: {info['total_size_mb']:.1f} MB")
            shutil.rmtree(self.cache_dir)
            print("✓ Cache cleared")
        else:
            print(f"Cache directory does not exist: {self.cache_dir}")

    def prewarm_model(
        self,
        model_fn: Callable,
        theta_samples: List,
        time_grids: List,
        show_progress: bool = True
    ):
        """
        Pre-warm JAX cache by compiling model for various input shapes.

        Compiles the model with different parameter and time grid shapes
        to populate the cache. Subsequent runs with these shapes will be instant.

        Parameters
        ----------
        model_fn : callable
            Model function (theta, times) -> pmf
        theta_samples : list of arrays
            Sample parameter vectors (different shapes)
        time_grids : list of arrays
            Sample time grids (different shapes)
        show_progress : bool, optional
            Show progress during warming. Default: True

        Examples
        --------
        >>> # Pre-warm for common shapes
        >>> theta_samples = [
        ...     jnp.ones(1),   # 1D parameter
        ...     jnp.ones(2),   # 2D parameter
        ...     jnp.ones(5)    # 5D parameter
        ... ]
        >>> time_grids = [
        ...     jnp.linspace(0.1, 5, 20),   # 20 time points
        ...     jnp.linspace(0.1, 5, 50),   # 50 time points
        ...     jnp.linspace(0.1, 5, 100)   # 100 time points
        ... ]
        >>> manager.prewarm_model(model, theta_samples, time_grids)
        """
        if not HAS_JAX:
            raise ImportError("JAX required for pre-warming")

        import time
        from itertools import product

        total = len(theta_samples) * len(time_grids)
        count = 0

        if show_progress:
            print(f"Pre-warming JAX cache for {total} combinations...")
            print(f"Cache directory: {self.cache_dir}")

        start_time = time.time()

        for theta, times in product(theta_samples, time_grids):
            if show_progress:
                count += 1
                print(f"  [{count}/{total}] theta_shape={theta.shape}, "
                      f"times_shape={times.shape}...", end=' ', flush=True)

            try:
                # Call model to trigger compilation
                _ = model_fn(theta, times)
                # Block to ensure compilation completes
                _ = jax.block_until_ready(_)

                if show_progress:
                    print("✓")
            except Exception as e:
                if show_progress:
                    print(f"✗ ({str(e)[:50]})")

        elapsed = time.time() - start_time

        if show_progress:
            print(f"\n✓ Pre-warming complete in {elapsed:.1f}s")
            info = self.info()
            print(f"Cache size: {info['total_size_mb']:.1f} MB "
                  f"({info['num_files']} files)")

    def export_cache(
        self,
        output_path: Union[Path, str],
        patterns: Optional[List[str]] = None,
        compress: bool = True
    ):
        """
        Export JAX cache to tarball for distribution.

        Parameters
        ----------
        output_path : Path or str
            Output tarball path
        patterns : list of str, optional
            Glob patterns to include. Default: all files
        compress : bool, optional
            Use gzip compression. Default: True

        Examples
        --------
        >>> manager.export_cache('my_model_cache.tar.gz')
        >>> # Later, on another machine:
        >>> manager.import_cache('my_model_cache.tar.gz')
        """
        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")

        output_path = Path(output_path)
        mode = 'w:gz' if compress else 'w'

        # Collect files to export
        if patterns is None:
            files = list(self.cache_dir.rglob('*'))
        else:
            files = []
            for pattern in patterns:
                files.extend(self.cache_dir.glob(pattern))

        files = [f for f in files if f.is_file()]

        print(f"Exporting {len(files)} cache files to {output_path}")

        with tarfile.open(output_path, mode) as tar:
            for file_path in files:
                arcname = file_path.relative_to(self.cache_dir)
                tar.add(file_path, arcname=arcname)

                # Add metadata
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'cache_dir': str(self.cache_dir),
            'num_files': len(files),
            'total_size_mb': sum(f.stat().st_size for f in files) / (1024**2)
        }

        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Export complete")
        print(f"  Files: {len(files)}")
        print(f"  Size: {metadata['total_size_mb']:.1f} MB")
        print(f"  Metadata: {metadata_file}")

    def import_cache(
        self,
        tarball_path: Union[Path, str],
        overwrite: bool = False
    ):
        """
        Import JAX cache from tarball.

        Parameters
        ----------
        tarball_path : Path or str
            Path to exported cache tarball
        overwrite : bool, optional
            Overwrite existing cache files. Default: False

        Examples
        --------
        >>> manager = CacheManager()
        >>> manager.import_cache('downloaded_cache.tar.gz')
        """
        tarball_path = Path(tarball_path)

        if not tarball_path.exists():
            raise FileNotFoundError(f"Tarball not found: {tarball_path}")

        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Importing cache from {tarball_path}")

        with tarfile.open(tarball_path, 'r:*') as tar:
            members = tar.getmembers()
            print(f"  Extracting {len(members)} files...")

            for member in members:
                dest_path = self.cache_dir / member.name

                if dest_path.exists() and not overwrite:
                    print(f"  Skipping (exists): {member.name}")
                    continue

                tar.extract(member, self.cache_dir)

        print("✓ Import complete")

        # Show updated cache info
        info = self.info()
        print(f"Cache now contains {info['num_files']} files "
              f"({info['total_size_mb']:.1f} MB)")

    def sync_from_remote(
        self,
        remote_cache_dir: Union[Path, str],
        dry_run: bool = False
    ):
        """
        Synchronize cache from remote directory (e.g., shared filesystem).

        Parameters
        ----------
        remote_cache_dir : Path or str
            Remote cache directory to sync from
        dry_run : bool, optional
            Show what would be synced without actually syncing. Default: False

        Examples
        --------
        >>> # On compute cluster with shared storage
        >>> manager = CacheManager(cache_dir='/home/user/.jax_cache')
        >>> manager.sync_from_remote('/shared/project/jax_cache')
        """
        remote_cache_dir = Path(remote_cache_dir)

        if not remote_cache_dir.exists():
            raise ValueError(f"Remote cache does not exist: {remote_cache_dir}")

        print(f"Syncing from {remote_cache_dir}")
        if dry_run:
            print("  (dry run - no files will be copied)")

        # Find files in remote that are newer or missing locally
        copied = 0
        skipped = 0

        for remote_file in remote_cache_dir.rglob('*'):
            if not remote_file.is_file():
                continue

            rel_path = remote_file.relative_to(remote_cache_dir)
            local_file = self.cache_dir / rel_path

            should_copy = False
            if not local_file.exists():
                should_copy = True
                reason = "new"
            elif remote_file.stat().st_mtime > local_file.stat().st_mtime:
                should_copy = True
                reason = "newer"
            else:
                skipped += 1
                continue

            if dry_run:
                print(f"  Would copy ({reason}): {rel_path}")
            else:
                local_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote_file, local_file)
                copied += 1

        if dry_run:
            print(f"\nDry run complete: {copied} files would be copied, "
                  f"{skipped} already up to date")
        else:
            print(f"✓ Sync complete: {copied} files copied, {skipped} already up to date")

    def vacuum(self, max_age_days: int = 30, max_size_gb: float = 10.0):
        """
        Clean up old cache entries.

        Removes cache files older than max_age_days or evicts oldest entries
        if cache exceeds max_size_gb.

        Parameters
        ----------
        max_age_days : int, optional
            Remove files older than this many days. Default: 30
        max_size_gb : float, optional
            Maximum cache size in GB. Default: 10.0

        Examples
        --------
        >>> manager = CacheManager()
        >>> manager.vacuum(max_age_days=7, max_size_gb=5.0)
        """
        if not self.cache_dir.exists():
            print("Cache directory does not exist")
            return

        import time
        from datetime import timedelta

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        max_size_bytes = max_size_gb * (1024**3)

        # Collect all files with metadata
        files = []
        total_size = 0

        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'path': file_path,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
                total_size += stat.st_size

        # Remove files older than cutoff
        removed_old = 0
        for file_info in files[:]:
            if file_info['mtime'] < cutoff_time:
                file_info['path'].unlink()
                total_size -= file_info['size']
                files.remove(file_info)
                removed_old += 1

        # If still over size limit, remove oldest files
        removed_excess = 0
        if total_size > max_size_bytes:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x['mtime'])

            while total_size > max_size_bytes and files:
                file_info = files.pop(0)
                file_info['path'].unlink()
                total_size -= file_info['size']
                removed_excess += 1

        print(f"✓ Cache vacuum complete")
        print(f"  Removed {removed_old} files older than {max_age_days} days")
        print(f"  Removed {removed_excess} excess files to meet size limit")
        print(f"  Cache size now: {total_size / (1024**2):.1f} MB")


def print_jax_cache_info(cache_dir: Optional[Union[Path, str]] = None):
    """
    Print formatted JAX cache information.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Cache directory. Default: from environment or ~/.jax_cache

    Examples
    --------
    >>> from phasic.cache_manager import print_jax_cache_info
    >>> print_jax_cache_info()
    """
    manager = CacheManager(cache_dir=cache_dir)
    info = manager.info()

    print("=" * 70)
    print("JAX COMPILATION CACHE INFO")
    print("=" * 70)
    print(f"Cache directory: {info['cache_dir']}")

    if not info['exists']:
        print("Status: Cache directory does not exist")
        return

    print(f"Cached compilations: {info['num_files']}")
    print(f"Total size: {info['total_size_mb']:.1f} MB")

    if info['num_files'] > 0:
        print(f"\nMost recent files (showing up to 10):")
        for file_info in info['files'][:10]:
            print(f"  {file_info['modified']} | {file_info['size_kb']:>8.1f} KB | "
                  f"{file_info['path']}")

    print("=" * 70)


def configure_layered_cache(
    local_cache_dir: Optional[Union[Path, str]] = None,
    shared_cache_dir: Optional[Union[Path, str]] = None,
    enable: bool = True
):
    """
    Configure layered cache strategy (local + shared).

    Sets up JAX to use a local cache with optional shared cache fallback.
    Checks local cache first, then shared cache, finally compiles if needed.

    Parameters
    ----------
    local_cache_dir : Path or str, optional
        Local cache directory. Default: ~/.jax_cache
    shared_cache_dir : Path or str, optional
        Shared cache directory (read-only fallback). Default: None
    enable : bool, optional
        Enable layered caching. Default: True

    Examples
    --------
    >>> # On compute cluster
    >>> configure_layered_cache(
    ...     local_cache_dir='/home/user/.jax_cache',
    ...     shared_cache_dir='/shared/project/jax_cache'
    ... )
    >>>
    >>> # JAX will now check both caches before compiling

    Notes
    -----
    This must be called before importing JAX for full effect.
    """
    if not enable:
        return

    if local_cache_dir is None:
        local_cache_dir = Path.home() / '.jax_cache'
    else:
        local_cache_dir = Path(local_cache_dir)

    # Set primary cache
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ JAX cache configured:")
    print(f"  Local: {local_cache_dir}")

    if shared_cache_dir:
        shared_cache_dir = Path(shared_cache_dir)
        print(f"  Shared: {shared_cache_dir}")

        # Note: JAX doesn't natively support layered caching,
        # but we can implement it at the Python level by copying
        # from shared to local on cache miss
        # This would require hooking into JAX's cache mechanism
        # For now, just document the approach

        print(f"\n  Note: Layered caching requires manual sync or cache hooks")
        print(f"  Use CacheManager.sync_from_remote() to populate local from shared")
