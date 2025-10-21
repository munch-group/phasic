"""
Model Distribution and Cache Management Utilities

Provides tools for:
- Exporting models with compilation configurations
- Generating warmup scripts for pre-populating JAX cache
- Managing JAX compilation cache

Note: This module provides a simplified API by wrapping CacheManager.
For advanced cache operations, use CacheManager directly.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

# Import CacheManager for internal use
from .cache_manager import CacheManager


def clear_cache(cache_dir: Optional[Union[Path, str]] = None, verbose: bool = True) -> None:
    """
    Clear JAX compilation cache.

    This is a simplified wrapper around CacheManager.clear().

    Parameters
    ----------
    cache_dir : Path or str, optional
        Cache directory to clear. If None, uses default from environment
        or ~/.jax_cache
    verbose : bool, optional
        Print information about cleared cache. Default: True

    Examples
    --------
    >>> from phasic import clear_cache
    >>> clear_cache()  # Clear default cache
    >>> clear_cache('/custom/cache/dir')  # Clear specific cache

    See Also
    --------
    CacheManager.clear : Advanced cache clearing with confirmation
    """
    manager = CacheManager(cache_dir=cache_dir)

    if not manager.cache_dir.exists():
        if verbose:
            print(f"Cache directory does not exist: {manager.cache_dir}")
        return

    # Get info before clearing
    if verbose:
        info = manager.info()
        print(f"Clearing cache: {manager.cache_dir}")
        print(f"  Files: {info['num_files']}")
        print(f"  Size: {info['total_size_mb']:.1f} MB")

    # Clear via CacheManager (which uses shutil.rmtree)
    manager.clear(confirm=True)

    if verbose:
        print(f"✓ Cache cleared successfully")


def cache_info(cache_dir: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Get information about JAX compilation cache.

    This is a simplified wrapper around CacheManager.info() with
    reformatted output for backward compatibility.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Cache directory to inspect. If None, uses default from environment
        or ~/.jax_cache

    Returns
    -------
    dict
        Cache statistics including:
        - 'exists': Whether cache directory exists
        - 'path': Cache directory path
        - 'num_files': Number of cached files
        - 'total_size_mb': Total cache size in megabytes
        - 'files': List of (filename, size_kb, modified_time) tuples

    Examples
    --------
    >>> from phasic import cache_info
    >>> info = cache_info()
    >>> print(f"Cache size: {info['total_size_mb']:.1f} MB")
    >>> print(f"Cached compilations: {info['num_files']}")

    See Also
    --------
    CacheManager.info : Returns info in slightly different format
    """
    manager = CacheManager(cache_dir=cache_dir)
    info = manager.info()

    # Reformat file list for backward compatibility
    # CacheManager returns list of dicts, we return list of tuples
    files_reformatted = []
    for file_dict in info['files']:
        files_reformatted.append((
            file_dict['path'],
            file_dict['size_kb'],
            file_dict['modified']
        ))

    return {
        'exists': info['exists'],
        'path': info['cache_dir'],
        'num_files': info['num_files'],
        'total_size_mb': info['total_size_mb'],
        'files': files_reformatted
    }


def print_cache_info(cache_dir: Optional[Union[Path, str]] = None, max_files: int = 10) -> None:
    """
    Print formatted cache information.

    This is a simplified wrapper around CacheManager functionality.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Cache directory to inspect. If None, uses default.
    max_files : int, optional
        Maximum number of files to display. Default: 10

    Examples
    --------
    >>> from phasic import print_cache_info
    >>> print_cache_info()  # Show cache statistics

    See Also
    --------
    print_jax_cache_info : Alternative from cache_manager module
    """
    info = cache_info(cache_dir)

    print("=" * 70)
    print("JAX COMPILATION CACHE INFO")
    print("=" * 70)
    print(f"Path: {info['path']}")

    if not info['exists']:
        print("Status: Cache directory does not exist")
        return

    print(f"Cached compilations: {info['num_files']}")
    print(f"Total size: {info['total_size_mb']:.1f} MB")

    if info['num_files'] > 0:
        print(f"\nMost recent files (showing {min(max_files, len(info['files']))}/{info['num_files']}):")
        for filename, size_kb, modified in info['files'][:max_files]:
            print(f"  {modified} | {size_kb:>8.1f} KB | {filename}")

    print("=" * 70)


def generate_warmup_script(
    output_path: Union[Path, str],
    model_code: str,
    theta_dim: int,
    n_particles: int = 100,
    data_shape: Optional[tuple] = None,
    config_path: Optional[Union[Path, str]] = None
) -> None:
    """
    Generate a Python script to pre-populate JAX compilation cache.

    This script can be distributed with models to allow users to pre-compile
    before running actual inference, making the first run much faster.

    Parameters
    ----------
    output_path : Path or str
        Path where warmup script will be saved
    model_code : str
        Python code to create the model (will be inserted in script)
    theta_dim : int
        Dimension of theta parameter
    n_particles : int, optional
        Number of particles for warmup compilation. Default: 100
    data_shape : tuple, optional
        Shape of observed data. If None, uses (20,) as default
    config_path : Path or str, optional
        Path to compilation config file (relative to script location)

    Examples
    --------
    >>> from phasic.model_export import generate_warmup_script
    >>>
    >>> model_code = '''
    ... def coalescent_callback(state, nr_samples=3):
    ...     if len(state) == 0:
    ...         return [(np.array([nr_samples]), 1.0, [1.0])]
    ...     if state[0] > 1:
    ...         n = state[0]
    ...         return [(np.array([n - 1]), 0.0, [n * (n - 1) / 2])]
    ...     return []
    ...
    ... graph = pta.Graph(callback=coalescent_callback, parameterized=True)
    ... model = pta.Graph.pmf_from_graph(graph, discrete=False)
    ... '''
    >>>
    >>> generate_warmup_script(
    ...     output_path='warmup.py',
    ...     model_code=model_code,
    ...     theta_dim=1,
    ...     n_particles=100
    ... )
    """
    output_path = Path(output_path)
    data_shape = data_shape or (20,)

    config_line = ""
    if config_path:
        config_line = f"compilation_config='{config_path}',\n        "

    script_content = f'''#!/usr/bin/env python3
"""
JAX Compilation Warmup Script

This script pre-compiles the model to populate JAX's compilation cache.
Run this once before using the model to avoid long compilation times
on first use.

Usage:
    python {output_path.name}

After running, JAX will have cached the compiled model and subsequent
runs will be much faster.
"""

import os
import sys
from time import time

print("=" * 70)
print("JAX COMPILATION WARMUP")
print("=" * 70)
print("This will pre-compile the model to populate JAX cache.")
print("This may take several minutes for large models...")
print()

# Import dependencies
try:
    import numpy as np
    import jax.numpy as jnp
    import phasic as pta
except ImportError as e:
    print(f"Error: Missing dependency: {{e}}")
    print("Install with: pip install phasic[jax]")
    sys.exit(1)

# Show cache location
cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                           os.path.expanduser('~/.jax_cache'))
print(f"JAX cache directory: {{cache_dir}}")
print()

# Create model
print("Creating model...")
{model_code}

# Generate dummy data for warmup
print(f"Generating dummy data (shape: {data_shape})...")
dummy_theta = jnp.ones({theta_dim})
dummy_times = jnp.linspace(0.1, 4.0, {data_shape[0]})
dummy_data = model(dummy_theta, dummy_times)
dummy_data = jnp.maximum(dummy_data, 1e-10)

# Warmup SVGD compilation
print(f"\\nWarming up SVGD with {n_particles} particles...")
print("This triggers compilation and caches the result.")
print()

start = time()

svgd = pta.SVGD(
    model=model,
    observed_data=dummy_data,
    theta_dim={theta_dim},
    n_particles={n_particles},
    n_iterations=2,  # Just enough to compile
    {config_line}verbose=True
)

# Run a few iterations to ensure full compilation
svgd.fit()

elapsed = time() - start

print(f"\\n" + "=" * 70)
print("WARMUP COMPLETE!")
print("=" * 70)
print(f"Time: {{elapsed:.1f}}s")
print(f"\\nJAX compilation cache has been populated at:")
print(f"  {{cache_dir}}")
print(f"\\nFuture runs with the same model shape will be much faster!")
print("=" * 70)
'''

    # Write script
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script_content)

    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(output_path, 0o755)

    print(f"✓ Warmup script saved to: {output_path}")


def export_model_package(
    output_dir: Union[Path, str],
    model_code: str,
    theta_dim: int,
    compilation_config: Optional[Any] = None,
    n_particles: int = 100,
    data_shape: Optional[tuple] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export a complete model package for distribution.

    Creates a directory with:
    - warmup.py: Script to pre-populate JAX cache
    - config.json: Compilation configuration
    - metadata.json: Model metadata and description

    Parameters
    ----------
    output_dir : Path or str
        Directory to create package in
    model_code : str
        Python code to create the model
    theta_dim : int
        Dimension of theta parameter
    compilation_config : CompilationConfig, optional
        Compilation configuration to include
    n_particles : int, optional
        Number of particles for warmup. Default: 100
    data_shape : tuple, optional
        Shape of observed data
    metadata : dict, optional
        Additional metadata to include (author, description, etc.)

    Examples
    --------
    >>> from phasic import CompilationConfig
    >>> from phasic.model_export import export_model_package
    >>>
    >>> model_code = '''...'''  # Model definition
    >>>
    >>> export_model_package(
    ...     output_dir='my_model_v1',
    ...     model_code=model_code,
    ...     theta_dim=1,
    ...     compilation_config=CompilationConfig.balanced(),
    ...     metadata={
    ...         'name': 'Coalescent Model',
    ...         'author': 'Your Name',
    ...         'version': '1.0.0',
    ...         'description': 'Population coalescent inference'
    ...     }
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating model package in: {output_dir}")

    # Save compilation config
    config_path = None
    if compilation_config is not None:
        config_path = output_dir / 'config.json'
        compilation_config.save_to_file(config_path)
        print(f"  ✓ Saved compilation config: {config_path.name}")

    # Generate warmup script
    warmup_path = output_dir / 'warmup.py'
    generate_warmup_script(
        output_path=warmup_path,
        model_code=model_code,
        theta_dim=theta_dim,
        n_particles=n_particles,
        data_shape=data_shape,
        config_path='config.json' if config_path else None
    )
    print(f"  ✓ Generated warmup script: {warmup_path.name}")

    # Save metadata
    full_metadata = {
        'created': datetime.now().isoformat(),
        'theta_dim': theta_dim,
        'n_particles': n_particles,
        'data_shape': data_shape,
    }
    if metadata:
        full_metadata.update(metadata)

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_path.name}")

    # Create README
    readme_path = output_dir / 'README.md'
    readme_content = f"""# {metadata.get('name', 'Model Package') if metadata else 'Model Package'}

{metadata.get('description', 'Exported model package') if metadata else 'Exported model package'}

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install phasic[jax]
   ```

2. **Run warmup script (optional but recommended):**
   ```bash
   python warmup.py
   ```
   This pre-compiles the model and populates JAX's cache. First run may take minutes,
   but subsequent runs will be instant.

3. **Use the model:**
   ```python
   import phasic as pta

   # Load with optimized configuration
   svgd = pta.SVGD(
       model=model,
       observed_data=data,
       theta_dim={theta_dim},
       compilation_config='config.json'
   )

   svgd.fit()
   ```

## Files

- `warmup.py` - Pre-compilation warmup script
- `config.json` - Optimized JAX compilation settings
- `metadata.json` - Model metadata and configuration
- `README.md` - This file

## Metadata

"""
    if metadata:
        for key, value in metadata.items():
            readme_content += f"- **{key}**: {value}\n"

    readme_content += f"\n---\n*Package created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Created README: {readme_path.name}")

    print(f"\n✓ Model package created successfully!")
    print(f"\nTo distribute:")
    print(f"  1. Compress: tar -czf {output_dir.name}.tar.gz {output_dir.name}/")
    print(f"  2. Users extract and run: python {output_dir.name}/warmup.py")
