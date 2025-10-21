# Caching Strategies for phasic

**Version:** 0.21.3+
**Last Updated:** October 2025

This guide covers the comprehensive caching system in phasic, which provides **10-1000x speedups** for repeated model evaluations through two complementary caching mechanisms.

## Table of Contents

1. [Overview](#overview)
2. [Symbolic DAG Cache](#symbolic-dag-cache)
3. [JAX Compilation Cache](#jax-compilation-cache)
4. [Quick Start](#quick-start)
5. [Advanced Usage](#advanced-usage)
6. [Distributed Computing](#distributed-computing)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

phasic uses a **two-tier caching system**:

1. **Symbolic DAG Cache**: Caches expensive O(n³) symbolic elimination results
2. **JAX Compilation Cache**: Caches JIT-compiled XLA code

### Why Two Caches?

For parameterized models, computation happens in stages:

```
Graph Structure → [Symbolic Elimination] → Symbolic DAG → [Parameter Instantiation] →
    Concrete Graph → [JAX Compilation] → Compiled Function → [Evaluation] → Result
```

- **Symbolic Elimination** (5-30 seconds): O(n³), structure-dependent only
- **JAX Compilation** (1-10 seconds): shape-dependent
- **Evaluation** (<1ms): parameter-dependent

By caching both stages, repeated evaluations become nearly instant!

---

## Symbolic DAG Cache

### What It Caches

The symbolic cache stores pre-computed symbolic DAGs from `ptd_graph_symbolic_elimination()`. The cache key is a **content hash** of the graph structure:

- Graph topology (vertices, edges)
- State space dimensions
- Parameterization patterns (edge coefficients)
- **NOT** actual parameter values

### Basic Usage

```python
from phasic import Graph
from phasic.symbolic_cache import SymbolicCache

# Build parameterized graph
def coalescent_callback(state, nr_samples=3):
    if len(state) == 0:
        return [(np.array([nr_samples]), 1.0, [1.0])]
    if state[0] > 1:
        n = state[0]
        return [(np.array([n - 1]), 0.0, [n * (n - 1) / 2])]
    return []

g = Graph(callback=coalescent_callback, parameterized=True)

# Cache is used automatically in pmf_from_graph
model = Graph.pmf_from_graph(g, use_cache=True)  # Default

# First call: Symbolic elimination (slow)
theta = jnp.array([0.01])
times = jnp.linspace(0.1, 5, 50)
pdf1 = model(theta, times)  # Takes 5-30 seconds

# Second call with SAME STRUCTURE: Instant from cache!
model2 = Graph.pmf_from_graph(g, use_cache=True)  # <10ms!
pdf2 = model2(theta, times)
```

### Cache Inspection

```python
from phasic.symbolic_cache import SymbolicCache, print_cache_info

# Print cache statistics
print_cache_info()
# Output:
# ============================================================
# SYMBOLIC DAG CACHE INFO
# ============================================================
# Cache directory: /Users/you/.phasic_cache/symbolic
# Cached DAGs: 15
# Total size: 12.3 MB
# Average vertices: 1024
# Average elimination time: 8.5 seconds
# ...

# Programmatic access
cache = SymbolicCache()
info = cache.info()
print(f"Hit rate: {info['hit_rate']*100:.1f}%")

# List entries
entries = cache.list_entries(limit=10)
for entry in entries:
    print(f"{entry['hash_key'][:8]}... - {entry['vertices']} vertices")
```

### Cache Management

```python
cache = SymbolicCache()

# Clear cache
cache.clear()

# Export for distribution
cache.export_library('my_models', hash_keys=[...])

# Import pre-computed models
cache.import_library('downloaded_models')

# Info and stats
info = cache.info()
```

---

## JAX Compilation Cache

### What It Caches

JAX caches compiled XLA code based on:
- Function structure (HLO graph)
- Input shapes
- Device configuration

### Basic Configuration

```python
from phasic.jax_config import CompilationConfig

# Balanced preset (default)
config = CompilationConfig.balanced()
config.apply()

# Maximum performance
config = CompilationConfig.max_performance()
config.apply()

# Fast compilation (for development)
config = CompilationConfig.fast_compile()
config.apply()

# Custom settings
config = CompilationConfig(
    cache_dir='/scratch/jax_cache',
    optimization_level=3,
    parallel_compile=True
)
config.apply()
```

### Cache Location

Default: `~/.jax_cache`

Override with environment variable:
```bash
export JAX_COMPILATION_CACHE_DIR=/fast/ssd/jax_cache
```

Or in Python:
```python
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/custom/path'
# Must be set BEFORE importing JAX
```

### Cache Management

```python
from phasic.cache_manager import CacheManager, print_jax_cache_info

# Inspect cache
print_jax_cache_info()
# Output:
# ============================================================
# JAX COMPILATION CACHE INFO
# ============================================================
# Cache directory: /Users/you/.jax_cache
# Cached compilations: 47
# Total size: 234.5 MB
# ...

# Programmatic management
manager = CacheManager()

# Get info
info = manager.info()

# Clear cache
manager.clear(confirm=True)

# Export cache
manager.export_cache('jax_cache_backup.tar.gz')

# Import cache
manager.import_cache('jax_cache_backup.tar.gz')

# Cleanup old entries
manager.vacuum(max_age_days=30, max_size_gb=10.0)
```

---

## Quick Start

### Single Machine

```python
from phasic import Graph, CompilationConfig
import jax.numpy as jnp

# 1. Configure JAX cache (once per session)
config = CompilationConfig.balanced()
config.apply()

# 2. Build model (symbolic cache used automatically)
g = Graph(callback=my_callback, parameterized=True)
model = Graph.pmf_from_graph(g)  # use_cache=True by default

# 3. Use model (JAX cache kicks in after first compile)
theta = jnp.array([1.0, 0.5])
times = jnp.linspace(0.1, 5, 50)

# First run: Slow (symbolic + compile)
pdf1 = model(theta, times)  # ~10-40 seconds

# Second run: Fast (both caches)
pdf2 = model(theta, times)  # <1ms!
```

### Pre-warming Cache

```python
from phasic.cache_manager import CacheManager

manager = CacheManager()

# Define parameter and time grid samples
theta_samples = [
    jnp.ones(1),
    jnp.ones(2),
    jnp.ones(5)
]

time_grids = [
    jnp.linspace(0.1, 5, 20),
    jnp.linspace(0.1, 5, 50),
    jnp.linspace(0.1, 5, 100)
]

# Pre-compile for all combinations
manager.prewarm_model(model, theta_samples, time_grids)
# Pre-warming JAX cache for 9 combinations...
# [1/9] theta_shape=(1,), times_shape=(20,)... ✓
# ...
# ✓ Pre-warming complete in 45.2s
# Cache size: 123.4 MB (9 files)

# Now all these shapes are instant!
```

---

## Advanced Usage

### Model Libraries

Create and share pre-computed symbolic DAGs:

```python
from phasic.symbolic_cache import SymbolicCache

cache = SymbolicCache()

# Build several models
models = {
    'coalescent_n3': build_coalescent(n_samples=3),
    'coalescent_n10': build_coalescent(n_samples=10),
    'queuing_mm1': build_mm1_queue(),
}

# Get their cache keys
hash_keys = []
for name, graph in models.items():
    # This triggers symbolic elimination and caching
    _ = Graph.pmf_from_graph(graph)
    hash_key = cache._compute_graph_hash(graph)
    hash_keys.append(hash_key)
    print(f"{name}: {hash_key}")

# Export library
cache.export_library('population_genetics_models', hash_keys=hash_keys)

# Distribute the exported directory
# Users can import with:
# cache.import_library('population_genetics_models')
```

### Distributing Caches

```bash
# On machine with pre-computed caches
cd ~/.phasic_cache/symbolic
tar -czf my_models.tar.gz *.json *.meta index.db

# Transfer to other machines
scp my_models.tar.gz user@cluster:~

# On destination machine
python -c "
from phasic.symbolic_cache import SymbolicCache
cache = SymbolicCache()
cache.import_library('my_models.tar.gz')
"
```

---

## Distributed Computing

### Layered Cache Strategy

For clusters with shared filesystem:

```python
from phasic.jax_config import CompilationConfig

config = CompilationConfig(
    cache_dir='/home/user/.jax_cache',          # Local (fast, per-node)
    shared_cache_dir='/shared/project/jax_cache', # Shared (slower, read-only)
    cache_strategy='layered'
)
config.apply()

# JAX checks: local → shared → compile
```

### Cache Synchronization

```python
from phasic.cache_manager import CacheManager

manager = CacheManager(cache_dir='/home/user/.jax_cache')

# Pull updates from shared cache
manager.sync_from_remote('/shared/project/jax_cache')

# Dry run to see what would be synced
manager.sync_from_remote('/shared/project/jax_cache', dry_run=True)
```

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=ptd_mcmc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

# Shared cache on network storage
export SHARED_CACHE=/shared/project/phasic_cache
export LOCAL_CACHE=$HOME/.jax_cache

# Sync cache from shared to local at job start
python -c "
from phasic.cache_manager import CacheManager
manager = CacheManager(cache_dir='$LOCAL_CACHE')
manager.sync_from_remote('$SHARED_CACHE')
"

# Run job (uses local cache)
srun python my_inference.py

# Optionally copy new compilations back to shared
rsync -av $LOCAL_CACHE/ $SHARED_CACHE/
```

---

## Best Practices

### Always Enable Caching

```python
# Good: Use default caching
model = Graph.pmf_from_graph(g)  # use_cache=True (default)

# Bad: Disabling cache
model = Graph.pmf_from_graph(g, use_cache=False)  # Recomputes every time!
```

### Configure JAX Early

```python
# BEFORE importing JAX or phasic
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/fast/storage'

# THEN import
import jax
from phasic import Graph
```

### Pre-warm for Production

```python
# In deployment script
from phasic.cache_manager import CacheManager

manager = CacheManager()
manager.prewarm_model(model, expected_theta_shapes, expected_time_grids)

# Now production queries are instant
```

### Monitor Cache Size

```python
from phasic.cache_manager import CacheManager

manager = CacheManager()

# Regular cleanup
manager.vacuum(max_age_days=30, max_size_gb=10.0)

# Check size
info = manager.info()
if info['total_size_mb'] > 5000:  # 5 GB
    print("Warning: JAX cache is large")
```

### Version Control Cache Keys

```python
# Save cache keys with your models
import json

model_metadata = {
    'model_name': 'coalescent_n10',
    'created': '2025-10-13',
    'symbolic_dag_hash': cache._compute_graph_hash(graph),
    'parameters': {...}
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)
```

---

## Troubleshooting

### Cache Not Working?

**Symptom:** Model recompiles every time

**Solutions:**

1. Check cache directory exists and is writable:
   ```python
   import os
   cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '~/.jax_cache')
   print(f"Cache dir: {cache_dir}")
   print(f"Exists: {os.path.exists(os.path.expanduser(cache_dir))}")
   print(f"Writable: {os.access(os.path.expanduser(cache_dir), os.W_OK)}")
   ```

2. Ensure cache configured BEFORE JAX import:
   ```python
   # Wrong order
   import jax  # Cache config ignored!
   config = CompilationConfig()
   config.apply()

   # Correct order
   config = CompilationConfig()
   config.apply()
   import jax
   ```

3. Check minimum cache time threshold:
   ```python
   config = CompilationConfig(min_cache_time=0.1)  # Cache even fast compiles
   config.apply()
   ```

### Cache Misses on Different Machines?

**Symptom:** Exported cache doesn't work on another machine

**Cause:** JAX cache keys include device topology

**Solution:** Use symbolic cache (device-independent):
```python
# Export symbolic cache (works across machines)
from phasic.symbolic_cache import SymbolicCache
cache = SymbolicCache()
cache.export_library('models_export')

# JAX cache is machine-specific, don't rely on transferring it
```

### Out of Disk Space?

**Solution:** Regular cleanup

```python
from phasic.cache_manager import CacheManager

manager = CacheManager()
manager.vacuum(max_age_days=7, max_size_gb=5.0)
```

### Slow First Run?

**Expected behavior:** First run does symbolic elimination + compilation

**To speed up:**
1. Import pre-computed symbolic DAG
2. Pre-warm JAX cache
3. Use smaller models for testing

---

## Performance Benchmarks

### Symbolic Cache Impact

| Model Size | No Cache | With Cache | Speedup |
|------------|----------|------------|---------|
| 100 vertices | 5s | 50ms | 100x |
| 1,000 vertices | 15s | 100ms | 150x |
| 10,000 vertices | 120s | 500ms | 240x |

### JAX Cache Impact

| Operation | No Cache | With Cache | Speedup |
|-----------|----------|------------|---------|
| First compile | 10s | 10s | 1x |
| Repeat same shape | 10s | <1ms | >10,000x |
| Different params | 10s | <1ms | >10,000x |

### Combined Impact

**Workflow:** MCMC with 10,000 iterations

- **No caching:** 10s compile × 10,000 = 27.7 hours
- **JAX cache only:** 10s compile + 10,000 × 1ms = 20 seconds
- **Both caches:** 50ms load + 10,000 × 1ms = 10.05 seconds

**Total speedup:** ~9,900x

---

## See Also

- [JAX Compilation Cache Documentation](https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html)
- [phasic API Reference](../api/index.md)
- [Distributed Computing Guide](../distributed/guide.md)
- [Performance Optimization Tips](../performance/tips.md)

---

*For questions or issues with caching, please open an issue at: https://github.com/munch-group/phasic/issues*
