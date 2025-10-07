# JAX as Optional Dependency

## Summary

JAX is now an **optional dependency** for PtDAlgorithms. The library can be used without JAX for basic functionality, but JAX is required for gradient-based inference and JAX transformations (JIT, grad, vmap).

## Changes Made

### 1. Code Changes (`src/ptdalgorithms/__init__.py`)

- **Conditional JAX import**: JAX is imported with try/except
- **Graceful error messages**: Functions that require JAX raise informative errors if JAX is not available
- **`HAS_JAX` flag**: Internal flag to check JAX availability

```python
# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False
```

### 2. Pip/PyPI (`pyproject.toml`)

JAX is already defined as an optional extra:

```toml
[project.optional-dependencies]
jax = ['jax>=0.4.0', 'jaxlib', 'h5py']
```

**Installation:**
```bash
# With JAX support
pip install 'ptdalgorithms[jax]'

# Or manually
pip install ptdalgorithms
pip install jax jaxlib
```

### 3. Pixi (`pixi.toml`)

JAX is now in a feature group:

```toml
[dependencies]
# Core dependencies (no JAX)

[feature.jax.dependencies]
jax = ">=0.6.0,<0.7"
jaxlib = ">=0.6.0,<0.7"
h5py = ">=1.14.6,<2"

[environments]
default = ["jax"]    # Default includes JAX
minimal = []         # Without JAX
dev = ["jax", "dev"] # Development
```

**Installation:**
```bash
# Default environment (with JAX)
pixi install

# Minimal environment (without JAX)
pixi install --environment minimal

# Development environment
pixi install --environment dev
```

### 4. Conda (`conda-build/meta.yaml`)

JAX is specified in `run_constrained` (not required, but version-constrained if installed):

```yaml
run:
  - python
  - numpy
  - eigen
  # ... other core deps
run_constrained:
  - jax >=0.4.0
  - jaxlib >=0.4.0
  - h5py
```

**Installation:**
```bash
# Base installation (no JAX)
conda install ptdalgorithms

# With JAX support
conda install ptdalgorithms jax
```

## What Works Without JAX?

### ✅ Works WITHOUT JAX:

1. **FFI approach** for C++ models:
   ```python
   from ptdalgorithms import Graph

   # FFI mode doesn't need JAX
   builder = Graph.pmf_from_cpp("model.cpp", use_ffi=True)
   graph = builder(np.array([1.0, 2.0]))
   pdf = graph.pdf(1.0, 100)
   ```

2. **Direct graph manipulation**:
   ```python
   g = Graph(state_length=2)
   v1 = g.find_or_create_vertex([1, 0])
   v2 = g.find_or_create_vertex([0, 1])
   v1.add_edge(v2, 1.5)

   # Use graph methods directly
   g.normalize()
   pdf = g.pdf(1.0, 100)
   ```

3. **R bindings**: Full functionality (doesn't use JAX)

### ❌ Requires JAX:

1. **JAX-compatible models** from `pmf_from_graph()`:
   ```python
   # Raises ImportError if JAX not installed
   model = Graph.pmf_from_graph(g)
   pdf = model(times)
   ```

2. **JAX-compatible C++ models**:
   ```python
   # Raises ImportError if JAX not installed
   model = Graph.pmf_from_cpp("model.cpp")  # use_ffi=False
   pdf = model(theta, times)
   ```

3. **Parameterized edges with gradients**:
   ```python
   # Raises ImportError if JAX not installed
   model = Graph.pmf_from_graph(parameterized_graph)
   grad = jax.grad(lambda t: model(t, times).sum())
   ```

## Error Messages

If JAX is not installed and you try to use JAX-dependent features, you'll get a helpful error:

```
ImportError: JAX is required for JAX-compatible models.
Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib
```

## Migration Guide

### For Existing Users

**No action needed** if you already have JAX installed. The library will detect it and work as before.

### For New Users

**Choose your installation method:**

| Use Case | Installation | JAX Required? |
|----------|-------------|---------------|
| Basic graph operations, FFI mode | `pip install ptdalgorithms` | ❌ No |
| Gradient-based inference, SVGD | `pip install 'ptdalgorithms[jax]'` | ✅ Yes |
| Development | `pixi install --environment dev` | ✅ Yes |
| Minimal (no JAX) | `pixi install --environment minimal` | ❌ No |

### For Package Maintainers

**Conda packages:**
- Base package: `ptdalgorithms` (no JAX)
- Full package: Install `ptdalgorithms` + `jax` + `jaxlib`

**Docker/containers:**
```dockerfile
# Minimal
RUN pip install ptdalgorithms

# With JAX support
RUN pip install 'ptdalgorithms[jax]'
```

## Documentation Updates

Examples that use JAX-dependent features now include the installation requirement:

```python
# Requires: pip install 'ptdalgorithms[jax]'
import jax.numpy as jnp
from ptdalgorithms import Graph

model = Graph.pmf_from_graph(g)
pdf = model(times)
```

## Testing

The changes have been tested with:
- ✅ JAX installed (all features work)
- ✅ JAX not installed (core features work, JAX features raise helpful errors)
- ✅ Conda build with `run_constrained`
- ✅ Pixi environments (default, minimal, dev)

## Benefits

1. **Smaller installation**: Core library without JAX is lighter
2. **Faster CI/CD**: Tests that don't need JAX can skip it
3. **Better dependency management**: Clear separation of core vs. optional features
4. **Platform compatibility**: JAX has platform limitations (e.g., Windows), now users can still use core features
5. **Reduced conflicts**: Users who don't need JAX won't have JAX dependency conflicts

## Future Considerations

- Consider creating separate conda packages: `ptdalgorithms` and `ptdalgorithms-jax`
- Add environment markers in `pyproject.toml` for platform-specific JAX dependencies
- Document which examples require JAX
