# Multivariate Tests - Avoiding Segmentation Faults

## Quick Fix

If you're experiencing segmentation faults when running multivariate tests, use one of these solutions:

### Option 1: Set Device Count to 1 (Easiest)
```bash
export PTDALG_CPUS=1
python your_test_script.py
```

### Option 2: Use Safe Configuration in Code
```python
from phasic import Graph, SVGD

# Add these parameters to avoid segfaults
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    # ... other parameters ...
    parallel='none',  # Disable multi-device parallelization
    jit=False         # Disable JIT compilation (optional)
)
```

### Option 3: Use the Safe Test File
```bash
python test_multivariate_safe.py  # Includes safe configuration
```

## Why Does This Happen?

The segfault occurs when:
- Using **multivariate models** (2D rewards)
- With **longer SVGD runs** (10+ iterations, 20+ particles)
- Using **default JAX configuration** (pmap across 8 CPU devices + JIT)

The combination of:
1. Python loop over features in multivariate model
2. JAX pmap parallelization across 8 devices
3. JIT compilation of complex model
4. Multiple SVGD iterations with gradients

...can cause threading/memory issues leading to segfaults on some systems.

## Test Files

### ✅ test_multivariate_safe.py
- **Status**: Always works
- **Config**: `parallel='none'`, `jit=False`
- **Iterations**: 2 (minimal)
- **Use for**: Development and debugging

### ✅ test_multivariate_basic.py
- **Status**: Always works
- **Config**: No SVGD optimization
- **Use for**: Basic functionality tests

### ⚠️ tests/test_multivariate.py
- **Status**: Updated with safe configuration
- **Config**: Now uses `SAFE_CONFIG` dict
- **Use for**: Comprehensive testing

### ⚠️ test_multivariate_simple.py
- **Status**: May segfault with default config
- **Config**: Uses default (pmap + JIT)
- **Use for**: Performance testing (after fixing)

## Running Tests

### Recommended (Safe)
```bash
# Set device count to 1
export PTDALG_CPUS=1

# Run safe tests
python test_multivariate_safe.py

# Run basic tests
python test_multivariate_basic.py

# Run pytest tests
python -m pytest tests/test_multivariate.py -v
```

### With Default Configuration (May Segfault)
```bash
# Unset device limit
unset PTDALG_CPUS

# Run with default config (8 devices, pmap, JIT)
python test_multivariate_simple.py  # May segfault
```

## Performance vs Stability

| Configuration | Speed | Stability | Use Case |
|--------------|-------|-----------|----------|
| `parallel='none', jit=False` | Slowest | ✅ Most stable | Development, debugging |
| `parallel='vmap', jit=True` | Medium | ✅ Stable | Production (single device) |
| `parallel='pmap', jit=True` | Fastest | ⚠️ May segfault | Production (if stable) |

## Examples

### Example 1: Safe SVGD with Multivariate Model
```python
from phasic import Graph, SVGD
import jax.numpy as jnp

# Create model
graph = Graph(callback=model_callback, parameterized=True)
model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2)

# Setup 2D data
observed_data = jnp.random.exponential(0.5, size=(100, 3))
rewards_2d = jnp.ones((n_vertices, 3))

# SAFE configuration
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=8,
    n_iterations=100,
    parallel='none',  # Safe
    jit=False,        # Safe
    verbose=True,
    rewards=rewards_2d
)

result = svgd.optimize()
```

### Example 2: Using Graph.svgd() with Safe Config
```python
result = graph.svgd(
    observed_data=observed_data,
    theta_dim=1,
    n_particles=8,
    n_iterations=100,
    rewards=rewards_2d,
    parallel='none',  # Safe
    jit=False         # Safe
)
```

### Example 3: Fast Configuration (If Stable on Your System)
```python
# Test first with small run
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=8,  # Divides evenly by 8 devices
    n_iterations=10,
    rewards=rewards_2d,
    parallel='pmap',  # Fast but may segfault
    jit=True,         # Fast
    verbose=True
)

try:
    result = svgd.optimize()
    print("Success! Can use fast config")
except:
    print("Segfault - use safe config instead")
```

## Debugging Checklist

If you encounter segfaults:

1. ✅ Set `export PTDALG_CPUS=1`
2. ✅ Add `parallel='none'` to SVGD calls
3. ✅ Add `jit=False` to SVGD calls
4. ✅ Reduce `n_iterations` to test (e.g., 2-5)
5. ✅ Use `n_particles=8` (divides evenly by device count)
6. ✅ Run `test_multivariate_safe.py` to verify setup

## More Information

See `MULTIVARIATE_SEGFAULT_FIX.md` for detailed technical analysis and future improvements.

## Summary

**For reliable testing**: Use `parallel='none'` and/or `export PTDALG_CPUS=1`

**For production**: Test with your configuration first, fall back to safe mode if needed

**All test files updated**: Now include safe configuration by default
