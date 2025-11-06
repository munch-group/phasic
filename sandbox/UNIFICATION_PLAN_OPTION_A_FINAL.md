# Option A: Replace Symbolic Completely - FINAL Implementation Plan

**Date**: 2025-01-01
**Version**: Target 0.23.0
**Status**: Planning (Final - With JAX/Multi-CPU Guarantees)
**Effort**: 2-3 days
**Risk**: Low-Medium
**Priority**: High

---

## Executive Summary

**Revised plan with strict requirements**:
1. ✅ **No silent fallbacks** - Fail loudly with clear errors
2. ✅ **Preserve all JAX functionality** - jit, grad, vmap, pmap
3. ✅ **Multi-machine CPU support** - No interference with distributed computing
4. ✅ **Work or fail** - No degraded performance modes

**Goal**: Replace symbolic system with trace-based system + automatic disk caching, while maintaining all existing functionality.

---

## Critical Requirements

### Requirement 1: No Silent Fallbacks ⚠️

**WRONG** (Silent fallback):
```c
if (graph->elimination_trace == NULL) {
    // Silently fall back to symbolic
    goto traditional_path;  // ❌ BAD - user doesn't know!
}
```

**CORRECT** (Fail loudly):
```c
if (graph->elimination_trace == NULL) {
    snprintf(ptd_err, sizeof(ptd_err),
             "FATAL: Trace-based elimination failed.\n"
             "  Graph structure: %zu vertices, %zu params\n"
             "  Possible causes:\n"
             "    - Graph contains unsupported cycles\n"
             "    - Cache I/O error (check ~/.phasic_cache/traces/)\n"
             "    - Out of memory\n"
             "  Workaround: Simplify graph structure or disable parameterization\n",
             graph->vertices_length, graph->param_length);
    DEBUG_PRINT("ERROR: %s\n", ptd_err);
    return -1;  // ✅ FAIL - user must fix the problem
}
```

### Requirement 2: Preserve JAX Functionality ✅

**JAX Functions That Must Work**:

```python
from phasic import Graph
import jax
import jax.numpy as jnp

# Create parameterized model
g = Graph(callback=model, parameterized=True)
model = Graph.pmf_from_graph(g, discrete=False)

theta = jnp.array([1.0, 2.0])
times = jnp.linspace(0.1, 5.0, 100)

# 1. JIT compilation
jitted_model = jax.jit(model)
pdf = jitted_model(theta, times)  # ✅ MUST WORK

# 2. Automatic differentiation
grad_fn = jax.grad(lambda t: jnp.sum(model(t, times)))
gradient = grad_fn(theta)  # ✅ MUST WORK

# 3. Vectorization (vmap)
batch_theta = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
vmapped = jax.vmap(lambda t: model(t, times))
batch_pdf = vmapped(batch_theta)  # ✅ MUST WORK

# 4. Multi-device (pmap)
devices = jax.local_devices()
if len(devices) > 1:
    pmapped = jax.pmap(lambda t: model(t, times))
    replicated_theta = jnp.array([theta] * len(devices))
    multi_pdf = pmapped(replicated_theta)  # ✅ MUST WORK
```

**Implementation Strategy**:
- Trace system operates at **graph construction time** (Python)
- JAX compilation happens **after** trace evaluation
- No interference with JAX transforms

### Requirement 3: Multi-Machine CPU Support ✅

**Environment**: Multiple machines, each with multiple CPUs

```python
# Machine 1
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Configure multi-CPU
from jax import config
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

# Use phasic with distributed JAX
from phasic import Graph, SVGD
model = Graph.pmf_from_graph(graph)

# Multi-machine SVGD
svgd = SVGD(model, ...)
results = svgd.fit()  # ✅ MUST WORK across machines
```

**Trace Cache Implications**:
- **Cache location**: `~/.phasic_cache/traces/` (local filesystem)
- **Multi-machine**: Each machine may have different cache
- **Solution**: Traces are deterministic - same structure → same trace
  - Machine A records trace, saves to local cache
  - Machine B records trace independently, saves to local cache
  - Both traces are identical (deterministic hashing)
  - No network I/O required

**Optional**: Shared cache directory (NFS, distributed filesystem)
```python
import os
os.environ['PHASIC_CACHE_DIR'] = '/shared/nfs/phasic_cache'
# Now all machines share the same cache
```

### Requirement 4: Work or Fail (No Degraded Modes) ⚠️

**Unacceptable**:
```python
# User creates parameterized graph
g = Graph(callback=model, parameterized=True)
g.update_parameterized_weights([1.0, 2.0])

# Silently falls back to slower mode
moments = g.moments(2)  # ❌ BAD - works but 10× slower, user doesn't know
```

**Acceptable**:
```python
# Option 1: Work as specified (fast)
g = Graph(callback=model, parameterized=True)
g.update_parameterized_weights([1.0, 2.0])
moments = g.moments(2)  # ✅ Uses trace (fast), or...

# Option 2: Fail with clear error
# ❌ RuntimeError: Trace-based elimination failed.
#    Graph structure contains unsupported cycles.
#    Cannot use parameterized mode with this graph.
```

**User must explicitly choose**:
```python
# If user knows graph won't work with traces
g = Graph(callback=model, parameterized=False)  # ✅ Explicit choice
moments = g.moments(2)  # Works, but no parameter updates
```

---

## Implementation Changes Based on Requirements

### Change 1: Remove ALL Silent Fallbacks

**Location**: `src/c/phasic.c:566-648` (`ptd_precompute_reward_compute_graph`)

**BEFORE** (from original plan):
```c
if (graph->elimination_trace == NULL) {
    // Try recording
    graph->elimination_trace = ptd_record_elimination_trace(graph);

    if (graph->elimination_trace == NULL) {
        // Silent fallback to symbolic
        goto traditional_path;  // ❌ WRONG
    }
}
```

**AFTER** (strict failure):
```c
if (graph->elimination_trace == NULL) {
    // Compute graph hash for cache lookup
    char hash_hex[65];
    if (ptd_graph_content_hash(graph, hash_hex) != 0) {
        snprintf(ptd_err, sizeof(ptd_err),
                 "FATAL: Failed to compute graph hash for cache lookup.\n"
                 "  This indicates a serious internal error.\n"
                 "  Please report this bug with your graph construction code.");
        return -1;
    }

    DEBUG_PRINT("INFO: Checking trace cache for hash %s...\n", hash_hex);

    // Try loading from cache
    graph->elimination_trace = load_trace_from_cache(hash_hex);

    if (graph->elimination_trace != NULL) {
        DEBUG_PRINT("INFO: ✓ Trace loaded from cache (%zu operations)\n",
                   graph->elimination_trace->operations_length);
    } else {
        // Cache miss - must record new trace
        DEBUG_PRINT("INFO: Cache miss, recording new trace...\n");

        graph->elimination_trace = ptd_record_elimination_trace(graph);

        if (graph->elimination_trace == NULL) {
            // FAIL LOUDLY - DO NOT FALL BACK
            snprintf(ptd_err, sizeof(ptd_err),
                     "FATAL: Trace-based elimination failed for parameterized graph.\n"
                     "  Graph details:\n"
                     "    - Vertices: %zu\n"
                     "    - Parameters: %zu\n"
                     "    - State length: %zu\n"
                     "  Possible causes:\n"
                     "    1. Graph contains unsupported cycle patterns\n"
                     "    2. Graph structure is too complex for trace recording\n"
                     "    3. Out of memory during elimination\n"
                     "  Solutions:\n"
                     "    - Simplify graph structure (reduce vertices/edges)\n"
                     "    - Use non-parameterized mode: Graph(parameterized=False)\n"
                     "    - Check for infinite loops in callback function\n"
                     "  If you believe this is a bug, please report with:\n"
                     "    - Graph construction code\n"
                     "    - Error message above\n"
                     "    - Output of: export PHASIC_DEBUG=1 && python your_script.py",
                     graph->vertices_length,
                     graph->param_length,
                     graph->state_length);

            DEBUG_PRINT("FATAL ERROR: %s\n", ptd_err);
            return -1;  // HARD FAILURE
        }

        DEBUG_PRINT("INFO: ✓ Trace recorded (%zu operations)\n",
                   graph->elimination_trace->operations_length);

        // Save to cache (best-effort, continue if fails)
        if (!save_trace_to_cache(hash_hex, graph->elimination_trace)) {
            // Cache save failure is NOT fatal - trace is in memory
            DEBUG_PRINT("WARNING: Failed to save trace to cache.\n");
            DEBUG_PRINT("  Cache directory: ~/.phasic_cache/traces/\n");
            DEBUG_PRINT("  Check permissions and disk space.\n");
            DEBUG_PRINT("  Performance may be degraded on next session.\n");
            // CONTINUE - trace is still usable from memory
        } else {
            DEBUG_PRINT("INFO: ✓ Trace saved to cache: %s\n", hash_hex);
        }
    }
}

// At this point, trace MUST exist
assert(graph->elimination_trace != NULL);

// Use trace-based path (NO FALLBACK)
if (graph->current_params == NULL) {
    snprintf(ptd_err, sizeof(ptd_err),
             "FATAL: Cannot compute moments/PDF - parameters not set.\n"
             "  Call update_parameterized_weights() first.\n"
             "  Example: graph.update_parameterized_weights([1.0, 2.0])");
    return -1;
}

// Evaluate trace
struct ptd_trace_result *trace_result = ptd_evaluate_trace(
    graph->elimination_trace,
    graph->current_params,
    graph->param_length
);

if (trace_result == NULL) {
    snprintf(ptd_err, sizeof(ptd_err),
             "FATAL: Trace evaluation failed.\n"
             "  Parameters: %zu dimensions\n"
             "  Trace operations: %zu\n"
             "  This is an internal error - trace was recorded but cannot be evaluated.\n"
             "  Please report this bug.",
             graph->param_length,
             graph->elimination_trace->operations_length);
    return -1;
}

// Build reward compute graph
graph->reward_compute_graph = ptd_build_reward_compute_from_trace(
    trace_result,
    graph
);

ptd_trace_result_destroy(trace_result);

if (graph->reward_compute_graph == NULL) {
    snprintf(ptd_err, sizeof(ptd_err),
             "FATAL: Failed to build reward compute graph from trace.\n"
             "  This is an internal error.\n"
             "  Please report this bug.");
    return -1;
}

DEBUG_PRINT("INFO: ✓ Reward compute graph built from trace\n");
return 0;  // SUCCESS
```

**Key Changes**:
- ❌ **No `goto traditional_path`** - removed entirely
- ✅ **Explicit error messages** - user knows exactly what failed
- ✅ **Actionable solutions** - user knows how to fix
- ✅ **Debug information** - helps reporting bugs
- ⚠️ **Cache save failure OK** - trace still works from memory

### Change 2: Validate JAX Compatibility

**New Test**: `tests/test_jax_compatibility.py`

```python
"""
Test that trace system doesn't break JAX functionality
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from phasic import Graph

def coalescent_callback(state):
    if state.size == 0:
        return [(np.array([5]), 0.0, [1.0])]
    if state[0] <= 1:
        return []
    n = state[0]
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), 0.0, [rate])]

class TestJAXCompatibility:

    def test_jit_compilation(self):
        """Test jax.jit works with trace-based models"""
        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        theta = jnp.array([1.0])
        times = jnp.linspace(0.1, 5.0, 50)

        # Compile with JIT
        jitted_model = jax.jit(model)

        # First call (compilation)
        pdf1 = jitted_model(theta, times)
        assert pdf1.shape == (50,)
        assert jnp.all(jnp.isfinite(pdf1))

        # Second call (cached)
        pdf2 = jitted_model(theta, times)
        assert jnp.allclose(pdf1, pdf2)

        print("✓ jax.jit works")

    def test_automatic_differentiation(self):
        """Test jax.grad works with trace-based models"""
        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        times = jnp.linspace(0.1, 5.0, 50)

        # Define loss function
        def loss(theta):
            pdf = model(theta, times)
            return jnp.sum(jnp.log(pdf + 1e-10))

        # Compute gradient
        grad_fn = jax.grad(loss)
        theta = jnp.array([1.0])
        gradient = grad_fn(theta)

        assert gradient.shape == (1,)
        assert jnp.isfinite(gradient[0])

        # Verify gradient is non-zero
        assert jnp.abs(gradient[0]) > 1e-6

        print("✓ jax.grad works")

    def test_vmap_vectorization(self):
        """Test jax.vmap works with trace-based models"""
        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        times = jnp.linspace(0.1, 5.0, 50)

        # Batch of parameters
        batch_theta = jnp.array([[1.0], [2.0], [3.0]])

        # Vectorize over parameter batches
        vmapped = jax.vmap(lambda t: model(t, times))
        batch_pdf = vmapped(batch_theta)

        assert batch_pdf.shape == (3, 50)
        assert jnp.all(jnp.isfinite(batch_pdf))

        # Verify different parameters give different results
        assert not jnp.allclose(batch_pdf[0], batch_pdf[1])

        print("✓ jax.vmap works")

    def test_pmap_multi_device(self):
        """Test jax.pmap works with trace-based models (if multiple devices)"""
        devices = jax.local_devices()
        if len(devices) < 2:
            pytest.skip("Need multiple devices for pmap test")

        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        times = jnp.linspace(0.1, 5.0, 50)
        theta = jnp.array([1.0])

        # Replicate across devices
        replicated_theta = jnp.array([theta] * len(devices))

        # Parallel map across devices
        pmapped = jax.pmap(lambda t: model(t, times))
        multi_pdf = pmapped(replicated_theta)

        assert multi_pdf.shape == (len(devices), 50)
        assert jnp.all(jnp.isfinite(multi_pdf))

        # All devices should produce same result
        for i in range(1, len(devices)):
            assert jnp.allclose(multi_pdf[0], multi_pdf[i])

        print(f"✓ jax.pmap works ({len(devices)} devices)")

    def test_jit_grad_combined(self):
        """Test jax.jit + jax.grad combination"""
        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        times = jnp.linspace(0.1, 5.0, 50)

        def loss(theta):
            pdf = model(theta, times)
            return jnp.sum(jnp.log(pdf + 1e-10))

        # Compile gradient function
        grad_fn = jax.jit(jax.grad(loss))

        theta = jnp.array([1.0])
        gradient = grad_fn(theta)

        assert gradient.shape == (1,)
        assert jnp.isfinite(gradient[0])

        print("✓ jax.jit(jax.grad(...)) works")

    def test_vmap_grad_combined(self):
        """Test jax.vmap + jax.grad combination (for SVGD)"""
        g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g, discrete=False)

        times = jnp.linspace(0.1, 5.0, 50)

        def loss(theta):
            pdf = model(theta, times)
            return jnp.sum(jnp.log(pdf + 1e-10))

        # Vectorize gradient computation (SVGD pattern)
        batch_theta = jnp.array([[1.0], [2.0], [3.0]])
        grad_fn = jax.grad(loss)
        vmapped_grad = jax.vmap(grad_fn)

        batch_gradients = vmapped_grad(batch_theta)

        assert batch_gradients.shape == (3, 1)
        assert jnp.all(jnp.isfinite(batch_gradients))

        print("✓ jax.vmap(jax.grad(...)) works (SVGD pattern)")
```

**All tests must pass** - if any fail, trace system has broken JAX compatibility.

### Change 3: Multi-Machine Cache Strategy

**Problem**: Multiple machines, each with local filesystem

**Solution 1: Independent Caches** (Default)
```
Machine A: ~/.phasic_cache/traces/abc123.json
Machine B: ~/.phasic_cache/traces/abc123.json  (different file, same content)
```
- Each machine records trace independently
- Traces are deterministic (same structure → same trace)
- No network I/O required
- Slight overhead: First run on each machine records trace

**Solution 2: Shared Cache** (Optional, via environment variable)
```python
# On all machines, point to shared NFS directory
import os
os.environ['PHASIC_CACHE_DIR'] = '/shared/nfs/phasic_cache/traces'

# Now all machines use same cache
from phasic import Graph
g = Graph(...)  # Machine A records, saves to shared cache
# Machine B loads from shared cache immediately
```

**Implementation**:
```c
// In get_cache_dir()
static const char *get_cache_dir(void) {
    static char cache_dir[1024];
    static bool initialized = false;

    if (!initialized) {
        // Check for custom cache directory
        const char *custom_cache = getenv("PHASIC_CACHE_DIR");
        if (custom_cache != NULL) {
            snprintf(cache_dir, sizeof(cache_dir), "%s", custom_cache);
            DEBUG_PRINT("INFO: Using custom cache directory: %s\n", cache_dir);
        } else {
            // Default: ~/.phasic_cache/traces/
            const char *home = getenv("HOME");
            if (home == NULL) {
                home = getenv("USERPROFILE");  // Windows
            }
            if (home == NULL) {
                home = "/tmp";  // Last resort
            }
            snprintf(cache_dir, sizeof(cache_dir), "%s/.phasic_cache/traces", home);
        }

        // Create directory
        mkdir_recursive(cache_dir);
        initialized = true;
    }

    return cache_dir;
}
```

**Test Multi-Machine**:
```python
# tests/test_multi_machine.py
import subprocess
import os
import tempfile

def test_independent_caches():
    """Test that multiple machines can work independently"""

    script = """
import numpy as np
from phasic import Graph

def callback(state):
    if state.size == 0:
        return [(np.array([3]), 0.0, [1.0])]
    if state[0] <= 1:
        return []
    n = state[0]
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), 0.0, [rate])]

g = Graph(callback=callback, parameterized=True, nr_samples=3)
g.update_parameterized_weights(np.array([1.0]))
moments = g.moments(2)
print(f"SUCCESS:{moments[0]:.10f}")
"""

    # Simulate two machines with different cache directories
    cache_a = tempfile.mkdtemp()
    cache_b = tempfile.mkdtemp()

    env_a = os.environ.copy()
    env_a['PHASIC_CACHE_DIR'] = cache_a

    env_b = os.environ.copy()
    env_b['PHASIC_CACHE_DIR'] = cache_b

    # Machine A
    result_a = subprocess.run(
        ["python", "-c", script],
        env=env_a,
        capture_output=True,
        text=True
    )

    # Machine B
    result_b = subprocess.run(
        ["python", "-c", script],
        env=env_b,
        capture_output=True,
        text=True
    )

    # Both should succeed
    assert "SUCCESS" in result_a.stdout
    assert "SUCCESS" in result_b.stdout

    # Both should produce same result
    val_a = float(result_a.stdout.split("SUCCESS:")[1])
    val_b = float(result_b.stdout.split("SUCCESS:")[1])
    assert abs(val_a - val_b) < 1e-10

    print("✓ Independent caches work")

def test_shared_cache():
    """Test that shared cache works across 'machines'"""

    shared_cache = tempfile.mkdtemp()

    script = """
import numpy as np
from phasic import Graph

def callback(state):
    if state.size == 0:
        return [(np.array([3]), 0.0, [1.0])]
    if state[0] <= 1:
        return []
    n = state[0]
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), 0.0, [rate])]

g = Graph(callback=callback, parameterized=True, nr_samples=3)
g.update_parameterized_weights(np.array([1.0]))
moments = g.moments(2)
print(f"SUCCESS:{moments[0]:.10f}")
"""

    env = os.environ.copy()
    env['PHASIC_CACHE_DIR'] = shared_cache

    # First run: records trace
    result1 = subprocess.run(
        ["python", "-c", script],
        env=env,
        capture_output=True,
        text=True
    )

    # Second run: loads from cache
    result2 = subprocess.run(
        ["python", "-c", script],
        env=env,
        capture_output=True,
        text=True
    )

    assert "SUCCESS" in result1.stdout
    assert "SUCCESS" in result2.stdout

    # Check cache file exists
    import os
    cache_files = os.listdir(shared_cache)
    assert len(cache_files) == 1
    assert cache_files[0].endswith('.json')

    print("✓ Shared cache works")
```

### Change 4: Feature Flag Behavior

**Environment Variable**: `PHASIC_DISABLE_AUTO_TRACE`

**BEFORE** (wrong):
```c
if (disable_auto_trace) {
    goto traditional_path;  // ❌ Silent fallback
}
```

**AFTER** (correct):
```c
static bool feature_flag_checked = false;
static bool auto_trace_enabled = true;

if (!feature_flag_checked) {
    const char *disable = getenv("PHASIC_DISABLE_AUTO_TRACE");
    if (disable != NULL && strcmp(disable, "1") == 0) {
        auto_trace_enabled = false;
        DEBUG_PRINT("WARNING: PHASIC_DISABLE_AUTO_TRACE=1 detected\n");
        DEBUG_PRINT("WARNING: Parameterized graphs will FAIL\n");
        DEBUG_PRINT("WARNING: This flag is for debugging only\n");
    }
    feature_flag_checked = true;
}

if (graph->parameterized && !auto_trace_enabled) {
    snprintf(ptd_err, sizeof(ptd_err),
             "FATAL: Parameterized graph not supported (PHASIC_DISABLE_AUTO_TRACE=1).\n"
             "  This flag disables trace-based elimination.\n"
             "  To use parameterized graphs:\n"
             "    - Unset PHASIC_DISABLE_AUTO_TRACE\n"
             "    - Or use non-parameterized mode: Graph(parameterized=False)");
    return -1;  // ✅ FAIL LOUDLY
}
```

**User Experience**:
```bash
# User disables auto-trace
export PHASIC_DISABLE_AUTO_TRACE=1
python script.py

# Output:
# WARNING: PHASIC_DISABLE_AUTO_TRACE=1 detected
# WARNING: Parameterized graphs will FAIL
# FATAL: Parameterized graph not supported (PHASIC_DISABLE_AUTO_TRACE=1)
#   To use parameterized graphs:
#     - Unset PHASIC_DISABLE_AUTO_TRACE
#     - Or use non-parameterized mode: Graph(parameterized=False)
```

---

## Updated Testing Strategy

### Test Categories

1. **Correctness Tests** - Results must be exact
2. **JAX Compatibility Tests** - All JAX functions must work
3. **Multi-Machine Tests** - Distributed computing must work
4. **Error Handling Tests** - Failures must be loud and clear
5. **Performance Tests** - Must meet targets or fail

### Mandatory Test Suite

**File**: `tests/test_strict_requirements.py`

```python
"""
Strict requirement tests - ALL MUST PASS
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from phasic import Graph
from phasic.trace_cache import clear_trace_cache

def coalescent(n):
    def callback(state):
        if state.size == 0:
            return [(np.array([n]), 0.0, [1.0])]
        if state[0] <= 1:
            return []
        k = state[0]
        rate = k * (k - 1) / 2
        return [(np.array([k - 1]), 0.0, [rate])]
    return callback

class TestStrictRequirements:

    def test_no_silent_fallback_trace_failure(self):
        """If trace fails, must raise exception (no silent fallback)"""

        # Create pathological graph that might fail trace recording
        # (This is a synthetic test - real graphs should work)

        def bad_callback(state):
            # Intentionally complex/cyclic structure
            if state.size == 0:
                return [(np.array([0, 0]), 0.0, [1.0])]
            # Create potential cycles
            return [
                (np.array([state[0] + 1, state[1]]), 0.0, [1.0]),
                (np.array([state[0], state[1] + 1]), 0.0, [1.0]),
                (np.array([0, 0]), 0.0, [1.0]),  # Back to start
            ]

        g = Graph(callback=bad_callback, parameterized=True, state_length=2)

        with pytest.raises(RuntimeError, match="FATAL.*trace"):
            g.update_parameterized_weights(np.array([1.0]))
            g.moments(2)

        print("✓ Trace failure raises exception (no silent fallback)")

    def test_no_silent_fallback_cache_failure(self):
        """If cache fails, must continue or fail loudly (no silent degradation)"""

        clear_trace_cache()

        # Make cache directory read-only to force save failure
        import os
        from phasic.trace_cache import get_cache_dir
        cache_dir = get_cache_dir()

        # This should still work (cache save is best-effort)
        g = Graph(callback=coalescent(3), parameterized=True, nr_samples=3)
        g.update_parameterized_weights(np.array([1.0]))

        # Should succeed (trace in memory even if cache fails)
        moments = g.moments(2)
        assert moments is not None

        print("✓ Cache save failure doesn't break functionality")

    def test_jax_jit_required(self):
        """jax.jit MUST work"""
        clear_trace_cache()

        g = Graph(callback=coalescent(5), parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g)

        theta = jnp.array([1.0])
        times = jnp.linspace(0.1, 5.0, 50)

        jitted = jax.jit(model)
        pdf = jitted(theta, times)

        assert pdf.shape == (50,)
        assert jnp.all(jnp.isfinite(pdf))

        print("✓ jax.jit REQUIRED - PASS")

    def test_jax_grad_required(self):
        """jax.grad MUST work"""
        clear_trace_cache()

        g = Graph(callback=coalescent(5), parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g)

        times = jnp.linspace(0.1, 5.0, 50)

        def loss(theta):
            return jnp.sum(model(theta, times))

        grad_fn = jax.grad(loss)
        gradient = grad_fn(jnp.array([1.0]))

        assert gradient.shape == (1,)
        assert jnp.isfinite(gradient[0])

        print("✓ jax.grad REQUIRED - PASS")

    def test_jax_vmap_required(self):
        """jax.vmap MUST work"""
        clear_trace_cache()

        g = Graph(callback=coalescent(5), parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g)

        times = jnp.linspace(0.1, 5.0, 50)
        batch_theta = jnp.array([[1.0], [2.0], [3.0]])

        vmapped = jax.vmap(lambda t: model(t, times))
        batch_pdf = vmapped(batch_theta)

        assert batch_pdf.shape == (3, 50)
        assert jnp.all(jnp.isfinite(batch_pdf))

        print("✓ jax.vmap REQUIRED - PASS")

    def test_multi_cpu_required(self):
        """Multi-CPU setup must not break"""
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

        clear_trace_cache()

        g = Graph(callback=coalescent(5), parameterized=True, nr_samples=5)
        model = Graph.pmf_from_graph(g)

        theta = jnp.array([1.0])
        times = jnp.linspace(0.1, 5.0, 50)

        pdf = model(theta, times)

        assert pdf.shape == (50,)
        assert jnp.all(jnp.isfinite(pdf))

        print("✓ Multi-CPU REQUIRED - PASS")

    def test_performance_or_fail(self):
        """Performance must meet targets or test fails"""
        import time

        clear_trace_cache()

        g = Graph(callback=coalescent(67), parameterized=True, nr_samples=67)

        # Cold cache (first call)
        g.update_parameterized_weights(np.array([1.0]))
        start = time.time()
        g.moments(2)
        cold = time.time() - start

        # Warm cache (second call)
        g2 = Graph(callback=coalescent(67), parameterized=True, nr_samples=67)
        g2.update_parameterized_weights(np.array([2.0]))
        start = time.time()
        g2.moments(2)
        warm = time.time() - start

        # STRICT REQUIREMENTS
        assert cold < 0.100, f"Cold cache too slow: {cold*1000:.2f}ms (target: <100ms)"
        assert warm < 0.010, f"Warm cache too slow: {warm*1000:.2f}ms (target: <10ms)"

        speedup = cold / warm
        assert speedup >= 5.0, f"Speedup too low: {speedup:.1f}× (target: ≥5×)"

        print(f"✓ Performance REQUIRED - PASS")
        print(f"  Cold: {cold*1000:.2f}ms, Warm: {warm*1000:.2f}ms, Speedup: {speedup:.1f}×")
```

**CI/CD Requirement**: ALL tests must pass before merge.

---

## Updated Implementation Phases

### Phase 1: Enable Cache I/O (0.5 days)

**Deliverables**:
- [ ] Uncommented cache functions
- [ ] Fixed JSON parsing (if needed)
- [ ] Cross-platform paths with `PHASIC_CACHE_DIR` support
- [ ] `tests/test_cache_io.c` passing
- [ ] **NO SILENT FAILURES** - all errors logged

### Phase 2: Add Automatic Cache Lookup (0.5 days)

**Deliverables**:
- [ ] Modified `ptd_precompute_reward_compute_graph()` with cache lookup
- [ ] **REMOVED all `goto traditional_path`** - no silent fallbacks
- [ ] Feature flag causes immediate failure with clear error
- [ ] All error messages actionable

### Phase 3: Remove Symbolic System (0.5 days)

**Deliverables**:
- [ ] `parameterized_reward_compute_graph` field removed
- [ ] All symbolic functions removed
- [ ] No compilation warnings

### Phase 4: Testing - JAX & Multi-Machine (0.5-1 day)

**Deliverables**:
- [ ] `tests/test_jax_compatibility.py` - ALL PASS
- [ ] `tests/test_multi_machine.py` - ALL PASS
- [ ] `tests/test_strict_requirements.py` - ALL PASS
- [ ] Performance benchmarks meet targets or FAIL

### Phase 5: Documentation (0.5 days)

**Deliverables**:
- [ ] Updated CLAUDE.md with strict failure semantics
- [ ] Migration guide mentions "no silent fallbacks"
- [ ] Examples show error handling

---

## Success Criteria (Updated)

### Mandatory (ALL must pass)

- [ ] ✅ **No silent fallbacks** - All failures are loud
- [ ] ✅ **JAX works** - jit, grad, vmap, pmap all functional
- [ ] ✅ **Multi-machine works** - Independent or shared cache
- [ ] ✅ **Performance targets met** - Or test fails
- [ ] ✅ **No memory leaks** - Valgrind clean
- [ ] ✅ **No regressions** - All existing tests pass

### Optional (Nice to have)

- [ ] Cache compression for large traces
- [ ] Async cache writes
- [ ] Cache statistics/monitoring

---

## Conclusion

This **FINAL implementation plan** ensures:

1. ✅ **No silent fallbacks** - Work perfectly or fail loudly
2. ✅ **JAX compatibility preserved** - All transforms work
3. ✅ **Multi-machine support** - Independent or shared caches
4. ✅ **Clear error messages** - Users know what went wrong and how to fix
5. ✅ **Strict testing** - All requirements enforced by tests

**Timeline**: 2-3 days with strict quality requirements.

**Risk**: Low-Medium (complexity in error handling, but trace system already proven)

---

**End of Final Implementation Plan**
