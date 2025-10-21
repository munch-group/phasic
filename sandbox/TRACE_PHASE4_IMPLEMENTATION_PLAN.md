# Phase 4 Implementation Plan: Full Phase-Type Likelihood

**Date:** 2025-10-15
**Status:** Ready to implement
**Prerequisites:** Phases 1-3 complete, CLAUDE.md reduced

---

## Context

### Phases 1-3 Summary

**Phase 1**: Trace recording and evaluation (constant weights)
- Status: ✅ Complete (14/14 tests passing)
- File: `src/phasic/trace_elimination.py`
- Key functions: `record_elimination_trace()`, `evaluate_trace()`

**Phase 2**: Parameterization support + JAX integration
- Status: ✅ Complete (12/12 tests passing)
- Added: `OpType.DOT`, `evaluate_trace_jax()`, `trace_to_jax_fn()`
- Supports: jax.jit, jax.grad, jax.vmap

**Phase 3**: SVGD integration
- Status: ✅ Complete
- Added: `trace_to_log_likelihood()` with **simplified exponential approximation**
- Performance: 37v <2s, 67v <5s for 1000 evaluations

**Current Limitation**: Line 1199-1201 of `trace_elimination.py`:
```python
# SIMPLIFIED - NOT EXACT PHASE-TYPE LIKELIHOOD
log_lik = jnp.sum(jnp.log(lambda_param) - lambda_param * observed_data)
```

This assumes exponential distribution. For complex phase-type models (Erlang, coalescent, etc.), this is computing the likelihood for the **wrong model**.

---

## Phase 4 Goal

Replace simplified exponential with **exact phase-type likelihood** using the existing C forward algorithm (`graph.pdf()`).

### Key Insight

```
trace + params → instantiate_from_trace() → graph → graph.pdf() → exact PDF
```

The C library already has the exact PDF computation (Algorithm 4 from paper). We just need to connect it to the trace system.

---

## Implementation Tasks

### Task 1: Update `trace_to_log_likelihood()` (PRIMARY)

**File**: `src/phasic/trace_elimination.py`
**Lines to modify**: 1116-1205

#### New Function Signature

```python
def trace_to_log_likelihood(trace: EliminationTrace, observed_data,
                           reward_matrix=None, granularity=0):
    """
    Convert trace to exact phase-type log-likelihood (univariate or multivariate)

    Parameters
    ----------
    trace : EliminationTrace
        From record_elimination_trace(graph, param_length=n)
    observed_data : array_like
        - 1D (n_obs,): univariate observations
        - 2D (n_obs, k): k-variate observations
    reward_matrix : array_like, optional
        - None: default rewards (all 1s) → univariate
        - 1D (n_vertices,): single reward → univariate
        - 2D (n_vertices, k): k rewards → k-variate
    granularity : int, default=0
        PDF uniformization granularity (0 = auto)

    Returns
    -------
    callable
        log_lik(params) -> scalar (JAX-compatible)
    """
```

#### Implementation Structure

```python
def trace_to_log_likelihood(trace, observed_data, reward_matrix=None, granularity=0):
    import jax
    import jax.numpy as jnp

    observed_data_np = np.asarray(observed_data)

    # Detect univariate vs multivariate
    is_multivariate = (reward_matrix is not None and
                      np.asarray(reward_matrix).ndim == 2)

    if is_multivariate:
        reward_matrix_np = np.asarray(reward_matrix)
        k_dims = reward_matrix_np.shape[1]

        # Validate shape
        if observed_data_np.ndim != 2 or observed_data_np.shape[1] != k_dims:
            raise ValueError(f"Shape mismatch: expected (n_obs, {k_dims})")

    def log_likelihood_impl(params_np):
        """Pure numpy/C++ implementation"""
        # 1. Instantiate graph from trace
        from . import Graph
        graph = instantiate_from_trace(trace, params_np)

        if is_multivariate:
            # Multivariate: sum k marginal log-likelihoods
            total_ll = 0.0
            for i in range(k_dims):
                # Transform with i-th reward vector
                graph_i = graph.reward_transform(reward_matrix_np[:, i])

                # Exact PDF via C forward algorithm
                pdf_i = graph_i.pdf(observed_data_np[:, i], granularity)

                # Accumulate log-likelihood
                total_ll += np.sum(np.log(np.maximum(pdf_i, 1e-10)))

            return total_ll

        else:
            # Univariate case
            if reward_matrix is not None:
                rewards_np = np.asarray(reward_matrix)
                graph = graph.reward_transform(rewards_np)

            # Exact PDF via C forward algorithm
            pdf_values = graph.pdf(observed_data_np, granularity)

            # Log-likelihood with stability
            return np.sum(np.log(np.maximum(pdf_values, 1e-10)))

    def log_likelihood_jax(params):
        """JAX wrapper with pure_callback"""
        result_shape = jax.ShapeDtypeStruct((), jnp.float64)

        return jax.pure_callback(
            lambda p: log_likelihood_impl(np.asarray(p)),
            result_shape,
            params,
            vmap_method='sequential'  # Enable vmap
        )

    return log_likelihood_jax
```

#### Key Points

1. **Always use exact PDF** - no `use_exact` flag (simplified exponential is just wrong for non-exponential models)
2. **Unified function** - handles both univariate and multivariate via `reward_matrix` shape
3. **JAX compatibility** - use `jax.pure_callback` like existing `ffi_wrappers.py`
4. **Leverage existing code** - `instantiate_from_trace()` and `graph.pdf()` already work
5. **Numerical stability** - `np.maximum(pdf, 1e-10)` before log

#### What to Remove

- **Lines 1179-1203**: The simplified exponential implementation
- Keep the docstring structure but update for new API

#### What to Keep

- **Lines 1208-1283**: `trace_to_pmf_function()` - unchanged
- **Lines 1286-1347**: `create_svgd_model_from_trace()` - update to use new signature

---

### Task 2: Update `create_svgd_model_from_trace()` (SECONDARY)

**File**: `src/phasic/trace_elimination.py`
**Lines**: 1286-1347

#### Changes Needed

```python
def create_svgd_model_from_trace(trace, model_type='log_likelihood', **kwargs):
    """Factory function for SVGD models"""

    if model_type == 'log_likelihood':
        observed_data = kwargs.get('observed_data')
        if observed_data is None:
            raise ValueError("observed_data required for log_likelihood")

        reward_matrix = kwargs.get('reward_matrix', None)  # Changed from reward_vector
        granularity = kwargs.get('granularity', 0)

        return trace_to_log_likelihood(trace, observed_data, reward_matrix, granularity)

    elif model_type in ['pmf', 'pdf']:
        # Unchanged
        times = kwargs.get('times')
        if times is None:
            raise ValueError("times required for pmf/pdf")
        discrete = model_type == 'pmf'
        return trace_to_pmf_function(trace, times, discrete)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

---

### Task 3: Create Test Suite - Validation

**New file**: `tests/test_trace_likelihood.py`

#### Test 1: Exponential Distribution Match

```python
def test_exact_matches_true_exponential():
    """For true exponential, exact PDF should work perfectly"""
    from phasic import Graph
    from phasic.trace_elimination import (
        record_elimination_trace,
        trace_to_log_likelihood
    )
    import numpy as np
    import jax.numpy as jnp

    # Build simple exponential model
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    start.add_edge(v1, 1.0)
    v1.add_edge_parameterized(v2, 0.0, [1.0])  # weight = θ[0]

    # Record trace
    trace = record_elimination_trace(g, param_length=1)

    # Generate test data from known exponential
    true_rate = 2.5
    observed = np.array([0.5, 1.0, 1.5, 2.0])

    # Compute log-likelihood
    log_lik = trace_to_log_likelihood(trace, observed)
    ll_value = log_lik(jnp.array([true_rate]))

    # Compare with analytical exponential log-likelihood
    analytical_ll = np.sum(np.log(true_rate) - true_rate * observed)

    assert np.abs(ll_value - analytical_ll) < 1e-6, \
        f"Exact PDF mismatch: {ll_value} vs {analytical_ll}"
```

#### Test 2: vs Direct graph.pdf()

```python
def test_trace_matches_direct_pdf():
    """Trace-based likelihood should match direct graph.pdf()"""
    # Build parameterized graph
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])

    start.add_edge(v1, 1.0)
    v1.add_edge_parameterized(v2, 0.0, [2.0])   # 2*θ
    v2.add_edge_parameterized(v3, 0.0, [1.5])   # 1.5*θ

    # Record trace
    trace = record_elimination_trace(g, param_length=1)

    # Test parameters
    theta = np.array([1.5])
    times = np.array([0.5, 1.0, 2.0])

    # Method 1: Via trace
    log_lik = trace_to_log_likelihood(trace, times)
    ll_trace = log_lik(jnp.array(theta))

    # Method 2: Direct (instantiate + pdf)
    from phasic.trace_elimination import instantiate_from_trace
    graph_direct = instantiate_from_trace(trace, theta)
    pdf_direct = graph_direct.pdf(times, granularity=0)
    ll_direct = np.sum(np.log(pdf_direct + 1e-10))

    assert np.abs(ll_trace - ll_direct) < 1e-8, \
        f"Trace vs direct mismatch: {ll_trace} vs {ll_direct}"
```

#### Test 3: Multivariate Support

```python
def test_multivariate_likelihood():
    """Test k-dimensional phase-type likelihood"""
    # Build graph with 3 vertices
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    start.add_edge(v1, 1.0)
    v1.add_edge_parameterized(v2, 0.0, [1.0])

    trace = record_elimination_trace(g, param_length=1)

    # Define 2-dimensional rewards
    reward_matrix = np.array([
        [1.0, 0.5],  # Vertex 0 (start)
        [2.0, 1.5],  # Vertex 1
        [0.0, 0.0],  # Vertex 2 (absorbing)
    ])

    # 2D observations
    observed_data = np.array([
        [1.0, 0.8],
        [1.5, 1.2],
        [0.5, 0.4],
    ])

    # Compute multivariate log-likelihood
    log_lik = trace_to_log_likelihood(trace, observed_data,
                                      reward_matrix=reward_matrix)

    ll_value = log_lik(jnp.array([2.0]))

    # Should be sum of two marginal log-likelihoods
    # (can verify by computing each marginal separately)
    assert np.isfinite(ll_value)
    assert ll_value < 0  # Log-likelihood should be negative
```

#### Test 4: JAX Transformations

```python
def test_jax_jit_vmap():
    """Test JAX transformations work"""
    import jax

    # Simple model
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    start.add_edge_parameterized(v1, 0.0, [1.0])

    trace = record_elimination_trace(g, param_length=1)
    observed = np.array([0.5, 1.0, 1.5])

    log_lik = trace_to_log_likelihood(trace, observed)

    # Test JIT
    jitted = jax.jit(log_lik)
    theta = jnp.array([2.0])
    ll1 = log_lik(theta)
    ll2 = jitted(theta)
    assert np.abs(ll1 - ll2) < 1e-10

    # Test vmap
    theta_batch = jnp.array([[1.0], [2.0], [3.0]])
    ll_batch = jax.vmap(log_lik)(theta_batch)
    assert ll_batch.shape == (3,)
```

#### Test 5: Numerical Stability

```python
def test_numerical_stability():
    """Test stability with extreme parameter values"""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    start.add_edge_parameterized(v1, 0.0, [1.0])

    trace = record_elimination_trace(g, param_length=1)
    observed = np.array([0.1, 0.5, 1.0])

    log_lik = trace_to_log_likelihood(trace, observed)

    # Very small rate (should handle without overflow)
    ll_small = log_lik(jnp.array([0.001]))
    assert np.isfinite(ll_small)

    # Very large rate (should handle without underflow)
    ll_large = log_lik(jnp.array([1000.0]))
    assert np.isfinite(ll_large)
```

---

### Task 4: Create Test Suite - Correctness

**New file**: `tests/test_trace_vs_matrix.py`

#### Test 1: vs Matrix Exponential

```python
def test_vs_matrix_exponential():
    """Compare exact trace likelihood vs matrix exponential"""
    import scipy.linalg

    # Build simple 3-state model
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    start.add_edge(v1, 1.0)
    v1.add_edge_parameterized(v2, 0.0, [2.0])  # 2*θ

    trace = record_elimination_trace(g, param_length=1)
    theta = np.array([1.5])
    times = np.array([0.5, 1.0])

    # Method 1: Trace + graph.pdf
    log_lik = trace_to_log_likelihood(trace, times)
    ll_trace = log_lik(jnp.array(theta))

    # Method 2: Matrix exponential
    # Build S matrix manually
    rate = 2.0 * theta[0]  # = 3.0
    S = np.array([
        [-rate, rate],
        [0.0, 0.0]
    ])
    alpha = np.array([1.0, 0.0])
    s_star = np.array([0.0, rate])

    # Compute PDF at each time
    pdf_matrix = []
    for t in times:
        exp_St = scipy.linalg.expm(S * t)
        pdf_t = alpha @ exp_St @ s_star
        pdf_matrix.append(pdf_t)

    ll_matrix = np.sum(np.log(np.array(pdf_matrix) + 1e-10))

    # Should match (within numerical precision)
    assert np.abs(ll_trace - ll_matrix) < 1e-5, \
        f"Matrix exponential mismatch: {ll_trace} vs {ll_matrix}"
```

#### Test 2: Rabbit Coalescent Model

```python
def test_rabbit_coalescent_correctness():
    """Validate on rabbit island model from paper"""

    def rabbit_callback(state):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [(np.array([n - 1]), 0.0, [rate])]

    # Build 5-rabbit model (22 vertices)
    g = Graph(
        state_length=1,
        callback=rabbit_callback,
        parameterized=True,
        nr_samples=5
    )

    trace = record_elimination_trace(g, param_length=1)

    # Test with known parameter
    theta = np.array([2.0])
    times = np.array([0.5, 1.0, 1.5])

    log_lik = trace_to_log_likelihood(trace, times)
    ll = log_lik(jnp.array(theta))

    # Should produce finite, negative log-likelihood
    assert np.isfinite(ll)
    assert ll < 0

    # Verify PDF sums correctly (integrate to ~1)
    from phasic.trace_elimination import instantiate_from_trace
    graph = instantiate_from_trace(trace, theta)
    fine_times = np.linspace(0.01, 10, 1000)
    pdf = graph.pdf(fine_times, granularity=200)
    integral = np.trapz(pdf, fine_times)
    assert 0.95 < integral < 1.05, f"PDF doesn't integrate to 1: {integral}"
```

---

### Task 5: Performance Benchmarks

**New file**: `tests/test_trace_phase4_performance.py`

```python
def benchmark_exact_vs_phase3_targets():
    """Verify Phase 4 still meets Phase 3 timing goals"""
    import time

    def rabbit_callback(state):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [(np.array([n - 1]), 0.0, [rate])]

    models = [
        (5, 22, "22 vertices (5 rabbits)"),
        (8, 37, "37 vertices (8 rabbits)"),
        (10, 67, "67 vertices (10 rabbits)"),
    ]

    for nr_rabbits, expected_vertices, desc in models:
        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print(f"{'='*60}")

        # Build model
        t0 = time.time()
        g = Graph(
            state_length=1,
            callback=rabbit_callback,
            parameterized=True,
            nr_samples=nr_rabbits
        )
        build_time = time.time() - t0

        actual_vertices = g.vertices_length()
        print(f"Vertices: {actual_vertices} (expected: {expected_vertices})")

        # Record trace
        t0 = time.time()
        trace = record_elimination_trace(g, param_length=1)
        record_time = time.time() - t0
        print(f"Trace recording: {record_time:.3f}s")

        # Generate test data
        observed = np.random.exponential(1.0, size=100)

        # Create log-likelihood
        t0 = time.time()
        log_lik = trace_to_log_likelihood(trace, observed, granularity=0)
        setup_time = time.time() - t0

        # Benchmark 1000 evaluations (SVGD workload)
        n_evals = 1000
        thetas = np.random.uniform(0.5, 2.0, size=(n_evals, 1))

        t0 = time.time()
        for theta in thetas:
            ll = log_lik(jnp.array(theta))
        eval_time = time.time() - t0

        total_time = record_time + setup_time + eval_time
        per_eval = eval_time / n_evals * 1000  # ms

        print(f"Evaluation time: {eval_time:.3f}s ({n_evals} evals)")
        print(f"Per evaluation: {per_eval:.2f}ms")
        print(f"Total time: {total_time:.3f}s")

        # Check against targets
        if expected_vertices <= 37:
            target = 300  # 5 minutes = 300 seconds
            status = "✓ PASS" if total_time < target else "✗ FAIL"
            print(f"Target <5 min: {status} ({total_time:.1f}s vs {target}s)")
        else:  # 67 vertices
            target = 1800  # 30 minutes
            status = "✓ PASS" if total_time < target else "✗ FAIL"
            print(f"Target <30 min: {status} ({total_time:.1f}s vs {target}s)")

if __name__ == '__main__':
    benchmark_exact_vs_phase3_targets()
```

---

### Task 6: Create Model Library (Simplified)

**Directory structure**:
```
src/phasic/models/
├── __init__.py
├── coalescent.py
├── queuing.py
└── reliability.py
```

#### File: `src/phasic/models/__init__.py`

```python
"""
PtDAlgorithms Model Library

Pre-built models for common phase-type distributions.
"""

from . import coalescent
from . import queuing
from . import reliability

__all__ = ['coalescent', 'queuing', 'reliability']
```

#### File: `src/phasic/models/coalescent.py`

```python
"""Coalescent models for population genetics"""

import numpy as np
from phasic import Graph

def constant_population(nr_samples, parameterized=True):
    """
    Kingman coalescent with constant population size

    Parameters
    ----------
    nr_samples : int
        Number of sampled lineages (n)
    parameterized : bool
        If True, coalescence rate = θ (parameter)
        If False, coalescence rate = 1.0

    Returns
    -------
    Graph
        Graph representing coalescent process

    Notes
    -----
    State: [n] = number of lineages
    Transitions: n → n-1 with rate n(n-1)/2 * θ
    """
    def callback(state):
        if len(state) == 0:
            # Initial state
            return [(np.array([nr_samples]), 1.0, [1.0] if parameterized else None)]

        n = state[0]
        if n <= 1:
            return []  # Absorbing (MRCA reached)

        rate = n * (n - 1) / 2

        if parameterized:
            # Weight = rate * θ
            return [(np.array([n - 1]), 0.0, [rate])]
        else:
            # Fixed weight
            return [(np.array([n - 1]), rate)]

    return Graph(
        state_length=1,
        callback=callback,
        parameterized=parameterized,
        nr_samples=nr_samples
    )


def exponential_growth(nr_samples, parameterized=True):
    """
    Coalescent with exponential population growth

    Parameters
    ----------
    nr_samples : int
        Number of lineages
    parameterized : bool
        If True, uses parameters [N0, growth_rate]

    Returns
    -------
    Graph
        Exponential growth coalescent

    Notes
    -----
    Population size: N(t) = N0 * exp(growth_rate * t)
    Rate: n(n-1)/(2*N0) adjusted for growth
    """
    # TODO: Implement exponential growth model
    raise NotImplementedError("Exponential growth coalescent not yet implemented")
```

#### File: `src/phasic/models/queuing.py`

```python
"""Queuing theory models"""

import numpy as np
from phasic import Graph

def mm1_queue(max_queue_size=10, parameterized=True):
    """
    M/M/1 queue (Poisson arrivals, exponential service, 1 server)

    Parameters
    ----------
    max_queue_size : int
        Maximum queue capacity
    parameterized : bool
        If True, uses parameters [arrival_rate, service_rate]

    Returns
    -------
    Graph
        M/M/1 queue model

    Notes
    -----
    State: [n] = number in system (0 to max_queue_size)
    Transitions:
    - n → n+1 with rate λ (arrivals)
    - n → n-1 with rate μ (service completions)
    """
    def callback(state):
        if len(state) == 0:
            # Start empty
            return [(np.array([0]), 1.0, [0.0, 0.0] if parameterized else None)]

        n = state[0]
        transitions = []

        if parameterized:
            # Arrival (if not at capacity)
            if n < max_queue_size:
                transitions.append((np.array([n + 1]), 0.0, [1.0, 0.0]))  # λ

            # Service (if queue not empty)
            if n > 0:
                transitions.append((np.array([n - 1]), 0.0, [0.0, 1.0]))  # μ
        else:
            # Fixed rates (example: λ=1, μ=2)
            if n < max_queue_size:
                transitions.append((np.array([n + 1]), 1.0))
            if n > 0:
                transitions.append((np.array([n - 1]), 2.0))

        return transitions

    return Graph(
        state_length=1,
        callback=callback,
        parameterized=parameterized,
        max_queue_size=max_queue_size
    )
```

#### File: `src/phasic/models/reliability.py`

```python
"""Reliability models for system failure analysis"""

import numpy as np
from phasic import Graph

def series_system(n_components, parameterized=True):
    """
    Series reliability system (fails if any component fails)

    Parameters
    ----------
    n_components : int
        Number of components in series
    parameterized : bool
        If True, each component has parameter θᵢ (failure rate)

    Returns
    -------
    Graph
        Series system reliability model

    Notes
    -----
    State: [k] = number of working components
    System fails when k=0
    Overall failure rate = sum of component failure rates
    """
    def callback(state):
        if len(state) == 0:
            # Start with all working
            init_params = [0.0] * n_components if parameterized else None
            return [(np.array([n_components]), 1.0, init_params)]

        k = state[0]
        if k == 0:
            return []  # System failed

        if parameterized:
            # Each working component can fail
            # Combined rate = θ₁ + θ₂ + ... + θₖ
            coeffs = [1.0] * k + [0.0] * (n_components - k)
            return [(np.array([k - 1]), 0.0, coeffs)]
        else:
            # Fixed failure rate per component
            return [(np.array([k - 1]), k * 1.0)]

    return Graph(
        state_length=1,
        callback=callback,
        parameterized=parameterized,
        n_components=n_components
    )
```

---

## Implementation Order

### Week 1: Core Implementation
1. **Day 1**: Implement new `trace_to_log_likelihood()` function
2. **Day 2**: Update `create_svgd_model_from_trace()`, remove old code
3. **Day 3-4**: Create `tests/test_trace_likelihood.py` (5 tests)
4. **Day 5**: Debug and fix any issues, ensure all tests pass

### Week 2: Validation & Library
5. **Day 1-2**: Create `tests/test_trace_vs_matrix.py` (correctness validation)
6. **Day 3**: Performance benchmarks `tests/test_trace_phase4_performance.py`
7. **Day 4-5**: Model library (3-5 models in `src/phasic/models/`)

---

## Success Criteria

- [ ] `trace_to_log_likelihood()` uses exact `graph.pdf()` (no simplified exponential)
- [ ] Univariate and multivariate support via `reward_matrix` shape
- [ ] All tests in `test_trace_likelihood.py` passing (5/5)
- [ ] All tests in `test_trace_vs_matrix.py` passing (2/2)
- [ ] Performance benchmarks show <10s for 67v/1000 evals (vs 5s Phase 3)
- [ ] Phase 3 targets still met: 37v <5min ✓, 67v <30min ✓
- [ ] 3-5 models in library (coalescent, queuing, reliability)
- [ ] All Phase 1-3 tests still passing (backward compatibility)
- [ ] TRACE_PHASE4_STATUS.md completed

---

## Key Implementation Notes

### graph.pdf() Usage

**Signature**: `graph.pdf(times, granularity=0)`
- `times`: numpy array of time points
- `granularity`: 0 = auto (2 × max_rate), or explicit value
- **Returns**: numpy array of PDF values (same shape as times)
- **Vectorized**: Handles arrays efficiently

**Example**:
```python
pdf_values = graph.pdf(np.array([0.5, 1.0, 1.5]), granularity=100)
# Returns: array([0.23, 0.15, 0.08])
```

### instantiate_from_trace() Usage

**Signature**: `instantiate_from_trace(trace, params)`
- Creates concrete (non-parameterized) graph
- Edge weights computed from trace evaluation
- Returns Graph object with all Python API methods

**Important**: Result is NOT normalized - don't call `normalize()` after instantiation (already handled in trace evaluation)

### JAX pure_callback Pattern

**Template** (from `ffi_wrappers.py`):
```python
import jax

def jax_wrapper(jax_params):
    result_shape = jax.ShapeDtypeStruct(output_shape, jnp.float64)

    return jax.pure_callback(
        lambda p: numpy_function(np.asarray(p)),
        result_shape,
        jax_params,
        vmap_method='sequential'  # KEY: enables vmap
    )
```

**Key points**:
- Convert JAX arrays to numpy inside lambda
- Specify output shape and dtype
- `vmap_method='sequential'` enables batching
- No gradients through pure_callback (numerical differentiation required)

---

## Files to Modify

### Modify Existing
- [ ] `src/phasic/trace_elimination.py` (lines 1116-1347)

### Create New
- [ ] `tests/test_trace_likelihood.py`
- [ ] `tests/test_trace_vs_matrix.py`
- [ ] `tests/test_trace_phase4_performance.py`
- [ ] `src/phasic/models/__init__.py`
- [ ] `src/phasic/models/coalescent.py`
- [ ] `src/phasic/models/queuing.py`
- [ ] `src/phasic/models/reliability.py`
- [ ] `TRACE_PHASE4_STATUS.md`

---

## Running Tests

```bash
# Phase 4 tests
pixi run python -m pytest tests/test_trace_likelihood.py -v
pixi run python -m pytest tests/test_trace_vs_matrix.py -v

# Performance benchmarks
pixi run python tests/test_trace_phase4_performance.py

# Verify backward compatibility
pixi run python -m pytest tests/test_trace_recording.py -v  # Phase 1
pixi run python -m pytest tests/test_trace_jax.py -v        # Phase 2
pixi run python tests/test_trace_svgd_benchmark.py          # Phase 3

# All tests
pixi run test
```

---

**Ready to implement - all design decisions made, code patterns established, success criteria clear.**
