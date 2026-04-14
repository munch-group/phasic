# Plan: Homogenize Graph Composition Methods

## Context

Make `laplace_transform`, `discretize`, `joint_prob_graph`, and a new `add_epoch` all return a new graph. Extra info (rewards) stored as attributes. No method requires pre-arranged state/coefficient layout. Composition pipeline:

```python
graph = Graph(coalescent)
g1 = graph.add_epoch(t1)
g2 = g1.add_epoch(t2)
g3 = g2.discretize(rate=0.1)
joint = g3.joint_prob_graph(indexer, tot_reward_limit=3)
```

## Migration Strategy

**Old implementations are commented out, not deleted.** When replacing a method (e.g., `discretize`), the original implementation is commented out in-place with a `# COMPOSABLE_MIGRATION: original implementation` marker. The new implementation is added directly below. This preserves the old code for reference and easy rollback.

**Call sites: old patterns are commented out.** When updating callers, the old line is commented out (with `# COMPOSABLE_MIGRATION: old call`) and the new line is added below it. This makes the diff reviewable and reversible.

**Equivalence tests:** Each batch includes tests that build the same graph using both the old (in-place) approach and the new (returns-new-graph) approach, then assert the resulting graphs are structurally identical (same vertices, edges, weights, rewards).

## C-Level Constraints

1. `param_length` locked after first non-IPV edge (`phasic.c:3076`) — must rebuild graph to widen
2. `add_edge_parameterized` rejects coefficients shorter than `param_length` (`phasic.c:2894`)
3. `update_weights` requires exact theta/param_length match (`phasic.c:3141`)
4. `edge_state(n)` returns min(n, actual) coefficients — must manually pad
5. State length fixed at `Graph(state_length)` — no resizing
6. `add_aux_vertex` on parameterized graphs requires coefficient vector, not scalar (`pybind:3484`)
7. Hash system is structure-based — works on any graph regardless of construction method

---

## Step 1: `_rebuild_with_wider_layout` helper

**File:** `src/phasic/__init__.py`, add after `clone()` method (line ~6394)

```python
def _rebuild_with_wider_layout(self, extra_state_dims=0, extra_coeff_slots=0,
                                state_fill=0, coeff_fill=0.0) -> 'Graph':
    """Rebuild graph with wider state vectors and/or coefficient arrays."""
    new_state_length = self.state_length() + extra_state_dims
    new_param_length = self.param_length() + extra_coeff_slots

    new_graph = Graph(new_state_length)

    # Set param_length BEFORE adding any edges
    if new_param_length > 0:
        new_graph.set_param_length(new_param_length)

    pad_state = np.full(extra_state_dims, state_fill, dtype=int)
    pad_coeff = [coeff_fill] * extra_coeff_slots

    # Map: old vertex index → new vertex object
    vertex_map = {}

    # Starting vertex maps to new starting vertex
    vertex_map[self.starting_vertex().index()] = new_graph.starting_vertex()

    # Create all non-starting vertices first
    for vertex in self.vertices():
        idx = vertex.index()
        if idx == self.starting_vertex().index():
            continue
        new_state = np.append(vertex.state(), pad_state).astype(int)
        new_vertex = new_graph.find_or_create_vertex(new_state)
        vertex_map[idx] = new_vertex

    # Copy all edges with padded coefficients
    for vertex in self.vertices():
        old_idx = vertex.index()
        new_vertex = vertex_map[old_idx]
        if self.parameterized():
            for edge in vertex.parameterized_edges():
                coeffs = list(edge.edge_state(self.param_length())) + pad_coeff
                new_vertex.add_edge(vertex_map[edge.to().index()], coeffs)
        else:
            for edge in vertex.edges():
                new_vertex.add_edge(vertex_map[edge.to().index()], edge.weight())

    # Copy Python metadata
    new_graph._callback = self._callback
    new_graph._callback_kwargs = self._callback_kwargs.copy() if self._callback_kwargs else {}
    new_graph._weight_mode = self._weight_mode
    new_graph._weight_callback = self._weight_callback
    new_graph.is_discrete = self.is_discrete
    new_graph._last_callback_vertices_length = new_graph.vertices_length()

    return new_graph
```

**Considerations:**
- `parameterized_edges()` returns only parameterized edges. `edges()` returns all edges (including parameterized ones with concrete weights). For parameterized graphs, we must use `parameterized_edges()` + `edge_state()` to get the coefficients.
- `edge_state(param_length)` returns the stored coefficients. We pad with `coeff_fill`.
- Vertex ordering in new graph matches old graph (same `find_or_create_vertex` order).

---

## Step 2: Update `laplace_transform` (line 2998)

**Current code:**
```python
def laplace_transform(self, theta: float) -> Self:
    if self.is_discrete:
        raise ValueError(...)
    return Graph(super().laplace_transform(theta))
```

**New code (old return statement commented out):**
```python
def laplace_transform(self, theta: float) -> Self:
    if self.is_discrete:
        raise ValueError(...)
    result = Graph(super().laplace_transform(theta))
    # Copy metadata from source graph
    result._callback = self._callback
    result._callback_kwargs = self._callback_kwargs.copy() if self._callback_kwargs else {}
    result._weight_mode = self._weight_mode
    result._weight_callback = self._weight_callback
    result._last_callback_vertices_length = result.vertices_length()
    # COMPOSABLE_MIGRATION: original implementation
    # return Graph(super().laplace_transform(theta))
    return result
```

**Change:** Replaces the bare `return Graph(...)` with a version that copies metadata. Old return commented out. No signature change. No call-site changes.

---

## Step 3: Update `discretize` (line 2858)

**Current signature:** `discretize(rate, skip_existing=False, **kwargs) -> NDArray[np.int64]`
**New signature:** `discretize(rate, skip_existing=False, **kwargs) -> Graph`

**New implementation:**

```python
def discretize(self, rate: float | Callable, skip_existing: bool = False, **kwargs: Any) -> 'Graph':
    """
    Create a discretized copy of this graph.

    Returns a new graph with auxiliary vertices added for each transient state.
    The original graph is not modified.

    Parameters
    ----------
    rate : float or callable
        Discretization rate. If callable, receives state array and **kwargs,
        must return a scalar rate (for non-parameterized graphs) or a
        coefficient vector (for parameterized graphs).
    skip_existing : bool, optional
        If True, skip vertices that already have auxiliary vertices.

    Returns
    -------
    Graph
        New discretized graph with `.rewards` attribute containing the
        reward vector (1 for auxiliary vertices, 0 otherwise).
    """
    if not callable(rate):
        if not isinstance(rate, (int, float, np.integer, np.floating)):
            raise TypeError(f"rate must be a number or callable, got {type(rate).__name__}")
        if rate <= 0 or rate >= 1:
            raise ValueError(f"rate must be in (0, 1), got {rate}")

    # Work on a clone so the original is not modified
    new_graph = self.clone()

    vlength = new_graph.vertices_length()
    aux_indices = []

    for vertex in new_graph.vertices():
        if vertex.index() == new_graph.starting_vertex().index() or not vertex.edges():
            continue

        if skip_existing:
            has_aux, is_aux = False, False
            for edge in vertex.edges():
                if edge.to().state().sum() == 0 and edge.to().edges_length() and edge.to().edges()[0].to().index() == vertex.index():
                    has_aux = True
                    aux_indices.append(edge.to().index())
                    vlength -= 1
                    break
            if vertex.state().sum() == 0:
                is_aux = True
            if has_aux or is_aux:
                continue

        _rate = rate(vertex.state(), **kwargs) if callable(rate) else rate
        aux_vertex = vertex.add_aux_vertex(_rate)
        aux_vertex.set_aux(True)
        aux_indices.append(aux_vertex.index())

    rewards = np.zeros(vlength + len(aux_indices), dtype=int)
    for index in aux_indices:
        rewards[index] = 1

    new_graph.normalize()

    new_graph.is_discrete = True
    new_graph.set_was_dph(True)

    new_graph.rewards = rewards
    return new_graph
```

**Key differences from current:**
1. Calls `self.clone()` instead of mutating `self`
2. Operates on `new_graph` throughout
3. Stores `rewards` as attribute on `new_graph`
4. Returns `new_graph` instead of `rewards`
5. No `@_invalidates_trace` needed (original graph untouched)

**Migration pattern in source:** The original `discretize` method is renamed to `_discretize_inplace` (keeping the `@_invalidates_trace` decorator) and its body is commented out with `# COMPOSABLE_MIGRATION: original implementation`. The new `discretize` method is added directly below. This keeps the old implementation accessible for equivalence testing via `graph._discretize_inplace(rate)`.

```python
@_invalidates_trace
def _discretize_inplace(self, rate: float | Callable, skip_existing: bool = False, **kwargs: Any) -> NDArray[np.int64]:
    """COMPOSABLE_MIGRATION: original in-place discretize, kept for equivalence testing."""
    # ... original implementation unchanged ...
```

---

## Step 4: `add_epoch` method

**File:** `src/phasic/__init__.py`, add after `discretize` (line ~2925)

```python
def add_epoch(self, time: float, callback: Callable | None = None, **kwargs: Any) -> 'Graph':
    """
    Add an epoch boundary, returning a new graph with epoch transition edges.

    Computes transition rates from stop_probability(time) / accumulated_occupancy(time)
    and wires up sister vertices in the next epoch. The new graph has one additional
    state dimension (epoch index) if this is the first epoch, and one additional
    coefficient slot (epoch transition rate).

    Parameters
    ----------
    time : float
        Time at which the epoch boundary occurs.
    callback : callable, optional
        Callback for building out new-epoch vertices. If None, uses the
        stored callback from graph construction.
    **kwargs
        Additional keyword arguments merged with stored callback kwargs.

    Returns
    -------
    Graph
        New graph with epoch transitions wired up.
    """
    if callback is None and self._callback is None:
        raise RuntimeError(
            "No callback available. Either construct the graph with a callback "
            "or provide one to add_epoch()."
        )

    # Determine if this is the first epoch (need to add state dimension)
    is_first_epoch = not hasattr(self, '_epoch_state_index')
    extra_state = 1 if is_first_epoch else 0

    # Rebuild with wider layout
    new_graph = self._rebuild_with_wider_layout(
        extra_state_dims=extra_state,
        extra_coeff_slots=1
    )

    # Track epoch metadata
    if is_first_epoch:
        new_graph._epoch_state_index = self.state_length()  # index of new epoch dim
        new_graph._n_epochs = 1
    else:
        new_graph._epoch_state_index = self._epoch_state_index
        new_graph._n_epochs = self._n_epochs + 1

    epoch_idx = new_graph._n_epochs  # current epoch number (0-based: 0 was original)
    epoch_state_idx = new_graph._epoch_state_index

    # Compute transition rates on the ORIGINAL graph
    # (must be done before rebuilding changes anything)
    stop_probs = np.array(self.stop_probability(time))
    acc_occ = np.array(self.accumulated_occupancy(time))

    with np.errstate(invalid='ignore'):
        transition_rates = stop_probs / acc_occ

    # New coefficient slot index (last slot in widened coeff array)
    new_coeff_idx = new_graph.param_length() - 1

    # Wire epoch transitions on the new graph
    # For each transient vertex in the previous epoch, add edge to sister in new epoch
    n_vertices_before_extend = new_graph.vertices_length()

    for i in range(1, n_vertices_before_extend):
        vertex = new_graph.vertex_at(i)
        state = vertex.state()

        # Skip absorbing vertices
        if vertex.edges_length() == 0:
            continue

        # Only process vertices in the previous epoch
        if state[epoch_state_idx] != epoch_idx - 1:
            continue

        # Skip if transition rate is NaN (unreachable state)
        if np.isnan(transition_rates[i]) if i < len(transition_rates) else True:
            continue

        # Create sister vertex in new epoch
        sister_state = state.copy()
        sister_state[epoch_state_idx] = epoch_idx
        child = new_graph.find_or_create_vertex(sister_state)

        # Add epoch transition edge with rate in the new coeff slot
        coeff = np.zeros(new_graph.param_length())
        coeff[new_coeff_idx] = transition_rates[i]
        vertex.add_edge(child, list(coeff))

    # Prepare callback for extending new-epoch vertices
    use_callback = callback if callback is not None else self._callback
    use_kwargs = {}
    if callback is None and self._callback_kwargs:
        use_kwargs = self._callback_kwargs.copy()
    use_kwargs.update(kwargs)

    # Wrap callback to:
    # 1. Strip epoch dimension from state before passing to original callback
    # 2. Re-append epoch index to returned states  
    # 3. Zero-pad coefficients to new width
    original_param_length = self.param_length()
    extra_coeff_slots_total = new_graph.param_length() - original_param_length

    def epoch_callback_wrapper(state, **kw):
        # Only generate transitions for vertices in the current epoch
        if state[epoch_state_idx] != epoch_idx:
            return []

        # Strip epoch dim(s) added by add_epoch before passing to original callback
        if is_first_epoch:
            base_state = np.delete(state, epoch_state_idx)
        else:
            base_state = np.delete(state, epoch_state_idx)

        # Call original callback
        transitions = use_callback(base_state, **kw)

        # Re-append epoch index and pad coefficients
        result = []
        for transition in transitions:
            child_state, coeffs = transition[0], transition[1]
            # Re-insert epoch dimension
            new_child = np.insert(np.asarray(child_state, dtype=int), 
                                  epoch_state_idx, epoch_idx)
            # Pad coefficient vector
            coeffs_list = list(coeffs)
            padded = coeffs_list + [0.0] * (new_graph.param_length() - len(coeffs_list))
            result.append([new_child, padded])
        return result

    # Extend graph to build out new-epoch vertices
    new_graph._callback = epoch_callback_wrapper
    new_graph._callback_kwargs = use_kwargs
    new_graph.extend(epoch_callback_wrapper, **use_kwargs)

    return new_graph
```

**How this maps to the tutorial's manual approach:**
- Tutorial's `coalescent_1param` pre-allocates `np.zeros(len(epochs)+1)` → replaced by automatic padding
- Tutorial's `add_epoch` manually computes `stop_probs / accum_v_time` → same logic here
- Tutorial's `graph.extend(callback, ...)` → same, but with wrapped callback
- Tutorial's `indexer.epoch` slot → replaced by `_epoch_state_index`

---

## Step 5: Update all `discretize` call sites

### `tests/pytest/test_exp_geom.py:23-25`

```python
# COMPOSABLE_MIGRATION: old call
# rewards = disc_graph.discretize(1-rate)
# assert disc_graph.expectation_discrete(rewards) == approx((1 - rate)  / rate)
# assert disc_graph.variance_discrete(rewards) == approx((1 - rate) / rate**2)
disc_graph = disc_graph.discretize(1-rate)
assert disc_graph.expectation_discrete(disc_graph.rewards) == approx((1 - rate)  / rate)
assert disc_graph.variance_discrete(disc_graph.rewards) == approx((1 - rate) / rate**2)
```

### `tests/pytest/test_api_comprehensive.py:398-402`

```python
# COMPOSABLE_MIGRATION: old call
# rewards = g.discretize(0.1)
g = g.discretize(0.1)
sample_result = g.sample(10)
```

### `tests/pytest/test_api_comprehensive.py:421-426`

```python
# COMPOSABLE_MIGRATION: old call
# rewards = g.discretize(0.1)
g = g.discretize(0.1)
assert g.is_discrete == True
vertices_length_after = g.vertices_length()
```

### `tests/pytest/test_comprehensive_api.py:333-336`

```python
# COMPOSABLE_MIGRATION: old call
# _ = g.discretize(0.1)
g = g.discretize(0.1)
pmf = g.pdf(5)
```

### `tests/pytest/test_comprehensive_api.py:381-385`

```python
# COMPOSABLE_MIGRATION: old call
# _ = g.discretize(0.1)
g = g.discretize(0.1)
assert all([0 <= prob <= 1 for prob in g.stop_probability(1)])
```

### `tests/pytest/test_comprehensive_api.py:564-568`

**Current (aspirational, currently broken):**
```python
g_discrete, rewards = g.discretize(reward_rate=0.1)
assert g_discrete is not None
assert g_discrete.vertices_length() >= g.vertices_length()
assert rewards.shape[1] == g_discrete.vertices_length()
```

**New:**
```python
g_discrete = g.discretize(0.1)
assert g_discrete is not None
assert g_discrete.vertices_length() >= g.vertices_length()
assert len(g_discrete.rewards) == g_discrete.vertices_length()
```

### `tests/pytest/test_comprehensive_api.py:580`

**New:**
```python
g_discrete = g.discretize(0.1)  # skip_states not in current API, remove or adapt
assert g_discrete is not None
```

### `tests/pytest/test_comprehensive_api.py:591`

**New:**
```python
g_discrete = g.discretize(0.1)  # skip_slots not in current API, remove or adapt
assert g_discrete is not None
```

### `tests/pytest/test_input_validation.py:264-288`

These test error-raising on bad inputs. The validation logic is unchanged (same checks at top of method), so these tests work as-is — `discretize` still raises before cloning if rate is invalid.

### Tutorial notebooks

#### `docs/pages/tutorial/discrete.ipynb`

**Cell 7** (code) — discretize + plot:
```python
# Current:
graph = exponential(0.4)
rewards = graph.discretize(1-rate)
graph.plot()

# New:
graph = exponential(0.4)
graph = graph.discretize(1-rate)
graph.plot()
```

**Cell 8** (markdown) — update description:
```
# Current: "The `discretize` method changes to graph in-place..."
# New: "The `discretize` method returns a new graph with auxiliary vertices..."
```

**Cell 9** (code) — uses `graph` and `rewards` after discretize:
```python
# Current:
cdf = graph.reward_transform(rewards).cdf(x)

# New:
cdf = graph.reward_transform(graph.rewards).cdf(x)
```

**Cell 11** (code) — second discretize example:
```python
# Current:
graph = exponential(rate)
rewards = graph.discretize(1-rate)
...
cdf = graph.reward_transform(rewards).cdf(x)

# New:
graph = exponential(rate)
graph = graph.discretize(1-rate)
...
cdf = graph.reward_transform(graph.rewards).cdf(x)
```

**Cell 19** (code) — callable rate:
```python
# Current:
rewards = mutation_graph.discretize(mutation, mutation_rate=mutation_rate)

# New:
mutation_graph = mutation_graph.discretize(mutation, mutation_rate=mutation_rate)
```

**Cell 21** (code) — uses `mutation_graph` after discretize:
```python
# Current:
mutation_graph.plot()

# No change needed (still references mutation_graph)
```

#### `docs/pages/tutorial/parametrization.ipynb`

**Cell 19** (code) — parameterized discretize:
```python
# Current:
rewards = mutation_graph.discretize(mutation_rate)

# New:
mutation_graph = mutation_graph.discretize(mutation_rate)
```

**Cell 20** (code) — uses `mutation_graph` + `rewards` after discretize:
```python
# Current:
mutation_graph.update_weights([3, 2])
rt_graph = mutation_graph.reward_transform(rewards)

# New:
mutation_graph.update_weights([3, 2])
rt_graph = mutation_graph.reward_transform(mutation_graph.rewards)
```

**Cell 21** (code) — same pattern:
```python
# Current:
mutation_graph.update_weights([3, 2])
rt_graph = mutation_graph.reward_transform(rewards)

# New:
mutation_graph.update_weights([3, 2])
rt_graph = mutation_graph.reward_transform(mutation_graph.rewards)
```

#### `docs/pages/tutorial/time-inhomogeneous.ipynb`

This notebook gets a larger rewrite. The "Moments of epoch-wise time homogeneous phase-type distributions" section (cells around the second `coalescent_1param` definition and `add_epoch` function) is replaced with the new `add_epoch` API. The "Discrete distributions" section with `coalescent_2param` and manual epoch/discretize wiring is also simplified.

See Step 7 for the full rewritten code.

#### `docs/pages/tutorial/laplace.ipynb`, `joint-probability.ipynb`, `svgd-joint-prob.ipynb`, `svgd-multi-param.ipynb`, `method-of-moments.ipynb`

These use `laplace_transform` and `joint_prob_graph` which already return new graphs. **No changes needed** — the API is unchanged for these methods.

#### `docs/pages/tutorial/finite-markov-chains.ipynb`

All discretize calls are commented out (lines 631, 664, 850). **No changes needed.**

---

## Step 6: New test file

**File:** `tests/pytest/test_composable_operations.py`

```python
"""Tests for composable graph operations."""
import numpy as np
from pytest import approx
from phasic import Graph, with_ipv


def _make_coalescent(nr_samples=4):
    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions
    return coalescent


class TestDiscretizeReturnsNewGraph:
    def test_returns_graph(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        v2 = g.find_or_create_vertex([2])
        v.add_edge(v2, 1.0)

        result = g.discretize(0.1)
        assert isinstance(result, Graph)
        assert hasattr(result, 'rewards')
        assert result.is_discrete

    def test_original_unchanged(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        v2 = g.find_or_create_vertex([2])
        v.add_edge(v2, 1.0)

        orig_n = g.vertices_length()
        orig_discrete = g.is_discrete

        _ = g.discretize(0.1)

        assert g.vertices_length() == orig_n
        assert g.is_discrete == orig_discrete


class TestAddEpoch:
    def test_single_epoch(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        assert g1.vertices_length() > graph.vertices_length()
        assert g1.param_length() == graph.param_length() + 1

    def test_original_unchanged(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        orig_n = graph.vertices_length()
        orig_pl = graph.param_length()

        _ = graph.add_epoch(1.0)

        assert graph.vertices_length() == orig_n
        assert graph.param_length() == orig_pl

    def test_chained_epochs(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        g2 = g1.add_epoch(2.0)
        assert g2.param_length() == graph.param_length() + 2
        assert g2.vertices_length() > g1.vertices_length()


class TestCompositionPipeline:
    def test_epoch_then_discretize(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        g1.update_weights([1.0, 1.0])  # coal_rate + epoch_trans

        g2 = g1.discretize(0.1)
        assert g2.is_discrete
        assert hasattr(g2, 'rewards')

    def test_laplace_preserves_metadata(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        lt = graph.laplace_transform(0.5)
        assert lt._callback is not None  # metadata preserved


class TestRebuildWithWiderLayout:
    def test_wider_state(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)

        wider = g._rebuild_with_wider_layout(extra_state_dims=1)
        assert wider.state_length() == 2
        assert wider.vertices_length() == g.vertices_length()

    def test_wider_coefficients(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)

        wider = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert wider.param_length() == graph.param_length() + 1
        assert wider.vertices_length() == graph.vertices_length()

    def test_rebuild_preserves_structure(self):
        """Vertex count, edge targets, and edge weights match original."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        rebuilt = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert rebuilt.vertices_length() == graph.vertices_length()
        for i in range(graph.vertices_length()):
            orig_v = graph.vertex_at(i)
            new_v = rebuilt.vertex_at(i)
            # Same number of edges
            assert orig_v.edges_length() == new_v.edges_length()
            # Edge targets match
            orig_targets = sorted([e.to().index() for e in orig_v.edges()])
            new_targets = sorted([e.to().index() for e in new_v.edges()])
            assert orig_targets == new_targets

    def test_rebuild_no_change(self):
        """extra_state_dims=0, extra_coeff_slots=0 produces equivalent graph."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([2.0])

        rebuilt = graph._rebuild_with_wider_layout()
        assert rebuilt.vertices_length() == graph.vertices_length()
        assert rebuilt.param_length() == graph.param_length()
        assert rebuilt.state_length() == graph.state_length()
        # Same moments (structural equivalence)
        assert rebuilt.moments(3) == approx(graph.moments(3), rel=1e-10)

    def test_rebuild_preserves_metadata(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph._weight_mode = 'log'

        rebuilt = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert rebuilt._callback is graph._callback
        assert rebuilt._weight_mode == 'log'
        assert rebuilt.is_discrete == graph.is_discrete


class TestEquivalenceOldVsNew:
    """Verify new implementations produce identical graphs to old implementations."""

    def test_discretize_equivalence(self):
        """New discretize produces same graph structure and rewards as old in-place version."""
        # Build two identical graphs
        def _build():
            g = Graph(1)
            start = g.starting_vertex()
            v1 = g.find_or_create_vertex([1])
            v2 = g.find_or_create_vertex([2])
            start.add_edge(v1, 2.0)
            v1.add_edge(v2, 2.0)
            return g

        # Old approach: in-place (call the commented-out _discretize_inplace)
        g_old = _build()
        rewards_old = g_old._discretize_inplace(0.1)

        # New approach: returns new graph
        g_orig = _build()
        g_new = g_orig.discretize(0.1)

        # Same vertices count
        assert g_new.vertices_length() == g_old.vertices_length()
        # Same rewards
        assert np.array_equal(g_new.rewards, rewards_old)
        # Same is_discrete flag
        assert g_new.is_discrete == g_old.is_discrete
        # Same edge structure
        for i in range(g_old.vertices_length()):
            v_old = g_old.vertex_at(i)
            v_new = g_new.vertex_at(i)
            assert v_old.edges_length() == v_new.edges_length()
            old_weights = sorted([e.weight() for e in v_old.edges()])
            new_weights = sorted([e.weight() for e in v_new.edges()])
            assert old_weights == approx(new_weights, rel=1e-10)
        # Same moments
        assert g_new.expectation_discrete(g_new.rewards) == approx(
            g_old.expectation_discrete(rewards_old), rel=1e-10)

    def test_discretize_equivalence_callable(self):
        """New discretize with callable rate matches old in-place version."""
        coalescent = _make_coalescent(4)

        g_old = Graph(coalescent)
        g_old.update_weights([1.0])
        def rate_fn(state):
            return [0, sum(state) * 0.1]
        rewards_old = g_old._discretize_inplace(rate_fn)

        g_orig = Graph(coalescent)
        g_orig.update_weights([1.0])
        g_new = g_orig.discretize(rate_fn)

        assert g_new.vertices_length() == g_old.vertices_length()
        assert np.array_equal(g_new.rewards, rewards_old)

    def test_laplace_equivalence(self):
        """New laplace_transform with metadata matches old output structurally."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        lt = graph.laplace_transform(0.5)

        # Structural test: same number of vertices/edges as before
        # (laplace_transform doesn't change structure, just adds absorbing edges)
        assert lt.vertices_length() == graph.vertices_length()
        # Metadata is now preserved
        assert lt._callback is not None

    def test_epoch_moments_match_tutorial(self):
        """New add_epoch produces same moments as tutorial's manual epoch wiring."""
        from phasic import StateIndexer, Property
        from itertools import combinations_with_replacement
        from functools import partial
        all_pairs = partial(combinations_with_replacement, r=2)

        nr_samples = 10
        epochs = [0, 1, 2]
        pop_sizes = [1, 5, 10]

        # --- Manual approach (from tutorial) ---
        indexer = StateIndexer(
            lineages=[Property('ton', min_value=1, max_value=nr_samples)],
            slots=['epoch']
        )

        def coalescent_1param(state, epochs=None, epoch_idx=None, indexer=None):
            transitions = []
            epoch_idx = int(epoch_idx)
            if state[indexer.epoch] != epoch_idx:
                return transitions
            for i, j in all_pairs(indexer.lineages):
                pi = indexer.lineages.index_to_props(i)
                pj = indexer.lineages.index_to_props(j)
                if state.sum() <= 1:
                    continue
                same = int(pi.ton == pj.ton)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                k = indexer.props_to_index(ton=pi.ton + pj.ton)
                new[k] += 1
                coeff = np.zeros(len(epochs) + 1)
                coeff[epoch_idx] = state[i] * (state[j] - same) / (1 + same)
                transitions.append([new, coeff])
            return transitions

        def manual_add_epoch(graph, callback, epochs, epoch_idx, indexer):
            epoch = epochs[epoch_idx]
            stop_probs = np.array(graph.stop_probability(epoch))
            accum_v_time = np.array(graph.accumulated_occupancy(epoch))
            with np.errstate(invalid='ignore'):
                epoch_trans_rates = stop_probs / accum_v_time
            for i in range(1, graph.vertices_length() - 1):
                if epoch_trans_rates is None or np.isnan(epoch_trans_rates[i]):
                    continue
                if graph.vertex_at(i).edges_length() == 0:
                    continue
                vertex = graph.vertex_at(i)
                state = vertex.state()
                if not state[indexer.epoch] == epoch_idx - 1:
                    continue
                sister_state = state.copy()
                sister_state[indexer.epoch] = epoch_idx
                child = graph.find_or_create_vertex(sister_state)
                coeff = np.zeros(len(epochs) + 1)
                coeff[-1] = epoch_trans_rates[i]
                vertex.add_edge(child, coeff)
            graph.extend(callback, epochs=epochs, epoch_idx=epoch_idx, indexer=indexer)

        ipv = [0] * indexer.state_length
        ipv[indexer.props_to_index(ton=1)] = nr_samples

        manual_graph = Graph(coalescent_1param, ipv=ipv,
                             epochs=epochs, epoch_idx=0, indexer=indexer)
        manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
        for ei in range(1, len(epochs)):
            manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
            manual_add_epoch(manual_graph, coalescent_1param, epochs, ei, indexer)
        manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
        manual_moments = manual_graph.moments(5)

        # --- New approach ---
        @with_ipv([nr_samples] + [0] * (nr_samples - 1))
        def coalescent_simple(state):
            transitions = []
            for i in range(state.size):
                for j in range(i, state.size):
                    same = int(i == j)
                    if same and state[i] < 2:
                        continue
                    if not same and (state[i] < 1 or state[j] < 1):
                        continue
                    new = state.copy()
                    new[i] -= 1
                    new[j] -= 1
                    new[i + j + 1] += 1
                    transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
            return transitions

        new_graph = Graph(coalescent_simple)
        new_graph.update_weights([1 / pop_sizes[0]])
        g1 = new_graph.add_epoch(epochs[1])
        g2 = g1.add_epoch(epochs[2])
        g2.update_weights([1 / pop_sizes[0], 1 / pop_sizes[1], 1 / pop_sizes[2]])
        new_moments = g2.moments(5)

        # Moments should match within numerical tolerance
        assert new_moments == approx(manual_moments, rel=1e-6)
```

---

## Step 7: Update tutorial notebook

**File:** `docs/pages/tutorial/time-inhomogeneous.ipynb`

The "Moments of epoch-wise time homogeneous phase-type distributions" section (cells around line 4445-4994) gets dramatically simplified. The new version:

```python
nr_samples = 10
epochs = [0, 1, 2]
pop_sizes = [1, 5, 10]

@with_ipv([nr_samples] + [0] * (nr_samples - 1))
def coalescent(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i + j + 1] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return transitions

graph = Graph(coalescent)

# Add epoch boundaries
g1 = graph.add_epoch(epochs[1])
g2 = g1.add_epoch(epochs[2])

# theta: [coal_rate_epoch0, coal_rate_epoch1(?), epoch1_trans, epoch2_trans]
# Actually: one coal rate param + 2 epoch trans params
g2.update_weights([1/pop_sizes[0], 1/pop_sizes[1], 1/pop_sizes[2]])

print(g2.moments(5))
```

Note: The exact theta layout depends on how the callback maps to coefficients. The key simplification is that users no longer define `add_epoch` or pre-allocate coefficient slots.

---

## Implementation Notes (from execution)

### Coefficient layout for `add_epoch`

Each `add_epoch` call adds `base_param_length + 1` coefficient slots:
- `base_param_length` slots for the new epoch's dynamics (same structure as original callback)
- 1 slot for the epoch transition rate

**Example with `base_param_length = 1` (single coalescent rate):**
- Original: `[coal_rate]` (1 slot)
- After 1st `add_epoch`: `[epoch0_coal, epoch1_coal, epoch1_trans]` (3 slots)
- After 2nd `add_epoch`: `[epoch0_coal, epoch1_coal, epoch1_trans, epoch2_coal, epoch2_trans]` (5 slots)

**Usage pattern:**
```python
graph = Graph(coalescent)
graph.update_weights([1/N0])           # set epoch 0 rate
g1 = graph.add_epoch(t1)              
g1.update_weights([1/N0, 1/N1, 1])    # set all rates before next epoch
g2 = g1.add_epoch(t2)
g2.update_weights([1/N0, 1/N1, 1, 1/N2, 1])  # final theta
g2.moments(5)
```

### `discretize` on parameterized graphs

When `discretize` is called on a parameterized graph with a scalar rate, `_rebuild_with_wider_layout(extra_coeff_slots=1)` is used to add a coefficient slot for the discretization rate. The scalar rate is placed in the new last slot.

### `_discretize_inplace` kept for equivalence testing

The original in-place `discretize` is preserved as `_discretize_inplace` with its implementation intact (not just commented out) so equivalence tests can call it.

---

## Execution: Batched with Test Gates

### Batch 1: `_rebuild_with_wider_layout` + tests

**Implement:**
- Add `_rebuild_with_wider_layout` method to `Graph` (Step 1)
- Add `TestRebuildWithWiderLayout` tests to `tests/pytest/test_composable_operations.py`

**Tests to pass before proceeding:**
- `TestRebuildWithWiderLayout.test_wider_state` — state dimensions widened correctly
- `TestRebuildWithWiderLayout.test_wider_coefficients` — coeff slots widened correctly
- `TestRebuildWithWiderLayout.test_rebuild_preserves_structure` — vertex count, edge count, edge targets match
- `TestRebuildWithWiderLayout.test_rebuild_preserves_metadata` — `_callback`, `_weight_mode`, `is_discrete` copied
- `TestRebuildWithWiderLayout.test_rebuild_no_change` — zero-widening produces structurally equivalent graph (same moments)
- `pixi run test` — all existing tests still pass (no existing code changed)

**Gate:** All above pass → proceed to Batch 2

---

### Batch 2: `laplace_transform` metadata + `discretize` rewrite

**Implement:**
- Comment out old `laplace_transform` return, add metadata-copying version (Step 2)
- Rename old `discretize` to `_discretize_inplace`, comment out its body (Step 3)
- Add new `discretize` that returns new graph (Step 3)
- Comment out old call patterns at ALL `discretize` call sites in test files, add new patterns (Step 5)
- Add `TestEquivalenceOldVsNew` and `TestDiscretizeReturnsNewGraph` tests

**Tests to pass before proceeding:**
- `TestEquivalenceOldVsNew.test_discretize_equivalence` — new discretize produces same graph/rewards as old `_discretize_inplace`
- `TestEquivalenceOldVsNew.test_discretize_equivalence_callable` — same with callable rate
- `TestEquivalenceOldVsNew.test_laplace_equivalence` — same structure, metadata preserved
- `TestCompositionPipeline.test_laplace_preserves_metadata` — callback/weight_mode copied
- `TestDiscretizeReturnsNewGraph.test_returns_graph` — returns Graph with `.rewards`
- `TestDiscretizeReturnsNewGraph.test_original_unchanged` — original graph not mutated
- `test_exp_geom.py` — geometric distribution moments still correct (updated call site)
- `test_api_comprehensive.py` — updated discretize calls work
- `test_comprehensive_api.py` — updated discretize calls work  
- `test_input_validation.py` — validation errors still raised correctly
- `pixi run test` — all tests pass

**Gate:** All above pass → proceed to Batch 3

---

### Batch 3: `add_epoch` implementation

**Implement:**
- Add `add_epoch` method (Step 4)
- Add `TestAddEpoch` and epoch-related `TestCompositionPipeline` tests

**Tests to pass before proceeding:**
- `TestAddEpoch.test_single_epoch` — graph grows, param_length increases by 1
- `TestAddEpoch.test_original_unchanged` — original not mutated
- `TestAddEpoch.test_chained_epochs` — two epochs, param_length increases by 2
- `TestCompositionPipeline.test_epoch_then_discretize` — `add_epoch → discretize` works
- `TestEquivalenceOldVsNew.test_epoch_moments_match_tutorial` — new `add_epoch` produces same moments as tutorial's manual approach (`[8.73, 173.4, ...]`). This test replicates the tutorial's manual epoch wiring and compares moments.
- `pixi run test` — all tests pass

**Gate:** All above pass → proceed to Batch 4

---

### Batch 4: Tutorial notebook updates

**Implement:**
- Update `docs/pages/tutorial/discrete.ipynb` (cells 7, 8, 9, 11, 19) — comment out old patterns, add new
- Update `docs/pages/tutorial/parametrization.ipynb` (cells 19, 20, 21) — comment out old patterns, add new
- Rewrite `docs/pages/tutorial/time-inhomogeneous.ipynb` epoch sections — comment out manual `add_epoch` function and old wiring, add new `graph.add_epoch()` API

**Verification:**
- Run each notebook end-to-end (or at least the modified cells) to confirm no errors
- `pixi run test` — all tests still pass
- Moments from time-inhomogeneous notebook match expected values
