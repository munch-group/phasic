# Plan: Composable Modeling Operations

## Problem

Phasic's modeling tools compose in a pipeline:

```
callback → Graph → [add_epoch] → [discretize] → [joint_prob_graph] → compute
```

Currently there is an inconsistency in how these composition steps handle state vectors and edge coefficient arrays:

- **`joint_prob_graph()`** creates a new graph, freely widening state vectors (adding reward dimensions) and coefficient arrays (adding a mutation rate slot). The user's callback doesn't need to know about these extra dimensions.
- **`discretize()`** mutates the graph in-place. For parameterized graphs with a callable rate, the callable must return a full-width coefficient vector — the mutation rate slot must already exist.
- **`add_epoch`** is not a method at all. The user must manually: pre-allocate coefficient slots in the callback for all epochs + transitions, compute `stop_probability/accumulated_occupancy` ratios, create sister vertices, handle NaN, and call `extend()`. This is ~30 lines of graph surgery per epoch (see `docs/pages/tutorial/time-inhomogeneous.ipynb`).

The root cause is that `param_length` is locked in C after the first edge is added, and vertex state lengths are fixed at graph creation. In-place operations cannot widen these. Only operations that create a new graph can freely redefine the layout.

## Design Principle

**Every composition operation returns a new graph**, free to widen state vectors and coefficient arrays as needed. The user's callback describes only the base dynamics — it never needs to pre-allocate slots for downstream composition steps.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `add_epoch` API | New Graph method returning new graph | Follows `joint_prob_graph` pattern; enables automatic slot management |
| `discretize` API | Returns `(new_graph, rewards)` instead of in-place mutation | Consistency; enables automatic coefficient expansion |
| `joint_prob_graph` | No change needed | Already returns new graph |
| Epoch transition rates | Always `stop_probability(t) / accumulated_occupancy(t)` | Covers the common case; simplest API |
| Epoch callback | Reuse stored base callback; accept optional override | Default covers common case; override enables different dynamics per epoch |
| Coefficient expansion | Rebuild graph (no C changes) | Pragmatic first step; C-level optimization can come later |

## Current State: How the Tutorial Does Epochs

From `docs/pages/tutorial/time-inhomogeneous.ipynb`, the user must:

### 1. Pre-allocate coefficient slots in the callback

```python
def coalescent_1param(state, epochs=None, epoch_idx=None, indexer=None):
    # ...
    coeff = np.zeros(len(epochs) + 1)  # +1 for epoch transition slot
    coeff[epoch_idx] = rate
    transitions.append([new, coeff])
```

The callback must know the total number of epochs upfront. Adding a 4th epoch means rewriting the callback.

### 2. Pre-allocate epoch slot in the state vector

```python
indexer = StateIndexer(
    lineages=[Property('ton', min_value=1, max_value=nr_samples)],
    slots=['epoch']  # Must exist before graph construction
)
```

### 3. Manually wire up epoch transitions (~30 lines)

```python
def add_epoch(graph, callback, epochs, epoch_idx, indexer):
    stop_probs = np.array(graph.stop_probability(epoch))
    accum_v_time = np.array(graph.accumulated_occupancy(epoch))

    with np.errstate(invalid='ignore'):
        epoch_trans_rates = stop_probs / accum_v_time

    for i in range(1, graph.vertices_length() - 1):
        if np.isnan(epoch_trans_rates[i]):
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
```

## Target State: New API

### `add_epoch`

```python
# Base callback — knows nothing about epochs
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

# Build base graph — 1 coefficient slot (coalescent rate), no epoch in state
graph = Graph(coalescent)

# Add epochs — each call returns a new graph with an extra coefficient slot
g1 = graph.add_epoch(t_start)        # 2 coeff slots: [coal, epoch1_trans]
g2 = g1.add_epoch(t_end)             # 3 coeff slots: [coal, epoch1_trans, epoch2_trans]

# theta has one entry per coefficient slot
g2.update_weights([1/N, 1/N_bottle, 1/N])
g2.moments(5)
```

### `discretize`

```python
# Current (in-place, user manages slots):
rewards = graph.discretize(mutation_rate_callable)  # callable returns full coeff vector

# New (returns new graph, auto-appends mutation slot):
disc_graph, rewards = graph.discretize(rate=0.1)
# or with callable that returns a scalar per vertex:
disc_graph, rewards = graph.discretize(rate=lambda state: 0.1 * sum(state))
```

### Full composition pipeline

```python
graph = Graph(coalescent)
g1 = graph.add_epoch(t1)
g2 = g1.add_epoch(t2)
g3, rewards = g2.discretize(rate=lambda state: mutation_rate)
joint = g3.joint_prob_graph(indexer, tot_reward_limit=3)
joint.update_weights([coal_rate, epoch1_trans, epoch2_trans, mutation_rate])
joint.moments(5, rewards=rewards)
```

Each step is independent — the callback, discretize callable, and joint_prob_graph configuration don't need to know about each other's coefficient slots.

## Implementation

### Step 1: Graph rebuild helper

**File**: `src/phasic/__init__.py` (private method)

```python
def _rebuild_with_wider_layout(self, extra_state_dims=0, extra_coeff_slots=0,
                                state_fill=0, coeff_fill=0.0) -> Graph:
```

This is the core utility that enables all composition operations. It:

1. Creates a new `Graph(self.state_length() + extra_state_dims)`
2. Iterates source vertices, creating corresponding vertices in the new graph with widened state vectors (appending `state_fill` values)
3. Iterates source edges, creating corresponding edges with widened coefficient arrays (appending `coeff_fill` values)
4. Preserves vertex ordering, starting vertex, edge weights
5. Copies metadata: `is_discrete`, `was_dph`, callback reference, kwargs, `weight_mode`, etc.
6. Sets `param_length` on the new graph to `self.param_length() + extra_coeff_slots`

This extracts the pattern already present in `joint_prob_graph` (which manually copies vertices and edges into a new graph) into a reusable helper.

**Note**: `joint_prob_graph` does more than just widening — it also builds a fundamentally different topology (reward-tracking vertices, trash states, etc.). The helper only covers the "copy with wider layout" part, not the topology transformation.

### Step 2: `Graph.add_epoch(time, callback=None, **kwargs)`

**File**: `src/phasic/__init__.py`

```python
@_invalidates_trace
def add_epoch(self, time: float, callback: Callable | None = None, **kwargs) -> Graph:
```

Algorithm:

1. **Rebuild** the graph with `_rebuild_with_wider_layout(extra_state_dims=1, extra_coeff_slots=1)`:
   - State vectors get an epoch index appended (0 for all existing vertices)
   - Coefficient arrays get one new slot (for this epoch's transition rate)

   **Exception for subsequent epochs**: If this graph already has an epoch dimension (from a previous `add_epoch` call), don't add another state dimension — just increment the epoch counter on sister vertices. Track this via a `_epoch_state_index` attribute set by the first `add_epoch` call.

2. **Compute transition rates** on the **original** (pre-rebuild) graph:
   ```python
   stop_probs = np.array(self.stop_probability(time))
   acc_occ = np.array(self.accumulated_occupancy(time))
   with np.errstate(invalid='ignore'):
       transition_rates = stop_probs / acc_occ
   ```

3. **Wire up epoch transitions** on the new graph:
   For each transient vertex `v` in the new graph (skipping starting vertex, absorbing vertices, and vertices with NaN transition rate):
   - Only process vertices in the current (latest) epoch
   - Create sister vertex: same state but epoch index incremented by 1
   - Add edge from `v` to sister with the transition rate in the new coefficient slot

4. **Extend** the new graph using the callback to build out new-epoch vertices:
   - Default: use `self._callback` (stored from construction)
   - Override: use provided `callback`
   - The callback returns coefficient vectors of the **base** length. Wrap it to zero-pad to the new `param_length`:
     ```python
     def padded_callback(state, **kw):
         transitions = original_callback(state, **kw)
         return [(s, list(c) + [0.0] * extra_slots) for s, c in transitions]
     ```
   - Pass the wrapped callback to `extend()`

5. **Set metadata**:
   - `new_graph._epoch_state_index = <index of epoch dim in state vector>`
   - `new_graph._n_epochs = self._n_epochs + 1` (or 1 if first call)
   - Store callback reference for subsequent `add_epoch` calls

6. **Return** the new graph.

### Step 3: Update `discretize()` to return new graph

**File**: `src/phasic/__init__.py`

Change from:
```python
def discretize(self, rate, skip_existing=False, **kwargs) -> NDArray:
    # Mutates self, returns rewards
```

To:
```python
def discretize(self, rate=None, skip_existing=False, **kwargs) -> tuple[Graph, NDArray]:
    # Returns (new_graph, rewards)
```

Algorithm:

1. If graph is parameterized and `rate` is callable:
   - Rebuild with `_rebuild_with_wider_layout(extra_coeff_slots=1)` (mutation rate slot)
   - The callable returns a **scalar** rate per vertex
   - When calling `add_aux_vertex`, construct coefficient vector: `[0, ..., 0, scalar_rate]`
2. If `rate` is a scalar:
   - Clone the graph (no coefficient expansion needed — rate is constant)
   - Call `add_aux_vertex(rate)` as before
3. Normalize, set `is_discrete=True`, `was_dph=True`
4. Compute rewards array
5. Return `(new_graph, rewards)`

**Backward compatibility**: This is a breaking change. The old signature returns just `rewards` and mutates in-place. Since this is pre-1.0 software, a clean break is acceptable. Update all internal call sites and tests. If gentler migration is desired, we could temporarily support both patterns via return type detection, but this adds complexity.

### Step 4: Tests

**File**: `tests/pytest/test_composable_epochs.py` (new)

```python
def test_single_epoch_moments():
    """add_epoch produces correct moments for single epoch change."""

def test_multiple_epochs_moments():
    """Chained add_epoch calls match manual tutorial results."""

def test_epoch_plus_discretize():
    """graph.add_epoch(t) then discretize(rate) pipeline."""

def test_epoch_plus_joint_prob_graph():
    """Full pipeline: add_epoch → discretize → joint_prob_graph."""

def test_custom_epoch_callback():
    """Override callback for different dynamics in new epoch."""

def test_original_unchanged():
    """add_epoch does not modify the original graph."""

def test_coefficient_layout():
    """Verify coefficient arrays are correctly widened at each step."""

def test_pool_nielsen_validation():
    """Compare against analytical formula for pairwise coalescence."""
```

**File**: `tests/pytest/test_discretize_composable.py` (new)

```python
def test_discretize_returns_tuple():
    """discretize returns (graph, rewards), original unchanged."""

def test_discretize_scalar_rate():
    """Scalar rate: no coefficient expansion."""

def test_discretize_callable_parameterized():
    """Callable rate on parameterized graph: auto-appends coeff slot."""

def test_discretize_after_epoch():
    """discretize on epoch-extended graph."""
```

### Step 5: Update existing tests and call sites

**Files**: All files that call `discretize()` in-place

Search for `= graph.discretize(` and `.discretize(` patterns. Update to unpack tuple:
```python
# Old:
rewards = graph.discretize(rate)
# New:
graph, rewards = graph.discretize(rate)
```

### Step 6: Update tutorial notebook

**File**: `docs/pages/tutorial/time-inhomogeneous.ipynb`

Replace the manual `add_epoch` function and all the pre-allocation boilerplate with the new API. The "Moments of epoch-wise time homogeneous phase-type distributions" section should simplify dramatically.

## Files to modify

| File | Change |
|------|--------|
| `src/phasic/__init__.py` | Add `_rebuild_with_wider_layout()`, `add_epoch()`, update `discretize()` |
| `tests/pytest/test_composable_epochs.py` | New: epoch composition tests |
| `tests/pytest/test_discretize_composable.py` | New: non-mutating discretize tests |
| `tests/pytest/*.py` | Update existing `discretize()` call sites |
| `docs/pages/tutorial/time-inhomogeneous.ipynb` | Rewrite to use new API |

No C/C++ changes required. The rebuild approach constructs a new graph from scratch, which works within the existing C API (graph creation + edge addition). C-level in-place coefficient expansion can be added later as a performance optimization.

## Verification

1. **Replicate tutorial results**: The new `add_epoch()` method should produce the same `moments(5)` output as the manual approach in the tutorial (approximately `[8.73, 173.4, 5234, 209924, 10506329]`)
2. **Pool & Nielsen**: Compare pairwise coalescence times against the analytical formula
3. **Original unchanged**: After `g1 = graph.add_epoch(t)`, verify `graph` still has original `param_length`, state vectors, and edge coefficients
4. **Full pipeline**: `Graph → add_epoch → add_epoch → discretize → joint_prob_graph → moments()` produces correct results
5. **Existing tests pass**: `pixi run test` after updating `discretize` call sites

## Open Questions

1. **State vector expansion for first vs. subsequent epochs**: The first `add_epoch` adds an epoch dimension to the state vector. Subsequent calls reuse it (incrementing the value). Need to define how `_rebuild_with_wider_layout` knows whether to add a state dimension or not. Proposal: `add_epoch` checks for `_epoch_state_index` attribute — if present, no state expansion; if absent, expand by 1.

2. **Callback epoch awareness**: The padded callback receives states with the epoch index appended. If the callback inspects the state (e.g., checks `state.sum()` for absorption), the extra dimension could affect its logic. The wrapper may need to strip the epoch dimension before passing to the original callback and re-append it to returned states. This needs careful handling.

3. **`extend()` behavior with padded callbacks**: Currently `extend()` calls the callback and adds edges with whatever coefficients the callback returns. With the padded wrapper, this should work — but need to verify that `extend()` doesn't enforce `param_length` matching internally (the C layer might reject edges with mismatched coefficient lengths if we don't pad correctly).

4. **Performance**: Rebuilding the entire graph for each `add_epoch` call may be expensive for large models. For a 500K-vertex graph, this involves 500K vertex creations + all edge copies. Acceptable for now; C-level in-place expansion is the optimization path if needed.
