# Restore JAX vmap/pmap for Parallel Trace Recording Across Machines

**Date**: 2025-11-13
**Status**: PLANNED (NOT YET IMPLEMENTED)

## Critical Understanding

**Before hierarchical caching**: Trace recording wasn't vmap/pmap compatible (it worked with Graph objects)

**Original hierarchical design (commit a564d48)**: Intended to make trace recording parallelizable by:
1. Serializing graphs to JSON
2. Distributing work via vmap/pmap
3. Deserializing on each worker
4. Recording trace independently
5. Caching results

**Current broken state**: Uses Graph objects instead of JSON → vmap/pmap disabled → sequential only

**User's requirement**: "computing the missing SCCs should happen in parallel on a large number of machines"
- This means trace **recording** must be vmap/pmap compatible
- Multi-machine pmap requires serialized inputs (JSON)
- Each worker independently deserializes → records trace → caches

## Architecture: Distributed Trace Recording

```
Master Process:
├─ Decompose graph into SCCs
├─ Check disk cache for each SCC
├─ Serialize missing SCCs to JSON
├─ Create work_units: Dict[hash, json_string]
└─ Distribute via pmap (multi-machine) or vmap (single machine)

Worker Processes (on different machines):
├─ Receive JSON string
├─ Deserialize to Graph object
├─ Call record_elimination_trace()  ← Python/C++ (not FFI)
├─ Cache trace to local disk
└─ Return trace

Master Process:
├─ Collect all traces
├─ Stitch traces together
└─ Cache final result
```

## No Silent Fallbacks Policy

**CRITICAL**: All operations must work as specified or fail loudly with clear error messages.

### Serialization/Deserialization
- If `Graph.from_serialized()` fails → **RAISE RuntimeError** with details
- If JSON is malformed → **RAISE ValueError** with details
- If array shapes mismatch → **RAISE ValueError** with expected vs actual
- **NO silent fallbacks** to default values

### Parallelization Strategy
- If `strategy='vmap'` but JAX not available → **RAISE ImportError**
- If `strategy='pmap'` but only 1 device → **RAISE RuntimeError** (don't silently use vmap)
- If `strategy='auto'` → Explicitly log which strategy selected: `logger.info("Auto-selected vmap (8 CPUs)")`
- **NO silent downgrades** from pmap → vmap → sequential

### Cache Operations
- If cache directory cannot be created → **RAISE PermissionError**
- If cache file corrupted → **RAISE RuntimeError**, delete file, log error
- If cache hash collision (different graphs, same hash) → **RAISE RuntimeError**
- **NO silent cache misses** without logging

### Worker Errors
- If deserialization fails on worker → **RAISE** with full stack trace
- If trace recording fails → **RAISE** with graph details
- If worker returns incomplete data → **RAISE ValueError**
- **NO silent skipping** of failed workers

### Error Messages Must Include
1. **What failed**: Specific operation/function name
2. **Why it failed**: Root cause (missing file, wrong type, etc.)
3. **Context**: Graph size, SCC index, hash, file paths
4. **Resolution**: What user can do to fix it

**Example Good Error**:
```
RuntimeError: Graph deserialization failed for SCC hash abc123...
  Problem: 'states' array shape mismatch
  Expected: (50, 3) from JSON metadata (n_vertices=50, state_length=3)
  Actual: (48, 3) from JSON 'states' field
  Context: Processing SCC 12/50 in hierarchical caching
  Resolution: This indicates corrupted cache or version mismatch.
    Try: phasic.clear_caches() and rebuild graph
```

**Example Bad Error** (NO):
```
Warning: Failed to load from cache, computing...  ← Silent fallback!
```

## Implementation Plan

### 1. **Add `Graph.from_serialized()` classmethod**
**File**: `src/phasic/__init__.py` (after line ~1815, after `serialize()`)

**Purpose**: Reconstruct Graph from `serialize()` output for distributed workers

**Implementation**: Rebuild graph from numpy arrays:
- Create empty graph with `Graph(state_length)`
- Reconstruct all vertices from `states` array
- Rebuild regular edges from `edges` array
- Rebuild parameterized edges from `param_edges` array
- Rebuild starting vertex edges

**Error handling**:
```python
@classmethod
def from_serialized(cls, data: Dict[str, Any]) -> 'Graph':
    """
    Reconstruct Graph from serialize() output.

    Raises
    ------
    ValueError
        If data is missing required fields or has wrong shapes
    RuntimeError
        If graph reconstruction fails (edges to non-existent vertices, etc.)
    """
    # Validate required fields
    required = ['states', 'edges', 'start_edges', 'param_edges',
                'start_param_edges', 'param_length', 'state_length', 'n_vertices']
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(
            f"Graph deserialization failed: missing required fields {missing}\n"
            f"  Available fields: {list(data.keys())}\n"
            f"  This usually indicates corrupted cache or version mismatch.\n"
            f"  Resolution: Clear cache and rebuild"
        )

    # Validate array shapes
    n_vertices = int(data['n_vertices'])
    state_length = int(data['state_length'])
    states = np.asarray(data['states'], dtype=np.int32)

    if states.shape != (n_vertices, state_length):
        raise ValueError(
            f"Graph deserialization failed: states array shape mismatch\n"
            f"  Expected: ({n_vertices}, {state_length}) from metadata\n"
            f"  Actual: {states.shape} from 'states' field\n"
            f"  Resolution: This indicates corrupted data. Clear cache and rebuild."
        )

    # ... rest of implementation with similar error checking ...
```

### 2. **Modify `collect_missing_traces_batch()` to serialize graphs**
**File**: `src/phasic/hierarchical_trace_cache.py` lines 282-284

**Current**: Stores Graph objects → cannot be sent across network
**New**: Stores JSON strings → can be sent across network via pmap

**Changes**:
```python
# Serialize enhanced subgraph for distributed computation
try:
    import json
    serialized_dict = enhanced_subgraph.serialize(param_length=graph.param_length())
    json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in serialized_dict.items()}
    json_str = json.dumps(json_dict)
    work_units[scc_hash] = json_str
    logger.debug("→ Serialized SCC %s (%d bytes)", scc_hash[:16], len(json_str))
except Exception as e:
    raise RuntimeError(
        f"Failed to serialize SCC {scc_hash[:16]} for distributed computation\n"
        f"  SCC size: {enhanced_subgraph.vertices_length()} vertices\n"
        f"  Param length: {graph.param_length()}\n"
        f"  Error: {e}\n"
        f"  This is a critical error - cannot proceed with hierarchical caching"
    ) from e
```

**Return type**: `Dict[str, str]` (was `Dict[str, 'Graph']`)

### 3. **Update `compute_trace_work_unit()` to deserialize**
**File**: `src/phasic/hierarchical_trace_cache.py` lines 317-352

**Current**: Takes `(hash, Graph)` → vmapping not possible
**New**: Takes `(hash, json_string)` → vmapping works!

**Implementation with strict error handling**:
```python
def compute_trace_work_unit(hash_and_json: Tuple[str, str]) -> Tuple[str, 'EliminationTrace']:
    """
    JAX-compatible trace recording work unit for distributed computation.

    Raises
    ------
    ValueError
        If JSON is malformed or deserialization fails
    RuntimeError
        If trace recording fails
    """
    from .trace_elimination import record_elimination_trace
    from . import Graph
    import json
    import numpy as np

    graph_hash, graph_json = hash_and_json

    # Check cache on this worker
    cached = _load_trace_from_cache(graph_hash)
    if cached is not None:
        logger.debug("Worker: Cache hit for %s", graph_hash[:16])
        return (graph_hash, cached)

    logger.debug("Worker: Cache miss for %s, deserializing...", graph_hash[:16])

    # Deserialize graph - MUST succeed or raise
    try:
        graph_dict = json.loads(graph_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Worker: JSON deserialization failed for hash {graph_hash[:16]}\n"
            f"  JSON length: {len(graph_json)} bytes\n"
            f"  Error at position {e.pos}: {e.msg}\n"
            f"  This indicates corrupted work unit data"
        ) from e

    # Convert lists → numpy arrays with validation
    try:
        graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
        graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
        graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
        graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
        graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"Worker: Array conversion failed for hash {graph_hash[:16]}\n"
            f"  Available fields: {list(graph_dict.keys())}\n"
            f"  Error: {e}"
        ) from e

    # Reconstruct graph - MUST succeed or raise (from_serialized handles errors)
    graph = Graph.from_serialized(graph_dict)
    logger.debug("Worker: Deserialized graph with %d vertices", graph.vertices_length())

    # Record trace - MUST succeed or raise
    try:
        trace = record_elimination_trace(graph, param_length=graph_dict['param_length'])
        logger.debug("Worker: Recorded trace with %d operations", len(trace.operations))
    except Exception as e:
        raise RuntimeError(
            f"Worker: Trace recording failed for hash {graph_hash[:16]}\n"
            f"  Graph size: {graph.vertices_length()} vertices\n"
            f"  Param length: {graph_dict['param_length']}\n"
            f"  Error: {e}"
        ) from e

    # Cache to local disk on this worker
    success = _save_trace_to_cache(graph_hash, trace)
    if not success:
        logger.warning("Worker: Failed to cache trace %s (non-fatal)", graph_hash[:16])

    return (graph_hash, trace)
```

### 4. **Remove sequential override with explicit strategy validation**
**File**: `src/phasic/hierarchical_trace_cache.py` lines 355-499

**DELETE lines 387-389**:
```python
# Force sequential processing for Graph objects
# TODO: Add serialization support for parallel processing
# strategy = 'sequential'
```

**ADD explicit strategy validation (before line 394)**:
```python
# Validate parallelization strategy
if strategy not in ('auto', 'vmap', 'pmap', 'sequential'):
    raise ValueError(
        f"Invalid parallelization strategy: '{strategy}'\n"
        f"  Must be one of: 'auto', 'vmap', 'pmap', 'sequential'\n"
        f"  Got: {strategy!r}"
    )

# Auto-detect strategy with explicit logging
if strategy == 'auto':
    if not HAS_JAX:
        strategy = 'sequential'
        logger.info("Parallelization: Auto-selected 'sequential' (JAX not available)")
    else:
        n_devices = jax.device_count()
        if n_devices > 1:
            strategy = 'pmap'
            logger.info("Parallelization: Auto-selected 'pmap' (%d devices)", n_devices)
        else:
            strategy = 'vmap'
            logger.info("Parallelization: Auto-selected 'vmap' (single device, multi-CPU)")

# Validate strategy can be executed
if strategy == 'vmap' and not HAS_JAX:
    raise ImportError(
        "Cannot use strategy='vmap': JAX not installed\n"
        "  Install JAX: pip install jax jaxlib\n"
        "  Or use: strategy='sequential'"
    )

if strategy == 'pmap':
    if not HAS_JAX:
        raise ImportError(
            "Cannot use strategy='pmap': JAX not installed\n"
            "  Install JAX: pip install jax jaxlib\n"
            "  Or use: strategy='sequential'"
        )
    n_devices = jax.device_count()
    if n_devices < 2:
        raise RuntimeError(
            f"Cannot use strategy='pmap': only {n_devices} device available\n"
            f"  pmap requires 2+ devices (GPUs/TPUs or multi-machine setup)\n"
            f"  Current devices: {jax.devices()}\n"
            f"  Use: strategy='vmap' for single-device multi-CPU\n"
            f"  Or: strategy='auto' to automatically select"
        )
    logger.info("Parallelization: Using 'pmap' with %d devices", n_devices)

# Log final strategy choice
logger.info("Computing %d missing traces using strategy='%s'", len(work_units), strategy)
```

### 5. **Update sequential fallback to deserialize with error handling**
**File**: `src/phasic/hierarchical_trace_cache.py` lines 449-491

```python
for graph_hash, graph_json in work_list:
    logger.debug("Sequential: Processing %s", graph_hash[:16])

    # Check cache
    cached = _load_trace_from_cache(graph_hash)
    if cached is not None:
        results[graph_hash] = cached
        logger.debug("Sequential: Cache hit for %s", graph_hash[:16])
        continue

    # Deserialize - same strict error handling as compute_trace_work_unit
    try:
        graph_dict = json.loads(graph_json)
        graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
        graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
        graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
        graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
        graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)
        graph = Graph.from_serialized(graph_dict)
    except Exception as e:
        raise RuntimeError(
            f"Sequential: Failed to deserialize hash {graph_hash[:16]}\n"
            f"  This is a critical error in hierarchical caching\n"
            f"  Original error: {e}"
        ) from e

    # Rest of logic (subdivision check, trace recording) with error propagation...
```

### 6. **Update type hints throughout**
- `collect_missing_traces_batch()`: Returns `Tuple[Dict[str, str], List[str], Optional[SCCGraph]]`
- `compute_missing_traces_parallel()`: Takes `work_units: Dict[str, str]`
- Update all docstrings to specify error conditions

## Error Propagation Matrix

| Error Type | Where It Occurs | What Happens |
|------------|----------------|--------------|
| **JSON malformed** | `compute_trace_work_unit()` | `ValueError` with details, worker stops |
| **Array shape mismatch** | `Graph.from_serialized()` | `ValueError` with expected vs actual |
| **Missing vertices** | `from_serialized()` edge creation | `RuntimeError` with vertex details |
| **Trace recording fails** | `record_elimination_trace()` | `RuntimeError` with graph details |
| **Cache write fails** | `_save_trace_to_cache()` | Log warning (non-fatal) |
| **Cache read fails** | `_load_trace_from_cache()` | Return None, log debug |
| **JAX not available** | `strategy='vmap'/'pmap'` | `ImportError` immediately |
| **Insufficient devices** | `strategy='pmap'` with 1 device | `RuntimeError` immediately |
| **Worker error** | Any worker in vmap/pmap | Propagate to master, fail entire batch |

## Why This Enables Multi-Machine pmap

| Feature | How It Works |
|---------|--------------|
| **Serialization** | JSON strings can be sent over network (Graph objects cannot) |
| **XLA compilation** | JAX compiles the distribution logic, not trace recording itself |
| **Worker independence** | Each worker deserializes + records independently |
| **No Python GIL** | C++ trace recording releases GIL, true parallelism |
| **Disk caching** | Each worker caches to its local disk |
| **Result collection** | pmap gathers all traces back to master |
| **Error handling** | Any worker error fails entire batch (no silent failures) |

## JAX Transformation Compatibility

| Transform | Trace Recording | Trace Evaluation |
|-----------|----------------|------------------|
| `jax.jit` | ❌ Not needed (one-time operation) | ✅ Yes (via FFI) |
| `jax.grad` | ❌ Discrete operation | ✅ Yes (via custom VJP) |
| `jax.vmap` | ✅ **Yes** (with JSON serialization) | ✅ Yes |
| `jax.pmap` | ✅ **Yes** (with JSON serialization) | ✅ Yes |
| Multi-machine | ✅ **Yes** (JSON over network) | ✅ Yes (FFI) |

## Example: Multi-Machine Parallel Trace Recording

```python
import jax
from phasic import Graph
from phasic.hierarchical_trace_cache import get_trace_hierarchical

# Build large parameterized graph
graph = Graph(callback=model, nr_samples=100)  # 10,000+ vertices

# Master process:
# 1. Decomposes into 50 SCCs
# 2. Serializes each to JSON (MUST succeed or raise)
# 3. Distributes via pmap to 8 machines (MUST have 8 devices or raise)

trace = get_trace_hierarchical(
    graph,
    parallel_strategy='pmap'  # Explicit - if unavailable, raises immediately
)

# Under the hood:
# - work_units = {hash1: json1, hash2: json2, ...}  ← 50 JSON strings
# - Validates pmap available (8 devices found)
# - Logs: "Parallelization: Using 'pmap' with 8 devices"
# - jax.pmap distributes to 8 devices (machines)
# - Each machine processes 6-7 SCCs:
#   * Deserializes JSON → Graph (MUST succeed or raise)
#   * Records trace (MUST succeed or raise)
#   * Caches to local disk
#   * Returns trace
# - If ANY worker fails, entire batch fails with full error
# - Master collects 50 traces
# - Master stitches traces together
# - Result: complete trace for 10K vertex graph
```

## Key Differences from FFI

| Aspect | Trace Recording (This PR) | Trace Evaluation (FFI) |
|--------|--------------------------|------------------------|
| **What** | Record elimination operations | Evaluate operations with θ |
| **Input** | Graph structure (JSON) | Trace + parameters |
| **Output** | EliminationTrace | PMF/PDF/moments |
| **Language** | Python → C++ (pybind11) | C++ (FFI, zero-copy) |
| **Parallelization** | vmap/pmap with JSON | vmap/pmap with XLA buffers |
| **Frequency** | Once per structure | Many times (SVGD) |
| **Duration** | Slow (seconds) | Fast (milliseconds) |
| **Error handling** | Explicit raises, no fallbacks | Explicit raises, no fallbacks |

**Both are needed**: Recording parallelizes structure processing, FFI parallelizes parameter evaluation

## Testing Strategy

### Unit Tests
1. `test_from_serialized_round_trip()` - Verify `Graph.from_serialized(g.serialize()) == g`
2. `test_serialize_parameterized_edges()` - Edge states preserved
3. `test_serialize_empty_graph()` - Handle edge cases
4. `test_deserialization_error_messages()` - Verify error messages are informative

### Integration Tests
5. `test_sequential_unchanged()` - Existing behavior works
6. `test_vmap_single_machine()` - 8 cores, 50 SCCs → 5-8x speedup
7. `test_pmap_multi_gpu()` - 4 GPUs, 50 SCCs → 3-4x speedup
8. `test_pmap_multi_machine()` - 8 machines, 50 SCCs → 6-8x speedup

### Error Handling Tests
9. `test_strategy_validation_fails()` - Invalid strategy raises ValueError
10. `test_pmap_insufficient_devices()` - 1 device with strategy='pmap' raises RuntimeError
11. `test_malformed_json_raises()` - Bad JSON raises ValueError with details
12. `test_array_shape_mismatch_raises()` - Wrong shapes raise ValueError with details
13. `test_worker_error_propagates()` - Worker failure fails entire batch

### Performance Validation
- **Baseline** (sequential): 50 SCCs, 2000 vertices total → 60 seconds
- **vmap** (8 cores): Expected ~10 seconds (6x speedup)
- **pmap** (8 machines): Expected ~8 seconds (7.5x speedup)

## Success Criteria

✅ JSON serialization/deserialization works correctly
✅ All error paths tested and produce informative messages
✅ No silent fallbacks anywhere
✅ Sequential mode unchanged (backwards compatible)
✅ vmap achieves near-linear speedup (single machine)
✅ pmap achieves near-linear speedup (multi-machine)
✅ Trace evaluation (FFI) unchanged (no regression)
✅ All existing tests pass
✅ New error handling tests pass

## Implementation Checklist

- [ ] 1. Implement `Graph.from_serialized()` with strict error checking
- [ ] 2. Add comprehensive unit tests for serialization round-trip
- [ ] 3. Add error handling tests (malformed JSON, shape mismatches)
- [ ] 4. Modify `collect_missing_traces_batch()` to serialize with error handling
- [ ] 5. Update `compute_trace_work_unit()` to deserialize with error handling
- [ ] 6. Add strategy validation with explicit logging
- [ ] 7. Remove sequential override (delete 3 lines)
- [ ] 8. Update sequential fallback to deserialize
- [ ] 9. Update all type hints and docstrings
- [ ] 10. Test sequential mode (ensure unchanged)
- [ ] 11. Test vmap parallelization
- [ ] 12. Test pmap parallelization
- [ ] 13. Test all error conditions
- [ ] 14. Update documentation
- [ ] 15. Commit with detailed message
