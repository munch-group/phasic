# JAX vmap/pmap Parallelization Implementation Progress

**Date**: 2025-11-13
**Status**: PARTIALLY COMPLETE (Steps 1-4 done, 5-10 remaining)

## Completed Tasks

### 1. ✅ Implemented `Graph.from_serialized()` classmethod
- **Location**: `src/phasic/__init__.py` lines 1815-2144
- **Features**:
  - Validates all required fields with informative error messages
  - Handles v0.22.0 format (no base_weight)
  - Handles empty arrays correctly (shape (0,) or (0, n_cols))
  - Handles param_length=0 case (creates (0, 0) arrays)
  - Reuses starting vertex when it appears in states array
  - Full error checking with detailed error messages

### 2. ✅ Added comprehensive unit tests
- **Location**: `tests/test_graph_serialization.py`
- **Coverage**:
  - 6 round-trip tests (simple, parameterized, mixed, JSON, empty, large state)
  - 7 error handling tests (missing fields, wrong shapes, invalid indices)
  - 2 callback-based tests (coalescent model serialization + trace recording)
  - **Results**: **15/15 tests pass** ✅

### 3. ✅ Understanding of serialization format
- **Non-parameterized edges**: `[from_idx, to_idx, weight]` shape (n, 3)
- **Start edges**: `[to_idx, weight]` shape (n, 2)
- **Parameterized edges** (v0.22.0+): `[from_idx, to_idx, coeff1, coeff2, ...]` shape (n, 2+param_length)
  - NO base_weight field (removed in v0.22.0)
  - Special case: param_length=0 creates (0, 0) arrays
- **Start param edges**: `[to_idx, coeff1, coeff2, ...]` shape (n, 1+param_length)

### 4. ✅ Fixed callback-based test API
- **Issue**: Tests were using incorrect callback API (missing `@callback` decorator)
- **Root Cause**: Callback functions need to use `@callback(ipv=...)` decorator OR provide `ipv=` argument to Graph constructor
- **Callback Format**:
  - `ipv`: 2-tuples `(state, probability)` - decorator adds empty list for non-parameterized
  - Parameterized transitions: 2-tuples `(state, coeffs_list)` - NOT 3-tuples
  - Example:
    ```python
    @callback(ipv=[([5], 1.0)])  # ipv is 2-tuple
    def coalescent_callback(state, **kwargs):
        rate = n * (n - 1) / 2
        return [([n - 1], [rate])]  # Transition is 2-tuple (state, coeffs)
    ```
- **Tests Fixed**: Both callback tests now pass (15/15 total)

## Remaining Tasks

### 5. Modify `collect_missing_traces_batch()` to serialize graphs
**File**: `src/phasic/hierarchical_trace_cache.py` lines 282-284

**Current**:
```python
work_units[scc_hash] = enhanced_subgraph  # Graph object
```

**New**:
```python
import json
serialized_dict = enhanced_subgraph.serialize(param_length=graph.param_length())
json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
             for k, v in serialized_dict.items()}
json_str = json.dumps(json_dict)
work_units[scc_hash] = json_str  # JSON string
logger.debug("Serialized SCC %s (%d bytes)", scc_hash[:16], len(json_str))
```

**Error handling**: Wrap in try/except, raise RuntimeError with details

### 6. Update `compute_trace_work_unit()` to deserialize JSON
**File**: `src/phasic/hierarchical_trace_cache.py` lines 317-352

**Current signature**: `(hash, Graph) -> (hash, trace)`
**New signature**: `(hash, json_string) -> (hash, trace)`

**Implementation**:
1. Check cache on worker
2. Deserialize JSON to dict
3. Convert lists → numpy arrays
4. Call `Graph.from_serialized(graph_dict)`
5. Record trace
6. Cache to local disk
7. Return (hash, trace)

**All steps must succeed or raise with detailed errors**

### 7. Add strategy validation with explicit logging
**File**: `src/phasic/hierarchical_trace_cache.py` before line 394

**Delete forced sequential override** (lines 387-389):
```python
# Force sequential processing for Graph objects
# TODO: Add serialization support for parallel processing
# strategy = 'sequential'
```

**Add validation**:
```python
# Validate parallelization strategy
if strategy not in ('auto', 'vmap', 'pmap', 'sequential'):
    raise ValueError(f"Invalid parallelization strategy: '{strategy}'...")

# Auto-detect with explicit logging
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
            logger.info("Parallelization: Auto-selected 'vmap' (single device)")

# Validate JAX available for vmap/pmap
if strategy == 'vmap' and not HAS_JAX:
    raise ImportError("Cannot use strategy='vmap': JAX not installed...")

if strategy == 'pmap':
    if not HAS_JAX:
        raise ImportError("Cannot use strategy='pmap': JAX not installed...")
    n_devices = jax.device_count()
    if n_devices < 2:
        raise RuntimeError(
            f"Cannot use strategy='pmap': only {n_devices} device available\n"
            f"  pmap requires 2+ devices...\n"
            f"  Use: strategy='vmap' or strategy='auto'"
        )
```

### 8. Remove sequential override
**File**: `src/phasic/hierarchical_trace_cache.py` lines 387-389

Delete the 3 lines that force sequential strategy.

### 9. Update sequential fallback to deserialize JSON
**File**: `src/phasic/hierarchical_trace_cache.py` lines 449-491

**Current**: Processes Graph objects directly
**New**: Deserialize JSON strings

```python
for graph_hash, graph_json in work_list:
    # Check cache
    cached = _load_trace_from_cache(graph_hash)
    if cached is not None:
        results[graph_hash] = cached
        continue

    # Deserialize JSON
    try:
        graph_dict = json.loads(graph_json)
        # Convert to arrays...
        graph = Graph.from_serialized(graph_dict)
    except Exception as e:
        raise RuntimeError(f"Sequential: Failed to deserialize {graph_hash[:16]}...") from e

    # Record trace (rest unchanged)
    ...
```

### 10. Update type hints and docstrings
- `collect_missing_traces_batch()`: Returns `Dict[str, str]` not `Dict[str, Graph]`
- `compute_missing_traces_parallel()`: Takes `work_units: Dict[str, str]`
- Update all docstrings to specify error conditions

### 11. Test that existing sequential mode still works
- Run existing hierarchical caching tests
- Verify backward compatibility
- Test vmap if JAX available
- Test error messages

## Key Design Decisions

1. **No silent fallbacks**: All operations must work as specified or fail with clear errors
2. **JSON serialization**: Required for network transmission via pmap
3. **Worker independence**: Each worker deserializes + records independently
4. **Disk caching**: Each worker caches to its local disk
5. **v0.22.0 format**: No base_weight in parameterized edges

## Testing Strategy

### Unit Tests (Done)
- ✅ Serialization round-trip
- ✅ JSON network transmission simulation
- ✅ Error handling (missing fields, wrong shapes, invalid indices)

### Integration Tests (TODO)
- Sequential mode unchanged (backward compatibility)
- vmap parallelization (single machine, multi-CPU)
- pmap parallelization (multi-device)
- Error propagation (worker failures)

### Performance Validation (TODO)
- Baseline (sequential): 50 SCCs → 60s
- vmap (8 cores): Target ~10s (6x speedup)
- pmap (8 machines): Target ~8s (7.5x speedup)

## Next Steps

1. Implement task #5: Modify `collect_missing_traces_batch()` to serialize
2. Implement task #6: Update `compute_trace_work_unit()` to deserialize
3. Implement task #7: Add strategy validation
4. Test sequential mode still works
5. Test vmap if available
6. Document final implementation

## Files Modified

- ✅ `src/phasic/__init__.py`: Added `Graph.from_serialized()` method (lines 1815-2144)
- ✅ `tests/test_graph_serialization.py`: Added 15 comprehensive tests (all passing)
- ⏳ `src/phasic/hierarchical_trace_cache.py`: Needs modifications for tasks 5-9

## Success Criteria

- ✅ JSON serialization/deserialization works correctly
- ✅ All error paths tested and produce informative messages
- ✅ No silent fallbacks anywhere
- ⏳ Sequential mode unchanged (needs testing)
- ⏳ vmap achieves near-linear speedup
- ⏳ pmap achieves near-linear speedup
- ✅ Trace evaluation (FFI) unchanged (no changes needed)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
