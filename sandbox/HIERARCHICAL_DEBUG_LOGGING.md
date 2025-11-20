# Comprehensive DEBUG Logging for Hierarchical Trace Caching

**Date:** 2025-11-09
**Status:** ✅ Complete

## Summary

Added comprehensive DEBUG logging to every step of the hierarchical trace caching system, including:
- Every cache query (hit/miss)
- Every SCC subdivision
- Every step of trace merging from cache components

## Changes Made

### 1. Cache Query Logging (`_load_trace_from_cache`, `_save_trace_to_cache`)

**Every cache operation is now logged:**

```python
# Cache load
logger.debug("Cache query: hash=%s...", graph_hash[:16])
# On hit:
logger.debug("  ✓ Cache HIT: hash=%s..., %d vertices, %d operations", ...)
# On miss:
logger.debug("  ✗ Cache MISS: hash=%s...", ...)

# Cache save
logger.debug("Saving trace to cache: hash=%s..., %d vertices, %d operations", ...)
logger.debug("  ✓ Cache save successful: hash=%s...", ...)
```

**Output example:**
```
[DEBUG] Cache query: hash=1b6082c8dc75eb58...
[DEBUG]   ✗ Cache MISS: hash=1b6082c8dc75eb58...
[DEBUG] Saving trace to cache: hash=1b6082c8dc75eb58..., 110 vertices, 25027 operations
[DEBUG]   ✓ Cache save successful: hash=1b6082c8dc75eb58...
```

### 2. SCC Decomposition Logging (`get_scc_graphs`)

**Logs each step of SCC extraction:**

```python
logger.debug("Starting SCC decomposition for graph with %d vertices", ...)
logger.debug("Computing SCC decomposition...")
logger.debug("SCC decomposition computed")
logger.debug("Processing %d SCCs in topological order...", ...)

# For each SCC:
logger.debug("  Extracting SCC %d/%d...", i + 1, total)
logger.debug("    → SCC %d: %d vertices, hash=%s...", ...)
```

**Output example:**
```
[DEBUG] Starting SCC decomposition for graph with 340 vertices
[DEBUG] Computing SCC decomposition...
[DEBUG] SCC decomposition computed
[DEBUG] Processing 125 SCCs in topological order...
[DEBUG]   Extracting SCC 1/125...
[DEBUG]     → SCC 1: 2 vertices, hash=754bb27832f37e5f...
[DEBUG]   Extracting SCC 2/125...
[DEBUG]     → SCC 2: 2 vertices, hash=350c5ca3bac9ae3f...
[INFO] SCC decomposition: found 125 components with sizes [2, 2, 4, ...]
```

### 3. Recursive Trace Collection Logging (`collect_missing_traces_batch`)

**Detailed logging for recursive subdivision:**

```python
logger.debug("Starting recursive trace collection: graph has %d vertices, min_size=%d", ...)

# In recursive function (with indentation based on depth):
logger.debug("%sChecking graph: %d vertices, hash=%s...", indent, ...)
logger.debug("%s✓ Cache hit for %d vertices", indent, ...)
# OR
logger.debug("%s✗ Cache miss for %d vertices", indent, ...)

# When subdividing:
logger.debug("%s→ Subdividing into SCCs (%d vertices >= min_size=%d)...", ...)
logger.debug("%s  Computing SCC decomposition...", indent)
logger.debug("%s  ✓ Found %d SCCs", indent, ...)
logger.debug("%s  SCC sizes: %s", indent, scc_sizes)

# Processing each SCC:
logger.debug("%s  Processing SCC %d/%d: %d vertices, hash=%s...", ...)
logger.debug("%s  ✓ Completed SCC %d/%d", indent, ...)
```

**Output example:**
```
[DEBUG] Starting recursive trace collection: graph has 340 vertices, min_size=10
[DEBUG] Checking graph: 340 vertices, hash=a1b2c3d4e5f6...
[DEBUG] ✗ Cache miss for 340 vertices
[DEBUG] → Subdividing into SCCs (340 vertices >= min_size=10)...
[DEBUG]   Computing SCC decomposition...
[DEBUG]   ✓ Found 125 SCCs
[DEBUG]   SCC sizes: [2, 2, 4, 4, 8, 4, 9, ...]
[DEBUG]   Processing SCC 1/125: 2 vertices, hash=754bb27832f37e5f...
[DEBUG]     Checking graph: 2 vertices, hash=754bb27832f37e5f...
[DEBUG]     ✓ Cache hit for 2 vertices
[DEBUG]   ✓ Completed SCC 1/125
[DEBUG]   Processing SCC 2/125: 2 vertices, hash=350c5ca3bac9ae3f...
[DEBUG]     Checking graph: 2 vertices, hash=350c5ca3bac9ae3f...
[DEBUG]     ✗ Cache miss for 2 vertices
[DEBUG]     → Added as work unit (2 vertices, below min_size)
[DEBUG]   ✓ Completed SCC 2/125
[INFO] Trace collection complete: 47 work units needed
[INFO] Cache statistics: 78 hits, 47 misses
[INFO] Cached vertices: 234/340 (68.8% of graph)
```

### 4. Trace Stitching Logging (`stitch_scc_traces`)

**Comprehensive logging for each merge step:**

```python
logger.debug("Starting trace stitching: %d SCCs, %d vertices total", ...)
logger.debug("SCC traces available: %d", ...)

# Step 1: Vertex mapping
logger.debug("Step 1: Building vertex mapping...")
logger.debug("  ✓ Vertex mapping: %d vertices", ...)

# Step 2: Initialize merged trace
logger.debug("Step 2: Initializing merged trace structure...")
logger.debug("  ✓ Created trace structure: %d vertices, param_length=%d", ...)
logger.debug("  Copying vertex states from original graph...")
logger.debug("  ✓ Copied %d vertex states", ...)

# Step 3: Process each SCC
logger.debug("Step 3: Processing %d SCCs in topological order...", ...)

# For each SCC:
logger.debug("SCC %d/%d: hash=%s..., %d internal vertices, %d connecting vertices", ...)
logger.debug("  %d operations to merge", ...)
logger.debug("  Operation offset: %d (merged trace has %d operations before this SCC)", ...)
logger.debug("  Copied %d operations (merged trace now has %d operations)", ...)
logger.debug("  Mapped %d trace vertices to original graph", ...)
logger.debug("  Internal vertices: %s", sorted_list_preview)
logger.debug("  Merging vertex data for internal vertices...")

# For connecting vertices:
logger.debug("    Skipping connecting vertex: orig_idx=%d (trace_idx=%d)", ...)

# For each vertex with edges:
logger.debug("    Vertex %d (orig=%d): setting rate and %d edges", ...)
logger.debug("      Edge %d: vertex %d → %d (op_idx=%d)", ...)

logger.debug("  ✓ Set vertex data for %d internal vertices, %d edges", ...)
logger.debug("  ✓ SCC %d/%d merge complete (total ops now: %d)", ...)

# Final validation
logger.debug("Setting starting vertex...")
logger.debug("  ✓ Starting vertex: %d (original=%d)", ...)
logger.debug("Validating merged trace...")
logger.debug("  Validation:")
logger.debug("    Vertices with rates set: %d / %d", ...)
logger.debug("    Total edges: %d", ...)
logger.debug("    Total operations: %d", ...)
logger.debug("    Operation remapping: %d entries, max SCC ops: %d", ...)

logger.info("✓ Trace stitching complete: %d vertices, %d operations", ...)
```

**Output example:**
```
[DEBUG] Starting trace stitching: 125 SCCs, 340 vertices total
[DEBUG] SCC traces available: 125
[DEBUG] Step 1: Building vertex mapping...
[DEBUG]   ✓ Vertex mapping: 340 vertices
[DEBUG] Step 2: Initializing merged trace structure...
[DEBUG]   ✓ Created trace structure: 340 vertices, param_length=1
[DEBUG]   Copying vertex states from original graph...
[DEBUG]   ✓ Copied 340 vertex states
[DEBUG] Step 3: Processing 125 SCCs in topological order...
[DEBUG] SCC 1/125: hash=754bb27832f37e5f..., 2 internal vertices, 0 connecting vertices
[DEBUG]   38 operations to merge
[DEBUG]   Operation offset: 0 (merged trace has 0 operations before this SCC)
[DEBUG]   Copied 38 operations (merged trace now has 38 operations)
[DEBUG]   Mapped 2 trace vertices to original graph
[DEBUG]   Internal vertices: [0, 1]
[DEBUG]   Merging vertex data for internal vertices...
[DEBUG]     Vertex 0 (orig=0): setting rate and 2 edges
[DEBUG]       Edge 0: vertex 0 → 1 (op_idx=3)
[DEBUG]       Edge 1: vertex 0 → 2 (op_idx=8)
[DEBUG]     Vertex 1 (orig=1): setting rate and 1 edges
[DEBUG]       Edge 0: vertex 1 → 2 (op_idx=16)
[DEBUG]   ✓ Set vertex data for 2 internal vertices, 3 edges
[DEBUG]   ✓ SCC 1/125 merge complete (total ops now: 38)
[DEBUG] SCC 2/125: hash=350c5ca3bac9ae3f..., 2 internal vertices, 1 connecting vertices
[DEBUG]   42 operations to merge
[DEBUG]   Operation offset: 38 (merged trace has 38 operations before this SCC)
[DEBUG]     Skipping connecting vertex: orig_idx=2 (trace_idx=0)
[DEBUG]     Vertex 2 (orig=2): setting rate and 2 edges
[DEBUG]       Edge 0: vertex 2 → 3 (op_idx=41)
[DEBUG]       Edge 1: vertex 2 → 4 (op_idx=46)
...
[DEBUG] Setting starting vertex...
[DEBUG]   ✓ Starting vertex: 0 (original=0)
[DEBUG] Validating merged trace...
[DEBUG]   Validation:
[DEBUG]     Vertices with rates set: 340 / 340
[DEBUG]     Total edges: 680
[DEBUG]     Total operations: 4827
[DEBUG]     Operation remapping: 1250 entries, max SCC ops: 42
[INFO] ✓ Trace stitching complete: 340 vertices, 4827 operations
[DEBUG] Final trace statistics:
[DEBUG]   Vertices: 340
[DEBUG]   Operations: 4827
[DEBUG]   Parameters: 1
[DEBUG]   State length: 2
[DEBUG]   Starting vertex: 0
[DEBUG]   Vertices with data: 340
[DEBUG]   Total edges: 680
```

## Usage

### Enable DEBUG Logging

```python
from phasic.logging_config import set_log_level
set_log_level('DEBUG')

import phasic
# Now all operations will be logged in detail
```

### View Logs in Notebook

```python
# In notebook
from phasic.logging_config import set_log_level
set_log_level('DEBUG')

import phasic
graph = phasic.Graph(...)
trace = graph.compute_trace(hierarchical=True)
# Logs will appear in the notebook output
```

### View Logs in Terminal

```bash
export PHASIC_LOG_LEVEL=DEBUG
python my_script.py
# All logs will appear on stderr
```

### Filter Logs for Specific Module

```python
from phasic.logging_config import set_log_level

# DEBUG for hierarchical cache only
set_log_level('DEBUG', module='hierarchical_trace_cache')

# INFO for everything else
set_log_level('INFO')
```

## Log Categories

### Cache Operations
- Cache queries (every hash lookup)
- Cache hits with size info
- Cache misses
- Cache saves with success/failure

### SCC Decomposition
- Graph sizes being decomposed
- Number of components found
- Individual SCC sizes and hashes
- Topological ordering

### Recursive Collection
- Depth-aware indentation for nested calls
- Graph size at each level
- Cache hit/miss for each component
- Subdivision decisions (below/above min_size)
- Work unit collection
- Summary statistics

### Trace Merging
- Initialization steps
- Vertex mapping setup
- Operation copying with offsets
- Vertex data merging (rates, edges)
- Edge connections (source → target)
- Connecting vs internal vertices
- Final validation
- Complete statistics

## Benefits

1. **Debugging** - Pinpoint exactly where caching/merging issues occur
2. **Performance Analysis** - See which components are cached vs recomputed
3. **Understanding** - Visualize the hierarchical decomposition process
4. **Verification** - Validate that traces are being merged correctly
5. **Optimization** - Identify inefficiencies in cache strategy

## Performance Impact

With default WARNING level:
- **Zero overhead** - All DEBUG calls are filtered before string formatting
- **No performance impact** - Early exit in logging framework

With DEBUG level enabled:
- **Minimal overhead** - String formatting only when logging is active
- **Acceptable for development** - Useful for debugging, not production

## Files Modified

1. `src/phasic/hierarchical_trace_cache.py` - Added comprehensive logging throughout
2. `src/phasic/trace_elimination.py` - Custom `__repr__()` for EliminationTrace (prevents 171MB output)

## Related Documentation

- `TRACE_LOGGING_COMPLETE.md` - Original trace logging implementation
- `LOGGING_CHANGES_SUMMARY.md` - Summary of all logging changes
- `NOTEBOOK_171MB_BUG_FIX.md` - Fix for massive notebook file size

## Testing

To test the comprehensive logging:

```python
from phasic.logging_config import set_log_level
set_log_level('DEBUG')

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np

# Create a graph large enough to trigger hierarchical caching
# (>50 vertices by default)
graph = phasic.Graph(...)

# Enable hierarchical caching with low min_size to see subdivisions
trace = graph.compute_trace(hierarchical=True, min_size=10)

# You should see:
# - Cache queries for the full graph and each SCC
# - SCC decomposition details
# - Recursive collection with indentation
# - Detailed merge steps showing operations and edges
# - Final validation and statistics
```

## Example Output Patterns

### Successful Cache Hit
```
[DEBUG] Cache query: hash=abc123...
[DEBUG]   ✓ Cache HIT: hash=abc123..., 25 vertices, 1234 operations
```

### Cache Miss → Subdivision
```
[DEBUG] Cache query: hash=def456...
[DEBUG]   ✗ Cache MISS: hash=def456...
[DEBUG] → Subdividing into SCCs (100 vertices >= min_size=50)...
[DEBUG]   Computing SCC decomposition...
[DEBUG]   ✓ Found 45 SCCs
[DEBUG]   SCC sizes: [2, 2, 4, 8, 3, ...]
```

### Nested Recursion (with indentation)
```
[DEBUG] Checking graph: 100 vertices, hash=abc...
[DEBUG] → Subdividing into SCCs...
[DEBUG]   Processing SCC 1/3: 40 vertices, hash=def...
[DEBUG]     Checking graph: 40 vertices, hash=def...
[DEBUG]     → Subdividing into SCCs...
[DEBUG]       Processing SCC 1/2: 20 vertices, hash=ghi...
[DEBUG]         Checking graph: 20 vertices, hash=ghi...
[DEBUG]         ✓ Cache hit for 20 vertices
```

### Merge Progress
```
[DEBUG] SCC 1/125 merge complete (total ops now: 38)
[DEBUG] SCC 2/125 merge complete (total ops now: 80)
[DEBUG] SCC 3/125 merge complete (total ops now: 122)
...
[DEBUG] SCC 125/125 merge complete (total ops now: 4827)
```

---

**Status:** ✅ Complete and ready for use

All hierarchical trace caching operations now have comprehensive DEBUG logging that shows every subdivision, cache query, and merge step in detail.
