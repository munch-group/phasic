# Slot Implementation for StateIndexer

## Summary

Added **Slot** functionality to `StateIndexer` for 1:1 label-to-index mappings, providing a lightweight alternative to PropertySets for simple metadata storage.

## Key Changes

### 1. Dynamic `IndexResult` Dataclass
- Result type for `index_to_props()` with PropertySet/Slot names as attributes
- Each PropertySet/Slot is an attribute on the result object
- Only the matching PropertySet/Slot is set, all others are None
- Allows direct access: `result.lineage.descendants` or `result.epoch`
- Created dynamically per StateIndexer instance via `make_dataclass()`

### 2. New `Slot` Dataclass
- Simple frozen dataclass with just a `name` field
- Each slot occupies exactly 1 index in concatenated space
- Unlike PropertySets which use combinatorial spaces

### 3. Updated `StateIndexer` Class

**New Parameters:**
- `slots: Optional[Union[List[str], List[Slot]]]` - accepts slot names or Slot objects

**Index Space Organization:**
- PropertySets allocated first (in insertion order)
- Slots allocated after all PropertySets
- Each slot gets exactly 1 index

**Updated Methods:**
- `__init__`: Processes slots parameter, tracks slot order and offsets
- `__getattr__`: Returns slot index (int) when accessing slot by name
- `__setattr__`: Prevents modification of slot attributes
- `__dir__`: Includes slot names for autocomplete
- `__contains__`: Checks both PropertySets and slots
- `__repr__`: Shows slots as `name(slot)` format

**Index Operations:**
- `_decompose_index()`: Returns `(slot_name, None)` for slot indices
- `_compose_index()`: Accepts optional `local_index=None` for slots
- `index_to_props()`: Returns `IndexResult` dataclass with PropertySet/Slot name attributes
- `_create_result_class()`: Creates and caches dynamic dataclass for results
- `index_ranges`: Includes slots with `(idx, idx)` ranges

## Usage Examples

```python
from phasic.state_indexing import StateIndexer, Property

# Create indexer with PropertySets and slots
indexer = StateIndexer(
    lineage=[Property('descendants', max_value=10)],  # indices 0-10
    metadata=[Property('time_bin', max_value=100)],   # indices 11-111
    slots=['epoch', 'branch_id']                       # indices 112, 113
)

# Access slot indices directly via attributes
epoch_idx = indexer.epoch        # Returns 112
branch_idx = indexer.branch_id   # Returns 113

# Use in state vector
state = np.zeros(indexer.n_states)
state[indexer.epoch] = 5         # Store epoch value
state[indexer.branch_id] = 42    # Store branch ID

# Query index information with direct attribute access
result = indexer.index_to_props(5)
result.lineage            # LineageProps(descendants=5)
result.lineage.descendants  # 5
result.metadata           # None (not in metadata range)
result.epoch              # None (not epoch slot)

# Different index in metadata range
result = indexer.index_to_props(15)
result.lineage            # None
result.metadata.time_bin  # 4
result.epoch              # None

# Slot index returns True
result = indexer.index_to_props(112)
result.lineage            # None
result.metadata           # None
result.epoch              # True

# Conditional pattern
if result.lineage:
    print(f"Descendants: {result.lineage.descendants}")

# Index ranges
ranges = indexer.index_ranges
# {'lineage': (0, 10), 'metadata': (11, 111),
#  'epoch': (112, 112), 'branch_id': (113, 113)}

# Slots-only indexer
simple = StateIndexer(slots=['epoch', 'sample_id', 'branch_id'])
simple.epoch       # Returns 0
simple.sample_id   # Returns 1
simple.branch_id   # Returns 2
```

## Design Decisions

### Ordering: Slots After PropertySets
Slots are always allocated after all PropertySets in the concatenated index space. This is because:
- Python's `**kwargs` doesn't preserve order relative to explicit parameters
- Keeps implementation simple and predictable
- Users can understand: "PropertySets first, then slots"

### Dynamic Result Class with PropertySet/Slot Attributes
`index_to_props()` returns dynamic dataclass with all PropertySet/Slot names as attributes:
- Direct access: `result.lineage.descendants` instead of `result.props.descendants`
- All PropertySet/Slot names are attributes (most are None, one is set)
- Supports conditional patterns: `if result.lineage: ...`
- Each StateIndexer creates its own result class (cached)

### Slot Values
Slots return `True` when their index is queried:
- PropertySets return their props object
- Slots return `True` (boolean indicator)
- All non-matching attributes return `None`

### Direct Attribute Access
Accessing a slot by name returns its index directly:
```python
indexer.epoch  # Returns int index, not Slot object
```
- Most ergonomic for common usage: `state[indexer.epoch] = value`
- PropertySets return PropertySet objects (different type helps distinguish)

## Implementation Details

**Files Modified:**
- `src/phasic/state_indexing.py` - All changes in this file

**New Internal State:**
- `_slots: Dict[str, Slot]` - slot name to Slot mapping
- `_slot_order: List[str]` - preserves slot insertion order
- `_offsets` includes both PropertySet and slot offsets
- `_result_class` - cached dynamic dataclass for index_to_props results

**Key Invariants:**
- `n_states` = sum of PropertySet n_states + number of slots
- Each slot occupies exactly 1 index
- No name conflicts between PropertySets and slots
- Slots always come after PropertySets in index space

## Testing

All 12 test scenarios passed:
1. ✓ Slot attribute access
2. ✓ n_states includes slots
3. ✓ index_ranges includes slots
4. ✓ _decompose_index handles slots
5. ✓ _compose_index handles slots
6. ✓ index_to_props handles slots
7. ✓ PropertySets still work
8. ✓ __repr__ includes slots
9. ✓ Slot objects work
10. ✓ Slots-only StateIndexer
11. ✓ __dir__ includes slots
12. ✓ __contains__ includes slots

## Backward Compatibility

Fully backward compatible:
- Existing code without slots works unchanged
- `slots` parameter is optional
- All existing methods work as before when no slots present
