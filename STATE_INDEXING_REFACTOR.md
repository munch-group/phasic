# State Indexing Refactoring

**Date:** 2025-11-29
**Status:** ✅ Complete

## Summary

Refactored `src/phasic/state_indexing.py` to remove the reserved index 0 for starting vertex, while preserving the mixed-radix numbering system for efficient property combination indexing.

## Issue Identified

The original implementation reserved index 0 for the starting vertex, which created unnecessary special cases:
- Index 0 returned `None` instead of a valid property combination
- All indices were offset by +1 throughout the codebase
- Size was calculated as `product_of_bases + 1`
- Created confusion about what index 0 represented

**Solution:** Remove the index 0 reservation and special handling, allowing index 0 to map to the first valid property combination (all properties at min_value).

## Key Changes

### 1. StateSpace Class Updates

**Removed:**
- Index 0 reservation (`if index == 0: return None`)
- +1 offset in `props_to_index()` return value
- -1 offset in `index_to_props()` input value
- +1 in `size` property calculation

**Preserved:**
- Mixed-radix numbering system (`_bases`, `_radix_powers`)
- `index_to_props()` and `props_to_index()` method names
- `size` property (now = product of bases, no +1)
- `as_dict` parameter in `index_to_props()`
- Property encoding/decoding system
- StateVector class with index-based interface

**Updated Behavior:**
- Index 0 now maps to first valid property combination: `{prop1: min_value1, prop2: min_value2, ...}`
- Valid indices are 0 to size-1 (instead of 1 to size)
- All property combinations have unique integer indices with no gaps

### 2. Property Class

**No changes** - `base` property and encode/decode methods remain unchanged.

### 3. StateVector Class

**No changes** - Still uses `index` parameter and `update_index()` method.

## API (Unchanged from Before)

### StateSpace

```python
# Create state space
state_space = StateSpace([
    Property('n_pop1', max_value=5, min_value=0),  # base 6
    Property('n_pop2', max_value=5, min_value=0)   # base 6
])

# Size is product of bases (no +1)
print(state_space.size)  # 36 (was 37 before)

# Index 0 maps to first valid state (not None!)
props = state_space.index_to_props(0)
print(props)  # {'n_pop1': 0, 'n_pop2': 0}

# Convert properties to index (no +1 offset)
idx = state_space.props_to_index({'n_pop1': 3, 'n_pop2': 2})
print(idx)  # 15 (calculated as 3 + 2*6 = 15, no +1)

# Round-trip conversion
props2 = state_space.index_to_props(idx)
assert props2 == {'n_pop1': 3, 'n_pop2': 2}
assert state_space.props_to_index(props2) == idx
```

### Usage Pattern

```python
from phasic.state_indexing import StateSpace, Property

# Define state space
state_space = StateSpace([
    Property('n', max_value=10, min_value=0)
])

# Map property combination to unique integer
for n in range(11):
    idx = state_space.props_to_index({'n': n})
    print(f"n={n} -> index {idx}")
    # n=0 -> index 0
    # n=1 -> index 1
    # ...
    # n=10 -> index 10
```

## Before vs After

### Before (with index 0 reserved):

```python
space = StateSpace([Property('a', max_value=2)])  # base 3

# Index 0 was reserved for starting vertex
space.index_to_props(0)  # None
space.size  # 4 = 3 + 1

# First valid property combination at index 1
space.index_to_props(1)  # {'a': 0}
space.props_to_index({'a': 0})  # 1 (offset by +1)
space.props_to_index({'a': 2})  # 3
```

### After (no index 0 reservation):

```python
space = StateSpace([Property('a', max_value=2)])  # base 3

# Index 0 maps to first valid property combination
space.index_to_props(0)  # {'a': 0}
space.size  # 3 (no +1)

# Direct mapping without offsets
space.props_to_index({'a': 0})  # 0
space.props_to_index({'a': 1})  # 1
space.props_to_index({'a': 2})  # 2
```

## Testing

All functionality verified with comprehensive test suite:

✅ Index 0 maps to valid property combination (not None)
✅ No +1/-1 offsets in conversion
✅ Size = product of bases (no +1)
✅ Properties with min_value handling
✅ Round-trip conversion for all indices
✅ Vectorized conversion
✅ StateVector wrapper class

## Benefits

1. **Simpler implementation:** No special cases for index 0
2. **Cleaner API:** No None returns, no confusing offsets
3. **Standard 0-indexing:** Consistent with Python/numpy conventions
4. **Better clarity:** Index directly corresponds to property combination, no mental +1/-1

## Backward Compatibility

⚠️ **Breaking changes:**
- `index_to_props(0)` now returns `{...}` instead of `None`
- All indices shifted down by 1 (what was index 5 is now index 4)
- `size` reduced by 1 for all state spaces

**Migration:**
- Any code checking for `if props is None` should be updated
- Any hardcoded indices need to be decremented by 1
- Any size-based loops need updating: `range(1, space.size)` → `range(space.size)`

## Files Modified

- `src/phasic/state_indexing.py` - Removed index 0 special handling (~15 lines changed)

---

**Conclusion:** This refactoring eliminates the confusing index 0 reservation, providing clean 0-indexed mapping from property combinations to integers using mixed-radix numbering.
