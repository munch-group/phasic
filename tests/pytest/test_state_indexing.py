"""
Tests for flexible state indexing system.

Verifies compatibility with original C++ implementation for all 4 scenarios:
1. Single locus
2. Two locus
3. Single locus with derived
4. Two locus with derived
"""

import pytest
import numpy as np
from phasic.state_indexing import Property, StateIndexer, StateVector


# ============================================================================
# Test Property Class
# ============================================================================

def test_property_basic():
    """Test basic Property functionality."""
    prop = Property('descendants', max_value=10)
    assert prop.name == 'descendants'
    assert prop.max_value == 10
    assert prop.base == 11
    assert prop.offset == 0


def test_property_with_offset():
    """Test Property with offset (1-indexed populations)."""
    prop = Property('population', max_value=2, offset=1)
    assert prop.base == 3

    # Encode: value 1 -> 0, value 2 -> 1, value 3 -> 2
    assert prop.encode_value(1) == 0
    assert prop.encode_value(2) == 1
    assert prop.encode_value(3) == 2

    # Decode: 0 -> value 1, 1 -> value 2, 2 -> value 3
    assert prop.decode_value(0) == 1
    assert prop.decode_value(1) == 2
    assert prop.decode_value(2) == 3


def test_property_validation():
    """Test property value validation."""
    prop = Property('descendants', max_value=10)

    # Valid values
    prop.validate_value(0)
    prop.validate_value(5)
    prop.validate_value(10)

    # Invalid values
    with pytest.raises(ValueError):
        prop.validate_value(-1)
    with pytest.raises(ValueError):
        prop.validate_value(11)


def test_property_validation_with_offset():
    """Test validation with offset."""
    prop = Property('population', max_value=2, offset=1)

    # Valid: 1, 2, 3
    prop.validate_value(1)
    prop.validate_value(2)
    prop.validate_value(3)

    # Invalid
    with pytest.raises(ValueError):
        prop.validate_value(0)
    with pytest.raises(ValueError):
        prop.validate_value(4)


# ============================================================================
# Test StateIndexer - Single Locus (Original C++ Scenario 1)
# ============================================================================

def test_single_locus_state_space():
    """Test single locus state space (matches C++ _index_to_props_single_locus)."""
    s = 10  # sample size

    # C++ formula: props_to_index = a * (s+1)^0 + (p-1) * (s+1)^1
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)  # populations 1, 2, 3
    ])

    # Test a few conversions
    # Index 0: descendants=0, population=1
    props = space.index_to_props(0)
    assert props['descendants'] == 0
    assert props['population'] == 1
    assert space.props_to_index(props) == 0

    # Index 11: descendants=0, population=2  (first index of pop 2)
    props = space.index_to_props(11)
    assert props['descendants'] == 0
    assert props['population'] == 2
    assert space.props_to_index(props) == 11

    # Index 15: descendants=4, population=2
    props = space.index_to_props(15)
    assert props['descendants'] == 4
    assert props['population'] == 2
    assert space.props_to_index(props) == 15


def test_single_locus_roundtrip():
    """Test single locus index ↔ props round-trips."""
    s = 10
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Test all indices
    for i in range(space.size):
        props = space.index_to_props(i)
        recovered_index = space.props_to_index(props)
        assert recovered_index == i, f"Failed at index {i}"


# ============================================================================
# Test StateIndexer - Two Locus (Original C++ Scenario 2)
# ============================================================================

def test_two_locus_state_space():
    """Test two locus state space (matches C++ _index_to_props_two_locus)."""
    s = 5  # sample size

    # C++ formula: props_to_index = a * (s+1)^0 + b * (s+1)^1 + (p-1) * (s+1)^2
    space = StateIndexer([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Index 0: l1=0, l2=0, pop=1
    props = space.index_to_props(0)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['population'] == 1

    # Index 7: l1=1, l2=1, pop=1  (1 + 1*6)
    props = space.index_to_props(7)
    assert props['descendants_l1'] == 1
    assert props['descendants_l2'] == 1
    assert props['population'] == 1

    # Index 36: l1=0, l2=0, pop=2  (first index of pop 2)
    props = space.index_to_props(36)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['population'] == 2


def test_two_locus_roundtrip():
    """Test two locus index ↔ props round-trips."""
    s = 5
    space = StateIndexer([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    for i in range(min(space.size, 100)):  # Test first 100 for speed
        props = space.index_to_props(i)
        recovered_index = space.props_to_index(props)
        assert recovered_index == i


# ============================================================================
# Test StateIndexer - Single Locus Derived (Original C++ Scenario 3)
# ============================================================================

def test_single_locus_derived_state_space():
    """Test single locus with derived (matches C++ _index_to_props_single_locus_derived)."""
    s = 5

    # C++ formula: a * (s+1)^0 + d * (s+1)^1 + (p-1) * (s+1)^2
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('is_derived', max_value=1),  # 0 or 1
        Property('population', max_value=2, offset=1)
    ])

    # Index 0: descendants=0, derived=0, pop=1
    props = space.index_to_props(0)
    assert props['descendants'] == 0
    assert props['is_derived'] == 0
    assert props['population'] == 1

    # Index 6: descendants=0, derived=1, pop=1
    props = space.index_to_props(6)
    assert props['descendants'] == 0
    assert props['is_derived'] == 1
    assert props['population'] == 1

    # Index 8: descendants=2, derived=1, pop=1
    props = space.index_to_props(8)
    assert props['descendants'] == 2
    assert props['is_derived'] == 1
    assert props['population'] == 1


def test_single_locus_derived_roundtrip():
    """Test single locus derived index ↔ props round-trips."""
    s = 5
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', max_value=2, offset=1)
    ])

    for i in range(space.size):
        props = space.index_to_props(i)
        recovered_index = space.props_to_index(props)
        assert recovered_index == i


# ============================================================================
# Test StateIndexer - Two Locus Derived (Original C++ Scenario 4)
# ============================================================================

def test_two_locus_derived_state_space():
    """Test two locus with derived (matches C++ _index_to_props_two_locus_derived)."""
    s = 3

    # C++ formula: a * (s+1)^0 + b * (s+1)^1 + d * (s+1)^2 + (p-1) * (s+1)^3
    space = StateIndexer([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', max_value=2, offset=1)
    ])

    # Index 0: l1=0, l2=0, derived=0, pop=1
    props = space.index_to_props(0)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['is_derived'] == 0
    assert props['population'] == 1

    # Test with derived=1
    props = space.index_to_props(16)  # (s+1)^2 = 16
    assert props['is_derived'] == 1


def test_two_locus_derived_roundtrip():
    """Test two locus derived index ↔ props round-trips."""
    s = 3
    space = StateIndexer([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', max_value=2, offset=1)
    ])

    for i in range(space.size):
        props = space.index_to_props(i)
        recovered_index = space.props_to_index(props)
        assert recovered_index == i


# ============================================================================
# Test StateIndexer - Vectorized Operations
# ============================================================================

def test_vectorized_index_to_props():
    """Test vectorized index → props conversion."""
    s = 5
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    indices = np.array([0, 1, 2, 11, 12, 22])
    props_list = space.index_to_props(indices)

    assert len(props_list) == 6
    assert props_list[0]['descendants'] == 0
    assert props_list[0]['population'] == 1
    assert props_list[3]['descendants'] == 0
    assert props_list[3]['population'] == 2


def test_vectorized_props_to_index():
    """Test vectorized props → index conversion."""
    s = 5
    space = StateIndexer([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Create array of encoded values
    encoded = np.array([
        [0, 0],  # descendants=0, pop=1 (encoded as 0)
        [1, 0],  # descendants=1, pop=1
        [0, 1],  # descendants=0, pop=2 (encoded as 1)
    ])

    indices = space.props_to_index(encoded)
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 3
    assert indices[0] == 0
    assert indices[1] == 1
    assert indices[2] == 6  # (s+1) = 6


# ============================================================================
# Test StateVector Class
# ============================================================================

def test_state_vector_from_index():
    """Test StateVector initialization from index."""
    space = StateIndexer([
        Property('descendants', max_value=5),
        Property('population', max_value=2, offset=1)
    ])

    state = StateVector(space, index=15)
    assert state.index == 15
    assert state['descendants'] == 3
    assert state['population'] == 3


def test_state_vector_from_props():
    """Test StateVector initialization from properties."""
    space = StateIndexer([
        Property('descendants', max_value=5),
        Property('population', max_value=2, offset=1)
    ])

    state = StateVector(space, props={'descendants': 3, 'population': 3})
    assert state.index == 15
    assert state['descendants'] == 3


def test_state_vector_modification():
    """Test StateVector property modification."""
    space = StateIndexer([
        Property('descendants', max_value=5),
        Property('population', max_value=2, offset=1)
    ])

    state = StateVector(space, index=0)
    assert state['descendants'] == 0

    # Modify property
    state['descendants'] = 3
    assert state['descendants'] == 3

    # Index not automatically updated
    assert state.index == 0

    # Manually update index
    state.update_index()
    assert state.index == 3


def test_state_vector_copy():
    """Test StateVector copying."""
    space = StateIndexer([
        Property('descendants', max_value=5),
        Property('population', max_value=2, offset=1)
    ])

    state1 = StateVector(space, index=15)
    state2 = state1.copy()

    assert state2.index == state1.index
    assert state2.props == state1.props

    # Modify copy doesn't affect original
    state2['descendants'] = 0
    assert state1['descendants'] == 3


# ============================================================================
# Test Edge Cases and Validation
# ============================================================================

def test_state_space_size():
    """Test state space size calculation."""
    space = StateIndexer([
        Property('a', max_value=2),  # base 3
        Property('b', max_value=3),  # base 4
    ])
    assert space.size == 3 * 4  # 12


def test_props_to_index_kwargs():
    """Test props_to_index with kwargs."""
    space = StateIndexer([
        Property('a', max_value=2),
        Property('b', max_value=2)
    ])

    # Dict syntax
    idx1 = space.props_to_index({'a': 1, 'b': 2})

    # Kwargs syntax
    idx2 = space.props_to_index(a=1, b=2)

    assert idx1 == idx2


def test_invalid_property_access():
    """Test invalid property access in StateVector."""
    space = StateIndexer([Property('a', max_value=2)])
    state = StateVector(space, index=0)

    with pytest.raises(KeyError):
        _ = state['nonexistent']

    with pytest.raises(KeyError):
        state['nonexistent'] = 1


def test_duplicate_property_names():
    """Test that duplicate property names raise error."""
    with pytest.raises(ValueError):
        StateIndexer([
            Property('a', max_value=2),
            Property('a', max_value=3)
        ])


# ============================================================================
# Integration Tests
# ============================================================================

def test_custom_property_combination():
    """Test custom property combination not in original C++."""
    space = StateIndexer([
        Property('descendants', max_value=10),
        Property('chromosome', max_value=21),  # 22 chromosomes
        Property('is_male', max_value=1),
        Property('age_bin', max_value=9),  # 10 age bins
        Property('population', max_value=4, offset=1)  # 5 populations
    ])

    # Test round-trip for random indices
    test_indices = [0, 1, 100, 1000, space.size - 1]
    for i in test_indices:
        if i < space.size:
            props = space.index_to_props(i)
            recovered = space.props_to_index(props)
            assert recovered == i


def test_all_scenarios_comprehensive():
    """Comprehensive test of all 4 original C++ scenarios with multiple indices."""
    scenarios = [
        # Single locus
        ([Property('descendants', max_value=10),
          Property('population', max_value=2, offset=1)],
         [0, 5, 11, 15, 22, 32]),

        # Two locus
        ([Property('descendants_l1', max_value=5),
          Property('descendants_l2', max_value=5),
          Property('population', max_value=2, offset=1)],
         [0, 1, 6, 7, 36, 37, 42]),

        # Single locus derived
        ([Property('descendants', max_value=5),
          Property('is_derived', max_value=1),
          Property('population', max_value=2, offset=1)],
         [0, 1, 6, 7, 12, 13, 18, 36]),

        # Two locus derived
        ([Property('descendants_l1', max_value=3),
          Property('descendants_l2', max_value=3),
          Property('is_derived', max_value=1),
          Property('population', max_value=2, offset=1)],
         [0, 1, 4, 5, 16, 17, 20, 32, 48])
    ]

    for properties, test_indices in scenarios:
        space = StateIndexer(properties)
        for idx in test_indices:
            if idx < space.size:
                props = space.index_to_props(idx)
                recovered = space.props_to_index(props)
                assert recovered == idx, f"Failed for {properties} at index {idx}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
