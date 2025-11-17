"""
Simple test script to verify state indexing implementation without pytest.
"""

import sys
import numpy as np

# Import directly from module file to avoid full package init
sys.path.insert(0, 'src')
from phasic.state_indexing import Property, StateSpace, StateVector


def test_single_locus():
    """Test single locus scenario (C++ _index_to_props_single_locus)."""
    print("Testing single locus scenario...")
    s = 10
    space = StateSpace([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Test specific indices
    test_cases = [
        (0, {'descendants': 0, 'population': 1}),
        (11, {'descendants': 0, 'population': 2}),
        (15, {'descendants': 4, 'population': 2}),
    ]

    for idx, expected in test_cases:
        props = space.index_to_props(idx)
        assert props == expected, f"Failed at index {idx}: got {props}, expected {expected}"
        recovered = space.props_to_index(props)
        assert recovered == idx, f"Round-trip failed: {idx} -> {recovered}"

    # Test all indices round-trip
    for i in range(space.size):
        props = space.index_to_props(i)
        recovered = space.props_to_index(props)
        assert recovered == i, f"Round-trip failed at index {i}"

    print(f"  ✓ All {space.size} indices passed round-trip test")


def test_two_locus():
    """Test two locus scenario (C++ _index_to_props_two_locus)."""
    print("Testing two locus scenario...")
    s = 5
    space = StateSpace([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Test specific cases
    props = space.index_to_props(0)
    assert props == {'descendants_l1': 0, 'descendants_l2': 0, 'population': 1}

    props = space.index_to_props(36)  # First index of population 2
    assert props['population'] == 2

    # Round-trip test (first 100 indices)
    for i in range(min(space.size, 100)):
        props = space.index_to_props(i)
        recovered = space.props_to_index(props)
        assert recovered == i, f"Round-trip failed at index {i}"

    print(f"  ✓ Tested {min(space.size, 100)} indices")


def test_single_locus_derived():
    """Test single locus with derived (C++ _index_to_props_single_locus_derived)."""
    print("Testing single locus with derived...")
    s = 5
    space = StateSpace([
        Property('descendants', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', max_value=2, offset=1)
    ])

    # Test specific cases
    test_cases = [
        (0, {'descendants': 0, 'is_derived': 0, 'population': 1}),
        (6, {'descendants': 0, 'is_derived': 1, 'population': 1}),
        (8, {'descendants': 2, 'is_derived': 1, 'population': 1}),
    ]

    for idx, expected in test_cases:
        props = space.index_to_props(idx)
        assert props == expected, f"Failed at index {idx}"

    # Round-trip all
    for i in range(space.size):
        props = space.index_to_props(i)
        recovered = space.props_to_index(props)
        assert recovered == i, f"Round-trip failed at index {i}"

    print(f"  ✓ All {space.size} indices passed")


def test_two_locus_derived():
    """Test two locus with derived (C++ _index_to_props_two_locus_derived)."""
    print("Testing two locus with derived...")
    s = 3
    space = StateSpace([
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', max_value=2, offset=1)
    ])

    # Test is_derived flag
    props = space.index_to_props(0)
    assert props['is_derived'] == 0

    props = space.index_to_props(16)  # (s+1)^2 = 16
    assert props['is_derived'] == 1

    # Round-trip all
    for i in range(space.size):
        props = space.index_to_props(i)
        recovered = space.props_to_index(props)
        assert recovered == i, f"Round-trip failed at index {i}"

    print(f"  ✓ All {space.size} indices passed")


def test_vectorized_operations():
    """Test vectorized conversions."""
    print("Testing vectorized operations...")
    s = 5
    space = StateSpace([
        Property('descendants', max_value=s),
        Property('population', max_value=2, offset=1)
    ])

    # Vectorized index_to_props
    indices = np.array([0, 1, 2, 11, 12, 22])
    props_list = space.index_to_props(indices)
    assert len(props_list) == 6
    assert props_list[0]['descendants'] == 0
    assert props_list[3]['population'] == 2

    # Vectorized props_to_index
    encoded = np.array([[0, 0], [1, 0], [0, 1]])
    indices = space.props_to_index(encoded)
    assert len(indices) == 3
    assert indices[0] == 0
    assert indices[1] == 1
    assert indices[2] == 6

    print("  ✓ Vectorized operations working")


def test_state_vector():
    """Test StateVector class."""
    print("Testing StateVector class...")
    space = StateSpace([
        Property('descendants', max_value=5),
        Property('population', max_value=2, offset=1)
    ])

    # From index
    state = StateVector(space, index=15)
    assert state.index == 15
    assert state['descendants'] == 3
    assert state['population'] == 3

    # From props
    state = StateVector(space, props={'descendants': 3, 'population': 3})
    assert state.index == 15

    # Modification
    state['descendants'] = 0
    state.update_index()
    assert state.index == 12

    # Copy
    state2 = state.copy()
    state2['descendants'] = 5
    assert state['descendants'] == 0  # Original unchanged

    print("  ✓ StateVector working")


def test_custom_combination():
    """Test custom property combination not in C++ code."""
    print("Testing custom property combination...")

    # Create a complex state space with 5 properties
    space = StateSpace([
        Property('descendants', max_value=10),
        Property('chromosome', max_value=21),  # 22 chromosomes (0-21)
        Property('is_male', max_value=1),
        Property('age_bin', max_value=9),  # 10 age bins (0-9)
        Property('population', max_value=4, offset=1)  # 5 populations (1-5)
    ])

    print(f"  State space size: {space.size:,} states")

    # Test a few random indices
    test_indices = [0, 1, 100, 1000, 10000, space.size - 1]
    for idx in test_indices:
        if idx < space.size:
            props = space.index_to_props(idx)
            recovered = space.props_to_index(props)
            assert recovered == idx, f"Failed at index {idx}"

    print(f"  ✓ Custom 5-property space working")


def test_property_validation():
    """Test property validation."""
    print("Testing property validation...")

    prop = Property('descendants', max_value=10)
    prop.validate_value(0)
    prop.validate_value(10)

    try:
        prop.validate_value(11)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    prop_offset = Property('population', max_value=2, offset=1)
    prop_offset.validate_value(1)
    prop_offset.validate_value(3)

    try:
        prop_offset.validate_value(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  ✓ Validation working")


def test_kwargs_interface():
    """Test kwargs interface for props_to_index."""
    print("Testing kwargs interface...")

    space = StateSpace([
        Property('a', max_value=2),
        Property('b', max_value=2)
    ])

    idx1 = space.props_to_index({'a': 1, 'b': 2})
    idx2 = space.props_to_index(a=1, b=2)
    assert idx1 == idx2

    print("  ✓ Kwargs interface working")


if __name__ == '__main__':
    print("="*60)
    print("Testing Flexible State Indexing System")
    print("="*60)
    print()

    try:
        test_single_locus()
        test_two_locus()
        test_single_locus_derived()
        test_two_locus_derived()
        test_vectorized_operations()
        test_state_vector()
        test_custom_combination()
        test_property_validation()
        test_kwargs_interface()

        print()
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ All 4 C++ scenarios verified")
        print("  ✓ Round-trip conversions working")
        print("  ✓ Vectorized operations working")
        print("  ✓ StateVector interface working")
        print("  ✓ Custom property combinations working")
        print()

    except AssertionError as e:
        print()
        print("="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        raise
