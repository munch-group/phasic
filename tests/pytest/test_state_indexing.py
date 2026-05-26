"""
Tests for flexible state indexing system.

Verifies the PropertySet / StateIndexer / Slot API for all 4 original C++
scenarios:
1. Single locus
2. Two locus
3. Single locus with derived
4. Two locus with derived
"""

import pytest
import numpy as np
from phasic import Property, PropertySet, StateIndexer
from phasic.state_indexing import Slot


# ============================================================================
# Test Property Class
# ============================================================================

def test_property_basic():
    prop = Property('descendants', max_value=10)
    assert prop.name == 'descendants'
    assert prop.max_value == 10
    assert prop.min_value == 0
    assert prop.base == 11


def test_property_with_min_value():
    prop = Property('population', min_value=1, max_value=3)
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
    prop = Property('descendants', max_value=10)

    prop.validate_value(0)
    prop.validate_value(5)
    prop.validate_value(10)

    with pytest.raises(ValueError):
        prop.validate_value(-1)
    with pytest.raises(ValueError):
        prop.validate_value(11)


def test_property_validation_with_min_value():
    prop = Property('population', min_value=1, max_value=3)

    prop.validate_value(1)
    prop.validate_value(2)
    prop.validate_value(3)

    with pytest.raises(ValueError):
        prop.validate_value(0)
    with pytest.raises(ValueError):
        prop.validate_value(4)


def test_property_iter():
    prop = Property('population', min_value=1, max_value=3)
    assert list(prop) == [1, 2, 3]


# ============================================================================
# Test PropertySet - Single Locus (Original C++ Scenario 1)
# ============================================================================

def test_single_locus_state_space():
    """Single locus PropertySet matches C++ _index_to_props_single_locus."""
    s = 10  # sample size

    # C++ formula: a * (s+1)^0 + (p-1) * (s+1)^1
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    # Index 0: descendants=0, population=1
    props = pset.index_to_props(0, as_dict=True)
    assert props['descendants'] == 0
    assert props['population'] == 1
    assert pset.props_to_index(props) == 0

    # Index 11: descendants=0, population=2
    props = pset.index_to_props(11, as_dict=True)
    assert props['descendants'] == 0
    assert props['population'] == 2
    assert pset.props_to_index(props) == 11

    # Index 15: descendants=4, population=2
    props = pset.index_to_props(15, as_dict=True)
    assert props['descendants'] == 4
    assert props['population'] == 2
    assert pset.props_to_index(props) == 15


def test_single_locus_roundtrip():
    s = 10
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    for i in range(pset.state_length):
        props = pset.index_to_props(i, as_dict=True)
        assert pset.props_to_index(props) == i, f"Failed at index {i}"


def test_single_locus_dataclass_roundtrip():
    """Default index_to_props returns a dataclass; p2i accepts it back."""
    s = 10
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    for i in range(pset.state_length):
        props = pset.index_to_props(i)  # dataclass
        assert pset.props_to_index(props) == i


# ============================================================================
# Test PropertySet - Two Locus (Original C++ Scenario 2)
# ============================================================================

def test_two_locus_state_space():
    """Two locus PropertySet matches C++ _index_to_props_two_locus."""
    s = 5

    # C++ formula: a * (s+1)^0 + b * (s+1)^1 + (p-1) * (s+1)^2
    pset = PropertySet('lineage', [
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    # Index 0: l1=0, l2=0, pop=1
    props = pset.index_to_props(0, as_dict=True)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['population'] == 1

    # Index 7: l1=1, l2=1, pop=1  (1 + 1*6)
    props = pset.index_to_props(7, as_dict=True)
    assert props['descendants_l1'] == 1
    assert props['descendants_l2'] == 1
    assert props['population'] == 1

    # Index 36: l1=0, l2=0, pop=2  (first index of pop 2)
    props = pset.index_to_props(36, as_dict=True)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['population'] == 2


def test_two_locus_roundtrip():
    s = 5
    pset = PropertySet('lineage', [
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    for i in range(min(pset.state_length, 100)):
        props = pset.index_to_props(i, as_dict=True)
        assert pset.props_to_index(props) == i


# ============================================================================
# Test PropertySet - Single Locus Derived (Original C++ Scenario 3)
# ============================================================================

def test_single_locus_derived_state_space():
    """Single locus with derived matches C++ _index_to_props_single_locus_derived."""
    s = 5

    # C++ formula: a * (s+1)^0 + d * (s+1)^1 + (p-1) * (s+1)^2
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('is_derived', max_value=1),  # 0 or 1
        Property('population', min_value=1, max_value=3),
    ])

    # Index 0: descendants=0, derived=0, pop=1
    props = pset.index_to_props(0, as_dict=True)
    assert props['descendants'] == 0
    assert props['is_derived'] == 0
    assert props['population'] == 1

    # Index 6: descendants=0, derived=1, pop=1
    props = pset.index_to_props(6, as_dict=True)
    assert props['descendants'] == 0
    assert props['is_derived'] == 1
    assert props['population'] == 1

    # Index 8: descendants=2, derived=1, pop=1
    props = pset.index_to_props(8, as_dict=True)
    assert props['descendants'] == 2
    assert props['is_derived'] == 1
    assert props['population'] == 1


def test_single_locus_derived_roundtrip():
    s = 5
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', min_value=1, max_value=3),
    ])

    for i in range(pset.state_length):
        props = pset.index_to_props(i, as_dict=True)
        assert pset.props_to_index(props) == i


# ============================================================================
# Test PropertySet - Two Locus Derived (Original C++ Scenario 4)
# ============================================================================

def test_two_locus_derived_state_space():
    """Two locus with derived matches C++ _index_to_props_two_locus_derived."""
    s = 3

    # C++ formula: a * (s+1)^0 + b * (s+1)^1 + d * (s+1)^2 + (p-1) * (s+1)^3
    pset = PropertySet('lineage', [
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', min_value=1, max_value=3),
    ])

    # Index 0: l1=0, l2=0, derived=0, pop=1
    props = pset.index_to_props(0, as_dict=True)
    assert props['descendants_l1'] == 0
    assert props['descendants_l2'] == 0
    assert props['is_derived'] == 0
    assert props['population'] == 1

    # Test with derived=1
    props = pset.index_to_props(16, as_dict=True)  # (s+1)^2 = 16
    assert props['is_derived'] == 1


def test_two_locus_derived_roundtrip():
    s = 3
    pset = PropertySet('lineage', [
        Property('descendants_l1', max_value=s),
        Property('descendants_l2', max_value=s),
        Property('is_derived', max_value=1),
        Property('population', min_value=1, max_value=3),
    ])

    for i in range(pset.state_length):
        props = pset.index_to_props(i, as_dict=True)
        assert pset.props_to_index(props) == i


# ============================================================================
# Test PropertySet - Vectorized Operations
# ============================================================================

def test_vectorized_index_to_props():
    s = 5
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    # For s=5, base = s+1 = 6, so first index of pop=2 is 6
    indices = np.array([0, 1, 2, 6, 12, 17])
    props_list = pset.index_to_props(indices, as_dict=True)

    assert len(props_list) == 6
    assert props_list[0]['descendants'] == 0
    assert props_list[0]['population'] == 1
    assert props_list[3]['descendants'] == 0
    assert props_list[3]['population'] == 2
    assert props_list[4]['descendants'] == 0
    assert props_list[4]['population'] == 3
    assert props_list[5]['descendants'] == 5
    assert props_list[5]['population'] == 3


def test_vectorized_index_to_props_as_values():
    s = 5
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    # For s=5, base = s+1 = 6, so first index of pop=2 is 6
    indices = np.array([0, 6, 7])
    values = pset.index_to_props(indices, as_values=True)

    # Decoded values: (descendants, population) — population is 1-based
    assert values.shape == (3, 2)
    assert tuple(values[0]) == (0, 1)
    assert tuple(values[1]) == (0, 2)
    assert tuple(values[2]) == (1, 2)


def test_vectorized_props_to_index():
    s = 5
    pset = PropertySet('lineage', [
        Property('descendants', max_value=s),
        Property('population', min_value=1, max_value=3),
    ])

    # Raw (decoded) property values: pop is 1-based
    raw = np.array([
        [0, 1],  # descendants=0, pop=1
        [1, 1],  # descendants=1, pop=1
        [0, 2],  # descendants=0, pop=2
    ])

    indices = pset.props_to_index(raw)
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 3
    assert indices[0] == 0
    assert indices[1] == 1
    assert indices[2] == 6  # (s+1) = 6


# ============================================================================
# Test PropertySet - Edge Cases and Validation
# ============================================================================

def test_state_space_size():
    pset = PropertySet('lineage', [
        Property('a', max_value=2),  # base 3
        Property('b', max_value=3),  # base 4
    ])
    assert pset.state_length == 3 * 4  # 12


def test_props_to_index_kwargs():
    pset = PropertySet('lineage', [
        Property('a', max_value=2),
        Property('b', max_value=2),
    ])

    idx_dict = pset.props_to_index({'a': 1, 'b': 2})
    idx_kwargs = pset.props_to_index(a=1, b=2)
    assert idx_dict == idx_kwargs


def test_partial_props_to_index():
    """Partial property specification returns array of matching indices."""
    pset = PropertySet('lineage', [
        Property('a', max_value=2),
        Property('b', max_value=2),
    ])

    # All indices where a=2: a=2,b=0; a=2,b=1; a=2,b=2 -> 2, 5, 8
    matches = pset.props_to_index({'a': 2})
    assert isinstance(matches, np.ndarray)
    assert sorted(matches.tolist()) == [2, 5, 8]


def test_duplicate_property_names():
    with pytest.raises(ValueError):
        PropertySet('lineage', [
            Property('a', max_value=2),
            Property('a', max_value=3),
        ])


# ============================================================================
# Test StateIndexer - PropertySet Access
# ============================================================================

def test_indexer_single_property_set():
    """StateIndexer with a single named PropertySet."""
    indexer = StateIndexer(
        lineage=[
            Property('descendants', max_value=10),
            Property('population', min_value=1, max_value=3),
        ]
    )

    assert indexer.state_length == 33  # 11 * 3
    assert indexer.lineage.state_length == 33
    assert isinstance(indexer.lineage, PropertySet)

    # Auto-detect props_to_index for single PropertySet
    assert indexer.props_to_index(descendants=0, population=1) == 0
    assert indexer.props_to_index({'descendants': 4, 'population': 2}) == 15


def test_indexer_multiple_property_sets():
    """StateIndexer concatenates multiple PropertySets."""
    indexer = StateIndexer(
        lineage=[Property('descendants', max_value=10)],     # 11 states
        metadata=[Property('time_bin', max_value=99)],       # 100 states
    )

    assert indexer.lineage.state_length == 11
    assert indexer.metadata.state_length == 100
    assert indexer.state_length == 111

    # Explicit PropertySet name
    assert indexer.props_to_index('lineage', {'descendants': 5}) == 5
    assert indexer.props_to_index('metadata', time_bin=4) == 11 + 4


def test_indexer_index_to_props_dispatch():
    """index_to_props dispatches to the matching PropertySet."""
    indexer = StateIndexer(
        lineage=[Property('descendants', max_value=10)],
        metadata=[Property('time_bin', max_value=99)],
    )

    # Index in lineage range
    result = indexer.index_to_props(5)
    assert result.lineage.descendants == 5
    assert result.metadata is None

    # Index in metadata range (offset by lineage.state_length=11)
    result = indexer.index_to_props(15)
    assert result.lineage is None
    assert result.metadata.time_bin == 4


def test_indexer_flatten():
    """flatten=True returns the PropertySet's props directly."""
    indexer = StateIndexer(
        lineage=[
            Property('descendants', max_value=5),
            Property('population', min_value=1, max_value=3),
        ]
    )

    props = indexer.index_to_props(15, flatten=True, as_dict=True)
    assert props == {'descendants': 3, 'population': 3}


# ============================================================================
# Test StateIndexer - Slots
# ============================================================================

def test_slot_via_positional():
    """Positional string args become Slots."""
    indexer = StateIndexer(
        'epoch', 'branch_id',
        lineage=[Property('descendants', max_value=10)],
    )

    # Slots come after the PropertySet block
    assert indexer.lineage.state_length == 11
    assert indexer.epoch == 11
    assert indexer.branch_id == 12
    assert indexer.state_length == 13  # 11 lineage states + 2 slots


def test_slot_via_slots_kwarg():
    """slots= keyword also accepts slot names."""
    indexer = StateIndexer(
        lineage=[Property('descendants', max_value=5)],  # 6 states
        slots=['epoch', 'branch_id'],
    )

    assert indexer.epoch == 6
    assert indexer.branch_id == 7
    assert indexer.state_length == 8


def test_slot_object_in_slots_kwarg():
    """slots= accepts Slot instances directly."""
    indexer = StateIndexer(
        lineage=[Property('descendants', max_value=2)],  # 3 states
        slots=[Slot('epoch')],
    )
    assert indexer.epoch == 3


def test_slot_index_to_props():
    """An index pointing to a slot marks that slot True in IndexResult."""
    indexer = StateIndexer(
        'epoch',
        lineage=[Property('descendants', max_value=10)],
    )

    result = indexer.index_to_props(indexer.epoch)
    assert result.lineage is None
    assert result.epoch is True


def test_duplicate_slot_name():
    with pytest.raises(ValueError):
        StateIndexer(
            'epoch', 'epoch',
            lineage=[Property('a', max_value=2)],
        )


def test_slot_collides_with_property_set():
    with pytest.raises(ValueError):
        StateIndexer(
            'lineage',
            lineage=[Property('a', max_value=2)],
        )


# ============================================================================
# Integration Tests
# ============================================================================

def test_custom_property_combination():
    pset = PropertySet('lineage', [
        Property('descendants', max_value=10),
        Property('chromosome', max_value=21),    # 22 chromosomes
        Property('is_male', max_value=1),
        Property('age_bin', max_value=9),        # 10 age bins
        Property('population', min_value=1, max_value=5),  # 5 populations
    ])

    test_indices = [0, 1, 100, 1000, pset.state_length - 1]
    for i in test_indices:
        if i < pset.state_length:
            props = pset.index_to_props(i, as_dict=True)
            assert pset.props_to_index(props) == i


def test_all_scenarios_comprehensive():
    """All 4 original C++ scenarios with multiple indices."""
    scenarios = [
        # Single locus
        ([Property('descendants', max_value=10),
          Property('population', min_value=1, max_value=3)],
         [0, 5, 11, 15, 22, 32]),

        # Two locus
        ([Property('descendants_l1', max_value=5),
          Property('descendants_l2', max_value=5),
          Property('population', min_value=1, max_value=3)],
         [0, 1, 6, 7, 36, 37, 42]),

        # Single locus derived
        ([Property('descendants', max_value=5),
          Property('is_derived', max_value=1),
          Property('population', min_value=1, max_value=3)],
         [0, 1, 6, 7, 12, 13, 18, 36]),

        # Two locus derived
        ([Property('descendants_l1', max_value=3),
          Property('descendants_l2', max_value=3),
          Property('is_derived', max_value=1),
          Property('population', min_value=1, max_value=3)],
         [0, 1, 4, 5, 16, 17, 20, 32, 48]),
    ]

    for properties, test_indices in scenarios:
        pset = PropertySet('lineage', properties)
        for idx in test_indices:
            if idx < pset.state_length:
                props = pset.index_to_props(idx, as_dict=True)
                assert pset.props_to_index(props) == idx, (
                    f"Failed for {properties} at index {idx}"
                )


# ============================================================================
# Concatenated layout: append / + preserves each operand's block layout
# (so a slot that is not at the end of self stays put, rather than being
# relocated past the appended block). Regression for joint_prob_graph mis-
# indexing the epoch slot after add_epoch.
# ============================================================================

def test_append_preserves_self_trailing_slot_position():
    """A trailing slot of self stays in place; other's psets follow after it."""
    # self mimics an epoch-augmented base indexer: one PropertySet then a slot.
    base = StateIndexer(
        'epoch',
        lineages=[Property('ton', min_value=1, max_value=4)],  # 4 states: 0..3
    )
    assert base.epoch == 4  # slot at the end of self (default layout)

    # other mimics the reward indexer appended by joint_prob_graph.
    reward = StateIndexer(
        lineages_ton=[Property('lineages_ton', min_value=1, max_value=4)],  # 4 states
    )

    combined = base + reward

    # Each operand keeps its layout as a contiguous block:
    #   self  : lineages(0..3), epoch(4)             -> indices [0, base.state_length)
    #   other : lineages_ton(5..8)                   -> offset by base.state_length
    assert combined.epoch == 4                       # NOT relocated to the end
    assert combined.index_ranges['lineages'] == (0, 4)
    assert combined.index_ranges['lineages_ton'] == (5, 9)
    assert combined.index_ranges['epoch'] == (4, 4)
    assert combined.state_length == base.state_length + reward.state_length == 9


def test_append_offsets_match_concatenated_state_vector():
    """combined offsets index np.append(self_block, other_block) correctly."""
    base = StateIndexer(
        'epoch',
        lineages=[Property('ton', min_value=1, max_value=4)],
    )
    reward = StateIndexer(
        lineages_ton=[Property('lineages_ton', min_value=1, max_value=4)],
    )
    combined = base + reward

    # The slot index decodes back to the slot, the reward indices decode to the
    # reward PropertySet, and the base indices decode to the base PropertySet.
    assert combined.index_to_props(combined.epoch).epoch is True
    assert combined.index_to_props(0).lineages is not None
    # First reward index sits immediately after the epoch slot.
    first_reward = combined.index_ranges['lineages_ton'][0]
    assert first_reward == base.state_length
    assert combined.index_to_props(first_reward).lineages_ton is not None


def test_append_roundtrips_interleaved_layout_through_to_dict():
    """to_dict/from_dict preserve a non-default (interleaved) layout."""
    base = StateIndexer(
        'epoch',
        lineages=[Property('ton', min_value=1, max_value=4)],
    )
    reward = StateIndexer(
        lineages_ton=[Property('lineages_ton', min_value=1, max_value=4)],
    )
    combined = base + reward

    restored = StateIndexer.from_dict(combined.to_dict())
    assert restored.index_ranges == combined.index_ranges
    assert restored.epoch == combined.epoch
    assert restored.state_length == combined.state_length
    assert restored == combined


def test_default_layout_still_psets_then_slots():
    """Plain construction is unchanged: slots come after the PropertySet block."""
    indexer = StateIndexer(
        'epoch', 'branch_id',
        lineage=[Property('descendants', max_value=10)],  # 11 states
    )
    assert indexer.epoch == 11
    assert indexer.branch_id == 12
    # to_dict records the default layout order explicitly.
    assert indexer.to_dict()['entity_order'] == ['lineage', 'epoch', 'branch_id']


def test_interleaved_layout_distinct_from_default_eq_hash():
    """Two indexers with the same psets/slots but different layouts differ."""
    base = StateIndexer('epoch', lineage=[Property('a', min_value=1, max_value=4)])
    reward = StateIndexer(reward=[Property('reward', min_value=1, max_value=4)])
    interleaved = base + reward  # layout: lineage, epoch, reward

    # A default-layout indexer with the same psets and slot: lineage, reward, epoch
    default_like = StateIndexer(
        'epoch',
        lineage=[Property('a', min_value=1, max_value=4)],
        reward=[Property('reward', min_value=1, max_value=4)],
    )

    assert interleaved._layout_order() == ['lineage', 'epoch', 'reward']
    assert default_like._layout_order() == ['lineage', 'reward', 'epoch']
    assert interleaved != default_like
    # Equality is reflexive and layout-preserving across a round-trip.
    assert interleaved == StateIndexer.from_dict(interleaved.to_dict())


def test_default_layout_hash_unchanged_by_entity_order():
    """The hash of a conventionally-built indexer omits the (default) layout."""
    # Two structurally identical default-layout indexers hash equal, and the
    # hash does not include an entity_order component (no cache-key churn).
    a = StateIndexer('epoch', lineage=[Property('d', max_value=3)])
    b = StateIndexer('epoch', lineage=[Property('d', max_value=3)])
    assert hash(a) == hash(b)
    assert a == b
    assert 'layout=' not in repr(a)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
