"""Review F p3: two calls that v3.1 would key identically but that build
different graphs, via a StateIndexer whose caller-owned property list was
mutated after the indexer was constructed (PropertySet stores the list
without copying, state_indexing.py:280; derived _bases/_radix_powers/_offsets
are computed once at construction, :288-289, :911-924).

The v3.1 StateIndexer key (plan §2.2.2 rule 8) is reproduced literally:
class token, PropertySet class names in order, to_dict(), plus each
Property's offset, after the admission checks (exact types, vars() subset
of the allow-lists). Master builds with graph_cache=False (no file).
"""
from phasic import Graph
from phasic.state_indexing import StateIndexer, PropertySet, StateSpace, Property

IX_ALLOW = {"_property_sets", "_slots", "_offsets", "_pset_order", "_slot_order",
            "_entity_order", "_result_class", "_total_state_length"}
PS_ALLOW = {"name", "properties", "property_dict", "_bases", "_radix_powers",
            "_props_dataclass"}


def v31_indexer_key(ix):
    assert type(ix) is StateIndexer and set(vars(ix)) <= IX_ALLOW
    for ps in ix._property_sets.values():
        assert type(ps) in (PropertySet, StateSpace) and set(vars(ps)) <= PS_ALLOW
        for p in ps.properties:
            assert type(p) is Property
    return ("StateIndexer",
            [type(ps).__name__ for ps in ix._property_sets.values()],
            ix.to_dict(),
            [[p.offset for p in ps.properties] for ps in ix._property_sets.values()])


def cb(state, indexer=None, **kw):
    k = int(state[0])
    if k + 1 < indexer.state_length:      # walks the whole index space
        return [([k + 1], 1.0)]
    return []


props = [Property("a", max_value=3)]
ixA = StateIndexer(lineage=props)
props[0] = Property("a", max_value=5)      # caller reuses / edits its list
ixB = StateIndexer(lineage=[Property("a", max_value=5)])

print("v3.1 indexer keys equal:", v31_indexer_key(ixA) == v31_indexer_key(ixB))
gA = Graph(cb, ipv=[[[0], 1.0]], indexer=ixA, graph_cache=False)
gB = Graph(cb, ipv=[[[0], 1.0]], indexer=ixB, graph_cache=False)
print("vertices A, B:", gA.vertices_length(), gB.vertices_length())
print("expectation A, B:", gA.expectation(), gB.expectation())
