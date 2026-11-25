"""reviewAC p8: (1) _bases/_radix_powers live on PropertySet, not the indexer;
(2) replacing a generated class (_props_dataclass / _result_class) with one of
the same field names changes what index_to_props returns while everything
rule 8 (a)-(c) compares stays equal to the canonical rebuild."""
import dataclasses, numpy as np
from phasic.state_indexing import StateIndexer, Property

ix = StateIndexer(lineage=[Property("n", max_value=3), Property("p", max_value=1)])
try:
    ix._bases
    print("ix._bases exists")
except AttributeError as e:
    print("ix._bases -> AttributeError:", e)
print("PropertySet has _bases:", hasattr(ix.lineage, "_bases"), ix.lineage._bases)

def canonical_view(ix):
    out = {"offsets": dict(ix._offsets), "tsl": ix._total_state_length,
           "orders": (list(ix._pset_order), list(ix._slot_order), list(ix._entity_order)),
           "result_fields": [f.name for f in dataclasses.fields(ix._result_class)]}
    for name in ix._pset_order:
        ps = ix._property_sets[name]
        out[name] = ([(p.name, p.min_value, p.max_value) for p in ps.properties],
                     sorted((k, (p.name, p.min_value, p.max_value)) for k, p in ps.property_dict.items()),
                     ps._bases.tolist(), ps._radix_powers.tolist(),
                     [f.name for f in dataclasses.fields(ps._props_dataclass)] if hasattr(ps, "_props_dataclass") else None)
    return out

before = ix.lineage.index_to_props(5)
print("index_to_props(5) before:", before)

@dataclasses.dataclass(frozen=True)
class Shifted:                     # same field names, different behaviour
    n: int
    p: int
    def __post_init__(self):
        object.__setattr__(self, "n", self.n + 100)

ix.lineage._props_dataclass = Shifted
after = ix.lineage.index_to_props(5)
print("index_to_props(5) after swap:", after)
c = StateIndexer.from_dict(ix.to_dict())
c.lineage.index_to_props(0)        # give c its lazy class too, for a like-for-like compare
v1, v2 = canonical_view(ix), canonical_view(c)
print("to_dict equal:", ix.to_dict() == c.to_dict())
print("all rule-8 (c) compared fields equal:", v1 == v2)
print("vars(pset) names:", sorted(vars(ix.lineage)))
