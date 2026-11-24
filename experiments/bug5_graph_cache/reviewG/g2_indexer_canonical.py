"""reviewG g2: prototype of plan v3.2 rule 8 (StateIndexer (a)-(c)) applied to indexers
that normal phasic usage produces; plus facts the builtin arm relies on."""
import math, itertools, sys, functools, warnings
import numpy as np
import phasic
from phasic import StateIndexer, Property, PropertySet, StateSpace
from phasic.state_indexing import Slot

FRESH_IX = set(vars(StateIndexer(a=[Property('x', 2)])))
FRESH_PS = set(vars(PropertySet('a', [Property('x', 2)])))
LAZY_PS = {'_props_dataclass'}

def check(ix, label):
    why = []
    if type(ix) is not StateIndexer:
        why.append("(a) indexer subclass")
    for ps in ix._property_sets.values():
        if type(ps) not in (PropertySet, StateSpace):
            why.append(f"(a) pset type {type(ps).__name__}")
        for p in ps.properties:
            if type(p) is not Property:
                why.append("(a) property subclass")
        extra = set(vars(ps)) - FRESH_PS - LAZY_PS
        if extra:
            why.append(f"(b) pset extra {extra}")
    extra = set(vars(ix)) - FRESH_IX
    if extra:
        why.append(f"(b) ix extra {extra}")
    try:
        c = StateIndexer.from_dict(ix.to_dict())
    except Exception as e:
        why.append(f"(c) from_dict raised {type(e).__name__}: {e}")
        print(f"{label:55s} REFUSED {why}")
        return
    if ix._offsets != c._offsets: why.append("(c) _offsets")
    if ix._total_state_length != c._total_state_length: why.append("(c) total")
    for a in ('_pset_order', '_slot_order', '_entity_order'):
        if getattr(ix, a) != getattr(c, a): why.append(f"(c) {a}")
    for name in c._pset_order:
        p1, p2 = ix._property_sets[name], c._property_sets[name]
        if p1.properties != p2.properties:
            why.append(f"(c) properties list ({type(p1.properties).__name__} vs list; "
                       f"{[(p.offset) for p in p1.properties]} vs {[(p.offset) for p in p2.properties]})")
        if list(p1.property_dict.items()) != list(p2.property_dict.items()):
            why.append("(c) property_dict")
        if not (np.array_equal(p1._bases, p2._bases) and p1._bases.dtype == p2._bases.dtype):
            why.append(f"(c) _bases {p1._bases.dtype} vs {p2._bases.dtype}")
        if not np.array_equal(p1._radix_powers, p2._radix_powers):
            why.append("(c) _radix_powers")
    print(f"{label:55s} {'REFUSED ' + str(why) if why else 'admitted'}")

P = Property
print("fresh StateIndexer vars:", sorted(FRESH_IX))
print("fresh PropertySet vars:", sorted(FRESH_PS))

ix = StateIndexer(lineage=[P('n', 4), P('pop', 2, min_value=1)]); check(ix, "plain kwargs indexer")
ix.lineage.index_to_props(3); ix.index_to_props(3) if hasattr(ix, 'index_to_props') else None
check(ix, "same, after index_to_props")
check(StateIndexer(property_sets=[P("n", 4)]), "legacy property_sets=[Property] ('default')")
check(StateIndexer('epoch', lineage=[P('n', 4)]), "positional slot")
check(StateIndexer(lineage=[P('n', 4)], slots=[Slot('b')]), "slots=[Slot]")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    check(StateIndexer(property_sets=[StateSpace([P('n', 4)])]), "StateSpace-based")
a = StateIndexer('epoch', lineage=[P('n', 4)]); b = StateIndexer(meta=[P('t', 3)])
check(a + b, "a + b (interleaved slot)")
check(a.append(b, prefix_other='o_'), "append prefix")
# add_epoch construction (src/phasic/__init__.py:3158)
base = StateIndexer(lineage=[P('n', 4)])
check(StateIndexer(*[s.name for s in base.slots()], 'epoch', property_sets=list(base.property_sets())),
      "add_epoch-style (shared PropertySets)")
# joint_prob_graph construction (:10700-10727)
from collections import OrderedDict
ps0 = base.property_sets()[0]
kw = OrderedDict((f'{ps0.name}_{p.name}', [P(f'{ps0.name}_{p.name}', min_value=p.min_value, max_value=p.max_value)]) for p in ps0.properties)
check(base + StateIndexer(**kw), "joint_prob-style base + reward")
# HexGrid
from phasic import HexGrid
try:
    g = HexGrid(3, 3)
    check(StateIndexer(lineage=[g.row_property, g.col_property, P('n', 2)]), "HexGrid row/col properties")
except Exception as e:
    print("HexGrid probe failed:", type(e).__name__, e)
# Edge cases
check(StateIndexer(lineage=(P('n', 4), P('m', 2))), "properties given as a TUPLE")
check(StateIndexer(lineage=[P('n', 3, offset=1)]), "Property(offset=1) (deprecated arg)")
check(StateIndexer(lineage=[P('n', np.int64(4))]), "np.int64 max_value")
check(StateIndexer(property_sets=[PropertySet('a', [P('n', 2)]), PropertySet('slots', [P('m', 2)])]),
      "PropertySet named 'slots'")

print("\n-- builtin arm facts --")
for name, f in [("np.exp", np.exp), ("np.sum", np.sum), ("np.linalg.solve", np.linalg.solve),
                ("math.sqrt", math.sqrt), ("len", len),
                ("itertools.cwr", itertools.combinations_with_replacement)]:
    print(f"{name:16s} type={type(f).__name__:28s} __module__={getattr(f,'__module__',None)!r:16} "
          f"__qualname__={getattr(f,'__qualname__',None)!r:14} __self__={type(getattr(f,'__self__',None)).__name__}")
print("cwr immutable flag:", bool(itertools.combinations_with_replacement.__flags__ & (1 << 8)))
print("Property immutable flag:", bool(Property.__flags__ & (1 << 8)))
