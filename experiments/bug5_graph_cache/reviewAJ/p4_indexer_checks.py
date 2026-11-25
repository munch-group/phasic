"""reviewAJ p4: a natural implementation of rule 8 (a)-(c) on ordinary indexers, incl. dill / deepcopy / pickle
round trips and an indexer whose PropertySet is shared with a second indexer."""
import os, copy, pickle
assert "scratchpad" in os.environ["HOME"]
import numpy as np
from phasic.state_indexing import StateIndexer, PropertySet, Property, Slot, StateSpace

PFIELDS = {"name", "max_value", "min_value", "offset"}

def arr_eq(a, b):
    return type(a) is type(b) is np.ndarray and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)

def check(ix, allow_ix, allow_ps):
    probs = []
    if type(ix) is not StateIndexer: probs.append("type")
    for nm in ("_property_sets", "_slots", "_offsets"):
        if type(getattr(ix, nm)) is not dict: probs.append(nm + " not dict")
    for nm in ("_pset_order", "_slot_order", "_entity_order"):
        if type(getattr(ix, nm)) is not list: probs.append(nm + " not list")
    if not set(vars(ix)) <= allow_ix: probs.append(("ix extra", set(vars(ix)) - allow_ix))
    c = StateIndexer.from_dict(ix.to_dict())
    for k, ps in ix._property_sets.items():
        if type(k) is not str: probs.append("key")
        if type(ps) not in (PropertySet, StateSpace): probs.append("ps type")
        if not set(vars(ps)) <= allow_ps: probs.append(("ps extra", set(vars(ps)) - allow_ps))
        if type(ps.properties) not in (list, tuple): probs.append("props type")
        for p in list(ps.properties) + list(ps.property_dict.values()):
            if type(p) is not Property or set(vars(p)) != PFIELDS: probs.append("prop")
        cps = c._property_sets[k]
        if [(p.name, p.min_value, p.max_value) for p in ps.properties] != [(p.name, p.min_value, p.max_value) for p in cps.properties]:
            probs.append("triples")
        if sorted((kk, p.name, p.min_value, p.max_value) for kk, p in ps.property_dict.items()) != \
           sorted((kk, p.name, p.min_value, p.max_value) for kk, p in cps.property_dict.items()):
            probs.append("pdict")
        if not arr_eq(ps._bases, cps._bases) or not arr_eq(ps._radix_powers, cps._radix_powers):
            probs.append("bases")
    for k, s in ix._slots.items():
        if type(s) is not Slot or set(vars(s)) != set(vars(Slot("x"))) or s.name != k: probs.append("slot")
    for nm in ("_offsets", "_total_state_length", "_pset_order", "_slot_order", "_entity_order"):
        if getattr(ix, nm) != getattr(c, nm): probs.append(nm + " differs")
    return probs or "PASS"

base = StateIndexer("epoch", lineage=[Property("descendants", max_value=4), Property("pop", max_value=2, min_value=1)])
ALLOW_IX = set(vars(base))
ALLOW_PS = set(vars(base.lineage)) | {"_props_dataclass"}
base.index_to_props(3)
print("fresh+used:", check(base, ALLOW_IX, ALLOW_PS))
try:
    import dill
    d = dill.loads(dill.dumps(base))
    print("dill:", check(d, ALLOW_IX, ALLOW_PS), "| index_to_props works:", d.index_to_props(3))
except Exception as e:
    print("dill raised", type(e).__name__, e)
print("deepcopy:", check(copy.deepcopy(base), ALLOW_IX, ALLOW_PS))
try:
    print("pickle:", check(pickle.loads(pickle.dumps(base)), ALLOW_IX, ALLOW_PS))
except Exception as e:
    print("pickle raised", type(e).__name__, str(e)[:120])
# shared PropertySet between two indexers (public constructor accepts PropertySet objects)
ps = PropertySet("lineage", [Property("descendants", max_value=4)])
a = StateIndexer(property_sets=[ps])
b = StateIndexer(property_sets=[ps], slots=["epoch"])
print("shared pset a:", check(a, ALLOW_IX, ALLOW_PS), "b:", check(b, ALLOW_IX, ALLOW_PS))
# append / + / joint-style
c2 = (base + StateIndexer(rew=[Property("r", max_value=3)]))
print("a+b:", check(c2, ALLOW_IX, ALLOW_PS))
