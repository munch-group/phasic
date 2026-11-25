"""Review AH p1: prototype plan v5.9 rule 8 (StateIndexer checks a-c + the
creation-time class registry) by monkeypatching state_indexing in-process,
on real indexers; then measure retention over many constructions.
Run with HOME redirected to a scratch dir. Writes nothing to disk."""
import os, sys, gc, copy, weakref, tracemalloc
assert "scratchpad" in os.environ.get("HOME", ""), "HOME must be scratch"
import dataclasses
import numpy as np
import phasic
from phasic import state_indexing as si
from phasic.state_indexing import StateIndexer, PropertySet, Property, Slot, StateSpace

# ---- registry as specified (module-level wrapper rebinding make_dataclass) ----
_real_md = dataclasses.make_dataclass
_GENERATED = weakref.WeakSet()
def _md_wrapper(cls_name, fields, *a, **k):
    cls = _real_md(cls_name, fields, *a, **k)
    snap = dict(vars(cls))
    names = tuple(f[0] if isinstance(f, tuple) else f for f in fields)
    cls.__phasic_snapshot__ = (snap, names)
    _GENERATED.add(cls)
    return cls
si.make_dataclass = _md_wrapper

class Refused(Exception):
    pass

def check_class(cls, expect_name, expect_fields):
    if type(cls) is not type:
        raise Refused("metaclass")
    if cls not in _GENERATED:
        raise Refused(f"class {cls!r} not registered")
    if cls.__bases__ != (object,):
        raise Refused("bases")
    if cls.__name__ != expect_name:
        raise Refused(f"name {cls.__name__} != {expect_name}")
    d = vars(cls)
    snap, names = d["__phasic_snapshot__"]
    if tuple(names) != tuple(expect_fields):
        raise Refused(f"fields {names} != {expect_fields}")
    keys = set(d) - {"__phasic_snapshot__"}
    if keys != set(snap):
        raise Refused(f"class dict keys differ: +{keys-set(snap)} -{set(snap)-keys}")
    for kk in keys:
        if d[kk] is not snap[kk]:
            raise Refused(f"class entry {kk} replaced")

FRESH = StateIndexer(lineage=[Property("a", max_value=2)], slots=["s"])
IX_ATTRS = set(vars(FRESH))
PS_ATTRS = set(vars(FRESH.lineage)) | {"_props_dataclass"}
SLOT_ATTRS = set(vars(Slot("x")))
PROP_ATTRS = {f.name for f in dataclasses.fields(Property)}

def check_prop(p):
    if type(p) is not Property:
        raise Refused("Property subclass")
    if set(vars(p)) != PROP_ATTRS:
        raise Refused("Property attrs")

def arr_eq(x, y):
    return (type(x) is type(y) is np.ndarray and x.dtype == y.dtype and x.shape == y.shape
            and np.array_equal(x, y))

def check_indexer(ix):
    try:
        # (a)
        if type(ix) is not StateIndexer:
            raise Refused("subclass")
        d = vars(ix)
        for n in ("_property_sets", "_slots", "_offsets"):
            if type(d[n]) is not dict:
                raise Refused(n)
        for n in ("_pset_order", "_slot_order", "_entity_order"):
            if type(d[n]) is not list:
                raise Refused(n)
        for k in d["_property_sets"]:
            if type(k) is not str:
                raise Refused("pset key")
        for k in d["_slots"]:
            if type(k) is not str:
                raise Refused("slot key")
        for k, v in d["_offsets"].items():
            if type(k) is not str or type(v) is not int:
                raise Refused(f"offsets {type(k)} {type(v)}")
        for name, ps in d["_property_sets"].items():
            if type(ps) is not PropertySet and type(ps) is not StateSpace:
                raise Refused("pset class")
            pv = vars(ps)
            if type(pv["property_dict"]) is not dict:
                raise Refused("property_dict")
            if type(pv["properties"]) not in (list, tuple):
                raise Refused("properties type")
            if type(pv["_bases"]) is not np.ndarray or type(pv["_radix_powers"]) is not np.ndarray:
                raise Refused("bases type")
            for k in pv["property_dict"]:
                if type(k) is not str:
                    raise Refused("pd key")
            for p in pv["properties"]:
                check_prop(p)
            for p in pv["property_dict"].values():
                check_prop(p)
            # (b)
            if not set(pv) <= PS_ATTRS:
                raise Refused(f"pset extra attrs {set(pv)-PS_ATTRS}")
        for k, s in d["_slots"].items():
            if type(s) is not Slot or set(vars(s)) != SLOT_ATTRS or s.name != k:
                raise Refused("slot")
        if not set(d) <= IX_ATTRS:
            raise Refused(f"indexer extra attrs {set(d)-IX_ATTRS}")
        if not IX_ATTRS <= set(d):
            raise Refused(f"indexer missing attrs {IX_ATTRS-set(d)}")
        td = ix.to_dict()  # would be fingerprinted here
        # (c)
        c = StateIndexer.from_dict(td)
        cd = vars(c)
        for n in ("_offsets", "_total_state_length", "_pset_order", "_slot_order", "_entity_order"):
            if d[n] != cd[n]:
                raise Refused(f"{n} differs from rebuild")
        for name, ps in d["_property_sets"].items():
            cps = cd["_property_sets"][name]
            t = lambda q: [(p.name, p.min_value, p.max_value) for p in q]
            if t(ps.properties) != t(cps.properties):
                raise Refused("properties differ from rebuild")
            if [(k, (p.name, p.min_value, p.max_value)) for k, p in ps.property_dict.items()] != \
               [(k, (p.name, p.min_value, p.max_value)) for k, p in cps.property_dict.items()]:
                raise Refused("property_dict differs")
            if not arr_eq(ps._bases, cps._bases) or not arr_eq(ps._radix_powers, cps._radix_powers):
                raise Refused("bases differ")
            pdc = vars(ps).get("_props_dataclass")
            if pdc is not None:
                check_class(pdc, f"{ps.name.capitalize()}Props", [p.name for p in ps.properties])
        check_class(d["_result_class"], "IndexResult", list(d["_pset_order"]) + list(d["_slot_order"]))
        return "ADMIT"
    except Refused as e:
        return f"REFUSE: {e}"
    except Exception as e:
        return f"REFUSE(wrapped {type(e).__name__}: {e})"

def used(ix):
    for ps in ix.property_sets():
        ps.index_to_props(0)
    ix.index_to_props(0)
    return ix

cases = {}
cases["plain fresh"] = StateIndexer(lineage=[Property("n", max_value=4), Property("p", max_value=2, min_value=1)])
cases["plain used"] = used(StateIndexer(lineage=[Property("n", max_value=4)], slots=["e"]))
cases["statespace"] = used(StateIndexer(property_sets=[StateSpace([Property("n", max_value=3)], name="ss")]))
a = StateIndexer("ep", lineage=[Property("n", max_value=3)])
b = StateIndexer(rew=[Property("r", max_value=2)])
cases["a+b interleaved slot"] = used(a + b)
cases["append prefix"] = used(a.append(StateIndexer(lineage=[Property("n", max_value=1)]), prefix_other="o_"))
cases["tuple props"] = used(StateIndexer(property_sets=[PropertySet("t", (Property("x", max_value=2),))]))
cases["deprecated offset"] = used(StateIndexer(lineage=[Property("x", max_value=2, offset=3)]))
base = StateIndexer(lineage=[Property("n", max_value=3), Property("d", max_value=2)])
cases["add_epoch style shared psets"] = used(StateIndexer("epoch", property_sets=list(base.property_sets())))
cases["deepcopy"] = copy.deepcopy(used(StateIndexer(lineage=[Property("n", max_value=3)])))
cases["copy.copy"] = copy.copy(StateIndexer(lineage=[Property("n", max_value=3)]))
cases["np.int64 max_value"] = StateIndexer(lineage=[Property("n", max_value=np.int64(3))])
cases["float max_value"] = StateIndexer(lineage=[Property("n", max_value=3.0)])
cases["bool max_value"] = StateIndexer(lineage=[Property("n", max_value=True)])
# Negative controls
e1 = StateIndexer(lineage=[Property("n", max_value=3)]); e1.lineage.properties.append(Property("z", max_value=1))
cases["NEG list appended"] = e1
e2 = used(StateIndexer(lineage=[Property("n", max_value=3)]))
object.__setattr__(e2, "_result_class", dataclasses.make_dataclass("IndexResult", [("lineage", object, None)], frozen=True))
cases["NEG unregistered result class"] = e2
e3 = used(StateIndexer(lineage=[Property("n", max_value=3)]))
e3._result_class.__init__ = lambda self, **k: None
cases["NEG __init__ replaced"] = e3
e4 = used(StateIndexer(lineage=[Property("n", max_value=3)]))
e4._result_class.__post_init__ = lambda self: None
cases["NEG __post_init__ added"] = e4

for k, v in cases.items():
    print(f"{k:32s} {check_indexer(v)}")

# Joint-prob graph indexer from real phasic code
@phasic.callback([[[3, 0], 1.0]])
def coal(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1, 0], [n * (n - 1) / 2]]]
try:
    ix = StateIndexer(lineage=[Property("n", max_value=3), Property("d", max_value=1)])
    print("real-graph indexer before build", check_indexer(ix))
except Exception as e:
    print("graph setup err", e)

# ---- retention ----
def churn(N):
    for i in range(N):
        bi = StateIndexer(lineage=[Property("n", max_value=5 + i % 3), Property("d", max_value=2)], slots=["epoch"])
        ri = StateIndexer(lineage_n=[Property("lineage_n", max_value=4)], lineage_d=[Property("lineage_d", max_value=2)])
        j = bi + ri
        used(bi); used(j)
        r = check_indexer(j)
        assert r == "ADMIT", r
gc.collect()
tracemalloc.start()
churn(50); gc.collect()
s0 = tracemalloc.get_traced_memory()[0]; n0 = len(_GENERATED)
churn(1000); gc.collect()
s1 = tracemalloc.get_traced_memory()[0]; n1 = len(_GENERATED)
print(f"retention over 1000 joint-pair constructions: +{(s1-s0)/1e6:.3f} MB traced, live registered {n0}->{n1}")
