"""Review AG p1: prototype StateIndexer checks (a)-(c) + the v5.8 creation-time
registry (by monkeypatching state_indexing.make_dataclass; src untouched),
on real indexers.  Also measures whether the registry leaks classes.

Run with HOME redirected (no graph is built here, no cache is touched).
"""
import gc
import sys
import warnings
import weakref

import numpy as np

import phasic.state_indexing as si
from phasic.state_indexing import (StateIndexer, Property, PropertySet,
                                   StateSpace, Slot)

# ---------------------------------------------------------------- registry
_GENERATED_CLASSES = {}
_orig_make_dataclass = si.make_dataclass


def _register(cls, field_names):
    key = id(cls)
    _GENERATED_CLASSES[key] = (weakref.ref(cls), tuple(field_names),
                               dict(vars(cls)))
    weakref.finalize(cls, _GENERATED_CLASSES.pop, key, None)


def _patched_make_dataclass(cls_name, fields, **kw):
    cls = _orig_make_dataclass(cls_name, fields, **kw)
    names = [f if isinstance(f, str) else f[0] for f in fields]
    _register(cls, names)
    return cls


si.make_dataclass = _patched_make_dataclass

PROP_FIELDS = {"name", "max_value", "min_value", "offset"}
SLOT_VARS = set(vars(Slot("x")))


class Refused(Exception):
    pass


def _cls_ok(cls, expected_fields, what):
    if type(cls) is not type:
        raise Refused(f"{what}: metaclass {type(cls)}")
    ent = _GENERATED_CLASSES.get(id(cls))
    if ent is None:
        raise Refused(f"{what}: not registered")
    ref, fields, snap = ent
    if ref() is not cls:
        raise Refused(f"{what}: stale registry entry")
    if tuple(fields) != tuple(expected_fields):
        raise Refused(f"{what}: fields {fields} != {expected_fields}")
    v = vars(cls)
    if set(v.keys()) != set(snap.keys()):
        raise Refused(f"{what}: key set differs: "
                      f"+{set(v) - set(snap)} -{set(snap) - set(v)}")
    for k, obj in snap.items():
        if v[k] is not obj:
            raise Refused(f"{what}: entry {k!r} replaced")


def _prop_ok(p, where):
    if type(p) is not Property:
        raise Refused(f"{where}: Property type {type(p)}")
    if set(vars(p)) != PROP_FIELDS:
        raise Refused(f"{where}: Property vars {set(vars(p))}")


# allow-lists from FRESH objects + documented lazy attribute
_fresh = StateIndexer(lineage=[Property("a", max_value=2)])
IX_ALLOW = set(vars(_fresh))
PS_ALLOW = set(vars(_fresh._property_sets["lineage"])) | {"_props_dataclass"}


def check(ix):
    try:
        # (a)
        if type(ix) is not StateIndexer:
            raise Refused("type")
        for n in ("_property_sets", "_slots", "_offsets"):
            if type(getattr(ix, n)) is not dict:
                raise Refused(f"{n} not dict")
        for n in ("_pset_order", "_slot_order", "_entity_order"):
            if type(getattr(ix, n)) is not list:
                raise Refused(f"{n} not list")
        for k in ix._property_sets:
            if type(k) is not str:
                raise Refused("pset key")
        for k in ix._slots:
            if type(k) is not str:
                raise Refused("slot key")
        for k, v in ix._offsets.items():
            if type(k) is not str or type(v) is not int:
                raise Refused(f"offsets {k!r}:{type(v)}")
        for name, ps in ix._property_sets.items():
            if type(ps) is not PropertySet and type(ps) is not StateSpace:
                raise Refused(f"pset type {type(ps)}")
            if type(ps.properties) not in (list, tuple):
                raise Refused("properties type")
            if type(ps.property_dict) is not dict:
                raise Refused("property_dict type")
            for k in ps.property_dict:
                if type(k) is not str:
                    raise Refused("property_dict key")
            for n in ("_bases", "_radix_powers"):
                if type(getattr(ps, n)) is not np.ndarray:
                    raise Refused(f"{n} type")
            for p in ps.properties:
                _prop_ok(p, f"{name}.properties")
            for p in ps.property_dict.values():
                _prop_ok(p, f"{name}.property_dict")
        for k, s in ix._slots.items():
            if type(s) is not Slot or set(vars(s)) != SLOT_VARS or s.name != k:
                raise Refused(f"slot {k}")
        # (b)
        if not set(vars(ix)) <= IX_ALLOW:
            raise Refused(f"ix extra attrs {set(vars(ix)) - IX_ALLOW}")
        for ps in ix._property_sets.values():
            if not set(vars(ps)) <= PS_ALLOW:
                raise Refused(f"ps extra attrs {set(vars(ps)) - PS_ALLOW}")
        # (c)
        d = ix.to_dict()
        c = StateIndexer.from_dict(d)
        for n in ("_offsets", "_total_state_length", "_pset_order",
                  "_slot_order", "_entity_order"):
            if getattr(ix, n) != getattr(c, n):
                raise Refused(f"(c) {n}: {getattr(ix, n)} vs {getattr(c, n)}")
        if list(ix._property_sets) != list(c._property_sets):
            raise Refused("(c) pset keys order")
        for name, ps in ix._property_sets.items():
            cps = c._property_sets[name]
            tr = lambda seq: [(p.name, p.min_value, p.max_value) for p in seq]
            if tr(ps.properties) != tr(cps.properties):
                raise Refused(f"(c) {name} props")
            trd = lambda dd: [(k, p.name, p.min_value, p.max_value)
                              for k, p in dd.items()]
            if trd(ps.property_dict) != trd(cps.property_dict):
                raise Refused(f"(c) {name} property_dict")
            for n in ("_bases", "_radix_powers"):
                a, b = getattr(ps, n), getattr(cps, n)
                if not (type(a) is type(b) and a.dtype == b.dtype
                        and a.shape == b.shape and np.array_equal(a, b)):
                    raise Refused(f"(c) {name}.{n}: {a!r} vs {b!r}")
            # registry: lazily created _props_dataclass
            if "_props_dataclass" in vars(ps):
                _cls_ok(ps._props_dataclass, [p.name for p in ps.properties],
                        f"{name}._props_dataclass")
        _cls_ok(ix._result_class, list(ix._pset_order) + list(ix._slot_order),
                "_result_class")
        return "PASS"
    except Refused as e:
        return f"REFUSED: {e}"
    except Exception as e:  # would be GraphCacheKeyError
        return f"ERROR->KeyError: {type(e).__name__}: {e}"


def case(label, ix):
    print(f"{label:55s} {check(ix)}")


warnings.simplefilter("ignore", DeprecationWarning)

a = StateIndexer(lineage=[Property("descendants", max_value=5)])
case("fresh plain", a)
a.index_to_props(3)
a.lineage.index_to_props(2)
case("after index_to_props (both generated classes)", a)

b = StateIndexer(lineages=[Property("ton", min_value=1, max_value=4)])
case("test_graph_cache.py indexer", b)
b.lineages.index_to_props(0); b.props_to_index(ton=1)
case("  ... used", b)

f = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=4)])
case("test_from_serialized_find_vertex indexer", f)

ss = StateSpace([Property("x", max_value=3)], name="lin")
s_ix = StateIndexer(property_sets=[ss])
case("StateSpace-based", s_ix)
s_ix.lin.index_to_props(1)
case("  ... used", s_ix)

m = StateIndexer(metadata=[Property("t", max_value=3)])
ab = a + m
case("a + b", ab)
ab.index_to_props(5)
case("  ... used", ab)

e1 = StateIndexer("epoch", lineage=[Property("d", max_value=3)])
e2 = StateIndexer(extra=[Property("q", max_value=2)])
inter = e1 + e2  # interleaved slot
case("a + b with interleaved slot", inter)
case("append(prefix_other=...)", a.append(a, prefix_other="o_"))

dep = StateIndexer(lineage=[Property("d", max_value=3, offset=2)])
case("deprecated offset Property", dep)
tup = StateIndexer(property_sets=[PropertySet("lin", (Property("d", max_value=3),))])
case("tuple of properties", tup)

# add_epoch construction (shares PropertySets with base)
ep = StateIndexer(*[s.name for s in a.slots()], "epoch",
                  property_sets=list(a.property_sets()))
case("add_epoch-style (shared psets)", ep)
# joint_prob_graph-style
from collections import OrderedDict
kw = OrderedDict()
for p in f.property_sets()[0].properties:
    nm = f"{f.property_sets()[0].name}_{p.name}"
    kw[nm] = [Property(nm, min_value=p.min_value, max_value=p.max_value)]
jp = f + StateIndexer(**kw)
case("joint_prob_graph-style base + reward", jp)
case("epoch indexer on joint (chained)", StateIndexer(
    *[s.name for s in jp.slots()], "epoch", property_sets=list(jp.property_sets())))

# canonical rebuild c is itself an indexer: check it too
case("from_dict rebuild", StateIndexer.from_dict(a.to_dict()))

# numpy-int Property values (e.g. from a numpy loop)
case("np.int64 max_value", StateIndexer(lin=[Property("d", max_value=np.int64(3))]))

# ---------------- negative controls of the registry
import dataclasses
neg = StateIndexer(lineage=[Property("d", max_value=3)])
neg._result_class.__init__ = lambda self, **k: None
case("NEG: _result_class.__init__ replaced", neg)
neg2 = StateIndexer(lineage=[Property("d", max_value=3)])
object.__setattr__(neg2, "_result_class",
                   dataclasses.make_dataclass("IndexResult", [("lineage", object, None)], frozen=True))
case("NEG: unregistered same-field class", neg2)

# ---------------- leak measurement
gc.collect()
n0 = len(_GENERATED_CLASSES)
for i in range(500):
    t = StateIndexer(lin=[Property("d", max_value=3)])
    t.lin.index_to_props(1)
    check(t)  # each check builds a canonical rebuild too
    del t
gc.collect()
n1 = len(_GENERATED_CLASSES)
alive = sum(1 for r, _, _ in _GENERATED_CLASSES.values() if r() is not None)
print(f"\nLEAK: registry entries before={n0} after 500 indexers+checks={n1} "
      f"(alive={alive})")
# same loop WITHOUT the registry, counting live IndexResult classes
si.make_dataclass = _orig_make_dataclass
_GENERATED_CLASSES.clear()
gc.collect()
def live_generated():
    return sum(1 for o in gc.get_objects()
               if type(o) is type and o.__name__ in ("IndexResult", "LinProps"))
L0 = live_generated()
for i in range(500):
    t = StateIndexer(lin=[Property("d", max_value=3)])
    t.lin.index_to_props(1)
    del t
gc.collect()
print(f"WITHOUT registry: live generated classes before={L0} after={live_generated()}")
# why: which snapshot entries reference the class?
si.make_dataclass = _patched_make_dataclass
t = StateIndexer(lin=[Property("d", max_value=3)])
cls = t._result_class
snap = _GENERATED_CLASSES[id(cls)][2]
holders = [k for k, v in snap.items() if any(r is cls for r in gc.get_referents(v))]
print("snapshot entries that strongly reference the class:", holders)
