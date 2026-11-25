"""reviewAC p1: numpy allow-set coverage, metaclass of admitted classes,
StateIndexer internals on ordinary constructions. No Graph built, no cache."""
import sys, types, functools, itertools
import numpy as np
print("python", sys.version.split()[0], "numpy", np.__version__)

ALLOW = frozenset(t for t in np.sctypeDict.values()
                  if getattr(t, "__module__", None) == "numpy") - {
    np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set size", len(ALLOW), sorted(t.__name__ for t in ALLOW))

# every builtin dtype char: is dtype.type in the allow-set?
for c in np.typecodes["All"]:
    try:
        dt = np.dtype(c)
    except TypeError:
        continue
    print(f"char {c!r:5} str={dt.str:6} type={dt.type.__name__:12} in_allow={any(dt.type is t for t in ALLOW)}")
for spec in ["datetime64[ns]", "timedelta64[s]", "U5", "S3", "<i8", ">i8", "=i4"]:
    dt = np.dtype(spec)
    print(f"spec {spec:15} type={dt.type.__name__} in_allow={any(dt.type is t for t in ALLOW)}")

# metaclass check on everything the plan admits
from phasic.state_indexing import StateIndexer, Property, PropertySet, StateSpace, Slot
objs = {
  "None": None, "int": 1, "float": 1.0, "str": "a", "list": [], "dict": {},
  "ndarray": np.zeros(2), "float64": np.float64(1), "int64": np.int64(1),
  "bool_": np.bool_(True), "func": (lambda s: s), "partial": functools.partial(int),
  "builtin": len, "ufunc": np.exp, "dispatcher": np.sum, "module": np,
  "Ellipsis": ..., "slice": slice(1, 2), "class cwr": itertools.combinations_with_replacement,
  "Property": Property("a", 3), "StateIndexer": StateIndexer(lineage=[Property("a", 3)]),
  "Slot": Slot("e"), "PropertySet": PropertySet("x", [Property("a", 2)]),
  "StateSpace": StateSpace([Property("a", 2)]), "bound": StateIndexer(lineage=[Property("a", 3)]).index_to_props,
  "complex": 1j, "bytes": b"x", "tuple": (),
}
for k, v in objs.items():
    print(f"metaclass {k:14} type(type(x)) is type: {type(type(v)) is type}")
for t in ALLOW:
    if type(t) is not type:
        print("NON-type metaclass scalar class:", t)

# ndarray dtype of ordinary internal arrays
ps_empty = PropertySet("e", [])
print("empty PropertySet _bases dtype", ps_empty._bases.dtype, "radix", ps_empty._radix_powers.dtype)
ix = StateIndexer("epoch", lineage=[Property("n", max_value=np.int64(4)), Property("p", 2, min_value=1)])
print("np.int64 max_value -> _bases dtype", ix.lineage._bases.dtype, type(ix.lineage._bases))
print("to_dict types:", {k: type(v).__name__ for k, v in ix.to_dict().items()})
print("to_dict pset_order is live list:", ix.to_dict()["pset_order"] is ix._pset_order)
print("offsets", ix._offsets, {k: type(v).__name__ for k, v in ix._offsets.items()})
print("vars(indexer) fresh", sorted(vars(ix)))
ix.index_to_props(0); ix.lineage.index_to_props(0)
print("vars(indexer) used ", sorted(vars(ix)))
print("vars(pset) used    ", sorted(vars(ix.lineage)))
# Property with np.int64 fields: vars and type
p = Property("n", max_value=np.int64(4))
print("Property vars", sorted(vars(p)), type(p.max_value).__name__)
