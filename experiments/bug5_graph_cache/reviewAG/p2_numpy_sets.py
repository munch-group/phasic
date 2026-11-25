"""Review AG p2: numpy allow-set construction as specified in rule 2, and
names the plan relies on (numpy, pybind, dataclass internals)."""
import sys, types, functools, itertools
import numpy as np

print("numpy", np.__version__, "python", sys.version.split()[0])
allow = frozenset(t for t in np.sctypeDict.values()
                  if getattr(t, "__module__", None) == "numpy")
allow = allow - {np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set:", sorted(t.__name__ for t in allow))
print("float64 in set:", np.float64 in allow, " longdouble is float64:",
      np.longdouble is np.float64)
print("type(np.sum):", type(np.sum), type(np.sum).__module__)
print("np.exp __self__:", getattr(np.exp, "__self__", "MISSING"))
print("immutable flag itertools.cwr:",
      bool(itertools.combinations_with_replacement.__flags__ & (1 << 8)))
# metaclass of numpy scalar classes: must be exactly `type` for the
# type(type(x)) is type pre-check to admit numpy scalars / arrays
for t in (np.float64, np.int64, np.bool_, np.str_, np.datetime64, np.ndarray):
    print(f"type({t.__name__}) =", type(t))
x = np.float64(1.0)
print("type(type(np.float64(1))) is type:", type(type(x)) is type)
print("type(type(np.zeros(1))) is type:", type(type(np.zeros(1))) is type)
print("type(type(functools.partial(len))) is type:",
      type(type(functools.partial(len))) is type)
print("type(type(np.exp)) is type:", type(type(np.exp)) is type)
import phasic.state_indexing as si
print("type(type(Property)) is type:", type(si.Property) is type)
