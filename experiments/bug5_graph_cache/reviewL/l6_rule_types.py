"""reviewL l6: how v3.7's exact-type rules classify some ordinary objects.

Pure classification against the rules as written in the plan (no phasic key
code exists yet): builtin arm = type is BuiltinFunctionType / np.ufunc /
type(np.sum), __self__ None or module, resolves by module+qualname to itself;
immutable type arm = isinstance(type) and __flags__ & (1<<8) and resolves.
"""
import sys, types, importlib, collections, functools
import numpy as np

IMMUT = 1 << 8


def resolves(obj):
    mod = getattr(obj, "__module__", None)
    qn = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if mod is None or qn is None or mod not in sys.modules:
        return False
    cur = sys.modules[mod]
    for part in qn.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return False
    return cur is obj


def builtin_arm(x):
    if not (type(x) is types.BuiltinFunctionType or type(x) is np.ufunc
            or type(x) is type(np.sum)):
        return f"not builtin arm (type {type(x).__module__}.{type(x).__qualname__})"
    s = getattr(x, "__self__", None)
    if not (s is None or isinstance(s, types.ModuleType)):
        return f"REFUSED (__self__ is {type(s).__name__})"
    return "ADMITTED by name" if resolves(x) else "REFUSED (does not resolve)"


def type_arm(x):
    if not isinstance(x, type):
        return "not a class"
    if not (x.__flags__ & IMMUT):
        return "REFUSED (mutable heap type)"
    return "ADMITTED by name" if resolves(x) else "REFUSED (does not resolve)"


print("-- compiled functions keyed by name (Cython binding=False style) --")
cands = []
for modname, attr in [("numpy.random._generator", "default_rng"),
                      ("numpy.random.mtrand", "seed"),
                      ("pandas._libs.lib", "is_scalar"),
                      ("pandas._libs.lib", "is_list_like"),
                      ("scipy.special._ufuncs", "gamma")]:
    try:
        m = importlib.import_module(modname)
        cands.append((f"{modname}.{attr}", getattr(m, attr)))
    except Exception as e:
        print(f"{modname}.{attr}: import failed {type(e).__name__}")
for name, obj in cands:
    print(f"{name}: {builtin_arm(obj)}")

print("-- ordinary values a callback may capture or receive --")
P = collections.namedtuple("P", "a b")
print("namedtuple instance: exact tuple?", type(P(1, 2)) is tuple)
mm = np.memmap("/dev/null", dtype="u1", mode="r", shape=(0,)) if False else None
print("np.memmap is ndarray subclass:", issubclass(np.memmap, np.ndarray) and np.memmap is not np.ndarray)
for cls in [collections.Counter, collections.defaultdict, collections.OrderedDict,
            collections.deque, functools.partial]:
    print(f"class {cls.__module__}.{cls.__qualname__}: {type_arm(cls)}")

from phasic.state_indexing import StateIndexer, Property
ix = StateIndexer(lineages=[Property("ton", min_value=1, max_value=4)])
p = ix.lineages.index_to_props(2)
print("index_to_props result type:", type(p).__module__, type(p).__qualname__,
      "-> rule 9 (refused)")
