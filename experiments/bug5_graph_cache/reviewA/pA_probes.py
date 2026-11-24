"""Review A probes (read-only w.r.t. src). Run with HOME=<scratch>."""
import os, sys, itertools, functools, math, types
import numpy as np
print("HOME =", os.environ.get("HOME"), " PHASIC_CACHE_DIR =", os.environ.get("PHASIC_CACHE_DIR"))
assert "scratchpad" in os.environ["HOME"]

print("\n[A] partial(combinations_with_replacement) func is a class:")
cwr = itertools.combinations_with_replacement
print("   isinstance(cwr, type) =", isinstance(cwr, type), "; partial.func is cwr:",
      functools.partial(cwr, r=2).func is cwr)

print("\n[B] builtin/ufunc/dispatcher attributes:")
def res(obj):
    mod = getattr(obj, "__module__", None); qn = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    m = sys.modules.get(mod) if isinstance(mod, str) else None
    o = m
    try:
        for part in (qn or "").split("."):
            o = getattr(o, part)
    except Exception as e:
        o = f"<err {e!r}>"
    return mod, qn, (o is obj)
for name, obj in [("len", len), ("math.sqrt", math.sqrt), ("np.exp", np.exp),
                  ("np.linalg.solve", np.linalg.solve), ("np.sum", np.sum), ("np.where", np.where),
                  ("np.zeros", np.zeros), ("np.array", np.array), ("np.isclose", np.isclose)]:
    print(f"   {name:16s} type={type(obj).__name__:24s} __self__={type(getattr(obj,'__self__',None)).__name__:10s} (module, qualname, resolves) = {res(obj)}")

print("\n[C] structured dtype: dtype.str vs str(dtype), identical bytes:")
a = np.zeros(2, dtype=[("x", "<i8"), ("y", "<f8")]); b = np.zeros(2, dtype=[("p", "<f8"), ("q", "<i8")])
print("   a.dtype.str =", a.dtype.str, " b.dtype.str =", b.dtype.str, " same bytes:", a.tobytes() == b.tobytes())
print("   str(a.dtype) =", str(a.dtype), "| str(b.dtype) =", str(b.dtype))

print("\n[D] int repr limit:")
try:
    repr(10**5000); print("   repr ok")
except ValueError as e:
    print("   repr(10**5000) raises ValueError:", str(e)[:70])

print("\n[E] clear_caches() with DEFAULT_CACHE_DIR monkeypatched only:")
home = os.environ["HOME"]
for d in (".phasic_cache/other", ".jax_cache"):
    os.makedirs(os.path.join(home, d), exist_ok=True)
    open(os.path.join(home, d, "sentinel"), "w").write("x")
import phasic, phasic.graph_cache as gc
from pathlib import Path
tmp = Path(home) / "redirected_graphs"
gc.DEFAULT_CACHE_DIR = tmp
os.environ.pop("PHASIC_CACHE_DIR", None)
phasic.clear_caches()
print("   ~/.phasic_cache/other/sentinel exists after:", os.path.exists(os.path.join(home, ".phasic_cache/other/sentinel")))
print("   ~/.jax_cache/sentinel exists after:", os.path.exists(os.path.join(home, ".jax_cache/sentinel")))

print("\n[F] module import order: is phasic_pybind imported when graph_cache's body runs?")
import ast
src = open(phasic.__file__).read().splitlines()
for i, l in enumerate(src, 1):
    if "phasic.graph_cache import" in l or l.startswith("from .phasic_pybind import *") or 'os.environ["OMP_NUM_THREADS"] =' in l:
        print(f"   __init__.py:{i}: {l.strip()}")

print("\n[G] Graph callback name check on a partial / callable instance:")
from phasic import Graph
p = functools.partial(lambda state, k=1: [], k=2)
try:
    Graph(p, ipv=[1])
except Exception as e:
    print("   Graph(partial, ipv=[1]) ->", type(e).__name__, str(e)[:80])

print("\n[H] int kwarg reaches callback as float (key sees int, graph sees float):")
seen = {}
def cb(state, n=None, flag=None, **kw):
    seen["n"] = n; seen["flag"] = flag
    return [] if state[0] <= 1 else [[[state[0]-1], 1.0]]
Graph(cb, ipv=[3], n=2, flag=True)
print("   callback received n=%r flag=%r" % (seen["n"], seen["flag"]))
