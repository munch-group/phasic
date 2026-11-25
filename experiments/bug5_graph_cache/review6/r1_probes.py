"""Review 6 probes for plan v2.2 (master code; HOME redirected)."""
import os, sys, json, math, operator, functools, random, time, tempfile, pathlib, traceback
assert "scratchpad" in os.environ["HOME"]
assert not os.environ.get("PHASIC_CACHE_DIR")
import numpy as np


def sec(t):
    print("\n=== " + t, flush=True)


sec("A. §2.2.1 builtin identity rule applied to candidates")
def resolve(obj):
    mod = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not mod or not name or mod not in sys.modules:
        return None
    cur = sys.modules[mod]
    for part in name.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur
def rule(obj):
    s = getattr(obj, "__self__", None)
    self_ok = s is None or type(s).__name__ == "module"
    r = resolve(obj)
    return self_ok and (r is obj), getattr(obj, "__module__", None), getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None), type(s).__name__
import scipy.special as sps
cands = {
    "len": len, "math.sqrt": math.sqrt, "operator.add": operator.add, "functools.reduce": functools.reduce,
    "np.exp": np.exp, "np.linalg.solve": np.linalg.solve, "np.linspace": np.linspace,
    "np.random.rand": np.random.rand, "random.random": random.random, "{}.get": {}.get,
    "frompyfunc": np.frompyfunc(lambda a: a, 1, 1), "dict.fromkeys": dict.fromkeys, "np.add.reduce": np.add.reduce,
    "sps.gamma": sps.gamma, "sps.binom": sps.binom, "sps.comb(py fn)": sps.comb, "time.time": time.time,
    "np.random.default_rng": np.random.default_rng, "math.comb": math.comb,
}
for k, v in cands.items():
    ok, mod, name, st = rule(v)
    print(f"  {k:22s} admitted={ok!s:5s} type={type(v).__name__:28s} module={mod} name={name} self={st}")

sec("B. read-error classes: invalid UTF-8 in the entry file")
d = pathlib.Path(tempfile.mkdtemp(dir=os.environ["HOME"]))
p = d / "x.json"; p.write_bytes(b'{"format": "2", "x": "\xff\xfe"}')
try:
    with open(p) as f:
        json.load(f)
except Exception as e:
    print("  json.load raised", type(e).__name__, "| is JSONDecodeError:", isinstance(e, json.JSONDecodeError),
          "| is ValueError:", isinstance(e, ValueError), "| is OSError:", isinstance(e, OSError))
p.write_text("[1, 2]")
with open(p) as f:
    v = json.load(f)
try:
    v["format"]
except Exception as e:
    print("  valid JSON list, entry['format'] ->", type(e).__name__)
p.unlink(); d.rmdir()

sec("C. umask has no read-only API")
print("  os.umask signature needs a mask (set-and-return):", os.umask.__doc__.strip().splitlines()[0])

sec("D. test 15: where does a monkeypatched _set_weight_tape fire on a FORMULA-mode hit? (master)")
import phasic
from phasic import Graph
from phasic.graph_cache import GraphCache
def pcb(state, **kw):
    k = state[0]
    if k <= 1:
        return []
    return [[[k - 1], [float(k)]]]
g = Graph(pcb, ipv=[4], graph_cache=True)                 # miss: saves linear-mode entry
g.weight_formula = "c0*t0"
print("  live graph weight_mode:", g._weight_mode)
GraphCache().save_graph(g, g._callback)                  # overwrite the entry with a formula-mode graph
entry = json.loads(next((pathlib.Path(os.environ["HOME"]) / ".phasic_cache" / "graphs").glob("*.json")).read_text())
print("  entry graph_data has weight_formula_tape:", "weight_formula_tape" in json.dumps(entry["graph_data"])[:10**7])
calls = []
orig = Graph._set_weight_tape
def boom(self, *a, **k):
    stack = "".join(traceback.format_stack(limit=6))
    calls.append("from_serialized" in stack)
    raise RuntimeError("boom")
Graph._set_weight_tape = boom
import logging
try:
    h = Graph(pcb, ipv=[4], graph_cache=True)
    print("  construction returned (master swallows); calls inside from_serialized:", calls)
except Exception as e:
    print("  raised", type(e).__name__, "; calls inside from_serialized:", calls)
Graph._set_weight_tape = orig
import shutil
shutil.rmtree(pathlib.Path(os.environ["HOME"]) / ".phasic_cache", ignore_errors=True)
print("\ncleaned")
