"""Review 5 probes of the plan v2.1 code/function fingerprint (prototype in fp_proto.py).
Usage: python q1_fp_probes.py            (main run; spawns subprocesses)
       python q1_fp_probes.py child      (prints fingerprints only)"""
import os, sys, subprocess, pathlib, time, random, functools, itertools, types
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
assert "scratchpad" in os.environ["HOME"]
from fp_proto import fingerprint, NotFP, func_fp
import q1_mod


def wrapped():
    import phasic
    return phasic.with_ipv([[[5], 1.0]])(q1_mod.cb)


def child():
    import phasic  # noqa
    print("cb", fingerprint(q1_mod.cb))
    print("wrapped", fingerprint(wrapped()))
    print("closure", fingerprint(q1_mod.make(2.5)))
    print("frozenset_iter_order", list(q1_mod.cb.__code__.co_consts[[type(c) for c in q1_mod.cb.__code__.co_consts].index(frozenset)]))


def sec(t):
    print("\n=== " + t, flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(); sys.exit(0)

sec("A. cross-process stability (PYTHONHASHSEED 0/1/2/random)")
outs = []
for seed in ("0", "1", "2", "random"):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    r = subprocess.run([sys.executable, __file__, "child"], env=env, capture_output=True, text=True)
    outs.append(r.stdout.strip().splitlines())
    print(seed, r.stdout.strip().replace("\n", " | "), r.stderr[-300:] if r.returncode else "")
fps = [[l for l in o if not l.startswith("frozenset_iter_order")] for o in outs]
print("fingerprints identical across seeds:", all(f == fps[0] for f in fps))

sec("B. 3.13 specialisation / instrumentation: co_code before vs after hot calls")
cb = q1_mod.cb
before_code = cb.__code__.co_code
fp0 = fingerprint(cb)
for _ in range(20000):
    cb([5], 1.0)
print("co_code unchanged after 20k calls:", cb.__code__.co_code == before_code, "| fp unchanged:", fingerprint(cb) == fp0)
# raw (adaptive) bytes differ?  _co_code_adaptive is private
adaptive = getattr(cb.__code__, "_co_code_adaptive", None)
print("adaptive bytecode differs from co_code (specialised):", adaptive is not None and adaptive != cb.__code__.co_code)
calls = []
sys.settrace(lambda *a: None)
try:
    cb([5], 1.0)
    print("under settrace: fp unchanged:", fingerprint(cb) == fp0)
finally:
    sys.settrace(None)
mon = sys.monitoring
mon.use_tool_id(3, "r5")
mon.set_events(3, mon.events.LINE | mon.events.PY_START)
mon.register_callback(3, mon.events.LINE, lambda *a: None)
cb([5], 1.0)
print("under sys.monitoring LINE: fp unchanged:", fingerprint(cb) == fp0)
mon.set_events(3, 0); mon.free_tool_id(3)

sec("C. the _callback wrapper closure")
w = wrapped()
print("wrapper freevars:", w.__code__.co_freevars,
      "cell types:", [type(c.cell_contents).__name__ for c in w.__closure__],
      "defaults:", w.__defaults__)
print("wrapper co_names (globals it reads):", w.__code__.co_names)

sec("D. builtin bound methods: module+qualname ignores __self__")
import fp_proto
print("(a) value rules as written (rule 6 has no builtin arm):")
try:
    fingerprint((lambda look: (lambda s: look("r")))({}.get)); print("  builtin in cell: fingerprintable")
except NotFP as e:
    print("  builtin in cell: NOT FINGERPRINTABLE", e)
print("(b) builtin cell routed to the §2.2.1 builtin arm:")
fp_proto.BUILTIN_AS_FUNCTION = True
d1, d2 = {"r": 1.0}, {"r": 9.0}
mk = lambda look: (lambda state: [[[state[0] - 1], look("r")]] if state[0] > 1 else [])
print("d1.get vs d2.get type:", type(d1.get).__name__, "| fingerprints equal:",
      fingerprint(mk(d1.get)) == fingerprint(mk(d2.get)))
r1, r2 = random.Random(1), random.Random(2)
print("Random(1).random vs Random(2).random equal:", fingerprint(mk(r1.random)) == fingerprint(mk(r2.random)))
print("np.frompyfunc ufunc name:", np.__name__ if False else __import__("numpy").frompyfunc(lambda a: a + 1, 1, 1).__name__)

sec("E. common closure contents that are refused")
def try_fp(label, f):
    try:
        fingerprint(f); print(f"  {label}: fingerprintable")
    except NotFP as e:
        print(f"  {label}: NOT FINGERPRINTABLE ({e})")
    except RecursionError:
        print(f"  {label}: RecursionError")
    except Exception as e:
        print(f"  {label}: ESCAPES as {type(e).__name__}: {e}")
@functools.lru_cache
def helper(k): return k * 1.0
def m1():
    h = helper
    return lambda s: [[[s[0] - 1], h(s[0])]] if s[0] > 1 else []
try_fp("lru_cache helper in cell", m1())
def m2():
    from itertools import combinations
    return lambda s: list(combinations(s, 2)) and []
try_fp("class (itertools.combinations) in cell", m2())
class Base:
    def cb(self, s): return []
class Model(Base):
    def to_dict(self): return {"x": 1}
    def cb(self, s): return super().cb(s)
try_fp("bound method using super() (__class__ cell)", Model().cb)
import numpy as np, math
def m3(fn):
    return lambda s: [[[s[0] - 1], fn(1.0)]] if s[0] > 1 else []
for lab, fn in [("np.exp (ufunc)", np.exp), ("np.linspace", np.linspace), ("math.comb", math.comb)]:
    try_fp(f"{lab} in cell (value rule has no builtin arm -> per proto)", m3(fn))
try:
    import scipy.special as sps, scipy.stats as st
    try_fp("scipy.special.comb in cell", m3(sps.comb))
    try_fp("scipy.stats.binom.pmf in cell", m3(st.binom.pmf))
except ImportError:
    print("  scipy not installed")

sec("F. recursion: self/mutual recursive closures; backref with/without index")
def m4():
    def a(s): return b(s)
    def b(s): return a(s) if s else []
    return a
try_fp("mutual recursion", m4())
def m5():
    def cb(s): return cb(s[1:]) if len(s) > 1 else []
    return cb
try_fp("self recursion", m5())

sec("G. edits: docstring / comment / whitespace; two lambdas on one line")
src_a = "def cb(state):\n    '''doc A'''\n    return [[[state[0]-1], 2.0]] if state[0] > 1 else []\n"
src_b = "def cb(state):\n    '''doc B'''\n    return [[[state[0]-1], 2.0]] if state[0] > 1 else []\n"
src_c = "\n\n# a comment\ndef cb(state):\n    '''doc A'''\n    # inner comment\n    return [[[state[0]-1],   2.0]] if state[0] > 1 else []\n"
src_d = "def cb(state):\n    '''doc A'''\n    return [[[state[0]-1], 3.0]] if state[0] > 1 else []\n"
def build(src, fn="<x>"):
    ns = {}; exec(compile(src, fn, "exec"), ns); return ns["cb"]
fa, fb, fc, fd = (fingerprint(build(s)) for s in (src_a, src_b, src_c, src_d))
print("docstring edit misses:", fa != fb, "| comment/whitespace edit hits:", fa == fc, "| constant edit misses:", fa != fd)
ns = {}; exec("l1, l2 = (lambda s: 1.0), (lambda s: 2.0)", ns)
print("two lambdas one line distinct:", fingerprint(ns["l1"]) != fingerprint(ns["l2"]))

sec("I. build-id stat record cost")
import phasic
pkg = pathlib.Path(phasic.__file__).parent
t0 = time.perf_counter()
st_ = [(p.relative_to(pkg), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(pkg.rglob("*.py")) if "__pycache__" not in p.parts]
sos = sorted(pkg.glob("phasic_pybind*.so"))
print(f"{len(st_)} py stats in {(time.perf_counter()-t0)*1000:.2f} ms; .so matches: {[s.name for s in sos]}")
print("phasic_pybind.__file__ under pkg:", pathlib.Path(phasic.phasic_pybind.__file__).parent == pkg)
print("non-py, non-so files in pkg (top):", sorted({p.suffix for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts}))
