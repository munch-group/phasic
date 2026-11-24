"""Review 4 probes for bug5-graph-cache-plan.md v2 (run on master code).

Run with HOME pointed at a scratch dir and PHASIC_CACHE_DIR unset.
"""
import os, sys, time, hashlib, importlib, textwrap, tempfile, pathlib, json
assert "phasic_cache" not in os.environ.get("PHASIC_CACHE_DIR", ""), "unset PHASIC_CACHE_DIR"
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]

SCR = pathlib.Path(os.environ["HOME"])

def section(t):
    print("\n=== " + t)

# ---------------------------------------------------------------
section("A. build-id cost: hash every *.py + phasic_pybind in installed pkg")
t0 = time.perf_counter()
import phasic
t_imp = time.perf_counter() - t0
pkg = pathlib.Path(phasic.__file__).parent
t0 = time.perf_counter()
h = hashlib.sha256()
files = sorted([p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts])
from phasic import phasic_pybind
so = pathlib.Path(phasic_pybind.__file__)
nbytes = 0
for p in files + [so]:
    b = p.read_bytes(); nbytes += len(b)
    h.update(str(p.relative_to(pkg)).encode()); h.update(b)
t_bid = time.perf_counter() - t0
print(f"import phasic {t_imp:.3f}s; build id over {len(files)} py + so = {nbytes/1e6:.2f} MB in {t_bid*1000:.1f} ms")
print("pybind file:", so.name, "inside pkg dir:", so.parent == pkg)

# ---------------------------------------------------------------
section("B. PTDConfigError vs ValueError for unknown configure() option")
from phasic.exceptions import PTDConfigError
print("PTDConfigError subclass of ValueError:", issubclass(PTDConfigError, ValueError))
try:
    phasic.configure(no_such_option=True)
except Exception as e:
    print("configure(no_such_option) raised", type(e).__name__, "| isinstance ValueError:", isinstance(e, ValueError))

# ---------------------------------------------------------------
section("C. AST key cannot see __defaults__ / closure cells (plan item 5 = same AST rule)")
from phasic.callback_hash import _normalize_ast
import ast, inspect

def ast_key(f):
    f = getattr(f, "__wrapped__", f)
    return hashlib.sha256(ast.dump(_normalize_ast(ast.parse(textwrap.dedent(inspect.getsource(f))))).encode()).hexdigest()[:16]

cbs_default = []
for n in (1.0, 5.0):
    def cb(state, rate=n):          # the classic loop-capture idiom
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], k * rate]]
    cbs_default.append(cb)
print("defaults idiom: __defaults__", [c.__defaults__ for c in cbs_default],
      "AST keys equal:", ast_key(cbs_default[0]) == ast_key(cbs_default[1]))

def make(rate):
    def cb(state):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], k * rate]]
    return cb
c1, c2 = make(1.0), make(5.0)
print("closure: cells", [c.__closure__[0].cell_contents for c in (c1, c2)],
      "AST keys equal:", ast_key(c1) == ast_key(c2))
g1 = phasic.Graph(c1, ipv=[4], graph_cache=False)
g2 = phasic.Graph(c2, ipv=[4], graph_cache=False)
print("  true expectations differ:", g1.expectation(), g2.expectation())

# ---------------------------------------------------------------
section("D. getsource reads the file on disk, not the running code")
moddir = SCR / "mods"; moddir.mkdir(exist_ok=True)
(moddir / "r4mod.py").write_text(textwrap.dedent("""
    def cb(state):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], 1.0 * k]]
"""))
sys.path.insert(0, str(moddir))
import r4mod
running_before = r4mod.cb.__code__.co_consts
key_before = ast_key(r4mod.cb)
(moddir / "r4mod.py").write_text(textwrap.dedent("""
    def cb(state):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], 7.0 * k]]
"""))
import linecache; linecache.checkcache()
key_after = ast_key(r4mod.cb)
print("running code unchanged:", r4mod.cb.__code__.co_consts == running_before,
      "| AST key changed after edit (no reload):", key_before != key_after)
print("  -> the running (old) code would build a graph saved under the NEW source's key")

# ---------------------------------------------------------------
section("E. _serialize_value (construction_params in the entry) vs plan's fingerprintable types")
import numpy as np
from phasic.graph_cache import _serialize_value
for name, v in [("np.bool_", np.bool_(True)), ("complex", 1+2j), ("bytes", b"ab"),
                ("np.complex128", np.complex128(1)), ("complex array", np.array([1j])),
                ("np.str_", np.str_("a")), ("nan", float("nan"))]:
    try:
        s = _serialize_value(v)
        json.dumps({"x": s})
        print(f"  {name}: serializable")
    except Exception as e:
        print(f"  {name}: FAILS {type(e).__name__}: {str(e)[:70]}")

# ---------------------------------------------------------------
section("F. master: does cache_key= reach the callback as an unknown kwarg?")
os.environ.pop("PHASIC_DISABLE_GRAPH_CACHE", None)
def cbk(state):
    k = state[0]
    if k <= 1:
        return []
    return [[[k - 1], 1.0 * k]]
try:
    phasic.Graph(cbk, ipv=[3], cache_key="x")
    print("  no error")
except Exception as e:
    print("  raised", type(e).__name__, str(e)[:100])
def cbkw(state, **kw):
    k = state[0]
    if k <= 1:
        return []
    return [[[k - 1], 1.0 * k]]
g = phasic.Graph(cbkw, ipv=[3], cache_key="x")
print("  with **kw: built,", g.vertices_length(), "vertices; files:",
      len(list((SCR / ".phasic_cache" / "graphs").glob("*.json"))))

# ---------------------------------------------------------------
section("G. kwargs popped before the key: cache_trace / hierarchical never reach it")
src = inspect.getsource(phasic.Graph.__init__)
i_pop = src.find("kwargs.pop('cache_trace'"); i_load = src.find("load_graph(")
print("  pop happens before load_graph:", 0 <= i_pop < i_load)

# ---------------------------------------------------------------
section("H. hit path bypasses the 'ipv with @callback wrapper' ValueError (master)")
@phasic.callback([4])
def wcb(state):
    k = state[0]
    if k <= 1:
        return []
    return [[[k - 1], 2.0 * k]]
try:
    phasic.Graph(wcb, ipv=[4], graph_cache=False)
    print("  uncached: no error")
except ValueError as e:
    print("  uncached: ValueError:", str(e)[:80])
phasic.Graph(wcb, graph_cache=True)      # populate
try:
    phasic.Graph(wcb, ipv=[9], graph_cache=True)
    print("  cached (warm): NO error, ipv=[9] silently ignored")
except ValueError as e:
    print("  cached: ValueError:", str(e)[:80])

# cleanup cache files written by this probe
import shutil
shutil.rmtree(SCR / ".phasic_cache", ignore_errors=True)
shutil.rmtree(moddir, ignore_errors=True)
print("\ncleaned")
