"""reviewM m6: a MODULE passed as a keyword argument (or held in a closure cell).
v3.8 rule 6b keys a module object by its NAME only, admitted iff
sys.modules[m.__name__] is m.  Show that (a) importlib.reload after editing the
file and (b) attribute assignment both keep that admission condition and the
name token unchanged, while the graph the callback builds changes.
Uncached builds only (graph_cache=False); HOME=scratch; module written to scratch."""
import os, sys, importlib, pathlib
assert os.environ["HOME"].startswith("/private/tmp/claude-501/"), os.environ["HOME"]
scratch = pathlib.Path(os.environ["HOME"]).parent / "m6mods"
scratch.mkdir(exist_ok=True)
sys.path.insert(0, str(scratch))
modfile = scratch / "reviewm_params.py"
modfile.write_text("RATE = 1.0\n")
import phasic
from phasic import Graph
import reviewm_params as P

def cb(state, params=None, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], params.RATE * n * (n - 1) / 2]]

def rule6b_token(m):
    assert sys.modules[m.__name__] is m, "rule 6b admission fails"
    return ("module", m.__name__)

def build():
    return float(Graph(cb, ipv=[[[5], 1.0]], params=P, graph_cache=False).expectation())

t0, e0 = rule6b_token(P), build()
print("initial          token", t0, "E[T]", e0)
P.RATE = 4.0
t1, e1 = rule6b_token(P), build()
print("after P.RATE=4   token", t1, "E[T]", e1, "same token:", t0 == t1, "same graph:", e0 == e1)
modfile.write_text("RATE = 10.0\n")
importlib.invalidate_caches()
P2 = importlib.reload(P)
t2, e2 = rule6b_token(P), build()
print("after edit+reload token", t2, "E[T]", e2, "same object:", P2 is P, "same token:", t0 == t2,
      "same graph:", e0 == e2)
for f in scratch.glob("*"):
    if f.is_file():
        f.unlink()
