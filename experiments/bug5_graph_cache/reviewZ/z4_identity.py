"""reviewZ z8/z9 (unmodified master; HOME redirected; no cache_key exists yet,
so builds use graph_cache=False -- this only shows the two calls build
DIFFERENT graphs from inputs whose content-only encoding (plan §2.2.2
"every sub-digest is a pure function of the object's content") is identical.

z8  identity: two content-identical closures used as dict keys.
z9  rule 2 reads x.dtype on a Python subclass of np.float64 before refusing it.
"""
import os, sys
import numpy as np


def z8():
    from phasic import Graph

    def make(c):
        def rate(k):
            return c * k
        return rate

    f, g = make(1.0), make(1.0)          # same code, same closure content, distinct objects
    assert f.__code__ is g.__code__
    assert [x.cell_contents for x in f.__closure__] == [x.cell_contents for x in g.__closure__]
    assert f.__defaults__ == g.__defaults__ and f.__dict__ == g.__dict__ == {}

    def cb(state, scale=None, pick=None, **kw):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], k * scale[pick]]]      # identity-based dict lookup

    scale = {f: 1.0, g: 5.0}
    e1 = Graph(cb, ipv=[5], scale=scale, pick=f, graph_cache=False).expectation()
    e2 = Graph(cb, ipv=[5], scale=scale, pick=g, graph_cache=False).expectation()
    print(f"[z8] pick=f E[T]={e1:.6f}   pick=g E[T]={e2:.6f}   different graphs: {abs(e1 - e2) > 1e-12}")
    print("[z8] content of kwargs identical under a content-only encoding: "
          "scale -> [(D(f),1.0),(D(g),5.0)] with D(f)==D(g); pick -> D(f) vs D(g), equal")


def z9():
    calls = []

    class F(np.float64):
        @property
        def dtype(self):
            calls.append("dtype property ran")
            return np.dtype("f8")

    x = F(1.0)
    gate1 = issubclass(type(x), np.generic)
    exact = type(x) is x.dtype.type        # plan rule 2 step 2, as written
    print(f"[z9] issubclass gate={gate1}; exact check={exact}; user code ran during check: {calls}")
    alt = type(x) is np.dtype(type(x)).type
    print(f"[z9] alternative gate type(x) is np.dtype(type(x)).type -> {alt}; user code ran: {len(calls) > 1}")


if __name__ == "__main__":
    assert "scratchpad" in os.environ.get("HOME", ""), "HOME must be redirected"
    for w in (sys.argv[1:] or ["z8", "z9"]):
        globals()[w]()
