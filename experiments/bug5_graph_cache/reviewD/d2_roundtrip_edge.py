"""Review D probe 2: graphs a fresh callback build accepts but that do not
survive serialize()->JSON->from_serialized() (= what a cache hit would do).
Unmodified master code; in-memory round trip only, no cache files."""
import os, json, traceback
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np, phasic
from phasic.graph_cache import _serialize_value, _deserialize_value

def rt(g):
    d = json.loads(json.dumps(_serialize_value(g.serialize())))
    return phasic.Graph.from_serialized(_deserialize_value(d))

def show(tag, g):
    try:
        print(tag, "n", g.vertices_length(), "param_length", g.param_length(),
              "parameterized", g.parameterized())
    except Exception as e:
        print(tag, "describe ERR", e)

def case(name, cb, **kw):
    print("==", name)
    try:
        g = phasic.Graph(cb, graph_cache=False, **kw)
    except Exception as e:
        print("  fresh build ERR", type(e).__name__, str(e)[:200]); return
    show("  fresh", g)
    try:
        h = rt(g)
    except Exception as e:
        print("  round trip ERR", type(e).__name__, str(e).splitlines()[0][:200]); return
    show("  hit  ", h)
    for theta in ([2.0], [2.0, 3.0], [2.0, 3.0, 4.0]):
        r = []
        for gg in (g, h):
            try:
                gg.update_weights(theta); r.append(round(float(gg.expectation()), 6))
            except Exception as e:
                r.append("ERR " + type(e).__name__)
        print("  theta", theta, "E fresh/hit", r)

def cb2(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0]]]
case("theta_dim=3 > coeff len 2", phasic.with_ipv([5])(cb2), theta_dim=3)
case("theta_dim=1 < coeff len 2", phasic.with_ipv([5])(cb2), theta_dim=1)

def cbmix(state, **kw):
    n = state[0]
    if n <= 1: return []
    if n % 2: return [[[n - 1], [n * (n - 1) / 2]]]
    return [[[n - 1], [n * (n - 1) / 2, 1.0]]]
case("mixed coefficient lengths", phasic.with_ipv([5])(cbmix))

def cbc(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], n * (n - 1) / 2]]
case("constant edges with theta_dim=2", phasic.with_ipv([5])(cbc), theta_dim=2)

# parameterized IPV form: [[state, [coeffs]]] -> wrapper's TypeError branch
def cbp(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2]]]
case("ipv with coefficient list", cbp, ipv=[[[5], [0.5]], [[4], [0.5]]])
# ipv with total prob < 1 (defective)
case("defective ipv", cbp, ipv=[[[5], 0.25], [[4], 0.5]])
