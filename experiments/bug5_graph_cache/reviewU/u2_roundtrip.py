# reviewU u2: does a serialize -> compact JSON -> ndarray-tag decode ->
# from_serialized round trip (what a v4.6 HIT returns) equal a fresh build
# in per-vertex edge ORDER, weights, coefficients, param_length, and
# downstream numbers (bitwise)?  No cache files are written.
import json, numpy as np, phasic
from phasic import Graph
from phasic.graph_cache import _serialize_value

def decode(v):
    if isinstance(v, dict):
        if v.get('__type__') == 'ndarray':
            return np.array(v['__data__'])
        return {k: decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode(x) for x in v]
    return v

def roundtrip(g):
    s = json.dumps(_serialize_value(g.serialize()), separators=(",", ":"))
    return Graph.from_serialized(decode(json.loads(s)))

def describe(g):
    out = []
    for v in g.vertices():
        es = [(tuple(e.to().state()), float(e.weight()), e.coefficients_length()) for e in v.edges()]
        ps = []
        try:
            for e in v.parameterized_edges():
                cl = e.coefficients_length()
                ps.append((tuple(e.to().state()), tuple(e.edge_state(cl))))
        except Exception as ex:
            ps = [repr(ex)]
        out.append((v.index(), tuple(v.state()), es, ps))
    return out

def compare(name, g, extra=lambda g: None):
    h = roundtrip(g)
    dg, dh = describe(g), describe(h)
    print(f"== {name}: n={g.vertices_length()} vs {h.vertices_length()}; "
          f"param_length {g.param_length()} vs {h.param_length()}; "
          f"parameterized {g.parameterized()} vs {h.parameterized()}")
    nd = 0
    for a, b in zip(dg, dh):
        if a != b:
            nd += 1
            if nd <= 3:
                print("  DIFF fresh:", a); print("       hit  :", b)
    print("  vertices with differing edge lists:", nd, "of", len(dg))
    try:
        print("  extra fresh:", extra(g)); print("  extra hit  :", extra(h))
    except Exception as e:
        print("  extra failed:", type(e).__name__, e)

# 1. constant-rate callback, edges deliberately returned in descending order
@phasic.with_ipv([[[4, 0], 1.0]])
def coal(state):
    n, m = state
    if n <= 1: return []
    out = []
    if m < 3: out.append([[n, m + 1], 0.7 + n])
    out.append([[n - 1, m], n * (n - 1) / 2])
    return out
g = Graph(coal)
compare("constant", g, lambda g: (g.expectation(), g.variance()))

# 2. parameterized callback, 2 params, edges in non-sorted order
@phasic.with_ipv([[[5], 1.0]])
def pcb(state):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 0.1 * n]]]
g2 = Graph(pcb)
def ex2(g):
    g.update_weights([1.0, 2.0]); return (g.expectation(), list(g.moments(2)))
compare("param", g2, ex2)

# 3. aux coefficients: theta_dim=2, coefficient length 3
@phasic.with_ipv([[[5], 1.0]])
def acb(state):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 0.1 * n, 7.0]]]
g3 = Graph(acb, theta_dim=2)
compare("aux theta_dim=2 coeff=3", g3)

# 4. parameterized with two out-edges per vertex in reverse order
@phasic.with_ipv([[[3, 0], 1.0]])
def two(state):
    a, b = state
    out = []
    if a > 0: out.append([[a - 1, b + 1], [float(a), 0.0]])
    if b > 0: out.append([[a, b - 1], [0.0, float(b)]])
    return list(reversed(out))
g4 = Graph(two)
compare("param two-edge reversed", g4, ex2)
