"""Review AH p3: is a cache hit (from_serialized(serialize()) via the JSON
encoding the cache uses) the same graph as a fresh build, for callback shapes
users write?  Compares per-vertex ordered edge lists (target state, weight,
coefficients) and moments bit-for-bit.  HOME redirected; no cache used."""
import os, json
assert "scratchpad" in os.environ.get("HOME", "")
import numpy as np
import phasic
from phasic import Graph
from phasic.graph_cache import _serialize_value, _deserialize_value

def jsonrt(d):
    s = json.dumps({k: _serialize_value(v) for k, v in d.items()}, separators=(",", ":"))
    return {k: _deserialize_value(v) for k, v in json.loads(s).items()}

def edges_of(g):
    out = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        es = []
        for e in v.edges():
            cl = e.coefficients_length() if hasattr(e, "coefficients_length") else -1
            es.append((tuple(e.to().state()), float(e.weight()), cl))
        pes = []
        try:
            for e in v.parameterized_edges():
                pes.append((tuple(e.to().state()), tuple(e.edge_state(e.coefficients_length()))))
        except Exception as ex:
            pes = [repr(ex)]
        out.append((tuple(v.state()), es, pes))
    return out

def compare(name, cb, ipv=None, theta=None, **kw):
    g = Graph(cb, ipv, **kw) if ipv is not None else Graph(cb, **kw)
    try:
        d = g.serialize()
    except Exception as ex:
        print(f"{name:40s} serialize raised {type(ex).__name__}: {str(ex)[:80]}")
        return
    h = Graph.from_serialized(jsonrt(d))
    if theta is not None:
        g.update_weights(theta); h.update_weights(theta)
    E1, E2 = edges_of(g), edges_of(h)
    same_struct = E1 == E2
    diff = None
    if not same_struct:
        for a, b in zip(E1, E2):
            if a != b:
                diff = (a, b); break
        if diff is None:
            diff = ("len", len(E1), len(E2))
    try:
        m1 = np.asarray(g.moments(2)); m2 = np.asarray(h.moments(2))
        mom = "bit-equal" if m1.tobytes() == m2.tobytes() else f"DIFF {m1} vs {m2}"
    except Exception as ex:
        mom = f"moments raised {type(ex).__name__}"
    print(f"{name:40s} V={g.vertices_length()}/{h.vertices_length()} edges-ordered-equal={same_struct} moments {mom}")
    if diff:
        print("    first diff:", str(diff)[:300])

# 1. constant coalescent
def coal(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], n * (n - 1) / 2]]
compare("constant coalescent", coal, [[[6], 1.0]])

# 2. parameterized coalescent
def coalp(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], [n * (n - 1) / 2]]]
compare("param coalescent", coalp, [[[6], 1.0]], theta=[2.0])

# 3. two-island parameterized (2 params), cyclic
def island(state, **kw):
    a, b = state
    out = []
    if a >= 2: out.append([[a - 1, b], [a * (a - 1) / 2, 0.0]])
    if b >= 2: out.append([[a, b - 1], [b * (b - 1) / 2, 0.0]])
    if a >= 1 and a + b > 1: out.append([[a - 1, b + 1], [0.0, a]])
    if b >= 1 and a + b > 1: out.append([[a + 1, b - 1], [0.0, b]])
    return out
compare("two-island param", island, [[[4, 0], 1.0]], theta=[1.0, 0.3])

# 4. duplicate targets emitted (loop over pairs), constant
def dup(state, **kw):
    n = state[0]
    if n <= 1: return []
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append([[n - 1], 1.0])
    return out
compare("duplicate targets, constant", dup, [[[5], 1.0]])

# 5. duplicate targets, parameterized
def dupp(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [1.0]] for _ in range(n * (n - 1) // 2)]
compare("duplicate targets, param", dupp, [[[5], 1.0]], theta=[1.5])

# 6. multiple starting states
compare("multi-start constant", coal, [[[6], 0.5], [[4], 0.5]])

# 7. zero-coefficient param edge
def zc(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], [n * 1.0, 0.0]], [[0], [0.0, 0.0]]] if n > 1 and n != 0 else []
compare("all-zero coefficient edge", zc, [[[4], 1.0]], theta=[1.0, 1.0])

# 8. weights with many digits / subnormal / huge
def odd(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], 0.1 + 1e-17 * n], [[0], 5e-324]]
compare("odd float weights", odd, [[[4], 1.0]])
