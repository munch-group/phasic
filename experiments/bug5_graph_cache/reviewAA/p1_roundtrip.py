"""reviewAA p1: does Graph.from_serialized(g.serialize()) (the hit path's
reconstruction) equal a fresh build?  Master code, no cache involved.
Run with HOME pointed at a scratch dir."""
import os, sys
assert "phasic_cache" not in os.environ.get("PHASIC_CACHE_DIR", "") or True
import numpy as np
import phasic
phasic.configure(graph_cache=False)
from phasic import Graph


def edges_of(g):
    out = []
    for v in g:
        row = []
        for e in v.edges():
            try:
                cl = e.coefficients_length()
                co = list(e.edge_state(cl)) if cl else []
            except Exception as ex:  # noqa
                cl, co = None, repr(ex)
            row.append((e.to().index(), float(e.weight()), cl, co))
        out.append((tuple(v.state()), row))
    return out


def compare(name, g):
    try:
        h = Graph.from_serialized(g.serialize())
    except Exception as ex:
        print(f"[{name}] roundtrip RAISES: {type(ex).__name__}: {ex}"[:300])
        return None
    print(f"[{name}] fresh: n={g.vertices_length()} param_length={g.param_length()} "
          f"parameterized={g.parameterized()} | hit: n={h.vertices_length()} "
          f"param_length={h.param_length()} parameterized={h.parameterized()}")
    ef, eh = edges_of(g), edges_of(h)
    print(f"[{name}] edge lists equal: {ef == eh}")
    if ef != eh:
        for a, b in zip(ef, eh):
            if a != b:
                print("   fresh:", a)
                print("   hit  :", b)
                break
    return h


# a) constant callback with theta_dim=2
def cb_const(state, **kw):
    if state[0] < 3:
        return [([state[0] + 1], 2.0)]
    return []
try:
    g = Graph(cb_const, ipv=[([0], 1.0)], theta_dim=2)
    compare("a_const_theta_dim2", g)
except Exception as ex:
    print("[a] build raises", type(ex).__name__, str(ex)[:120])

# b) parameterized: mixed param + constant edges to the SAME target
def cb_mixed(state, **kw):
    if state[0] == 0:
        return [([1], [1.0, 0.0]), ([1], 3.0), ([2], [0.0, 1.0])]
    return []
try:
    g = Graph(cb_mixed, ipv=[([0], 1.0)])
    compare("b_mixed_same_target", g)
except Exception as ex:
    print("[b] build raises", type(ex).__name__, str(ex)[:200])

# b2) two parameterized edges to the same target
def cb_dup(state, **kw):
    if state[0] == 0:
        return [([1], [1.0, 0.0]), ([1], [0.0, 1.0])]
    return []
g = Graph(cb_dup, ipv=[([0], 1.0)])
compare("b2_dup_param_same_target", g)
for th in ([1.0, 2.0],):
    g.update_weights(th)
    h = Graph.from_serialized(g.serialize()); h.update_weights(th)
    print("  b2 expectation fresh vs hit:", g.expectation(), h.expectation())

# c) decoupled theta_dim < coefficient length
def cb_dec(state, **kw):
    if state[0] < 3:
        return [([state[0] + 1], [1.0, 0.5, 7.0])]
    return []
g = Graph(cb_dec, ipv=[([0], 1.0)], theta_dim=2)
compare("c_decoupled", g)

# d) 3-tuple with a nonzero vestigial weight
def cb_3t(state, **kw):
    if state[0] < 2:
        return [([state[0] + 1], 5.0, [1.0])]
    return []
g = Graph(cb_3t, ipv=[([0], 1.0)])
h = compare("d_3tuple_weight", g)
print("  d expectation before update_weights fresh/hit:", g.expectation(), h.expectation() if h else None)

# e) parameterized IPV (coefficients on start edges)
def cb_e(state, **kw):
    if state[0] < 2:
        return [([state[0] + 1], [1.0])]
    return []
try:
    g = Graph(cb_e, ipv=[([0], [0.3]), ([1], [0.7])])
    h = compare("e_param_ipv", g)
    g.update_weights([2.0]); h.update_weights([2.0])
    print("  e expectation fresh/hit:", g.expectation(), h.expectation())
except Exception as ex:
    print("[e] raises", type(ex).__name__, str(ex)[:200])

# f) zero-coefficient param edge / all-zero
def cb_f(state, **kw):
    if state[0] == 0:
        return [([1], [0.0, 0.0]), ([2], [1.0, 1.0])]
    return []
g = Graph(cb_f, ipv=[([0], 1.0)])
compare("f_zero_coeff", g)

# g) constant graph with zero-weight edge
def cb_g(state, **kw):
    if state[0] == 0:
        return [([1], 0.0), ([2], 1.0)]
    return []
g = Graph(cb_g, ipv=[([0], 1.0)])
compare("g_zero_weight", g)
