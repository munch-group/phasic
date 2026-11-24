"""reviewH p1: does a hit (serialize -> JSON -> from_serialized -> copy-init)
equal a fresh build, for graph kinds a callback can produce?  Runs on master
code via the real Graph(..., graph_cache=True) path; HOME must be a scratch dir.
Compares per-vertex states, per-edge (to, weight, coefficients), starting
vertex edges, param_length, parameterized(), expectation() before and after
update_weights."""
import os, sys, json
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph


def dump(g):
    out = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        es = []
        for e in v.edges():
            try:
                cl = e.coefficients_length()
                co = list(e.edge_state(cl)) if cl else []
            except Exception as ex:
                cl, co = "?", repr(ex)
            es.append((tuple(e.to().state()), float(e.weight()), cl, tuple(co)))
        out.append((tuple(v.state()), es))
    return out


def safe(f):
    try:
        return f()
    except Exception as ex:
        return f"ERR {type(ex).__name__}: {str(ex)[:80]}"


def compare(name, make, theta=None):
    g1 = make()  # cold: builds and saves
    g2 = make()  # warm: hit
    d1, d2 = dump(g1), dump(g2)
    same_struct = d1 == d2
    rows = [f"== {name}: n={g1.vertices_length()} struct_equal={same_struct}"]
    if not same_struct:
        for i, (a, b) in enumerate(zip(d1, d2)):
            if a != b:
                rows.append(f"   first diff at vertex {i}:\n     fresh={a}\n     hit  ={b}")
                break
    for attr in ("param_length", "parameterized"):
        rows.append(f"   {attr}: fresh={safe(getattr(g1, attr))} hit={safe(getattr(g2, attr))}")
    rows.append(f"   expectation (no update): fresh={safe(g1.expectation)} hit={safe(g2.expectation)}")
    if theta is not None:
        safe(lambda: g1.update_weights(theta)); safe(lambda: g2.update_weights(theta))
        rows.append(f"   expectation (theta={theta}): fresh={safe(g1.expectation)} hit={safe(g2.expectation)}")
        d1, d2 = dump(g1), dump(g2)
        rows.append(f"   struct after update equal: {d1 == d2}")
    print("\n".join(rows), flush=True)


# 1. constant coalescent
def coal_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

compare("constant", lambda: Graph(coal_const, ipv=[5], graph_cache=True))

# 2. parameterized coalescent, one coefficient
def coal_param(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]

compare("param1", lambda: Graph(coal_param, ipv=[5], graph_cache=True), theta=[2.0])

# 3. two coefficients, theta_dim given
def two(state, **kw):
    n, m = state
    out = []
    if n > 1:
        out.append(([n - 1, m], [n * (n - 1) / 2, 0.0]))
    if n > 0 and m < 3:
        out.append(([n, m + 1], [0.0, 1.0]))
    return out

compare("param2", lambda: Graph(two, ipv=[[[4, 0], 1.0]], theta_dim=2, graph_cache=True), theta=[1.0, 3.0])

# 4. theta_dim smaller than coefficient length (aux coefficient)
def aux(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 7.0])]

compare("aux_coeff", lambda: Graph(aux, ipv=[5], theta_dim=1, graph_cache=True))

# 5. duplicate edges to the same child
def dup(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 1.0), ([n - 1], 2.0)]

compare("dup_const", lambda: Graph(dup, ipv=[4], graph_cache=True))

def dupp(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [1.0]), ([n - 1], [2.0])]

compare("dup_param", lambda: Graph(dupp, ipv=[4], graph_cache=True), theta=[3.0])

# 6. multi-state ipv with probabilities, some starting mass defect
def ms(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [float(n)])]

compare("multi_ipv", lambda: Graph(ms, ipv=[[[3], 0.5], [[5], 0.3]], graph_cache=True), theta=[1.5])

# 7. zero-coefficient edge and a zero constant rate
def zc(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [0.0]), ([n - 2 if n > 2 else 0], [1.0])]

compare("zero_coeff", lambda: Graph(zc, ipv=[5], graph_cache=True), theta=[2.0])

def z0(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 0.0), ([0], 1.0)]

compare("zero_const", lambda: Graph(z0, ipv=[4], graph_cache=True))

# 8. a vertex revisited / cyclic graph
def cyc(state, **kw):
    a, = state
    if a == 0:
        return []
    if a == 1:
        return [([2], [1.0]), ([0], [0.5])]
    return [([1], [2.0])]

compare("cyclic", lambda: Graph(cyc, ipv=[1], graph_cache=True), theta=[1.0])

import shutil
cd = os.path.join(os.environ["HOME"], ".phasic_cache")
print("cache files written:", sum(len(f) for _, _, f in os.walk(cd)))
shutil.rmtree(cd, ignore_errors=True)
