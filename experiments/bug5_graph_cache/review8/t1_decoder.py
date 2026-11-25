"""Does serialize() output of every graph kind need only the 'ndarray' tag, and does a
restricted decoder (plain JSON + ndarray only) round-trip it? (master code)"""
import os, json, itertools
assert "scratchpad" in os.environ["HOME"]
assert not os.environ.get("PHASIC_CACHE_DIR")
import numpy as np
import phasic
from phasic import Graph, StateIndexer, Property, with_ipv
from phasic.graph_cache import _serialize_value


def tags(x, acc):
    if isinstance(x, dict):
        if "__type__" in x:
            acc.add(x["__type__"])
        for v in x.values():
            tags(v, acc)
    elif isinstance(x, list):
        for v in x:
            tags(v, acc)
    return acc


def restricted(x):
    if isinstance(x, list):
        return [restricted(v) for v in x]
    if isinstance(x, dict):
        if "__type__" in x and "__data__" in x:
            if x["__type__"] != "ndarray":
                raise ValueError("tag " + x["__type__"])
            return np.array(restricted(x["__data__"]))
        return {k: restricted(v) for k, v in x.items()}
    return x


def same(a, b):
    if a.keys() != b.keys():
        return f"keys differ {set(a) ^ set(b)}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            va, vb = np.asarray(va), np.asarray(vb)
            if va.shape != vb.shape and va.size == vb.size == 0:
                continue
            if va.shape != vb.shape or not np.array_equal(va, vb, equal_nan=True):
                return f"{k} differs"
        elif va != vb:
            return f"{k}: {va!r} vs {vb!r}"
    return "identical"


def check(label, g):
    try:
        d = g.serialize()
    except Exception as e:
        print(f"  {label:28s} serialize() raised {type(e).__name__}: {str(e)[:70]}"); return
    blob = json.dumps(_serialize_value(d))
    t = tags(json.loads(blob), set())
    dec = restricted(json.loads(blob))
    g2 = Graph.from_serialized(dec)
    print(f"  {label:28s} tags={sorted(t)} keys+={sorted(set(d) - {'states','vertex_indices','edges','constant_edges','start_edges','param_edges','start_param_edges','param_length','state_length','n_vertices','weight_mode','dyn_ordering','is_discrete','was_dph','is_aux'})} roundtrip={same(d, g2.serialize())}")


def pcb(state, **kw):
    k = state[0]
    return [] if k <= 1 else [[[k - 1], [float(k), 0.5]]]
def ccb(state, **kw):
    k = state[0]
    return [] if k <= 1 else [[[k - 1], float(k)]]

check("parameterized linear", Graph(pcb, ipv=[5], graph_cache=False))
check("constant", Graph(ccb, ipv=[5], graph_cache=False))
g = Graph(pcb, ipv=[5], graph_cache=False); g.weight_formula = "c0*t0 + c1"
check("formula mode", g)
g = Graph(pcb, ipv=[5], graph_cache=False); g.weight_mode = "log"
check("log mode", g)
g = Graph(pcb, ipv=[5], graph_cache=False); g.weight_mode = "callback"
check("callback mode", g)
g = Graph(ccb, ipv=[5], graph_cache=False)
try:
    check("discretize()'d (was_dph, aux)", g.discretize(2.0))
except Exception as e:
    print("  discretize failed", type(e).__name__, str(e)[:80])
gd = Graph(ccb, ipv=[5], graph_cache=False); gd.is_discrete = True
check("is_discrete flag", gd)

nr = 3
indexer = StateIndexer(lineages=[Property("ton", min_value=0, max_value=nr)] * 0 or [Property(f"t{i}", max_value=nr) for i in range(nr)]) if False else None
from functools import partial
all_pairs = partial(itertools.combinations_with_replacement, r=2)
idx = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
@with_ipv([nr] + [0] * (nr - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same_ = int(i == j)
        if same_ and state[i] < 2: continue
        if not same_ and (state[i] < 1 or state[j] < 1): continue
        new = state.copy(); new[i] -= 1; new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same_) / (1 + same_)]])
    return t
try:
    base = Graph(coal, indexer=idx, graph_cache=False)
    check("indexer-built base", base)
    jpg = base.joint_prob_graph(idx, reward_limit=3, mutation_rate=1e-2)
    check("joint_prob_graph", jpg)
except Exception as e:
    print("  joint-prob setup failed", type(e).__name__, str(e)[:120])
