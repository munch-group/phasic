import time, os, sys, shutil
import numpy as np
from itertools import combinations_with_replacement
from functools import partial
import phasic
from phasic import Graph, StateIndexer, Property, with_ipv
from phasic.callback_hash import cache_entry_key
from phasic.graph_cache import GraphCache
all_pairs = partial(combinations_with_replacement, r=2)

def make_coal(n, param=True):
    @with_ipv([n] + [0] * (n - 1))
    def coal(state, indexer=None):
        tr = []
        for i, j in all_pairs(range(indexer.lineage.state_length)):
            p1 = indexer.lineage.index_to_props(i); p2 = indexer.lineage.index_to_props(j)
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1
            d = p1.descendants + p2.descendants
            if d > n: continue
            new[indexer.lineage.props_to_index(descendants=d)] += 1
            r = state[i] * (state[j] - same) / (1 + same)
            tr.append([new, [r] if param else r])
        return tr
    ix = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=n)])
    return coal, ix

def ser_equal(g1, g2):
    s1, s2 = g1.serialize(), g2.serialize()
    assert s1.keys() == s2.keys(), (s1.keys() ^ s2.keys())
    bad = []
    for k in s1:
        a, b = s1[k], s2[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
                bad.append((k, a.shape, b.shape, a.dtype, b.dtype))
        elif a != b:
            bad.append((k, a, b))
    return bad

def run(label, n, param):
    cb, ix = make_coal(n, param)
    t = time.perf_counter(); g0 = Graph(cb, indexer=ix); tb = time.perf_counter() - t
    ts = []
    for _ in range(3):
        t = time.perf_counter(); d = cache_entry_key("cost", cb, indexer=ix); ts.append(time.perf_counter() - t)
    from phasic import callback_hash as ch
    stats = dict(ch._LAST_WALK_STATS)
    t = time.perf_counter(); g1 = Graph(cb, indexer=ix, cache_key="cost-" + label); tmiss = time.perf_counter() - t
    t = time.perf_counter(); g2 = Graph(cb, indexer=ix, cache_key="cost-" + label); thit = time.perf_counter() - t
    path = g2._graph_cache_entry
    size = os.path.getsize(path)
    bad = ser_equal(g0, g2)
    if param:
        g0.update_weights([1.3]); g2.update_weights([1.3])
    e0, e2 = g0.expectation(), g2.expectation()
    print("RESULT", dict(model=label, n=n, vertices=g0.vertices_length(), build_s=round(tb, 3),
          key_ms=round(min(ts) * 1e3, 2), miss_total_s=round(tmiss, 3), hit_total_s=round(thit, 3),
          entry_MB=round(size / 2**20, 2), serialize_mismatch=bad, E_rel=abs(e0 - e2) / abs(e0), walk=stats))
    os.unlink(path)

for n in (8, 30):
    run(f"coal{n}-param", n, True)
run("coal12-const", 12, False)

# tiny graph kinds: no edges at all besides start; a 1-state absorbing
@with_ipv([1])
def trivial(state, **kw):
    return []
g0 = Graph(trivial); Graph(trivial, cache_key="triv"); g = Graph(trivial, cache_key="triv")
print("RESULT trivial", ser_equal(g0, g)); os.unlink(g._graph_cache_entry)
