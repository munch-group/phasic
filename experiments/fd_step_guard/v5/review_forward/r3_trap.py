"""R3: what the rate-1.0 t<->aux trap loop does, and whether the two candidate replacements keep the epoch path's results."""
import sys, numpy as np, math
from itertools import combinations_with_replacement
from scipy.linalg import expm
from phasic import Graph, Property, StateIndexer

def cb(state, indexer=None):
    tr = []
    for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
        same = int(i == j)
        if same and state[i] < 2: continue
        if not same and (state[i] < 1 or state[j] < 1): continue
        new = state.copy(); new[i] -= 1; new[j] -= 1; new[min(i + j + 1, state.size - 1)] += 1
        tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return tr

n = 5
indexer = StateIndexer(lineages=[Property('descendants', min_value=1, max_value=n)])
ipv0 = [0] * indexer.state_length; ipv0[indexer.lineages.props_to_index(descendants=1)] = n
graph = Graph(cb, ipv=ipv0, indexer=indexer)
cj = graph.joint_prob_graph(mutation_rate=1.0, reward_limit=2, discrete=False)
ctx = cj.epoch_context(); jsp = ctx._graph
print('JSP vertices', jsp.vertices_length(), 't-vertices', len(ctx._t_vertex_indices))

def gen(g):
    vs = list(g.vertices()); N = len(vs); Q = np.zeros((N, N)); loop = np.zeros((N, N), bool)
    aux = set(ctx._t_aux_map.values()); tset = set(ctx._t_aux_map.keys())
    for v in vs:
        if v.index() == 0: continue
        for e in v.edges():
            i, j = v.index(), e.to().index(); Q[i, j] += e.weight()
            if (i in tset and j in aux) or (i in aux and j in tset): loop[i, j] = True
    ipv = np.zeros(N)
    for e in vs[0].edges(): ipv[e.to().index()] += e.weight()
    return Q, loop, ipv

def collapse(raw):
    return np.array([raw[t] + raw[ctx._t_aux_map[t]] for t in ctx._t_vertex_indices])
def next_ipv(raw):
    out = []
    for v in ctx._ipv_target_indices:
        p = raw[v]
        if v in ctx._t_aux_map: p += raw[ctx._t_aux_map[v]]
        out.append(p)
    return np.array(out)

for theta, T in (([2.0, 1.0], 0.7), ([1e-4, 1e-4], 1e4 * 0.7), ([300.0, 150.0], 0.7 / 150)):
    ctx.update_weights(theta); R, loop, ipv = gen(jsp)
    print(f'## theta={theta} T={T:g}: loop edge weights {np.unique(R[loop])}, max non-loop exit rate {np.where(loop, 0, R).sum(1).max():g}, ipv sum {ipv.sum():g}')
    res = {}
    for name, c in (('loop rate 1.0 (shipped)', 1.0), ('rate 0 / mass retained', 0.0), ('loop rate = max rate', np.where(loop, 0, R).sum(1).max()), ('loop rate 1e6', 1e6)):
        Rc = np.where(loop, c, R); Qc = Rc - np.diag(Rc.sum(1))
        raw = ipv @ expm(Qc * T); res[name] = (collapse(raw), next_ipv(raw), raw)
        Gm = 1.02 * Rc.sum(1).max(); lam = Gm * T
        print(f'   {name:26s}: G={Gm:.4g} Lambda={lam:.4g} approx steps {int(lam + 8.5 * math.sqrt(lam) + 10)}; max |collapsed - shipped-loop| {np.abs(res[name][0] - res["loop rate 1.0 (shipped)"][0]).max():.1e}; next-ipv diff {np.abs(res[name][1] - res["loop rate 1.0 (shipped)"][1]).max():.1e}; split t/aux of 1st outcome {raw[ctx._t_vertex_indices[0]]:.3e}/{raw[ctx._t_aux_map[ctx._t_vertex_indices[0]]]:.3e}')
    try:
        sh = ctx.cumulative_probs(T, table=False)
        print(f'   shipped Euler cumulative_probs vs expm (loop 1.0): max abs diff {np.abs(sh - res["loop rate 1.0 (shipped)"][0]).max():.1e}, max rel {np.max(np.abs(sh - res["loop rate 1.0 (shipped)"][0]) / res["loop rate 1.0 (shipped)"][0]):.1e}')
    except Exception as e:
        print('   shipped ERR', str(e)[:150])

# does the EXISTING step keep mass on a vertex with (i) no edges, (ii) a zero-weight edge, (iii) a self-loop?
def tiny(kind):
    g = Graph(1); s = g.starting_vertex(); A = g.find_or_create_vertex([1]); Tv = g.find_or_create_vertex([2]); X = g.find_or_create_vertex([3])
    s.add_edge(A, 1.0); A.add_edge(Tv, 1.0); A.add_edge(X, 0.5)
    if kind == 'zero-weight edge': Tv.add_edge(X, 0.0)
    if kind == 'self-loop rate 3': Tv.add_edge(Tv, 3.0)
    if kind == 'self-loop rate 0': Tv.add_edge(Tv, 0.0)
    return g
ex = (1.0 / 1.5) * (1 - math.exp(-1.5 * 2.0))
for kind in ('no edges', 'zero-weight edge', 'self-loop rate 3', 'self-loop rate 0'):
    try:
        g = tiny(kind); sp = np.array(g.stop_probability(2.0)); print(f'existing stop_probability(2.0), outcome vertex with {kind}: {sp[2]:.6f} (exact retained mass {ex:.6f})')
    except Exception as e:
        print(f'{kind}: ERR {str(e)[:150]}')
