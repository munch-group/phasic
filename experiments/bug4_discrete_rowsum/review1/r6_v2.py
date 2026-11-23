"""Review 1 of plan v2: (A) is was_dph carried on every route a converted graph can take, and does a native
graph stay un-latched; (B) negative edge weights: 'rejected at construction/update today'?;
(C) which v2 'fail on master' stale-context tests already pass; (D) sampler cost model."""
import sys, pickle, copy, random, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from phasic import Graph
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
def t(label, f):
    try: print(f'    {label}: {f()}')
    except BaseException as e: print(f'    {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}')
def flags(g): return f'is_discrete={getattr(g, "is_discrete", "?")} was_dph={g.get_was_dph()}'
def native(param=True):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0] if param else 0.5); g.is_discrete = True
    if param: g.update_weights([0.5])
    return g
def conv():
    g0 = coalescent_graph(5); g0.update_weights([0.7]); d = g0.discretize(0.1); return d if isinstance(d, tuple) else (d, None)
print('## A. flag propagation')
d, rw = conv(); J = coalescent_indexed(3)[0].joint_prob_graph(coalescent_indexed(3)[1], reward_limit=2)
for name, g in (('discretize() result', d), ('fresh joint_prob_graph(discrete)', J), ('native parameterized', native()), ('native constant', native(False))):
    print(f'# {name}: {flags(g)}')
    t('clone()', lambda: flags(g.clone()))
    t('copy()', lambda: flags(g.copy()))
    t('copy.copy', lambda: flags(copy.copy(g)))
    t('copy.deepcopy', lambda: flags(copy.deepcopy(g)))
    t('pickle round trip', lambda: flags(pickle.loads(pickle.dumps(g))))
    t('from_serialized(serialize())', lambda: flags(Graph.from_serialized(g.serialize())))
    def drop():
        s = dict(g.serialize()); s.pop('was_dph', None); return flags(Graph.from_serialized(s))
    t('from_serialized without was_dph key (old cache)', drop)
    nv = g.vertices_length(); r = np.ones(nv, dtype=int); r[0] = 0
    if name.startswith('discretize') and rw is not None: r = np.asarray(rw, dtype=int)
    t('reward_transform_discrete(r)', lambda: flags(g.reward_transform_discrete(r)))
    t('reward_transform(r)', lambda: flags(g.reward_transform(r)))
    t('Graph(g) re-wrap', lambda: flags(Graph(g)))
print('## A2. elimination read-outs on a converted graph whose rows are NOT normalised (exempt under v2 2.3)')
h = Graph(1); s = h.starting_vertex(); a = h.find_or_create_vertex([1]); b = h.find_or_create_vertex([2]); s.add_edge(a, 1.0); a.add_edge(b, 3.0)
dd = h.discretize(1.0); dd = dd[0] if isinstance(dd, tuple) else dd
print('    discretize(1.0) of rate-3 chain rows:', [[e.weight() for e in dd.vertex_at(i).edges()] for i in range(dd.vertices_length())])
t('expectation / variance (normalised)', lambda: (dd.expectation(), dd.variance()))
print('## B. negative edge weights')
def neg_const():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, -0.5); a.add_edge(c, 0.9); b.add_edge(c, 0.5); g.is_discrete = True; return g
def neg_param(th):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); b.add_edge(c, [0.0, 0.5]); g.is_discrete = True; g.update_weights(th); return g
t('constant add_edge(-0.5): build', lambda: 'accepted' if neg_const() is not None else '')
t('constant: stop_probability(1)', lambda: np.asarray(neg_const().stop_probability(1)).tolist())
t('constant: pdf(1..3)', lambda: np.asarray(neg_const().pdf(np.array([1, 2, 3]))).tolist())
t('linear update_weights([-0.5, 0.9]): build', lambda: [[e.weight() for e in neg_param([-0.5, 0.9]).vertex_at(i).edges()] for i in range(4)])
t('param: stop_probability(1)  (vertex 2 is transient)', lambda: np.asarray(neg_param([-0.5, 0.9]).stop_probability(1)).tolist())
t('param: pdf(1..3)', lambda: np.asarray(neg_param([-0.5, 0.9]).pdf(np.array([1, 2, 3]))).tolist())
t('param: expectation, variance', lambda: (neg_param([-0.5, 0.9]).expectation(), neg_param([-0.5, 0.9]).variance()))
print('## C. stale contexts on master')
g = native(); t('accumulated_visits(3) valid', lambda: np.asarray(g.accumulated_visits(3)).tolist()); g.update_weights([1.5])
t('accumulated_visits(5) after update_weights([1.5])  (v2 test 9, listed as failing on master)', lambda: np.asarray(g.accumulated_visits(5)).tolist())
g = native(); t('pdf valid', lambda: g.pdf(2)); g.update_weights([1.5]); t('pdf after update (v2 guard 12)', lambda: g.pdf(2))
g = native(); c = g.distribution_context(); c.step(); g.update_weights([1.5])
def st():
    o = []
    for _ in range(3): c.step(); o.append(c.pmf())
    return o
t('user-held distribution_context stepped after update_weights([1.5])  (not addressed in v2)', st)
g = native(); c = g.distribution_context(); c.step(); g.update_weights([-0.5]); t('user-held context after update_weights([-0.5])', st)
print('## D. cost of a whole-graph row-sum pass relative to one discrete sample (helper placement in the samplers)')
g0 = coalescent_graph(40); g0.update_weights([0.7]); d = g0.discretize(0.01); d = d[0] if isinstance(d, tuple) else d
V = d.vertices_length(); E = sum(len(d.vertex_at(i).edges()) for i in range(V))
random.seed(1); n = 2000; t0 = time.perf_counter(); x = d.sample(n); t1 = time.perf_counter() - t0
print(f'    coalescent n=40 discretize: V={V} E={E}; sample({n}) takes {t1*1e3:.1f} ms = {t1/n*1e6:.1f} us/sample; mean path length {float(np.mean(x)):.0f} jumps')
print(f'    a pass over E={E} edges per SAMPLE would add about {E*1.5e-9*1e6:.0f} us/sample (1.5 ns/edge assumed) -> at 60,000 vertices / ~300,000 edges ~450 us/sample')
