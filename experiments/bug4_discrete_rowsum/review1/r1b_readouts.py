"""Review 1: every discrete read-out on a native geometric chain with row sum
p in {0.5, 1+1e-5, 1.5}. Unmodified code."""
import sys, random, warnings
import numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
def geom(p, start=1.0):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, start); a.add_edge(b, p); g.is_discrete = True; return g
def t(label, f):
    try:
        v = f()
        if isinstance(v, (list, tuple, np.ndarray)): v = np.asarray(v, dtype=float).round(8).tolist()
        print(f'    {label}: {v}')
    except BaseException as e: print(f'    {label}: {type(e).__name__}: {str(e).splitlines()[0][:100]}')
def ctx(g, k):
    c = g.distribution_context(); out = []
    for _ in range(k): c.step(); out.append(c.pmf())
    return out
probes = [
 ('pdf(1..3)', lambda g: g.pdf(np.array([1,2,3]))),
 ('cdf(1..3)', lambda g: g.cdf(np.array([1,2,3]))),
 ('expectation', lambda g: g.expectation()),
 ('expectation(rewards)', lambda g: g.expectation([0,2,0])),
 ('variance', lambda g: g.variance()),
 ('variance(rewards)', lambda g: g.variance([0,2,0])),
 ('covariance', lambda g: g.covariance([0,1,0],[0,2,0])),
 ('moments(2)', lambda g: g.moments(2)),
 ('moments(2,discrete=True)', lambda g: g.moments(2, discrete=True)),
 ('expected_waiting_time', lambda g: g.expected_waiting_time()),
 ('expected_sojourn_time', lambda g: g.expected_sojourn_time()),
 ('stop_probability(1)', lambda g: g.stop_probability(1)),
 ('stop_probability(2)', lambda g: g.stop_probability(2)),
 ('accumulated_visits(3)', lambda g: g.accumulated_visits(3)),
 ('accumulated_occupancy(3)', lambda g: g.accumulated_occupancy(3)),
 ('sample(4)', lambda g: (random.seed(1), g.sample(4))[1]),
 ('sample(4,rewards)', lambda g: (random.seed(1), g.sample(4, rewards=[0,2,0]))[1]),
 ('random_sample_discrete_stop_vertex(3)', lambda g: g.random_sample_discrete_stop_vertex(3)),
 ('distribution_context step x3 pmf', lambda g: ctx(g, 3)),
 ('reward_transform_discrete([0,2,0]).pdf(1..4)', lambda g: g.reward_transform_discrete(np.array([0,2,0])).pdf(np.array([1,2,3,4]))),
 ('reward_transform([0,2,0]).pdf(1..4)', lambda g: g.reward_transform(np.array([0,2,0])).pdf(np.array([1,2,3,4]))),
 ('normalize()', lambda g: g.clone().normalize() if hasattr(g,'clone') else None),
 ('as_matrices', lambda g: (lambda m: [np.asarray(m.ipv).tolist(), np.asarray(m.sim).tolist()])(g.as_matrices())),
 ('defect', lambda g: g.defect()),
 ('sample_path', lambda g: str(g.sample_path(1))[:80]),
 ('serialize->from_serialized pdf', lambda g: (lambda h: h.pdf(np.array([1,2])))(Graph.from_serialized(g.serialize()))),
 ('discretize(0.1)', lambda g: g.discretize(0.1)),
]
SKIP = set(sys.argv[1:])
for p in (0.5, 1 + 1e-5, 1.5):
    print(f'## native geometric p = {p!r}')
    for label, f in probes:
        if p == 1.5 and label in SKIP: print(f'    {label}: SKIPPED here (segfaults, see r1_crash.log)'); continue
        t(label, lambda: f(geom(p)))
print('## STARTING VERTEX sum (p = 0.5 chain)')
for s0 in (0.5, 1 + 1e-12, 1 + 1e-5, 1.0002, 1.5):
    print(f'# start edge = {s0!r}')
    for label in ('pdf(1..3)', 'cdf(1..3)', 'expectation', 'variance', 'sample(4)', 'stop_probability(1)', 'moments(2,discrete=True)', 'reward_transform_discrete([0,2,0]).pdf(1..4)'):
        f = dict(probes)[label]
        t(label, lambda: f(geom(0.5, s0)))
    t('sum pmf(0..400)', lambda: float(np.sum(geom(0.5, s0).pdf(np.arange(0, 401)))))
