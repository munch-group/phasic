"""Starting-vertex (IPV) row: is it checked today? (plan's helper says 'every non-starting vertex')"""
import numpy as np, warnings, random
warnings.filterwarnings('ignore')
from phasic import Graph
def g2(ipv, p=0.5):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, ipv); a.add_edge(b, p); g.is_discrete = True; return g
def t(label, f):
    try: print(f'  {label}: OK {f()!r}'[:150])
    except BaseException as e: print(f'  {label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:110]}')
for ipv in (1.0, 1 + 1e-12, 1 + 1e-5, 1.5, 0.5):
    print('IPV sum =', ipv)
    t('pdf(1..3)', lambda: np.asarray(g2(ipv).pdf(np.arange(1, 4))))
    t('cdf(200)', lambda: float(g2(ipv).cdf(200)))
    t('sample', lambda: (random.seed(1), g2(ipv).sample(3))[1])
    t('rtd.pdf', lambda: np.asarray(g2(ipv).reward_transform_discrete(np.array([0, 1, 0])).pdf(np.arange(1, 4))))
    t('expectation', lambda: g2(ipv).expectation())
