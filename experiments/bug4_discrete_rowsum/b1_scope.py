"""Bug 4 scoping on unmodified code: what do the discrete read-outs return
when a vertex's outgoing probabilities sum to slightly more than 1?"""
import sys, random, warnings
import numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
print('phasic from', phasic.__file__)
def geom(p):                       # S -> a --p--> absorbing   (stay with 1-p)
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
def two(p, q):                     # a --p--> b --q--> absorbing, a --(extra) --> absorbing with 0.3
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, p); a.add_edge(c, 0.3); b.add_edge(c, q); g.is_discrete = True; return g
def t(label, f):
    try: v = f(); print(f'    {label}: {v}')
    except Exception as e: print(f'    {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}')
for p in (0.5, 1.0, 1 + 1e-12, 1 + 1e-9, 1 + 1e-6, 1 + 1e-5, 1.0001, 1.0002, 1.5):
    print(f'## geometric, p = 1 + {p-1:.3g}' if p > 1 else f'## geometric, p = {p}')
    g = geom(p)
    t('pmf(1..4)', lambda: np.asarray(g.pdf(np.array([1, 2, 3, 4]))))
    t('cdf(1..4)', lambda: np.asarray(g.cdf(np.array([1, 2, 3, 4]))))
    t('expectation', lambda: g.expectation())
    t('variance', lambda: g.variance())
    t('moments(2)', lambda: g.moments(2))
    random.seed(1); t('sample(5)', lambda: g.sample(5))
    t('reward_transform_discrete([0,1,0]).pdf(1..3)', lambda: np.asarray(g.reward_transform_discrete(np.array([0, 1, 0])).pdf(np.array([1, 2, 3]))))
print('## two-step chain, row of a = p + 0.3')
for p in (0.7, 0.7 + 1e-9, 0.7 + 1e-5, 0.7001, 0.71):
    g = two(p, 0.5)
    try: v = np.asarray(g.pdf(np.array([1, 2, 3, 4, 5]))); print(f'  row sum 1 + {p + 0.3 - 1:.3g}: pmf = {v.round(8)}; sum over 1..200 = {float(np.sum(np.asarray(g.pdf(np.arange(1, 201))))):.10f}')
    except Exception as e: print(f'  row sum 1 + {p + 0.3 - 1:.3g}: {type(e).__name__}: {str(e).splitlines()[0][:110]}')
