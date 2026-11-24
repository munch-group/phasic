"""Plan guard 6: closed-form geometric to 1e-14 -- what errors does the unmodified build have?"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from phasic import Graph
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
k = np.arange(1, 201)
for p in (0.1, 0.3, 0.5, 0.9, 1.0):
    g = geom(p); pm = np.asarray(g.pdf(k)); cd = np.asarray(g.cdf(k))
    tp = p * (1 - p) ** (k - 1); tc = 1 - (1 - p) ** k
    e, v = g.expectation(), g.variance(); m = np.asarray(g.moments(2, discrete=True)); m0 = g.moments(2)
    print(f'p={p}: pmf abs {np.abs(pm-tp).max():.2e} rel {np.max(np.abs(pm-tp)/np.where(tp>0,tp,1)):.2e} | cdf abs {np.abs(cd-tc).max():.2e} | E={e!r} (true {1/p!r}) abs {abs(e-1/p):.2e} | var={v!r} true (1-p)/p^2={(1-p)/p**2!r} abs {abs(v-(1-p)/p**2):.2e} rel {abs(v-(1-p)/p**2)/max((1-p)/p**2,1e-300):.2e} | moments(2,discrete)={m} moments(2)={m0}')
