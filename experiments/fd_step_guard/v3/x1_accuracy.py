"""X1/X2: is the shipped continuous pdf an Euler discretization (first-order
in 1/G)?  Does Poisson mixing of the SAME iterates remove the error, for any
G, across rate magnitudes?  Closed form: hypoexponential(l1, l2)."""
import sys, time
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
from unif import pdf_cdf, max_rate, unif_G
from phasic import Graph

def hypo(l1, l2):
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, l1); b.add_edge(c, l2)
    return g
def exact(l1, l2, t):
    return l1 * l2 / (l2 - l1) * (np.exp(-l1 * t) - np.exp(-l2 * t))

print('## shipped pdf vs closed form at explicit granularity (l1=1, l2=3, t=0.7)')
g = hypo(1.0, 3.0); f = exact(1.0, 3.0, 0.7)
for G in (1024, 2048, 4096, 8192):
    print(f'  G={G:5d}: shipped relerr {abs(g.pdf(0.7, G) - f)/f:.3e}   poisson-mix relerr {abs(pdf_cdf(g, [0.7], G)[0][0] - f)/f:.3e}')
print('## poisson-mix with different G (must agree): ', [f'{pdf_cdf(g, [0.7], G)[0][0]:.15f}' for G in (4, 7, 50, 1024)], f'exact {f:.15f}')
print('## across rate magnitudes (l2 = 3*l1, t chosen so l1*t = 0.7): relative pdf error and steps')
for l1 in (1e-9, 1e-6, 1e-3, 1.0, 1e2, 1e4, 1e6):
    l2 = 3 * l1; t = 0.7 / l1
    g = hypo(l1, l2); f = exact(l1, l2, t)
    steps_shipped = t * max(1000, int(2 * max(512, max_rate(g))))
    if steps_shipped > 2e7:
        es = 'skipped'
    else:
        try: es = f'{abs(g.pdf(t) - f)/f:.2e}'
        except Exception as e: es = 'ERR ' + str(e)[:40]
    # The context takes an integer granularity, so the prototype rescales time
    # (pdf_theta(t) = s * pdf_{theta/s}(t*s), s = max rate); native code would
    # use a real-valued uniformization rate instead.
    s_ = max_rate(g); gs = hypo(l1 / s_, l2 / s_)
    t0 = time.time(); p, c, n = pdf_cdf(gs, [t * s_]); dt = time.time() - t0
    print(f'  l1={l1:7.0e} t={t:8.1e}: shipped relerr {es:>9s} ({steps_shipped:.3g} steps)   poisson-mix relerr {abs(p[0]*s_-f)/f:.2e} ({n} steps)', flush=True)
