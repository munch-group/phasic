"""E11: auto granularity = int(2*max_rate) above max_rate 512 makes the pdf
piecewise in theta. Measure (i) the jump size, (ii) FD derivative error of the
absolute 1e-7 step vs relative 3e-3*theta step, against the derivative of the
SAME pdf at a FIXED granularity (smooth in theta)."""
import sys
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import two_island_graph
g = two_island_graph(6)
T = [0.002, 0.005, 0.01]
def F(th, gran=0):
    g.update_weights(list(th))
    return float(np.sum(np.log([g.pdf(t, gran) for t in T])))
def maxrate(th):
    g.update_weights(list(th))
    return max(sum(e.weight() for e in g.vertex_at(i).edges()) for i in range(g.vertices_length()))
for scale in (20.0, 200.0, 2000.0):
    th = np.array([1.0*scale, 0.5*scale]); R = maxrate(th); G = int(2*R)
    # jump size: f just below vs just above a granularity boundary
    xs = th[0] * (1 + np.linspace(-1e-3, 1e-3, 2001))
    fv = np.array([F([x, th[1]]) for x in xs])
    d = np.abs(np.diff(fv)); typical = np.median(d)
    jumps = d[d > 20*typical]
    def fd(h, gran=0):
        return (F([th[0]+h, th[1]], gran) - F([th[0]-h, th[1]], gran))/(2*h)
    ref = (4*fd(th[0]*1e-4, G) - fd(th[0]*2e-4, G))/3   # fixed granularity: smooth
    for lab, h in (('abs 1e-7', 1e-7), ('rel 3e-3', 3e-3*th[0]), ('rel 1e-5', 1e-5*th[0])):
        v = fd(h)
        print(f'scale={scale:g} maxrate={R:.4g} G={G}: {lab:9s} dF={v:.8e} ref(fixed G)={ref:.8e} relerr={abs(v-ref)/abs(ref):.2e}')
    print(f'   jumps in a +-0.1% window: {len(jumps)} (median |jump| {np.median(jumps) if len(jumps) else 0:.2e}, typical step diff {typical:.2e}, |F|={abs(fv[1000]):.3g})')
