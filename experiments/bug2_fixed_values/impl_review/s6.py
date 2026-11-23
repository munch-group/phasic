import sys
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/impl_review')
from common import *
from phasic.svgd import SparseObservations
KW = dict(n_particles=6, n_iterations=4, learning_rate=0.01, progress=False, seed=1)
def g3():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(c, [1.0, 0.0]); a.add_edge(b, [0.0, 1.0]); b.add_edge(c, [1.0, 0.0])
    return g
def e1():
    g = g3(); nv = g.vertices_length()
    R = np.tile(np.arange(1.0, nv+1.0), (2,1))
    obs = np.column_stack([REAL, REAL*1.5])
    s = g.svgd(obs, rewards=R, fixed=[(1, 1e-10)], prior=[gauss(), None], theta_dim=2, **KW); report(s,[1])
scen("E1 multivariate 2-D rewards via Graph.svgd, fixed 1e-10", e1)
def e2():
    g = g3(); nv = g.vertices_length()
    R = np.tile(np.arange(1.0, nv+1.0), (2,1))
    sp = SparseObservations(values=np.array([0.5,1.0,0.7,1.2]), features=np.array([0,0,1,1]), n_features=2, slices=((0,2),(2,4)))
    s = g.svgd(sp, rewards=R, fixed=[(1, 1e-10)], prior=[gauss(), None], theta_dim=2, **KW); report(s,[1])
scen("E2 sparse observations, fixed 1e-10", e2)
def e3():
    for pr in (gauss(), [gauss(), None]):
        m = Graph.pmf_and_moments_from_graph(g3(), nr_moments=2, theta_dim=2)
        s = SVGD(model=m, observed_data=REAL, theta_dim=2, prior=pr, fixed=[(1, 2.0)], n_particles=6, n_iterations=12, learning_rate=0.01, progress=False, seed=1); s.optimize()
        print("  prior", "scalar" if not isinstance(pr, list) else "list", "free particles", repr(np.asarray(s.get_results()['particles'])[:,0]))
scen("E3 scalar vs list prior, fixed 2.0 (compare across builds)", e3)
def e4():
    s = g3().svgd(REAL, fixed=[(1, 0.3)], prior=[gauss(), None], theta_dim=2, regularization=0.5, preconditioner='fisher', return_history=False, **KW); report(s,[1])
scen("E4 Graph.svgd regularization+fisher+no history", e4)
