# reviewAK p6: master's real hit path (graph_cache=True, scratch HOME) vs a fresh uncached build:
# bit-equality of expectation/variance/pdf for a constant and a parameterized model.
import numpy as np, sys
from phasic import Graph, with_ipv
sys.path.insert(0, "experiments/bug5_graph_cache/reviewAK")
from p4_roundtrip_fidelity import cb_const, cb_param

def two_island_const(state, n=6):
    a1, a2 = state  # lineages in pop1, pop2
    out = []
    if a1 > 1: out.append([[a1 - 1, a2], a1 * (a1 - 1) / 2])
    if a2 > 1: out.append([[a1, a2 - 1], a2 * (a2 - 1) / 2 * 0.3])
    if a1 > 0 and a1 + a2 > 1: out.append([[a1 - 1, a2 + 1], 0.25 * a1])
    if a2 > 0 and a1 + a2 > 1: out.append([[a1 + 1, a2 - 1], 0.4 * a2])
    return out

def _main():
    ipv2 = [[[40, 0], 1.0]]
    for name, cb, ipv, theta in [("const", cb_const, [[[6, 0], 0.7], [[5, 1], 0.3]], None),
                                 ("param", cb_param, [[[6, 0], 0.7], [[5, 1], 0.3]], [1.3, 0.7]),
                                 ("island40_const", two_island_const, ipv2, None)]:
        w = with_ipv(ipv)(cb)
        miss = Graph(w, graph_cache=True)
        hit = Graph(w, graph_cache=True)
        fresh = Graph(w, graph_cache=False)
        gs = dict(miss=miss, hit=hit, fresh=fresh)
        if theta is not None:
            for g in gs.values(): g.update_weights(theta)
        E = {k: g.expectation() for k, g in gs.items()}
        V = {k: g.variance() for k, g in gs.items()}
        t = np.array([0.2, 1.0, 3.0])
        P = {k: np.asarray(g.pdf(t)) for k, g in gs.items()}
        print(f"{name:15s} V={fresh.vertices_length():5d} E hit==fresh {E['hit']==E['fresh']} (rel {abs(E['hit']-E['fresh'])/abs(E['fresh']):.2e})"
              f"  Var hit==fresh {V['hit']==V['fresh']} (rel {abs(V['hit']-V['fresh'])/abs(V['fresh']):.2e})"
              f"  pdf bitwise {bool(np.array_equal(P['hit'], P['fresh']))}  miss==fresh E {E['miss']==E['fresh']}")
if __name__ == "__main__":
    _main()
