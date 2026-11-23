"""Review v5 rollout: what does a 1e11 edge-step ceiling permit on small graphs?
Measures per-step cost of the shipped native step on tiny graphs (the new
method reuses the same iterates) and derives steps / seconds / bytes of the
cached scalar sequence (8 B per step, plan 3.1) at the ceiling.
Also: the shipped divergence-test model (test_svgd_divergence_robustness.py)
at its pinned thetas under uniformization (steps ~ 1.02*Lambda + tail)."""
import time, math
import numpy as np
from phasic import Graph

def coal(state, indexer=None):
    k = int(state[0])
    return [] if k <= 1 else [([k - 1], [k * (k - 1) / 2.0])]

CEIL = 1e11
def n_edges(g):
    return sum(len(v.edges()) for v in g.vertices())

rows = []
for n in (4, 10, 30):
    g = Graph(coal, ipv=[n])
    g.update_weights([1.0])
    V = g.vertices_length(); E = n_edges(g)
    ctx = g.distribution_context()
    N = 400000
    t0 = time.perf_counter()
    for _ in range(N):
        ctx.step()
    dt = time.perf_counter() - t0
    # python loop overhead calibration
    t0 = time.perf_counter()
    for _ in range(N):
        pass
    ov = time.perf_counter() - t0
    ns = (dt - ov) / N * 1e9
    steps = CEIL / E
    print(f"coalescent n={n}: V={V} E={E}: {ns:.0f} ns/step (incl. pybind call) = {ns/E:.1f} ns/edge; "
          f"at ceiling 1e11 edge-steps: steps={steps:.2e}, time~{steps*ns*1e-9/60:.0f} min, "
          f"scalar sequence 8B*steps={8*steps/1e9:.1f} GB")

print()
print("Shipped divergence-test model: coalescent n=4, times up to 2.0, rate = C(4,2)*theta")
g = Graph(coal, ipv=[4]); g.update_weights([1.0]); E = n_edges(g)
for th in (1e8, 1e9, 1e10, 1e15, 1e31):
    lam = 6.0 * th * 2.0
    steps = 1.02 * lam + 6 * math.sqrt(1.02 * lam)
    work = steps * E
    verdict = "UNDER ceiling -> computed" if work <= CEIL else "over ceiling -> raises"
    print(f"  theta={th:.0e}: Lambda={lam:.1e} steps={steps:.2e} work={work:.2e} edge-steps: {verdict}; "
          f"m_k cache={8*steps/1e9:.3g} GB; today: fail-soft penalty (instant)")
