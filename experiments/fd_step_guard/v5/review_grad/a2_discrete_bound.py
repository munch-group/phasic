"""A2 (prototype worktree): DISCRETE model, parameter is a probability with an upper bound.
Graph.svgd default = softplus -> the plan declares (0, inf) -> discrete = "jump-free" -> FULL log step.
PHASIC_FD_RULE=rel, c=1e-4 ~ the plan's log step s=1e-4; 'old' = today.
Model: geometric with success probability c*theta (c = 1 and c = 3): pmf(n) = q (1-q)^(n-1), q = c*theta.
d log pmf(n) / d theta = 1/theta - (n-1) c / (1 - c theta)."""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph
def geo(c):
    g = Graph(1); s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0); v2.add_edge(v1, [float(c)])
    return Graph.pmf_and_moments_from_graph(g, nr_moments=1, discrete=True, theta_dim=1)
N = jnp.array([1, 2, 3])
for c in (1.0, 3.0):
    M = geo(c)
    print(f'### success probability = {c:g} * theta (upper bound theta < {1/c:.4g})')
    for gap in (1e-1, 1e-3, 2e-4, 9e-5, 5e-5, 1e-6, 5e-8):
        th = (1.0 - gap) / c
        try:
            fwd = np.asarray(M(jnp.array([th]), N)[0])
        except Exception as e:
            fwd = f'FWD {type(e).__name__}'
        ex = sum(1/th - (n - 1) * c / (1 - c * th) for n in (1, 2, 3))
        row = []
        for name, env in (('today', ('old', '1')), ('log-step s=1e-4', ('rel', '1e-4'))):
            os.environ['PHASIC_FD_RULE'], os.environ['PHASIC_FD_REL'] = env
            try:
                v = float(jax.grad(lambda t: jnp.sum(jnp.log(M(t, N)[0])))(jnp.array([th]))[0])
                row.append(f'{name}: {v:+.5e} (rel err {abs(v-ex)/abs(ex):.1e})')
            except Exception as e:
                row.append(f'{name}: RAISES {type(e).__name__}: {str(e)[:70]!r}')
        print(f' 1-c*theta={gap:g}: exact {ex:+.5e} | ' + ' | '.join(row), flush=True)
    for over in (1e-5, 1e-2):
        th = (1.0 + over) / c
        try:
            print(f' forward ABOVE the bound (c*theta = 1+{over:g}):', np.asarray(M(jnp.array([th]), N)[0]))
        except Exception as e:
            print(f' forward ABOVE the bound (c*theta = 1+{over:g}): RAISES {type(e).__name__}: {str(e)[:120]!r}')
