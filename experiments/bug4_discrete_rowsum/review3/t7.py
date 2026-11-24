"""Tight rule (patched copy, raise above 1+1e-9): gradient at a VALID native-DPH theta close to the row-sum bound."""
import numpy as np, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.CRITICAL)
import phasic; from phasic import Graph
import jax, jax.numpy as jnp
print(phasic.__file__)
def pgeom():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); return g
T = jnp.asarray([1., 2., 3.])
def t(label, f):
    try: print(f'  {label}: OK {np.asarray(f())}')
    except BaseException as e: print(f'  {label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:100]}')
for th0 in (0.5, 0.99, 0.999999, 1 - 2e-7, 1 - 1.5e-7, 1 - 5e-8, 1.0):
    print(f'theta = 1 - {1-th0:.3g}')
    q = jnp.asarray([th0])
    t('value pmf', lambda: Graph.pmf_from_graph(pgeom(), discrete=True)(q, T))
    t('grad pmf[0] (pmf_from_graph)', lambda: jax.grad(lambda x: Graph.pmf_from_graph(pgeom(), discrete=True)(x, T)[0])(q))
    for ffi in (False, True):
        m = Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi=ffi)
        t(f'grad pmf[0] (pmf_and_moments use_ffi={ffi})', lambda: jax.grad(lambda x: m(x, T)[0][0])(q))
        t(f'grad E[N^2] (pmf_and_moments use_ffi={ffi})', lambda: jax.grad(lambda x: m(x, T)[1][1])(q))
