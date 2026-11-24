import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, phasic
from phasic import Graph
def pgeom():
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0]); g.is_discrete = True
    return g
def tryit(label, f):
    try: r = f(); print(label, "->", np.round(np.asarray(r, dtype=float), 6).tolist())
    except Exception as e: print(label, "RAISE", type(e).__name__, str(e)[:400].replace("\n"," "))
# Build a model first: phasic enables jax_enable_x64 lazily when jax was
# imported before it, so arrays must be created after that.
m = Graph.pmf_from_graph(pgeom(), discrete=True)
K = jnp.array([1.0, 2.0, 3.0]); assert K.dtype == jnp.float64
for th in (1 + 1e-5, 1 + 1e-12, 0.5):
    tryit(f"pmf_from_graph th={th}", lambda: m(jnp.array([th]), K))
for ffi in (False, True):
    mm = Graph.pmf_and_moments_from_graph(pgeom(), discrete=True, use_ffi=ffi)
    for th in (1 + 1e-5, 1 + 1e-12, 1.5):
        tryit(f"pmf_and_moments ffi={ffi} th={th}", lambda: mm(jnp.array([th]), K)[0])
        tryit(f"pmf_and_moments EMPTY ffi={ffi} th={th}", lambda: mm(jnp.array([th]), jnp.array([]))[1])
    tryit(f"vmap ffi={ffi}", lambda: jax.vmap(lambda t: mm(t, K)[0])(jnp.array([[1 + 1e-5], [1 + 1e-5]])))
for gap in (5e-8, 2e-7):
    tryit(f"grad gap={gap}", lambda: jax.grad(lambda t: jnp.sum(jnp.log(m(t, K))))(jnp.array([1 - gap])))
    tryit(f"value gap={gap}", lambda: m(jnp.array([1 - gap]), K))
try:
    m(jnp.array([1 + 1e-5]), K)
except Exception as e:
    print("is_rate_blowup:", phasic._is_rate_blowup(e), type(e).__name__)
