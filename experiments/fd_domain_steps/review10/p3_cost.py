# Cost of the edge-distance d_k = min_e |C theta|_e / |c_ek| in jnp under jit(vmap) over particles,
# with C closed over as a build-time constant (as a builder closure would). Synthetic sparse C:
# each edge carries one slot (coalescent-like). Reports compile time, run time, peak RSS.
import sys, time, resource
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
E, P, N = int(float(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
C = np.zeros((E, P)); C[np.arange(E), rng.integers(0, P, E)] = rng.uniform(0.5, 5, E)
absC = np.abs(C)
def d_of(theta):
    w = jnp.abs(jnp.asarray(C) @ theta)                       # (E,)
    ratio = jnp.where(absC > 0, w[:, None] / jnp.where(absC > 0, absC, 1.0), jnp.inf)  # (E,P)
    return jnp.min(ratio, axis=0)
f = jax.jit(jax.vmap(d_of))
th = jnp.asarray(rng.uniform(0.1, 2, (N, P)))
t0 = time.time(); f.lower(th).compile(); tc = time.time() - t0
t0 = time.time(); f(th).block_until_ready(); t1 = time.time() - t0
t0 = time.time(); f(th).block_until_ready(); t2 = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"E={E:.0e} P={P} N={N}: compile {tc:.1f}s, first {t1:.2f}s, second {t2:.2f}s, peak RSS {rss:.0f} MB, C bytes {C.nbytes/2**20:.0f} MB")
