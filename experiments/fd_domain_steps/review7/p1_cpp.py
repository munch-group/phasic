import os, sys, traceback
import jax, jax.numpy as jnp
from phasic import Graph
here = os.path.dirname(os.path.abspath(__file__))
m = Graph.pmf_from_cpp(os.path.join(here, "m_cpp.cpp"))
t = jnp.array([0.5, 1.0, 2.0])
for th in ([1.0, 1.0], [1.0, 2e-7], [1.0, 5e-8], [1.0, 0.0]):
    th = jnp.array(th)
    try:
        print("fwd", th, m(th, t))
    except Exception as e:
        print("fwd RAISE", th, type(e).__name__, str(e)[:200]); continue
    try:
        g = jax.grad(lambda p: jnp.sum(m(p, t)))(th)
        print("grad", th, g)
    except Exception as e:
        print("grad RAISE", th, type(e).__name__, str(e)[:300])
