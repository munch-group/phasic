"""Anomalies from the refusal grid (g3_summary.txt):
(a) linear continuous + 1-D rewards: the forward refused -1e-9 and -1e-7 but
    accepted -5e-8. Finer scan with full messages.
(b) today's gradient at theta = 0 on linear pmf_and_moments raises a message
    containing 'nan': full message, and the same for moments_from_graph's
    primal at 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from g1_refusal_grid import build, rewards_for, TIMES


def full(f, *a):
    try:
        v = f(*a)
        return "ok " + str(np.round(np.asarray(jax.tree_util.tree_leaves(v)[0]), 8))
    except Exception as e:  # noqa: BLE001
        m = str(e).replace("\n", " ")
        i = m.find("RuntimeError:")
        return "RAISE " + (m[i:i + 260] if i >= 0 else m[-260:])


g = build("linear", "cont")
rw = rewards_for(g)
print("rewards", np.asarray(rw), "vertices", g.vertices_length())
m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
print("(a) forward with 1-D rewards, slot 1 = p")
for p in (-1e-12, -1e-10, -1e-9, -5e-9, -1e-8, -2e-8, -4e-8, -5e-8, -6e-8, -8e-8, -1e-7, -2e-7, -1e-3):
    print(f"   p={p:.0e}:", full(lambda th: m(th, TIMES, rewards=rw), jnp.asarray([1.0, p])))
print("   same, without rewards, p=-5e-8:", full(lambda th: m(th, TIMES), jnp.asarray([1.0, -5e-8])))
print("(b) gradient at slot = 0")
L = lambda th: jnp.sum(m(th, TIMES)[0]) + 1e-3 * jnp.sum(m(th, TIMES)[1])
print("   pam grad at [1, 0]:", full(jax.grad(L), jnp.asarray([1.0, 0.0])))
print("   pam forward at [1, 0]:", full(lambda th: m(th, TIMES), jnp.asarray([1.0, 0.0])))
m2 = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
L2 = lambda th: jnp.sum(m2(th, TIMES)[0]) + 1e-3 * jnp.sum(m2(th, TIMES)[1])
print("   pam grad at [1, 0], exact_moment_grad=False:", full(jax.grad(L2), jnp.asarray([1.0, 0.0])))
L3 = lambda th: jnp.sum(m(th, TIMES)[0])
print("   pam grad at [1, 0], pdf-only loss:", full(jax.grad(L3), jnp.asarray([1.0, 0.0])))
if os.environ.get("PHASIC_SOURCE_DIR"):
    mm = Graph.moments_from_graph(g, nr_moments=2)
    print("   moments_from_graph forward at [1, 0]:", full(mm, jnp.asarray([1.0, 0.0])))
    print("   moments_from_graph forward at [1, 1e-12]:", full(mm, jnp.asarray([1.0, 1e-12])))
