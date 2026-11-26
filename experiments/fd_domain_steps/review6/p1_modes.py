"""Review 6 probe: today's FD backward at small theta across weight modes,
on the Euler sites (pmf_from_graph, pmf_and_moments_from_graph), plus a
direct forward at the probe point the plan's 'unconstrained' rule
(theta - 1e-7, no clamp) would use. Master, no source change."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")


def g2(coeffs_ab=(1.0, 1.0), coeffs_ac=(2.0, 1.0)):
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, list(coeffs_ab))
    b.add_edge(c, list(coeffs_ac))
    return g


def make(mode):
    g = g2()
    if mode == 'log':
        g.weight_mode = 'log'
    elif mode == 'formula_div':
        g.weight_formula = "c0*t0/(c1*t1)"      # rate ~ 1/t1 (e.g. 1/N)
    elif mode == 'formula_sqrt':
        g.weight_formula = "c0*t0*sqrt(c1*t1)"
    elif mode == 'formula_lin':
        g.weight_formula = "c0*t0 + c1*t1"
    return g


times = jnp.array([0.5, 1.0, 2.0])


def run(label, f, th):
    try:
        v = f(jnp.array(th))
        print(f"{label:58s} theta={th} -> {np.asarray(v)}", flush=True)
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"{label:58s} theta={th} -> RAISES {type(e).__name__}: {msg[:110]}", flush=True)


for mode in ('linear', 'log', 'formula_lin', 'formula_div', 'formula_sqrt'):
    g = make(mode)
    m1 = Graph.pmf_from_graph(g, theta_dim=2)
    m2 = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
    m2fd = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2,
                                            exact_moment_grad=False)

    def l1(th, m=m1):
        return jnp.sum(jnp.log(m(th, times)))

    def l2(th, m=m2):
        p, mo = m(th, times)
        return jnp.sum(jnp.log(p)) + jnp.sum(mo) * 1e-3

    def l2fd(th, m=m2fd):
        p, mo = m(th, times)
        return jnp.sum(jnp.log(p)) + jnp.sum(mo) * 1e-3

    def fwd2(th, m=m2):
        p, mo = m(th, times)
        return p
    for th in ([1.0, 1.0], [1.0, 5e-8], [1.0, 1e-7], [1.0, 1e-12]):
        run(f"[{mode}] pmf_from_graph grad", jax.grad(l1), th)
        run(f"[{mode}] pmf_and_moments grad (exact_mom)", jax.grad(l2), th)
        run(f"[{mode}] pmf_and_moments grad (FD mom)", jax.grad(l2fd), th)
    # plan's 'unconstrained' probe point at theta=[1,5e-8]: theta-1e-7, no clamp
    run(f"[{mode}] pmf_and_moments FORWARD at unconstrained probe", fwd2,
        [1.0, 5e-8 - 1e-7])
    # capped-positive rule at theta=1e9: exp(+-u) with u=log1p(1e-7/1e9)
    th = 1e9
    u = np.log1p(1e-7 / th)
    print(f"[{mode}] capped probe at 1e9: up-th={th*np.exp(u)-th:.3e} "
          f"dn-th={th*np.exp(-u)-th:.3e}; jnp: "
          f"{float(jnp.float64(th)*jnp.exp(jnp.float64(u)) - th):.3e}", flush=True)
