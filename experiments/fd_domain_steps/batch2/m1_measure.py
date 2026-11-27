"""Plan A Batch 2 measurements (sets the test tolerances).

1. declared ('positive') vs undeclared (today) gradients at fixed particles,
   per Graph.svgd leaf, at theta scale 1, 1e-3, 1e-4, 1e-5, plus a
   Richardson reference (log-coordinate central differences) to show which
   one is closer;
2. the band rule on refusing models (pole in log mode, offset in formula
   mode) and the plain step on a non-refusing offset, at 5e-8, 1e-9, 1e-13;
3. the D-NZ sign pin (linear, no rewards, exact_moment_grad=False, pole).

Run: pixi run python experiments/fd_domain_steps/batch2/m1_measure.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "pytest"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph
import _fd_golden_cells as C

TIMES = jnp.asarray(C.TIMES)
RW = jnp.asarray(C.REWARDS)


def leaves_loss(f):
    """SVGD's likelihood term: sum of log pdf / probability (the first
    output only; moments enter SVGD only through regularization)."""
    return lambda th: jnp.sum(jnp.log(f(th)[0]))


def rich(L, th, i, s=1e-2):
    def cd(h):
        tp = th.at[i].set(th[i] * np.exp(h)); tm = th.at[i].set(th[i] * np.exp(-h))
        return float((L(tp) - L(tm)) / (tp[i] - tm[i]))
    a, b = cd(s), cd(s / 2)
    return (4 * b - a) / 3


def leaf(name, doms):
    g = C.beside()
    if name == "pam":
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, param_domains=doms)
        return lambda th: m(th, TIMES)
    if name == "pam_rw":
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, param_domains=doms)
        return lambda th: m(th, TIMES, rewards=RW)
    if name == "multivar":
        m = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, theta_dim=2,
                                                          param_domains=doms)
        return lambda th: m(th, TIMES, rewards=jnp.stack([RW, 2 * RW]))
    if name == "jix":
        jpg, oi = C.joint_graphs()
        m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2, param_domains=doms)
        return lambda th: m(th, jnp.asarray(oi))
    if name == "epoch":
        jpg, oi = C.joint_graphs()
        d4 = None if doms is None else doms * 2
        m, _, _, _ = jpg._daisy_chain_svgd_model(
            observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
            user_prior=None, user_fixed=None, param_domains=d4)
        return lambda th: m(th, None)
    raise KeyError(name)


def part1():
    print("== 1. declared vs today at fixed particles (rel diff; rel err vs Richardson)")
    for name in ("pam", "pam_rw", "multivar", "jix", "epoch"):
        P = 4 if name == "epoch" else 2
        for scale in (1.0, 1e-3, 1e-4, 1e-5):
            base = np.array([1.0, 0.6, 1.3, 0.8][:P]) * scale
            th = jnp.asarray(base)
            Ld = leaves_loss(leaf(name, ["positive"] * 2))
            Lu = leaves_loss(leaf(name, None))
            gd = np.asarray(jax.grad(Ld)(th)); gu = np.asarray(jax.grad(Lu)(th))
            rel = np.abs(gd - gu) / np.maximum(np.abs(gu), 1e-300)
            ref = np.array([rich(Lu, th, i) for i in range(P)])
            ed = np.abs(gd - ref) / np.abs(ref); eu = np.abs(gu - ref) / np.abs(ref)
            print(f"{name:9s} scale={scale:7.0e} max rel(decl-today)={rel.max():.2e}  "
                  f"err decl={ed.max():.2e} today={eu.max():.2e}", flush=True)


def pole_log():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); b.add_edge(c, [2.0])
    g.weight_mode = "log"
    return g


def offset(mode):
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([2]); a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0, 1.0])
    if mode == "formula":
        g.weight_formula = "c0*t0 + c1*t1"
    return g


def mean_grad(g, th, doms, i):
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=len(th),
                                         exact_moment_grad=False, param_domains=doms)
    gr = np.asarray(jax.grad(lambda t: m(t, TIMES)[1][0])(jnp.asarray(th)))
    return float(gr[i]), m


def part2():
    print("== 2. band rule (refusing) and plain step (non-refusing)")
    for v in (5e-8, 1e-9, 1e-13):
        gx, m = mean_grad(pole_log(), [v], ["unconstrained"], 0)
        ex = -1.5 / v ** 2
        print(f"log pole  t={v:.0e}: fd={gx:.10e} exact={ex:.10e} rel={abs(gx/ex-1):.2e} "
              f"tie={m._fd_stats.tie}", flush=True)
    for v in (5e-8, 1e-9, 1e-13):
        gx, m = mean_grad(offset("formula"), [1.0, v], ["unconstrained"] * 2, 1)
        ex = -1.0 / (1.0 + v) ** 2
        print(f"formula offset t1={v:.0e}: fd={gx:.10e} exact={ex:.10e} rel={abs(gx/ex-1):.2e} "
              f"tie={m._fd_stats.tie}", flush=True)
    for v in (5e-8, 1e-9, 1e-13):
        gx, m = mean_grad(offset("linear"), [1.0, v], ["unconstrained"] * 2, 1)
        gt, _ = mean_grad(offset("linear"), [1.0, v], None, 1)
        ex = -1.0 / (1.0 + v) ** 2
        print(f"linear offset t1={v:.0e}: plain={gx:.10e} today={gt:.10e} rel plain={abs(gx/ex-1):.2e} "
              f"today={abs(gt/ex-1):.2e}", flush=True)


def two_stage():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0])
    return g


def part3():
    print("== 3. D-NZ pin: two-stage, E[T] = 1/t0 + 1/t1, slot 1 at 5e-8")
    gx, _ = mean_grad(two_stage(), [1.0, 5e-8], ["unconstrained"] * 2, 1)
    gt, _ = mean_grad(two_stage(), [1.0, 5e-8], None, 1)
    print(f"plain={gx:.10e} today(clamp)={gt:.10e} exact={-1/(5e-8)**2:.10e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "part1":
        part1()
    else:
        part3(); part2(); part1()
