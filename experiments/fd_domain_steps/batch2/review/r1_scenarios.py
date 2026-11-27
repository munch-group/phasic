"""Batch 2 review probes: declaration/containment scenarios."""
import sys
sys.path.insert(0, "/Users/kmt/phasic-fdsteps/tests/pytest")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph, SVGD, GaussPrior, BetaPrior
import _fd_golden_cells as C

DATA = np.asarray([0.3, 0.5, 0.8, 1.1, 1.7, 2.4])
FIT = dict(n_particles=4, n_iterations=2, progress=False, seed=1, learning_rate=1e-6)
gp = lambda: GaussPrior(1.0, 0.5, lower=0.0)


def run(name, fn):
    try:
        out = fn()
        print(f"[{name}] OK: {out}", flush=True)
    except Exception as e:
        print(f"[{name}] RAISE {type(e).__name__}: {str(e)[:400]}", flush=True)


g = C.beside()


def s1():
    s = g.svgd(DATA, prior=[gp(), gp()], **FIT)
    SVGD(model=s.model, observed_data=DATA, theta_dim=2, prior=GaussPrior(1, 1),
         positive_params=False, n_particles=4, n_iterations=1, progress=False)
    return s.model._param_domains
run("reuse_graph_svgd_model_positive_params_False", s1)


def s1b():
    s = g.svgd(DATA, prior=[gp(), None], fixed=[(1, 0.5)], **FIT)
    SVGD(model=s.model, observed_data=DATA, theta_dim=2, prior=[gp(), gp()],
         n_particles=4, n_iterations=1, progress=False)
run("reuse_nested_model_for_full", s1b)

_orig = Graph.__dict__['pmf_and_moments_from_graph']
_orig_fn = Graph.pmf_and_moments_from_graph
cap = {}


def spy(*a, **k):
    cap['d'] = k.get('param_domains')
    return _orig_fn(*a, **k)


Graph.pmf_and_moments_from_graph = staticmethod(spy) if isinstance(_orig, staticmethod) else spy


def s2b():
    cap.clear()
    g.svgd(DATA, prior=[gp(), BetaPrior(2, 2)], exposure=2.0, exposure_param_index=1, **FIT)
    return dict(cap)
run("single_epoch_exposure_beta_declared", s2b)


def s3():
    cap.clear()
    g.svgd(DATA, prior=[gp(), None], fixed=[0, 1], **FIT)
    return dict(cap)
run("fixed_as_mask", s3)


def s4():
    cap.clear()
    ti = np.full((4, 2), 0.7)
    g.svgd(DATA, prior=[BetaPrior(2, 2), gp()], theta_init=ti, n_iterations=1,
           progress=False, learning_rate=1e-6)
    return dict(cap)
run("theta_init_only", s4)


def s5():
    cap.clear()
    g.svgd(DATA, prior=[GaussPrior(0, 1), None], positive_params=False,
           param_transform=lambda p: jnp.exp(p), fixed=[(1, 0.5)], **FIT)
    return dict(cap)
run("custom_transform_fixed", s5)

Graph.pmf_and_moments_from_graph = _orig

jpg, oi = C.joint_graphs()
rows = C.joint_rows()


def s6():
    s = jpg.svgd(rows, epoch_starts=[0.0, 0.5], prior=[BetaPrior(2, 2), None],
                 fixed=[(1, 1e-2)], **FIT)
    em = jpg.epoch_model(rows, [0.0, 0.5], prior=[BetaPrior(2, 2), None], fixed=[(1, 1e-2)])
    th = jnp.asarray([0.3, 1e-2, 0.6, 1e-2])
    L = lambda m: (lambda t: jnp.sum(jnp.log(m(t, None)[0])))
    ga = np.asarray(jax.grad(L(s.model))(th))
    gb = np.asarray(jax.grad(L(em.model))(th))
    return s.model._param_domains, em._param_domains, ga, gb, (ga - gb)[::2] / ga[::2]
run("epoch_parity_betaprior", s6)


def s7():
    em = jpg.epoch_model(rows, [0.0, 0.5], fixed=[(1, 1e-2)])
    em.fit(positive_params=False, **FIT)
run("epoch_model_pp_false_default_prior", s7)


def s8():
    em = jpg.epoch_model(rows, [0.0, 0.5], fixed=[(1, 1e-2)], param_domains=['unit', 'positive'],
                         exposure=np.linspace(0.5, 3.0, len(rows)), exposure_param_index=0)
    r = em.fit(prior=[BetaPrior(2, 2), None, BetaPrior(2, 2), None], **FIT)
    return em._param_domains, np.asarray(r.theta_mean)
run("epoch_model_unit_on_exposure_slot", s8)


def s9():
    flat = [gp(), None, BetaPrior(2, 2), None]
    s = jpg.svgd(rows, epoch_starts=[0.0, 0.5], prior=flat, fixed=[(1, 1e-2)],
                 tied=[(0, [0, 1])], **FIT)
    return s.model._param_domains
run("tied_flat_prior_slave_beta", s9)
