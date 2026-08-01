"""Adversarial verification of the exact_moment_grad root-cause claims, from
an independent review of the B3 default-flip work (commits f89b5b2b onward).
Originally written to characterize two bugs; both are now fixed, so this
script demonstrates the FIXED behavior (kept as a regression/reference tool,
not a bug demo) -- see phasic/svgd.py's _sanitize_and_clip_grad_log_p (the
gradient-norm clip, section D) and pmf_and_moments_from_graph's
_rewards_provided guard (section E).

Run with:  pixi run python experiments/dr_exact_grad_svgd_divergence_verify.py

Sections:
  A) three-way gradient comparison at theta=0.0068 (exact vs FD vs closed
     form) -- confirms the moment-regularization loss has a genuine
     singularity there and exact grad computes it correctly (matches the
     closed form to ~machine precision; FD has ~1e-10 relative truncation
     error).
  B) systematic-bias sweep of exact vs FD over a theta grid (non-singular
     region) -- same pattern holds broadly, not just at the one point.
  C) bit-identity of initial particles between exact/FD SVGD constructions
     -- rules out PRNG-state divergence as an alternative explanation.
  D) short manual reproduction (140 steps) of both trajectories via
     svgd_step directly -- with the gradient-norm clip in place, neither
     trajectory now diverges (compare to the pre-fix behavior recorded in
     the fix(svgd) commit message: the exact trajectory used to reach
     theta~5.4e6 by step 97 and get stuck).
  E) rewards asymmetry probe -- with the _rewards_provided guard in place,
     exact and FD now agree when rewards are supplied (pre-fix, exact
     silently ignored rewards entirely and returned a materially wrong
     gradient).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import phasic                      # must precede jax array creation (x64)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph, SVGD, ExpStepSize


def build_exp_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


np.random.seed(42)
DATA = np.random.exponential(scale=0.5, size=120)
TIMES = jnp.asarray(DATA)
SM = jnp.array([np.mean(DATA), np.mean(DATA ** 2)])
REG = 5.0
print("sample moments m0=%.12g  m1=%.12g" % (float(SM[0]), float(SM[1])))

model_exact = Graph.pmf_and_moments_from_graph(
    build_exp_graph(), nr_moments=2, discrete=False, theta_dim=1,
    exact_moment_grad=True)
model_fd = Graph.pmf_and_moments_from_graph(
    build_exp_graph(), nr_moments=2, discrete=False, theta_dim=1,
    exact_moment_grad=False)


def moment_loss(model):
    def f(theta):
        _, m = model(theta, TIMES)
        return REG * jnp.sum((m[:2] - SM) ** 2)
    return f


def closed_form_loss(theta):
    m = jnp.array([1.0 / theta[0], 2.0 / theta[0] ** 2])
    return REG * jnp.sum((m - SM) ** 2)


# ---------------- A) three-way comparison at the launch point ----------------
print("\n=== A) d(moment_loss)/dtheta ===")
for th in (0.0068, 0.0067668, 0.05, 0.2, 0.5, 2.0):
    t = jnp.array([th])
    ge = float(jax.grad(moment_loss(model_exact))(t)[0])
    gf = float(jax.grad(moment_loss(model_fd))(t)[0])
    gc = float(jax.grad(closed_form_loss)(t)[0])
    print(f"theta={th:<10g} exact={ge: .10e} fd={gf: .10e} closed={gc: .10e} "
          f"| rel(exact,closed)={abs(ge-gc)/abs(gc):.3e} "
          f"rel(fd,closed)={abs(gf-gc)/abs(gc):.3e} "
          f"rel(exact,fd)={abs(ge-gf)/abs(gc):.3e}")

# ------- B) systematic bias in the NON-singular region (chaos vs bias) -------
print("\n=== B) exact-vs-FD relative difference across theta (moments only) ===")
def dm_dtheta(model, t):
    # Jacobian of the raw moment vector, not the loss
    return jax.jacobian(lambda th: model(th, TIMES)[1])(t)
for th in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
    t = jnp.array([th])
    Je = np.asarray(dm_dtheta(model_exact, t)).ravel()
    Jf = np.asarray(dm_dtheta(model_fd, t)).ravel()
    Ja = np.array([-1.0 / th**2, -4.0 / th**3])
    print(f"theta={th:<6g} exact={Je} fd={Jf} analytic={Ja} "
          f"| relerr_exact={np.max(np.abs(Je-Ja)/np.abs(Ja)):.3e} "
          f"relerr_fd={np.max(np.abs(Jf-Ja)/np.abs(Ja)):.3e}")

# ---------------- C) initial particles bit-identical? ----------------
print("\n=== C) initial particles ===")
def _flat_prior(phi):
    return -0.5 * jnp.sum((phi / 10.0) ** 2)

def make_svgd(model):
    return SVGD(model=model, observed_data=DATA, prior=_flat_prior,
                theta_dim=1, n_particles=60, n_iterations=1500,
                learning_rate=ExpStepSize(first_step=0.01, last_step=0.001, tau=500.0),
                seed=21, verbose=False, regularization=REG, nr_moments=2)

se, sf = make_svgd(model_exact), make_svgd(model_fd)
pe, pf = np.asarray(se.theta_init), np.asarray(sf.theta_init)
print("bitwise identical initial particles:", np.array_equal(pe, pf),
      "| max|diff| =", np.max(np.abs(pe - pf)))
print("preconditioner scaling exact:", getattr(se, 'preconditioner', None))

# ---------------- D) 140-step manual reproduction, both paths ----------------
print("\n=== D) 140-step trajectories ===")
from phasic.svgd import svgd_step, SVGDKernel

def trajectory(svgd, n_steps=140):
    lp = svgd._precompile_unified(2, SM, REG, None)
    kern = SVGDKernel(bandwidth='median_per_dim', preconditioner=None)
    parts = svgd.theta_init
    hist = []
    for step in range(n_steps):
        # A schedule (e.g. ExpStepSize) lives in step_schedule with
        # learning_rate left None; a constant lr lives in learning_rate
        # directly with step_schedule wrapping it -- step_schedule is always
        # the callable-or-None source of truth (see SVGD.__init__).
        lr = float(svgd.step_schedule(step, parts)) if svgd.step_schedule is not None \
            else float(svgd.learning_rate)
        # lp is the pre-compiled GRADIENT function (svgd._precompile_unified),
        # so it goes in compiled_grad, not log_prob_fn (svgd_step would
        # otherwise try grad(lp) -- differentiating an already-differentiated
        # function, which fails since the model's pure_callback only defines
        # a VJP, not a JVP).
        parts = svgd_step(parts, None, kern, lr, compiled_grad=lp)
        lo, hi = float(jnp.min(parts)), float(jnp.max(parts))
        hist.append((lo, hi))
        if step % 10 == 0 or abs(hi) > 1e3 or abs(lo) > 1e3:
            print(f"  step {step:4d}  phi range [{lo: .6g}, {hi: .6g}]")
    return hist

print("-- exact --"); he = trajectory(se)
print("-- fd    --"); hf = trajectory(sf)
print("exact final range:", he[-1], " fd final range:", hf[-1])

# ---------------- E) rewards asymmetry ----------------
print("\n=== E) rewards: exact Jacobian vs FD Jacobian ===")
g = build_exp_graph()
nv = g.vertices_length()
rw = jnp.array([1.0] + [3.0] * (nv - 1))       # non-unit reward vector
me = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
                                      theta_dim=1, exact_moment_grad=True)
mf = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
                                      theta_dim=1, exact_moment_grad=False)
t = jnp.array([2.0])
Je = np.asarray(jax.jacobian(lambda th: me(th, TIMES, rewards=rw)[1])(t)).ravel()
Jf = np.asarray(jax.jacobian(lambda th: mf(th, TIMES, rewards=rw)[1])(t)).ravel()
print("with rewards -> exact:", Je, " fd:", Jf,
      "\n  AGREE?" , np.allclose(Je, Jf, rtol=1e-4))
