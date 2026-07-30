"""Pinning tests for the finite-difference (FD) gradient defect at mixed scales.

Every SVGD-facing gradient in phasic is a ``custom_vjp`` whose backward re-runs
the opaque C++/FFI forward at theta +/- eps and takes a CENTRAL DIFFERENCE with
``eps = 1e-7`` ABSOLUTE, the SAME step for every parameter (moments backward
``__init__.py`` ~:6786, continuous pdf ~:3681, daisy-chain ~:4649/:4880,
reward-visit ``ffi_wrappers.py`` ~:1296).

An absolute step cannot serve parameters of very different magnitude at once — a
per-generation mutation rate (~1e-8) beside a coalescent rate (~1), which is the
regime the library exists for. These tests document, against an EXACT oracle,
that the FD gradient is:

  * accurate at moderate/uniform scale (the regime the existing fixtures test)
    -- the two ``*_accurate_*`` tests PASS;
  * a DEFECT at mixed scale -- the two ``*_mixed_scale_*`` tests are
    ``xfail(strict=True)``: today the FD gradient is grossly wrong (Part A) or
    HARD-CRASHES because the step drives a small rate negative (Part B, the
    daisy path). Each asserts the CORRECT behaviour, so when a B3 exact-AD /
    scale-aware fix lands they XPASS and the strict marker forces flipping them
    to normal tests.

Provenance: audit-situation-map.md:165,196 ("broken in BOTH step regimes … no
central-difference step can fix this"); reproductions in
scratchpad/fd_repro_A.py, fd_repro_B.py.
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


# --------------------------------------------------------------------------- #
# Part A — general FD path vs a CLOSED-FORM exact oracle
# --------------------------------------------------------------------------- #
# Model: S -> [3] --t0--> [2] --t1--> [1(absorbing)].  Time to absorption =
# Exp(t0) + Exp(t1), so E[T] = 1/t0 + 1/t1 (verified: model forward == closed
# form), and the EXACT gradient is [-1/t0^2, -1/t1^2].
def _two_stage_model():
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v1, [0.0, 1.0])
    return Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)


_MODEL_A = _two_stage_model()
_TIMES = jnp.array([1.0])


def _ET(theta):
    _pmf, moments = _MODEL_A(theta, _TIMES)
    return moments[0]


def _oracle(theta):
    return np.array([-1.0 / theta[0] ** 2, -1.0 / theta[1] ** 2])


_grad_A = jax.grad(_ET)


def test_fd_gradient_accurate_at_moderate_scale():
    """Sanity/positive control: FD == exact oracle when parameters are ~1e-4
    (the regime the existing fixtures live in)."""
    theta = [1.0, 1e-4]
    fd = np.array(_grad_A(jnp.array(theta)))
    np.testing.assert_allclose(fd, _oracle(theta), rtol=1e-3)


@pytest.mark.xfail(strict=True, reason="FD gradient broken at mixed parameter "
                   "scales (B3): abs eps=1e-7 gives 4-9% error at theta=[1,1e-8]")
def test_fd_gradient_correct_at_mixed_scale():
    """At theta=[1, 1e-8] the absolute-step central difference is 4-9% wrong on
    BOTH components (the small param's step even straddles a negative rate).
    A correct (analytic) gradient would match the oracle to ~1e-9."""
    theta = [1.0, 1e-8]
    fd = np.array(_grad_A(jnp.array(theta)))
    np.testing.assert_allclose(fd, _oracle(theta), rtol=1e-2)


def test_fd_no_single_step_works_at_mixed_scale():
    """Positive documentation test (PASSES): prove the 'broken in both step
    regimes' claim directly — no single absolute central-difference step gives
    an accurate gradient for BOTH a ~1 and a ~1e-8 parameter simultaneously."""
    theta = np.array([1.0, 1e-8])
    o0, o1 = _oracle(theta)

    def cfd(i, eps):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        if tm[i] <= 0 or tp[i] <= 0:
            return None                      # step drives a rate <= 0
        return (float(_ET(jnp.array(tp))) - float(_ET(jnp.array(tm)))) / (2 * eps)

    def ok(i, o, eps):
        v = cfd(i, eps)
        return v is not None and abs(v - o) / abs(o) < 1e-2

    steps = [1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    both_ok = [eps for eps in steps if ok(0, o0, eps) and ok(1, o1, eps)]
    assert both_ok == [], (
        f"expected NO single step to work for both params, but {both_ok} did"
    )


# --------------------------------------------------------------------------- #
# Part B — the daisy-chain FD gradient specifically
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _daisy():
    from phasic import with_ipv, StateIndexer, Property, set_log_level
    set_log_level("ERROR")
    all_pairs = partial(combinations_with_replacement, r=2)
    nr = 4
    indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])

    @with_ipv([nr] + [0] * (nr - 1))
    def coal(state, indexer=None):
        t = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy(); new[i] -= 1; new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return t

    graph = Graph(coal, indexer=indexer)
    mu = 1e-4
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=mu)
    jpg_d.update_weights([1 / 10_000, mu])
    obs = jpg_d.joint_prob_table().iloc[:12][jpg_d.joint_prob_table().columns[:-1]].to_numpy().tolist()
    jpg = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=mu, discrete=False)
    model = jpg.epoch_model(obs, epoch_starts=[0, 0.5]).model   # 4 free flat slots

    def loglik(theta):
        per_obs, _ = model(theta, None)
        return jnp.sum(jnp.log(per_obs + 1e-300))

    return loglik, jax.grad(loglik)


def _scale_matched_ref(loglik, theta, rel=1e-4):
    theta = np.array(theta, float)
    g = np.full(len(theta), np.nan)
    for i in range(len(theta)):
        eps = rel * abs(theta[i])
        if eps == 0 or theta[i] - eps <= 0:
            continue
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        g[i] = (float(loglik(jnp.array(tp))) - float(loglik(jnp.array(tm)))) / (2 * eps)
    return g


def test_daisy_fd_accurate_at_moderate_scale(_daisy):
    """Positive control: daisy FD gradient matches a scale-matched reference
    when all params are ~1e-4 (the tested regime)."""
    loglik, grad_fd = _daisy
    theta = [1e-4, 1e-4, 1e-4, 1e-4]
    fd = np.array(grad_fd(jnp.array(theta)))
    ref = _scale_matched_ref(loglik, theta)
    np.testing.assert_allclose(fd, ref, rtol=1e-3)


@pytest.mark.xfail(strict=True, reason="daisy-chain FD gradient broken at mixed "
                   "scale (B3): abs eps=1e-7 drives mu~1e-8 to a NEGATIVE "
                   "transition -> DaisyChainSojournFfiImpl aborts the backward")
def test_daisy_fd_correct_at_mixed_scale(_daisy):
    """At theta=[1, 1e-8, 1, 1e-8] the production daisy FD backward HARD-CRASHES
    (the -1e-7 step drives mu = 1e-8 negative and the FFI rejects it), even
    though the true gradient is finite and computable (the scale-matched
    reference returns it). A scale-aware / analytic fix would compute it."""
    loglik, grad_fd = _daisy
    theta = [1.0, 1e-8, 1.0, 1e-8]
    fd = np.array(grad_fd(jnp.array(theta)))          # currently raises ValueError
    ref = _scale_matched_ref(loglik, theta)
    assert np.all(np.isfinite(fd))
    np.testing.assert_allclose(fd, ref, rtol=1e-2)
