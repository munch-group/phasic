"""Batch G.1 -- public ``Graph.svgd(exact_final_grad=...)`` plumbing.

Plan of record: b3-batchG1-plan.md v2 (tests numbered as its I3 list).
Mirrors the D.4 suite's shape (test_svgd_exact_moment_grad_kwarg.py).
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv
from phasic.exceptions import SvgdConfigError

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

set_log_level("WARNING")

NR = 3
MU = 1e-4
TH = 1 / 10_000


def _base_graph():
    all_pairs = partial(combinations_with_replacement, r=2)
    indexer = StateIndexer(
        lineages=[Property("descendants", min_value=1, max_value=NR)])

    @with_ipv([NR] + [0] * (NR - 1))
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

    return Graph(coal, indexer=indexer), indexer


@pytest.fixture(scope="module")
def fx():
    np.random.seed(17)
    graph, indexer = _base_graph()
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU)
    jpg_d.update_weights([TH, MU])
    jpt = jpg_d.joint_prob_table()
    p = jpt["prob"] / jpt["prob"].sum()
    sample = np.random.choice(jpt.index.values, 30, p=p.to_numpy())
    obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    jpg = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU,
                                 discrete=False)
    jsp = jpg.joint_stop_prob_graph()
    return dict(graph=graph, jpg=jpg, jsp=jsp, obs=obs)


def _svgd(g, obs, **kw):
    # priors only where the caller sets one; svgd's own default (data-
    # informed) covers each leaf's shape correctly.
    kw.setdefault("n_iterations", 1)
    kw.setdefault("n_particles", 4)
    return g.svgd(obs, **kw)


# ---------------------------------------------------------------- test 1
def test_r30_rejections_where_r30_speaks(fx):
    jpg, obs, graph = fx["jpg"], fx["obs"], fx["graph"]
    # joint-prob, no epochs (baked leaf)
    with pytest.raises(SvgdConfigError, match="exact_final_grad requires epoch_starts"):
        _svgd(jpg, obs, exact_final_grad=True)
    with pytest.raises(SvgdConfigError, match="exact_final_grad requires epoch_starts"):
        _svgd(jpg, obs, exact_final_grad=False)
    # plain moments leaf (standard graph, continuous times)
    times = [0.5, 1.0, 1.5]
    with pytest.raises(SvgdConfigError, match="exact_final_grad requires epoch_starts"):
        _svgd(graph, times, exact_final_grad=True)
    # rewards 1-D and 2-D on the standard graph
    n = graph.vertices_length()
    with pytest.raises(SvgdConfigError, match="exact_final_grad requires epoch_starts"):
        _svgd(graph, times, rewards=np.ones(n), exact_final_grad=True)
    with pytest.raises(SvgdConfigError, match="exact_final_grad requires epoch_starts"):
        _svgd(graph, [[0.5, 1.0]] * 3, rewards=np.ones((n, 2)),
              exact_final_grad=True)


# ---------------------------------------------------------------- test 2
def test_r9_jsp_arm_rejects_with_its_own_message(fx):
    jsp, obs = fx["jsp"], fx["obs"]
    with pytest.raises(SvgdConfigError,
                       match="joint-stop-probability graph is not supported"):
        _svgd(jsp, obs, exposure=np.ones(len(obs)), exposure_param_index=1)
    # the jsp message must NOT prescribe epoch_starts as a same-graph fix
    with pytest.raises(SvgdConfigError, match="source joint-prob graph"):
        _svgd(jsp, obs, exposure=np.ones(len(obs)), exposure_param_index=1)
    # existing joint_prob arm untouched
    with pytest.raises(SvgdConfigError, match="vanilla joint-prob"):
        _svgd(fx["jpg"], obs, exposure=np.ones(len(obs)),
              exposure_param_index=1)


# ---------------------------------------------------------------- test 3
def test_frontdoor_single_epoch_exposure_exact(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    import phasic.ffi_wrappers as fw
    calls = []
    orig = fw.compute_daisy_chain_sojourn_ffi

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    fw.compute_daisy_chain_sojourn_ffi = spy
    try:
        expo = np.random.default_rng(0).uniform(0.5, 2.0, len(obs))
        fit_t = _svgd(jpg, obs, exposure=expo, exposure_param_index=1,
                      epoch_starts=[0.0], exact_final_grad=True)
        n_true = len(calls)
        assert n_true > 0, "spy dead"
        assert np.all(np.isfinite(np.asarray(fit_t.particles)))
        calls.clear()
        fit_f = _svgd(jpg, obs, exposure=expo, exposure_param_index=1,
                      epoch_starts=[0.0], exact_final_grad=False)
        n_false = len(calls)
        assert n_false > n_true, (
            f"expected FEWER traced FFI ops with exact ({n_true}) than FD "
            f"({n_false}) -- final-slot FD not removed")
        # G4 fold: pin the ABSOLUTE delta (plan's zero-FD claim). Measured
        # single-epoch: True=8, False=16; delta 8 = 2 central-difference
        # probes x 2 final-epoch slots x 2 gradient traces. A partial
        # removal (off-by-one in _efg_final_lo) changes the delta.
        assert n_false - n_true == 8, (n_true, n_false)
        # value check (G4 fold): the two fits' MODELS agree at a fixed
        # theta (particle positions are separately RNG-seeded per svgd()
        # call and cannot be compared). Tolerance anchored to G4-measured
        # actuals: max rel diff ~1e-6 -> 1e-4 gives ~100x headroom.
        th = jnp.asarray([TH, MU])

        def g(f):
            def loss(t):
                out, _ = f.model(t, None)
                return jnp.sum(jnp.log(out + 1e-300))
            return np.asarray(jax.grad(loss)(th))

        ga, gb = g(fit_t), g(fit_f)
        assert np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))
        assert np.max(np.abs(ga - gb)) / np.max(np.abs(gb)) < 1e-4, (ga, gb)
    finally:
        fw.compute_daisy_chain_sojourn_ffi = orig


# ---------------------------------------------------------------- test 4
def test_frontdoor_multi_epoch_final_fd_removed(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    import phasic.ffi_wrappers as fw
    calls = []
    orig = fw.compute_daisy_chain_sojourn_ffi

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    fw.compute_daisy_chain_sojourn_ffi = spy
    try:
        calls.clear()
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=True)
        n_true = len(calls)
        calls.clear()
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=False)
        n_false = len(calls)
        assert n_true > 0 and n_false > 0, "spy dead"
        assert n_true < n_false, (
            f"final-epoch FD ops not removed (True={n_true}, False={n_false})")
        # G4 fold: pin the delta (measured multi-epoch: True=20, False=28;
        # delta 8 = 2 probes x 2 FINAL-epoch slots x 2 traces -- earlier
        # slots keep FD in BOTH fits, so only the final slots differ).
        assert n_false - n_true == 8, (n_true, n_false)
    finally:
        fw.compute_daisy_chain_sojourn_ffi = orig


# ---------------------------------------------------------------- test 5
def test_explicit_false_bitwise_equals_default(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    # front-door: build via svgd; compare gradients of the fit-exposed
    # models at a fixed theta.
    fit_default = _svgd(jpg, obs, epoch_starts=[0.0, 0.5])
    fit_false = _svgd(jpg, obs, epoch_starts=[0.0, 0.5],
                      exact_final_grad=False)
    th = jnp.asarray([TH, MU, TH * 0.5, MU * 1.3])

    def g(fit):
        def loss(t):
            out, _ = fit.model(t, None)
            return jnp.sum(jnp.log(out + 1e-300))
        return np.asarray(jax.grad(loss)(th))

    np.testing.assert_array_equal(g(fit_default), g(fit_false))


# ---------------------------------------------------------------- test 6
def test_ledger_default_vs_user(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    fit_d = _svgd(jpg, obs, epoch_starts=[0.0, 0.5])
    opts_d = fit_d.effective_options(return_dict=True)
    assert "exact_final_grad" in opts_d
    assert opts_d["exact_final_grad"]["status"] == "default"
    fit_u = _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=True)
    opts_u = fit_u.effective_options(return_dict=True)
    assert opts_u["exact_final_grad"]["status"] == "user"
    assert opts_u["exact_final_grad"]["value"] is True


# ---------------------------------------------------------------- test 8
def test_blast_radius_raise_legible_frontdoor(fx, monkeypatch):
    jpg, obs = fx["jpg"], fx["obs"]
    fit = _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=True)
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        lambda self, *a, **k: [])
    th = jnp.stack([jnp.asarray([TH, MU, TH, MU]),
                    jnp.asarray([TH, MU, TH, MU]) * 1.2])
    with pytest.raises(Exception, match="exact sojourn adjoint declined"):
        jax.vmap(jax.jit(jax.grad(
            lambda t: jnp.sum(jnp.log(fit.model(t, None)[0] + 1e-300))
        )))(th)


# ---------------------------------------------------------------- test 9
def test_tied_exposure_exact_frontdoor(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    expo = np.random.default_rng(1).uniform(0.5, 2.0, len(obs))
    fit = _svgd(jpg, obs, exposure=expo, exposure_param_index=1,
                epoch_starts=[0.0, 0.5], tied=[(0, [0, 1])],
                exact_final_grad=True)
    assert np.all(np.isfinite(np.asarray(fit.particles)))
    # parity vs the FD model through the tied scatter. G4-measured
    # actuals: max rel diff 9.99e-7 (max ABS ~1.1e-4 -- the original
    # comment misreported abs as rel); tolerance 1e-4 gives ~100x
    # headroom. (The plan's oracle-column-sum variant is covered at the
    # builder level by test_exact_grad_daisy_final.py's tied test +
    # the verbatim-forwarding proof in test 11 -- deviation recorded in
    # the merge review.)
    fit_fd = _svgd(jpg, obs, exposure=expo, exposure_param_index=1,
                   epoch_starts=[0.0, 0.5], tied=[(0, [0, 1])],
                   exact_final_grad=False)
    th = jnp.asarray([TH, MU, TH, MU])

    def g(f):
        def loss(t):
            out, _ = f.model(t, None)
            return jnp.sum(jnp.log(out + 1e-300))
        return np.asarray(jax.grad(loss)(th))

    ga, gb = g(fit), g(fit_fd)
    assert np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))
    denom = np.max(np.abs(gb))
    assert np.max(np.abs(ga - gb)) / denom < 1e-4, (ga, gb)


# --------------------------------------------------------------- test 10
def test_constructor_guard_names_graph_svgd(fx):
    from phasic.svgd import SVGD

    def dummy_model(theta, times, rewards=None):
        return jnp.ones(3), jnp.zeros(2)

    with pytest.raises(TypeError, match="Graph.svgd"):
        SVGD(model=dummy_model, observed_data=jnp.asarray([1.0, 2.0, 3.0]),
             theta_dim=2, exact_final_grad=True)


# --------------------------------------------------------------- test 11
def test_forwarding_discrimination(fx, monkeypatch):
    jpg, obs = fx["jpg"], fx["obs"]
    seen = {}
    orig = Graph._daisy_chain_svgd_model

    def spy(self, **kw):
        seen.update(kw)
        return orig(self, **kw)

    monkeypatch.setattr(Graph, "_daisy_chain_svgd_model", spy)
    seen.clear()
    _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=False)
    assert seen.get("exact_final_grad") is False, seen.keys()
    seen.clear()
    _svgd(jpg, obs, epoch_starts=[0.0, 0.5])
    assert "exact_final_grad" not in seen, "None must NOT be forwarded"


# --------------------------------------------------------------- test 12
def test_preemption_branches_config_level(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    with pytest.raises(SvgdConfigError, match="final_read='sojourn'"):
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], final_read="stopprob",
              exact_final_grad=True)
    with pytest.raises(SvgdConfigError, match="weight_mode='linear'"):
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5],
              weight_formula="t0 * c0 + t1 * c1", exact_final_grad=True)
    # G4 fold: R30 is deliberately stricter than the plan's letter -- it
    # rejects ANY non-None (the R29 discipline), so explicit False on
    # these cells rejects too (tested, recorded in the merge review).
    with pytest.raises(SvgdConfigError, match="final_read='sojourn'"):
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], final_read="stopprob",
              exact_final_grad=False)
    with pytest.raises(SvgdConfigError, match="weight_mode='linear'"):
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5],
              weight_formula="t0 * c0 + t1 * c1", exact_final_grad=False)


def test_r30_jsp_no_epochs_kind_aware_message(fx):
    # G4 fold: without this branch, a jsp user would be told to "add
    # epoch_starts", which R1 forbids on jsp -- the same two-hop dead
    # end this batch's R9 jsp arm fixes.
    with pytest.raises(SvgdConfigError, match="source joint-prob graph"):
        _svgd(fx["jsp"], fx["obs"], exact_final_grad=True)


# --------------------------------------------------------------- test 13
@pytest.mark.parametrize("leaf", ["moments", "rewards1d", "rewards2d",
                                  "joint_prob", "jsp", "epoch"])
def test_none_never_raises(fx, leaf):
    graph, jpg, jsp, obs = fx["graph"], fx["jpg"], fx["jsp"], fx["obs"]
    times = [0.5, 1.0, 1.5]
    n = graph.vertices_length()
    if leaf == "moments":
        _svgd(graph, times, exact_final_grad=None)
    elif leaf == "rewards1d":
        _svgd(graph, times, rewards=np.ones(n), exact_final_grad=None)
    elif leaf == "rewards2d":
        # validation-level (a full 2-D-rewards fit needs the multivariate
        # fixture; the None-inertness property is a validation property).
        from phasic.svgd_config import from_svgd_call, validate
        validate(from_svgd_call(graph, [[0.5, 1.0]] * 3,
                                rewards=np.ones((n, 2)),
                                exact_final_grad=None))
    elif leaf == "joint_prob":
        _svgd(jpg, obs, exact_final_grad=None)
    elif leaf == "jsp":
        # A full jsp FIT needs jsp-specific observation semantics (the
        # jpg outcome tuples don't map); the None-inertness property is
        # a VALIDATION property, so validate the config directly.
        from phasic.svgd_config import from_svgd_call, validate
        validate(from_svgd_call(jsp, obs, exact_final_grad=None))
    elif leaf == "epoch":
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_final_grad=None)


# --------------------------------------------------------------- test 14
def test_fixed_and_scalar_exposure_frontdoor(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    # fixed mutation slot + per-obs exposure + single epoch + exact:
    fit = _svgd(jpg, obs,
                exposure=np.random.default_rng(2).uniform(0.5, 2.0, len(obs)),
                exposure_param_index=1, epoch_starts=[0.0],
                fixed=[(1, MU)], exact_final_grad=True)
    th = jnp.asarray([TH, MU])

    def loss(t):
        out, _ = fit.model(t, None)
        return jnp.sum(jnp.log(out + 1e-300))

    g = np.asarray(jax.grad(loss)(th))
    assert g[1] == 0.0, f"fixed slot gradient must be exactly 0.0, got {g[1]}"
    assert np.isfinite(g[0]) and g[0] != 0.0
    # scalar exposure (broadcast + K_unique == 1) + exact:
    fit_s = _svgd(jpg, obs, exposure=2.0, exposure_param_index=1,
                  epoch_starts=[0.0], exact_final_grad=True)
    assert np.all(np.isfinite(np.asarray(fit_s.particles)))
