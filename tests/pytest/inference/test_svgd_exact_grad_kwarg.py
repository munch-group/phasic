"""Batch E -- public ``Graph.svgd(exact_grad=...)`` plumbing (leaf 2, baked).

Plan of record: b3-batchE-plan.md v2 (I3, svgd level). Mirrors the
G.1/D.4 kwarg-suite shape.
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
MU, TH = 1e-4, 1 / 10_000


def _base():
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
    graph, indexer = _base()
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU)
    jpg_d.update_weights([TH, MU])
    jpt = jpg_d.joint_prob_table()
    p = jpt["prob"] / jpt["prob"].sum()
    sample = np.random.choice(jpt.index.values, 30, p=p.to_numpy())
    obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    jpg = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU,
                                 discrete=False)
    return dict(graph=graph, jpg=jpg, jpg_default=jpg_d,
                jsp=jpg.joint_stop_prob_graph(), obs=obs)


def _svgd(g, obs, **kw):
    kw.setdefault("n_iterations", 1)
    kw.setdefault("n_particles", 4)
    return g.svgd(obs, **kw)


# ---------------------------------------------------------------- R31
def test_r31_rejections(fx):
    graph, jpg, jpg_d, obs = fx["graph"], fx["jpg"], fx["jpg_default"], fx["obs"]
    times = [0.5, 1.0, 1.5]
    n = graph.vertices_length()
    # epochs -> exact_final_grad pointer
    with pytest.raises(SvgdConfigError, match="exact_final_grad"):
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_grad=True)
    # moments leaf
    with pytest.raises(SvgdConfigError, match="exact_moment_grad"):
        _svgd(graph, times, exact_grad=True)
    # rewards 1-D and 2-D
    with pytest.raises(SvgdConfigError, match="exact_grad"):
        _svgd(graph, times, rewards=np.ones(n), exact_grad=True)
    with pytest.raises(SvgdConfigError, match="exact_grad"):
        _svgd(graph, [[0.5, 1.0]] * 3, rewards=np.ones((n, 2)),
              exact_grad=True)
    # DEFAULT (discrete) jpg -> the rebuild message (the CRITICAL cell)
    with pytest.raises(SvgdConfigError, match="discrete=False"):
        _svgd(jpg_d, [1, 2, 1], exact_grad=True)
    with pytest.raises(SvgdConfigError, match="discrete=False"):
        _svgd(jpg_d, [1, 2, 1], exact_grad=False)
    # weight-formula cell (config-level pre-emption)
    with pytest.raises(SvgdConfigError, match="linear"):
        _svgd(jpg, obs, weight_formula="t0 * c0 + t1 * c1", exact_grad=True)


def test_r28_shadows_r31_on_joint_index_false(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    with pytest.raises(SvgdConfigError):
        # R28 (joint_index=False on a joint-prob graph) fires before R31;
        # deliberate shadowing (plan scope 2).
        _svgd(jpg, obs, joint_index=False, exact_grad=True)


@pytest.mark.parametrize("leaf", ["moments", "rewards1d", "rewards2d",
                                  "joint_prob", "jpg_default", "jsp",
                                  "epoch"])
def test_none_never_raises(fx, leaf):
    graph, jpg, jpg_d, jsp, obs = (fx["graph"], fx["jpg"],
                                   fx["jpg_default"], fx["jsp"], fx["obs"])
    times = [0.5, 1.0, 1.5]
    n = graph.vertices_length()
    from phasic.svgd_config import from_svgd_call, validate
    if leaf == "moments":
        _svgd(graph, times, exact_grad=None)
    elif leaf == "rewards1d":
        _svgd(graph, times, rewards=np.ones(n), exact_grad=None)
    elif leaf == "rewards2d":
        validate(from_svgd_call(graph, [[0.5, 1.0]] * 3,
                                rewards=np.ones((n, 2)), exact_grad=None))
    elif leaf == "joint_prob":
        _svgd(jpg, obs, exact_grad=None)
    elif leaf == "jpg_default":
        _svgd(jpg_d, obs, exact_grad=None)
    elif leaf == "jsp":
        # validation-level: no jsp fit exists (jpg outcome tuples do not
        # map on jsp -- the G.1 precedent, reason unchanged)
        validate(from_svgd_call(jsp, obs, exact_grad=None))
    elif leaf == "epoch":
        _svgd(jpg, obs, epoch_starts=[0.0, 0.5], exact_grad=None)


# ---------------------------------------------------- forwarding contract
def test_forwarding_discrimination(fx, monkeypatch):
    jpg, obs = fx["jpg"], fx["obs"]
    seen = []
    orig = Graph.pmf_from_graph_joint_index.__func__

    def spy(cls, *a, **kw):
        seen.append(dict(kw))
        return orig(cls, *a, **kw)

    monkeypatch.setattr(Graph, "pmf_from_graph_joint_index",
                        classmethod(spy))
    _svgd(jpg, obs, exact_grad=False)
    assert seen and seen[-1].get("exact_grad") is False, seen[-1].keys()
    seen.clear()
    _svgd(jpg, obs)
    assert seen and "exact_grad" not in seen[-1], \
        "None must NOT be forwarded"


# --------------------------------------------------------------- ledger
def test_ledger_default_vs_user(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    fit_d = _svgd(jpg, obs)
    opts = fit_d.effective_options(return_dict=True)
    assert "exact_grad" in opts
    assert opts["exact_grad"]["status"] == "default"
    fit_u = _svgd(jpg, obs, exact_grad=True)
    opts = fit_u.effective_options(return_dict=True)
    assert opts["exact_grad"]["status"] == "user"
    assert opts["exact_grad"]["value"] is True


# ---------------------------------------------------- constructor guard
def test_constructor_guard_names_graph_svgd():
    from phasic.svgd import SVGD

    with pytest.raises(TypeError, match="Graph.svgd"):
        SVGD(model=lambda th, t, rewards=None: (jnp.ones(3), jnp.zeros(2)),
             observed_data=jnp.asarray([1.0, 2.0, 3.0]),
             theta_dim=2, exact_grad=True)


# -------------------------------------------------- front-door commit
def test_frontdoor_commit_and_value(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    import phasic.ffi_wrappers as fw
    lengths = []
    orig = fw.compute_sojourn_times_ffi

    def spy(sd, t, vi, *a, **k):
        try:
            lengths.append(int(np.asarray(vi).size))
        except Exception:
            pass
        return orig(sd, t, vi, *a, **k)

    fw.compute_sojourn_times_ffi = spy
    try:
        fit_t = _svgd(jpg, obs, exact_grad=True)
        assert np.all(np.isfinite(np.asarray(fit_t.particles)))
        n_obs = len(obs)
        n_uniq = len(set(map(tuple, obs)))
        assert lengths, "spy dead"
        assert any(l == n_uniq for l in lengths) and n_uniq < n_obs, (
            f"dedup lengths not seen: {sorted(set(lengths))[:8]} "
            f"(n_obs={n_obs}, n_uniq={n_uniq})")
        # value: model gradients True-vs-False agree loosely (FD is
        # inaccurate, not wrong; measured at authoring ~1e-6 rel).
        fit_f = _svgd(jpg, obs, exact_grad=False)
        th = jnp.asarray([TH, MU])

        def g(f):
            def loss(t):
                out, _ = f.model(t, None)
                return jnp.sum(jnp.log(out + 1e-300))
            return np.asarray(jax.grad(loss)(th))

        ga, gb = g(fit_t), g(fit_f)
        assert np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))
        assert np.max(np.abs(ga - gb)) / np.max(np.abs(gb)) < 1e-4
    finally:
        fw.compute_sojourn_times_ffi = orig


# ------------------- decline handling (Batch E user decision 2026-08-14)
def test_svgd_decline_falls_back_to_fd_with_warning(fx, monkeypatch):
    """The svgd entry uses exact_grad_decline='fd': a declined particle
    gets a HOST-side FD gradient + a WARNING, and the fit proceeds --
    it must NOT raise (the model-level default 'raise' contract is
    tested in test_exact_grad_joint_index_baked.py)."""
    import logging

    class _H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, r):
            self.records.append(r)

    jpg, obs = fx["jpg"], fx["obs"]
    fit = _svgd(jpg, obs, exact_grad=True)
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        lambda self, *a, **k: [])
    lg = logging.getLogger("phasic")
    h = _H(); h.setLevel(logging.WARNING)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.WARNING)
    try:
        th2 = jnp.stack([jnp.asarray([TH, MU])] * 2)
        g = np.asarray(jax.vmap(jax.jit(jax.grad(
            lambda t: jnp.sum(jnp.log(fit.model(t, None)[0] + 1e-300))
        )))(th2))
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    assert np.all(np.isfinite(g))
    msgs = [r.getMessage() for r in h.records]
    assert any("finite-difference fallback" in m for m in msgs), msgs


# --------------------------- explicit False == no-kwarg (bitwise)
def test_explicit_false_bitwise_equals_default(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    fit_d = _svgd(jpg, obs)
    fit_f = _svgd(jpg, obs, exact_grad=False)
    th = jnp.asarray([TH, MU])

    def g(f):
        def loss(t):
            out, _ = f.model(t, None)
            return jnp.sum(jnp.log(out + 1e-300))
        return np.asarray(jax.grad(loss)(th))

    np.testing.assert_array_equal(g(fit_d), g(fit_f))


# --------- G4 fold: R31 kind-tuple regression guard (jsp, validation-level)
def test_r31_accepts_jsp_kind_at_validation(fx):
    """A joint_stop_prob graph is joint-prob-KIND for R31 (the R29/R9
    classifier-hole lesson): explicit exact_grad must NOT be rejected by
    the kind arm. Validation-level (no jsp fit exists); a regression of
    the kind tuple to ('joint_prob',) fails here."""
    from phasic.svgd_config import from_svgd_call, validate
    jsp, obs = fx["jsp"], fx["obs"]
    # jsp graphs are continuous (not was_dph), linear mode, no epochs/
    # exposure -> R31's accept path; validate() must not raise.
    validate(from_svgd_call(jsp, obs, exact_grad=True))
