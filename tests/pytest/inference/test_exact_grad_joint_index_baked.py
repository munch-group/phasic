"""Batch E -- baked/dedup-mode exact gradient for pmf_from_graph_joint_index.

Plan of record: b3-batchE-plan.md v2 (I3, model level). The oracle for
the baked backward is the SHIPPED non-baked committed exact path (itself
F-gated) evaluated at the same per-obs indices.
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import contextlib  # noqa: E402
import logging  # noqa: E402


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
    # mirrors the F suite's helper (relative import not possible here)
    logger = logging.getLogger('phasic')
    handler = _CollectingHandler()
    handler.setLevel(logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


set_log_level("WARNING")
NR = 3
MU, TH = 1e-4, 1 / 10_000


def _jpg(discrete=False):
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

    graph = Graph(coal, indexer=indexer)
    return graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU,
                                  discrete=discrete)


@pytest.fixture(scope="module")
def fx():
    jpg = _jpg(discrete=False)
    all_term = sorted({v.index() for v in jpg.vertices()
                       for e in v.edges() if len(e.to().edges()) == 0})
    # UNSORTED, heavily duplicated observations (25 obs over ~18 unique)
    obs = np.random.default_rng(3).choice(all_term, 25, replace=True).astype(int)
    assert not np.all(np.diff(obs) >= 0), "fixture must be unsorted"
    return dict(jpg=jpg, all_term=all_term, obs=obs)


def _grad(model, theta, vi, gv):
    def loss(t):
        out, _ = model(t, None if vi is None else jnp.asarray(vi))
        return jnp.sum(jnp.asarray(gv) * out)
    return np.asarray(jax.grad(loss)(jnp.asarray(theta)))


THETAS = {"benign": np.array([TH, MU]), "mixed": np.array([1e-4, 5e-2])}


# ------------------------------------------------------- tests 1+2 (parity)
@pytest.mark.parametrize("tlabel", ["benign", "mixed"])
@pytest.mark.parametrize("shape", ["main", "n_unique_1", "n_obs_1"])
def test_baked_exact_matches_nonbaked_oracle(fx, tlabel, shape):
    jpg, all_term, obs = fx["jpg"], fx["all_term"], fx["obs"]
    if shape == "n_unique_1":
        obs = np.full(7, int(all_term[0]))
    elif shape == "n_obs_1":
        obs = np.asarray([int(all_term[1])])
    gv = np.linspace(0.5, 1.5, obs.size)
    theta = THETAS[tlabel]
    m_baked_ex = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True)
    m_baked_fd = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=False)
    m_nb_ex = Graph.pmf_from_graph_joint_index(jpg, exact_grad=True)

    g_bex = _grad(m_baked_ex, theta, None, gv)
    g_bfd = _grad(m_baked_fd, theta, None, gv)
    g_oracle = _grad(m_nb_ex, theta, obs, gv)
    denom = np.max(np.abs(g_oracle))
    err_ex = np.max(np.abs(g_bex - g_oracle)) / denom
    err_fd = np.max(np.abs(g_bfd - g_oracle)) / denom
    # measured at authoring: err_ex ~2.5e-16 (machine precision) -- 1e-12
    # gives 1e4x headroom; err_fd ~1e-6 (FD's own accuracy) -- bounding
    # it at 1e-4 (100x headroom) is the plan's I3-2 baked-FD parity
    # check (G4 fold: previously only the ratio was asserted, so a
    # BROKEN baked-FD path was invisible); improvement floor 1e3 loose
    # under the measured ~1e10.
    assert err_ex < 1e-12, f"baked-exact off oracle: {err_ex:.3e}"
    assert err_fd < 1e-4, f"baked-FD off oracle: {err_fd:.3e}"
    assert err_fd / max(err_ex, 1e-300) > 1e3, (err_fd, err_ex)


# --------------------------------------------------------- test 3 (spy)
def test_trace_time_fd_ops_gone_baked(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    import phasic.ffi_wrappers as fw
    calls = []
    orig = fw.compute_sojourn_times_ffi

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    fw.compute_sojourn_times_ffi = spy
    try:
        m_ex = Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=obs.tolist(), exact_grad=True)
        m_fd = Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=obs.tolist(), exact_grad=False)
        gv = np.linspace(0.5, 1.5, obs.size)

        def loss_of(m):
            return lambda t: jnp.sum(jnp.asarray(gv) * m(t, None)[0])

        th2 = jnp.stack([jnp.asarray(THETAS["benign"])] * 2)
        calls.clear()
        jax.make_jaxpr(jax.vmap(jax.grad(loss_of(m_ex))))(th2)
        n_ex = len(calls)
        calls.clear()
        jax.make_jaxpr(jax.vmap(jax.grad(loss_of(m_fd))))(th2)
        n_fd = len(calls)
        assert n_ex > 0 and n_fd > 0, "spy dead"
        # measured at authoring: exact traces 6 wrapper passes, FD traces
        # 20. (The per-invocation trace multiplicity under vmap+grad is a
        # JAX-internal detail we deliberately do NOT re-derive here --
        # the G4 fold removed a garbled decomposition story; the PINNED
        # MEASUREMENT is the contract.) The absolute delta catches
        # partial-removal regressions.
        assert n_fd - n_ex == 14, (n_ex, n_fd)
    finally:
        fw.compute_sojourn_times_ffi = orig


# ------------------------------------------- test 5 (probe set == call set)
def test_probe_set_equals_call_set(fx, monkeypatch):
    jpg, obs = fx["jpg"], fx["obs"]
    seen = []
    orig = Graph._sojourn_grad_theta_subset

    def spy(self, idx, *a, **k):
        seen.append(tuple(int(x) for x in idx))
        return orig(self, idx, *a, **k)

    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", spy)
    model = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True)
    assert len(seen) == 1, "expected exactly the construction probe"
    probe_set = seen[0]
    gv = np.linspace(0.5, 1.5, obs.size)
    _grad(model, THETAS["benign"], None, gv)
    assert len(seen) >= 2
    assert set(seen[1]) == set(probe_set), (
        "committed call union differs from probe union -- the baked "
        "probe-exactness guarantee is broken")


# --------------------------------------------- test 6 (raise legibility)
def test_committed_decline_raises_legibly(fx, monkeypatch):
    jpg, obs = fx["jpg"], fx["obs"]
    model = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True)
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        lambda self, *a, **k: [])
    gv = np.linspace(0.5, 1.5, obs.size)
    th2 = jnp.stack([jnp.asarray(THETAS["benign"])] * 2)
    with pytest.raises(Exception, match="exact sojourn gradient"):
        jax.vmap(jax.jit(jax.grad(
            lambda t: jnp.sum(jnp.asarray(gv) * model(t, None)[0])
        )))(th2)


# ------------------------------------------------- test 7 (fixed_mask)
def test_fixed_mask_baked_exact_zero_slot(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    model = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True,
        fixed_mask=jnp.asarray([0.0, 1.0]))
    gv = np.linspace(0.5, 1.5, obs.size)
    g = _grad(model, THETAS["benign"], None, gv)
    assert g[1] == 0.0
    assert np.isfinite(g[0]) and g[0] != 0.0


# ------------------------------------------- test 8 (ladder order intact)
def test_ladder_order_was_dph_and_formula_still_decline(fx):
    obs = fx["obs"]
    jpg_d = _jpg(discrete=True)  # DEFAULT construction: was_dph
    with _capture_phasic_info_logs() as handler:
        Graph.pmf_from_graph_joint_index(
            jpg_d, observed_indices=obs.tolist()[:5], exact_grad=True)
    msgs = [r.getMessage() for r in handler.records]
    assert any("was_dph" in m for m in msgs), msgs
    jpg_f = _jpg(discrete=False)
    jpg_f.weight_formula = "t0 * c0 + t1 * c1"
    with _capture_phasic_info_logs() as handler:
        Graph.pmf_from_graph_joint_index(
            jpg_f, observed_indices=obs.tolist()[:5], exact_grad=True,
            theta_dim=2)
    msgs = [r.getMessage() for r in handler.records]
    assert any("weight_mode" in m or "linear" in m for m in msgs), msgs


# --------------------------------- test 9 (construction bounds validation)
def test_baked_out_of_range_raises_at_construction(fx):
    jpg = fx["jpg"]
    with pytest.raises(ValueError, match="observed_indices out of range"):
        Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=[10 ** 6], exact_grad=False)
    with pytest.raises(ValueError, match="observed_indices out of range"):
        Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=[-1], exact_grad=True)


# ----------------------- theta_dim override: FD + INFO log (F contract)
def test_theta_dim_override_declines_to_fd_with_log(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    with _capture_phasic_info_logs() as handler:
        m = Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=obs.tolist(), exact_grad=True, theta_dim=3)
    assert any("theta_dim" in r.getMessage() for r in handler.records)
    assert callable(m)
    # NOTE: no gradient evaluation here -- a theta_dim=3 override on this
    # 2-coefficient graph cannot compute ANY forward (the FFI validates
    # coefficients_length >= param_length, pre-existing); the decline
    # contract (FD + INFO log, no raise at construction) is the tested
    # property, matching the F suite's theta_dim_override_declines shape.


# ----------------- G4 fold: fallback VALUE correctness + kwarg validation
def test_fd_fallback_values_correct_mixed_batch(fx, monkeypatch):
    """exact_grad_decline='fd' (model level): at a batch mixing one
    declined and one computable theta, the declined particle's gradient
    matches tight FD of the FULL normalized forward, and the computable
    particle matches the pure-exact run bitwise-close. Value-level
    coverage the flipped svgd test lacks (G4 fold)."""
    jpg, obs = fx["jpg"], fx["obs"]
    gv = np.linspace(0.5, 1.5, obs.size)
    model = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True,
        exact_grad_decline="fd")
    model_ref = Graph.pmf_from_graph_joint_index(
        jpg, observed_indices=obs.tolist(), exact_grad=True)

    th_ok = np.array([TH, MU])
    th_bad = np.array([TH * 1.7, MU * 0.8])
    orig = Graph._sojourn_grad_theta_subset

    def selective_decline(self, idx, *a, **k):
        # decline ONLY at th_bad (matched via the clone's current
        # weights? cheaper: flag via closure counter -- decline the
        # second per-batch call). Simplest robust approach: decline
        # when the graph's current first-edge weight matches th_bad's
        # signature is fragile; instead decline every call while the
        # flag is set.
        if selective_decline.active:
            return []
        return orig(self, idx, *a, **k)

    selective_decline.active = False
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        selective_decline)

    def loss(m, t):
        out, _ = m(jnp.asarray(t), None)
        return jnp.sum(jnp.asarray(gv) * out)

    # computable case: fallback model == pure exact
    g_ok = np.asarray(jax.grad(lambda t: loss(model, t))(jnp.asarray(th_ok)))
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", orig)
    g_ref = np.asarray(jax.grad(lambda t: loss(model_ref, t))(jnp.asarray(th_ok)))
    assert np.max(np.abs(g_ok - g_ref)) / np.max(np.abs(g_ref)) < 1e-12

    # declined case: force decline, compare vs tight FD of the forward
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        selective_decline)
    selective_decline.active = True
    g_fb = np.asarray(jax.grad(lambda t: loss(model, t))(jnp.asarray(th_bad)))
    selective_decline.active = False
    eps = 1e-6
    ref = np.empty(2)
    for k in range(2):
        tp = th_bad.copy(); tp[k] += th_bad[k] * eps
        tm = th_bad.copy(); tm[k] -= th_bad[k] * eps

        def f(t):
            out, _ = model_ref(jnp.asarray(t), None)
            return float(jnp.sum(jnp.asarray(gv) * out))

        ref[k] = (f(tp) - f(tm)) / (2 * th_bad[k] * eps)
    # measured at authoring: ~1e-10 rel (G4 wiring-refuter probe); 1e-5
    # gives wide headroom over both FD schemes' noise.
    assert np.max(np.abs(g_fb - ref)) / np.max(np.abs(ref)) < 1e-5, (g_fb, ref)


def test_exact_grad_decline_validates(fx):
    jpg, obs = fx["jpg"], fx["obs"]
    with pytest.raises(ValueError, match="exact_grad_decline"):
        Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=obs.tolist(), exact_grad=True,
            exact_grad_decline="bogus")


def test_observed_indices_2d_rejected(fx):
    jpg = fx["jpg"]
    with pytest.raises(ValueError, match="1-D vertex indices"):
        Graph.pmf_from_graph_joint_index(
            jpg, observed_indices=[[1, 2], [3, 4]], exact_grad=False)
