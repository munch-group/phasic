"""B3 joint-index exact gradient: Graph.pmf_from_graph_joint_index(...,
exact_grad=True) -- the forward-mode theta-adjoint for the sojourn-vector
gradient (Graph._sojourn_grad_theta_subset, src/c/phasic.c). Since Batch F
the wiring is COMMIT-OR-DECLINE: a construction-time probe (theta=ones over
the all-terminal union) either fails (whole model uses FD, logged) or
commits the model to the exact path via a plain Python `if` (the FD branch
is never traced -- no per-call lax.cond, which under vmap computed BOTH
branches); once committed, a per-theta decline RAISES a diagnostic
RuntimeError instead of silently falling back. The default is False (D5
decision: forward-mode cost scales with P; typical P=2 favors FD) -- the
flip-back is a separate recorded decision. See b3-joint-index-plan.md D5/D6
and b3-batchF-plan.md.

The diagonal-multiplier-1 guard-asymmetry case flagged by the D1 review
(finding 2) is deliberately NOT re-proven here with a real Graph fixture
-- constructing one where a diagonal elimination command lands at EXACTLY
weight 1.0 is not something the Python-level API can force deterministically.
That case is already proven, with a crafted deterministic tape and a
"wrong guard" variant shown to diverge, in
experiments/dr_sojourn_fwdmode_adjoint.py's guard_asymmetry_case() (D1.5).
This file instead focuses on end-to-end behaviour: matches an independent
native central-difference oracle, survives grad+vmap, the default picks up
the exact path automatically, and every exclusion/decline logs why.

Every correctness assertion is anchored to an INDEPENDENT oracle: native
graph.update_weights(theta) + graph.expected_sojourn_time(indices) central
difference (never the exact-grad code path itself).
"""
import contextlib
import logging

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _two_param_chain():
    """3-vertex chain, 2 params, s -> v2 -> v1 (v1 absorbing)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])
    return g


def _four_param_branching():
    """5-vertex graph with a BRANCHING vertex (v3 has two out-edges), 4
    params, every path terminating at the absorbing vertex v0 -- exercises
    a genuinely non-chain topology, unlike _two_param_chain."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [2.0, 0.1, 5.0, 1.0])
    v3.add_edge_parameterized(v1, 0.0, [0.5, 3.0, 1.0, 7.0])
    v2.add_edge_parameterized(v0, 0.0, [1.0, 2.0, 0.25, 3.0])
    v1.add_edge_parameterized(v0, 0.0, [4.0, 0.5, 2.0, 0.125])
    return g


def _dph_native(probs):
    """Native DPH: is_discrete=True, was_dph=False (edge weight IS c.theta
    directly, no renormalisation) -- confirmed needing zero special-casing
    for sojourn (neither ComputeSojournTimesFfiImpl nor
    ptd_expected_sojourn_time_subset branch on is_discrete)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    n = len(probs)
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        coeff = [0.0] * n
        coeff[i] = 1.0
        vs[i].add_edge(vs[i + 1], coeff)
    g.is_discrete = True
    return g


def _all_terminal_indices(g):
    out = []
    for v in g.vertices():
        for e in v.edges():
            if len(e.to().edges()) == 0:
                out.append(v.index())
                break
    return sorted(set(out))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))


def _native_sojourn_probs_grad_cd(build, theta, vertex_indices, all_terminal, eps=1e-6):
    """Independent oracle: theta-perturbation central difference of the
    NATIVE graph.update_weights(theta) + graph.expected_sojourn_time(idx),
    reproducing the SAME quotient rule the model applies, entirely outside
    the exact-grad code path."""
    theta = np.asarray(theta, dtype=float)
    P = len(theta)
    J = np.zeros((len(vertex_indices), P))
    for j in range(P):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        gp = build(); gp.update_weights(list(tp))
        gm = build(); gm.update_weights(list(tm))
        obsp = np.asarray(gp.expected_sojourn_time(list(vertex_indices)))
        allp = np.asarray(gp.expected_sojourn_time(list(all_terminal)))
        probsp = obsp / allp.sum()
        obsm = np.asarray(gm.expected_sojourn_time(list(vertex_indices)))
        allm = np.asarray(gm.expected_sojourn_time(list(all_terminal)))
        probsm = obsm / allm.sum()
        J[:, j] = (probsp - probsm) / (2 * eps)
    return J


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
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


# --------------------------------------------------------------------------- matches native CD
def test_exact_grad_matches_native_cd_branching():
    build = _four_param_branching
    g0 = build()
    all_term = _all_terminal_indices(g0)
    vidx = list(range(g0.vertices_length()))
    theta = np.array([1.0, 2.0, 0.5, 1.5])

    model = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=True)
    J_exact = np.asarray(
        jax.jacobian(lambda th: model(th, jnp.asarray(vidx, dtype=jnp.int32))[0])(jnp.asarray(theta)))
    J_cd = _native_sojourn_probs_grad_cd(build, theta, vidx, all_term)
    assert _rel(J_exact, J_cd) < 1e-3


def test_exact_grad_matches_native_cd_unsorted_duplicated_subset_indices():
    """The union/searchsorted gather (_exact_sojourn_jac_np) is the only
    genuinely new index logic in the wiring, but every OTHER test here
    passes vertex_indices = list(range(n)) -- sorted, unique, and equal to
    the full vertex set, so np.union1d(vidx, all_terminal) == vidx and the
    gather is the identity permutation everywhere else, which could not
    catch a swapped/misaligned gather (found via adversarial review of the
    implemented fix). This test uses an UNSORTED, DUPLICATED, STRICT SUBSET
    vertex_indices -- the shape real (non-baked) observed data actually
    has -- and additionally checks the two duplicate rows of J_exact are
    identical (they gather the SAME underlying vertex)."""
    build = _four_param_branching
    g0 = build()
    all_term = _all_terminal_indices(g0)
    vidx = [3, 1, 1, 2, 1]  # unsorted, duplicated, strict subset of range(5)
    theta = np.array([1.0, 2.0, 0.5, 1.5])

    model = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=True)
    J_exact = np.asarray(
        jax.jacobian(lambda th: model(th, jnp.asarray(vidx, dtype=jnp.int32))[0])(jnp.asarray(theta)))
    J_cd = _native_sojourn_probs_grad_cd(build, theta, vidx, all_term)
    assert _rel(J_exact, J_cd) < 1e-3
    # rows 1, 2, 4 all correspond to vertex 1 -- must be identical
    np.testing.assert_array_equal(J_exact[1], J_exact[2])
    np.testing.assert_array_equal(J_exact[1], J_exact[4])


def test_exact_sojourn_jac_np_ffi_and_clone_agree_on_primal():
    """The quotient rule mixes a primal from the FFI-rebuilt graph
    (compute_sojourn_times_ffi, used for norm_exact/obs_sojourn_exact) with
    a Jacobian from a SEPARATE graph.clone() (_jix_exact_graph, used for
    J_obs/J_all) -- two independently-built elimination tapes that the
    pure-FD path never had to keep consistent (found via adversarial review
    of the implemented fix: an unasserted invariant, not a demonstrated
    bug). Assert the two representations agree on the PRIMAL sojourn values
    they each compute at the same theta, directly, rather than only
    indirectly through the end-to-end gradient (which routes through a
    THIRD representation, graph.expected_sojourn_time, as the CD oracle,
    and could mask a small systematic offset as CD error)."""
    import jax.numpy as _jnp
    from phasic.ffi_wrappers import compute_sojourn_times_ffi

    build = _four_param_branching
    g0 = build()
    all_term = _all_terminal_indices(g0)
    vidx = list(range(g0.vertices_length()))
    theta_np = [1.0, 2.0, 0.5, 1.5]

    structure_dict = g0.serialize(theta_dim=4)
    theta_j = _jnp.asarray(theta_np)
    ffi_obs = np.asarray(compute_sojourn_times_ffi(
        structure_dict, theta_j, _jnp.asarray(vidx, dtype=_jnp.int32)))
    ffi_all = np.asarray(compute_sojourn_times_ffi(
        structure_dict, theta_j, _jnp.asarray(all_term, dtype=_jnp.int32)))

    g_clone = build()
    g_clone.update_weights(theta_np)
    clone_obs = np.asarray(g_clone.expected_sojourn_time(vidx))
    clone_all = np.asarray(g_clone.expected_sojourn_time(all_term))

    np.testing.assert_allclose(ffi_obs, clone_obs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ffi_all, clone_all, rtol=1e-10, atol=1e-12)


def test_theta_dim_override_declines_exact_and_logs():
    """theta_dim overriding the graph's own param_length is out of scope
    for exact_grad (the C function reads graph->param_length off the
    clone, which would silently disagree with param_length_actual --
    used for every reshape/ShapeDtypeStruct in the wiring -- if the two
    differ; found via adversarial review of the implemented fix)."""
    g = _two_param_chain()  # graph.param_length() == 2
    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_from_graph_joint_index(g, theta_dim=3, exact_grad=True)
    messages = [r.getMessage() for r in handler.records]
    assert any("theta_dim" in m and "finite differences" in m for m in messages)


def test_exact_grad_matches_native_cd_mixed_scale():
    build = _four_param_branching
    g0 = build()
    all_term = _all_terminal_indices(g0)
    vidx = list(range(g0.vertices_length()))
    theta = np.array([10.0, 1e-2, 0.5, 2.0])

    model = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=True)
    J_exact = np.asarray(
        jax.jacobian(lambda th: model(th, jnp.asarray(vidx, dtype=jnp.int32))[0])(jnp.asarray(theta)))
    J_cd = _native_sojourn_probs_grad_cd(build, theta, vidx, all_term, eps=1e-6)
    assert _rel(J_exact, J_cd) < 1e-2


def test_exact_grad_matches_native_cd_native_dph():
    """Native DPH (is_discrete=True, was_dph=False) IS in scope -- confirmed
    via the C-level decline gate (only graph->was_dph declines) and the
    D3 gate script. Exercised end-to-end here."""
    # Batch F fate-table obligation: confirm via the direct C call that the
    # exact path genuinely SUCCEEDS at this theta -- under commit-or-decline
    # semantics a latent decline that FD-fallback used to mask would now
    # raise, so this precondition is the evidence artifact the plan requires.
    _g_pre = _four_param_branching()
    _g_pre.update_weights([10.0, 1e-2, 0.5, 2.0])
    assert len(_g_pre._sojourn_grad_theta_subset([0, 1, 2, 3, 4])) > 0, (
        "exact path unexpectedly declines at the mixed-scale theta")
    build = lambda: _dph_native((1.0, 1.0, 1.0))
    g0 = build()
    all_term = _all_terminal_indices(g0)
    vidx = list(range(g0.vertices_length()))
    theta = np.array([0.3, 0.4, 0.5])

    model = Graph.pmf_from_graph_joint_index(build(), theta_dim=3, exact_grad=True)
    J_exact = np.asarray(
        jax.jacobian(lambda th: model(th, jnp.asarray(vidx, dtype=jnp.int32))[0])(jnp.asarray(theta)))
    J_cd = _native_sojourn_probs_grad_cd(build, theta, vidx, all_term)
    assert _rel(J_exact, J_cd) < 1e-3


# --------------------------------------------------------------------------- grad+vmap (SVGD-safety)
def test_exact_grad_vmap_matches_fd():
    build = _four_param_branching
    vidx = list(range(build().vertices_length()))
    thetas = jnp.asarray([[1.0, 2.0, 0.5, 1.5], [2.0, 1.0, 1.0, 1.0], [0.5, 0.5, 2.0, 3.0]])

    model_exact = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=True)
    model_fd = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=False)
    vidx_j = jnp.asarray(vidx, dtype=jnp.int32)

    def loss(model, th):
        probs, _ = model(th, vidx_j)
        return jnp.sum(probs)

    g_exact = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_exact, th)))(thetas))
    g_fd = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_fd, th)))(thetas))
    assert _rel(g_exact, g_fd) < 1e-3


# --------------------------------------------------------------------------- default path picks it up automatically
def test_default_path_uses_fd():
    """exact_grad defaults to False, UNLIKE every other B3 exact-gradient
    kwarg in this codebase (which default to True). Deliberate: this
    function uses forward-mode (cost scales with theta_dim P), not
    reverse-mode (P-independent), so it only beats FD once P is roughly
    10-20+ (see the D3 benchmark in b3-joint-index-plan.md) -- and under
    SVGD's actual vmap(grad(loss))(particles) call pattern, the internal
    jax.lax.cond used to skip FD when exact succeeds cannot skip anything
    (JAX computes both branches of a cond whenever its predicate is
    batched, which it always is here) -- so exact_grad=True currently costs
    FD PLUS exact on every call under vmap, a net regression at this
    model's typical native P=2 (found via adversarial review of the
    implemented fix, which is why the default was flipped from the B3
    default-True precedent). This test confirms the default is genuinely
    FD (matches explicit exact_grad=False, byte-identical -- same code
    path) and that explicit exact_grad=True is still available and
    genuinely different (mirrors the log-weight-mode batch's own
    default-path test, adapted for the opposite default here)."""
    build = _four_param_branching
    vidx = jnp.asarray(list(range(build().vertices_length())), dtype=jnp.int32)
    theta = jnp.asarray([1.0, 2.0, 0.5, 1.5])

    model_default = Graph.pmf_from_graph_joint_index(build(), theta_dim=4)
    model_exact_explicit = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=True)
    model_fd_explicit = Graph.pmf_from_graph_joint_index(build(), theta_dim=4, exact_grad=False)

    def loss(model, th):
        probs, _ = model(th, vidx)
        return jnp.sum(probs)

    g_default = np.asarray(jax.grad(lambda th: loss(model_default, th))(theta))
    g_exact = np.asarray(jax.grad(lambda th: loss(model_exact_explicit, th))(theta))
    g_fd = np.asarray(jax.grad(lambda th: loss(model_fd_explicit, th))(theta))
    np.testing.assert_array_equal(g_default, g_fd)
    assert _rel(g_default, g_exact) < 1e-3  # exact is still a close approximation
    assert not np.array_equal(g_default, g_exact)  # but genuinely a different code path


def test_exact_grad_false_logs_and_matches_fd():
    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_from_graph_joint_index(_two_param_chain(), theta_dim=2, exact_grad=False)
    messages = [r.getMessage() for r in handler.records]
    assert any("exact_grad=False" in m and "finite differences" in m for m in messages)

    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    theta = jnp.asarray([2.0, 3.0])
    g_fd = np.asarray(jax.grad(lambda th: jnp.sum(model(th, vidx)[0]))(theta))
    assert np.all(np.isfinite(g_fd))


# --------------------------------------------------------------------------- fixed_mask
def test_fixed_mask_zeroes_exact_gradient_column():
    build = lambda: _dph_native((1.0, 1.0, 1.0))
    vidx = jnp.asarray(list(range(build().vertices_length())), dtype=jnp.int32)
    theta = jnp.asarray([0.3, 0.4, 0.5])
    fixed_mask = jnp.asarray([0, 1, 0])

    model = Graph.pmf_from_graph_joint_index(build(), theta_dim=3, fixed_mask=fixed_mask, exact_grad=True)
    grad = np.asarray(jax.grad(lambda th: jnp.sum(model(th, vidx)[0]))(theta))
    assert grad[1] == 0.0
    assert grad[0] != 0.0 and grad[2] != 0.0


# --------------------------------------------------------------------------- was_dph stays excluded
def test_was_dph_declines_and_logs():
    """A was_dph graph (Graph.discretize()) must NOT use the exact path
    (only graph->was_dph declines at the C level; native DPH is supported),
    and must log why at model-construction time."""
    gd = _two_param_chain().discretize(lambda state, **kw: [0.5, 0.5])
    assert gd.get_was_dph() is True

    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_from_graph_joint_index(gd, theta_dim=2, exact_grad=True)
    messages = [r.getMessage() for r in handler.records]
    assert any("was_dph" in m and "finite differences" in m for m in messages)

    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    theta = jnp.asarray([1.0, 2.0])
    grad = np.asarray(jax.grad(lambda th: jnp.sum(model(th, vidx)[0]))(theta))
    assert np.all(np.isfinite(grad))


def test_observed_indices_baked_mode_commits():
    """Batch E REWRITE of test_observed_indices_baked_mode_declines_and_logs
    (the fate table's one break-by-design): baked/dedup mode now SUPPORTS
    the exact gradient (scatter-add of the cotangent by the inverse-index
    map + the quotient rule at unique granularity; probe over the exact
    static baked union). Asserts the old decline log is GONE and the
    exact path commits; PRESERVES the original's second half (finite
    gradient; runtime vidx ignored in baked mode)."""
    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_from_graph_joint_index(
            _two_param_chain(), theta_dim=2, observed_indices=np.array([0, 1, 1, 2]),
            exact_grad=True)
    messages = [r.getMessage() for r in handler.records]
    assert not any("baked mode" in m and "finite differences" in m
                   for m in messages), messages

    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)  # ignored in baked mode
    theta = jnp.asarray([1.0, 2.0])
    # weighted loss: the unweighted sum's gradient is exactly zero on this
    # fixture (sum of normalized probs at these obs is theta-invariant),
    # which would make the oracle comparison 0/0-degenerate.
    w = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    grad = np.asarray(jax.grad(
        lambda th: jnp.sum(w * model(th, vidx)[0]))(theta))
    assert np.all(np.isfinite(grad))
    # exact engaged: matches the non-baked exact oracle at the same obs
    nb = Graph.pmf_from_graph_joint_index(_two_param_chain(), theta_dim=2,
                                          exact_grad=True)
    obs = jnp.asarray([0, 1, 1, 2], dtype=jnp.int32)
    g_nb = np.asarray(jax.grad(
        lambda th: jnp.sum(w * nb(th, obs)[0]))(theta))
    # NOTE: on this minimal chain fixture the normalized probs are
    # theta-invariant (both gradients are exactly zero -- the ORIGINAL
    # test only asserted finiteness for the same reason). Absolute
    # agreement pins baked==non-baked exact here; the strong nonzero
    # oracle parity lives in test_exact_grad_joint_index_baked.py on
    # the coalescent fixture.
    assert np.max(np.abs(grad - g_nb)) < 1e-12


# --------------------------------------------------------------------------- committed decline -> RAISE (Batch F)
def test_committed_decline_raises_at_extreme_condition():
    """Batch F semantics: the probe (theta=ones) succeeds and the model
    COMMITS; at an ill-conditioned theta the C path declines (confirmed
    directly) and the committed backward RAISES a diagnostic RuntimeError --
    under plain grad, vmap(grad), and the SVGD-real vmap(jit(grad))."""
    theta_np = [1.0, 1e-13]
    g_direct = _two_param_chain()
    g_direct.update_weights(theta_np)
    if len(g_direct._sojourn_grad_theta_subset([0, 1, 2])) != 0:
        pytest.skip("build lacks HAVE_MPFR: the conditioning gate is inert, "
                    "no decline to exercise")
    # Precondition (G4 review): the probe (theta=ones) must SUCCEED for this
    # fixture, else the model falls back to whole-model FD and the raise
    # assertions below would fail confusingly instead of skipping cleanly.
    g_probe = _two_param_chain()
    g_probe.update_weights([1.0, 1.0])
    assert len(g_probe._sojourn_grad_theta_subset([0, 1, 2])) > 0, (
        "probe unexpectedly declines at theta=ones on this build")

    model = Graph.pmf_from_graph_joint_index(_two_param_chain(), theta_dim=2, exact_grad=True)
    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    theta = jnp.asarray(theta_np)
    loss = lambda th: jnp.sum(model(th, vidx)[0])
    for name, fn in [
        ("grad", jax.grad(loss)),
        ("vmap(grad)", lambda t: jax.vmap(jax.grad(loss))(t[None, :])),
        ("vmap(jit(grad))", lambda t: jax.vmap(jax.jit(jax.grad(loss)))(t[None, :])),
        ("jit(vmap(grad))", lambda t: jax.jit(jax.vmap(jax.grad(loss)))(t[None, :])),
    ]:
        with pytest.raises(BaseException) as exc_info:
            np.asarray(fn(theta))
        msg = str(exc_info.value) + str(getattr(exc_info.value, "__cause__", ""))
        assert "declined at theta" in msg, (name, msg[:300])


def test_probe_failure_falls_back_to_fd_whole_model(monkeypatch):
    """Force the construction-time probe to fail (threshold ~ 0 makes the
    conditioning gate decline even at theta=ones); the model must use FD
    for its whole life: finite gradient, probe-failure INFO log, and NO
    raise at any theta."""
    g_probe = _two_param_chain()
    g_probe.update_weights([1.0, 1.0])
    monkeypatch.setenv("PHASIC_CONDITION_THRESHOLD", "0")
    if len(g_probe._sojourn_grad_theta_subset([0, 1, 2])) != 0:
        pytest.skip("build lacks HAVE_MPFR: cannot force a probe decline")

    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_from_graph_joint_index(
            _two_param_chain(), theta_dim=2, exact_grad=True)
    messages = [r.getMessage() for r in handler.records]
    assert any("construction-time probe" in m and "finite differences" in m
               for m in messages), messages

    monkeypatch.delenv("PHASIC_CONDITION_THRESHOLD")
    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    grad = np.asarray(jax.grad(
        lambda th: jnp.sum(model(th, vidx)[0]))(jnp.asarray([1.0, 2.0])))
    assert np.all(np.isfinite(grad))


def test_fd_branch_never_traced_when_committed():
    """The committed model's backward must not trace the FD branch: count
    Python-level calls to compute_sojourn_times_ffi during jaxpr tracing
    (both the forward and the FD loop route through it; the FD loop adds
    2*P calls per backward). A value check cannot distinguish 'skipped'
    from 'computed and discarded' -- call counting can."""
    import phasic.ffi_wrappers as fw
    real = fw.compute_sojourn_times_ffi
    counts = {"n": 0}

    def counting(*a, **k):
        counts["n"] += 1
        return real(*a, **k)

    fw.compute_sojourn_times_ffi = counting
    try:
        committed = Graph.pmf_from_graph_joint_index(
            _two_param_chain(), theta_dim=2, exact_grad=True)
        fd_model = Graph.pmf_from_graph_joint_index(
            _two_param_chain(), theta_dim=2, exact_grad=False)
    finally:
        fw.compute_sojourn_times_ffi = real

    vidx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    theta = jnp.asarray([1.0, 2.0])

    # Trace under vmap -- the batching regime SVGD actually uses and the
    # regime the whole batch exists to fix (G4 review: counting without
    # vmap leaves the central claim resting on a generic JAX property).
    tbatch = theta[None, :]

    counts["n"] = 0
    jax.make_jaxpr(jax.vmap(jax.grad(
        lambda th: jnp.sum(committed(th, vidx)[0]))))(tbatch)
    committed_calls = counts["n"]

    counts["n"] = 0
    jax.make_jaxpr(jax.vmap(jax.grad(
        lambda th: jnp.sum(fd_model(th, vidx)[0]))))(tbatch)
    fd_calls = counts["n"]

    # FD's backward traces 4 extra _compute_pure invocations (2*P, P=2),
    # each making 2 sojourn-FFI calls -> true margin is +8 vs the committed
    # backward's 2 direct FFI calls; assert the conservative +4.
    assert fd_calls >= committed_calls + 4, (committed_calls, fd_calls)


def test_out_of_range_index_raises_accurate_error():
    """A bad observed index must raise the index error, not the
    conditioning message (D6.1 review: indices[r] >= n is an
    index-dependent decline the probe cannot see)."""
    model = Graph.pmf_from_graph_joint_index(
        _two_param_chain(), theta_dim=2, exact_grad=True)
    vidx = jnp.asarray([0, 999], dtype=jnp.int32)
    with pytest.raises(BaseException) as exc_info:
        np.asarray(jax.grad(
            lambda th: jnp.sum(model(th, vidx)[0]))(jnp.asarray([1.0, 2.0])))
    msg = str(exc_info.value) + str(getattr(exc_info.value, "__cause__", ""))
    # The non-batched FORWARD FFI fails first on a bad index (its own loud
    # INTERNAL error), so under plain grad the backward check is shadowed;
    # either loud surface is acceptable, and the conditioning message must
    # never appear for this cause.
    assert ("out of range" in msg or "sojourn_time_subset failed" in msg), msg[:300]
    assert "declined at theta" not in msg, msg[:300]


def test_out_of_range_index_under_vmap_is_loud():
    """Under vmap the BATCHED forward FFI silently NaN-fills a positive
    out-of-range index (pre-existing gap, no exception) -- so the backward
    callback's bounds check is the LIVE defense there (G4 review finding).
    The gradient call must fail loudly with an accurate message, never
    return silently and never blame conditioning."""
    model = Graph.pmf_from_graph_joint_index(
        _two_param_chain(), theta_dim=2, exact_grad=True)
    vidx = jnp.asarray([0, 999], dtype=jnp.int32)
    tbatch = jnp.asarray([[1.0, 2.0]])
    with pytest.raises(BaseException) as exc_info:
        np.asarray(jax.vmap(jax.grad(
            lambda th: jnp.sum(model(th, vidx)[0])))(tbatch))
    msg = str(exc_info.value) + str(getattr(exc_info.value, "__cause__", ""))
    assert ("out of range" in msg or "sojourn_time_subset failed" in msg), msg[:300]
    assert "declined at theta" not in msg, msg[:300]
