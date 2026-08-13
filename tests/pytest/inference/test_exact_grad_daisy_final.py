"""Batch H I3 -- exact FINAL-epoch gradient for daisy-chain SVGD models.

Plan of record: b3-batchH-plan.md v3.1 (tests 1-9). Evidence base:
b3-batchH-findings.md (H0 oracle, I1 micro-gates). The kwarg under test
is the INTERNAL `_daisy_chain_svgd_model(exact_final_grad=...)` (public
Graph.svgd plumbing is Batch G leaf 1).

Fixture: a small coalescent joint-prob epoch model (nr_samples=3 for
speed; same construction family as tests/pytest/inference/test_lrt_at.py's
epoch_tied_free_pair and the H0 oracle fixture).
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402

set_log_level("WARNING")

NR = 3
EPOCH_STARTS = [0.0, 0.5]
DT = 0.5
THETA_BENIGN = np.array([1e-4, 1e-4, 0.5e-4, 1.3e-4])
THETA_MIXED = np.array([2e-5, 1e-4, 3e-4, 5e-2])


def _build_jpg():
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
    return graph.joint_prob_graph(
        indexer, reward_limit=4, mutation_rate=1e-4, discrete=False)


@pytest.fixture(scope="module")
def jpg():
    return _build_jpg()


@pytest.fixture(scope="module")
def obs(jpg):
    return [int(x) for x in jpg.joint_stop_prob_graph()._t_vertex_indices][:5]


def _models(jpg, obs, **kw):
    model, theta_dim, prior, fixed = jpg._daisy_chain_svgd_model(
        observed_indices=obs, epoch_starts=EPOCH_STARTS, **kw)
    return model, theta_dim


# ------------------------------------------------------------------ oracle
def _dense_parts(g, P):
    ser = g.serialize(theta_dim=P)
    n = int(ser["n_vertices"])
    ce = np.asarray(ser["edges"], dtype=np.float64).reshape(-1, 3)
    pe = np.asarray(ser["param_edges"], dtype=np.float64)
    if pe.size == 0:
        pe = pe.reshape(0, 2 + P)
    se = np.asarray(ser["start_edges"], dtype=np.float64).reshape(-1, 2)
    return n, ce, pe, se


def _oracle_per_obs(jpg, obs):
    """Dense differentiable replication of the full chain -> per-obs
    values, as in experiments/dr_batchH_oracle.py (H0 tier-2)."""
    P = jpg.param_length()
    jsp = jpg.joint_stop_prob_graph()
    sg = jpg.joint_sojourn_graph()
    iti = [int(x) for x in jsp._ipv_target_indices]
    tam = {int(k): int(v) for k, v in jsp._t_aux_map.items()}
    tvi = [int(x) for x in jsp._t_vertex_indices]
    jsp_states = jsp.states(); sg_states = sg.states()
    ipv_pos = {tuple(int(x) for x in jsp_states[v]): k for k, v in enumerate(iti)}
    s2i = {tuple(int(x) for x in sg_states[v]): v for v in range(sg.vertices_length())}
    gather = [int(ipv_pos[s]) for s in sg._ipv_target_states]
    sti = [int(s2i[tuple(int(x) for x in jsp_states[t])]) for t in tvi]
    obs_pos = [tvi.index(o) for o in obs]

    full = np.zeros(jpg.vertices_length())
    for e in jpg.starting_vertex().edges():
        full[e.to().index()] = e.weight()
    ipv0 = full[np.asarray(iti)]

    n_jsp, jsp_ce, jsp_pe, _ = _dense_parts(jsp, P)
    n_sg, sg_ce, sg_pe, sg_se = _dense_parts(sg, P)
    outdeg = np.zeros(n_sg)
    for arr in (sg_ce, sg_pe):
        for row in arr:
            outdeg[int(row[0])] += 1
    trans = np.where(outdeg > 0)[0]
    tp_ = {int(v): i for i, v in enumerate(trans)}
    t_pos = np.asarray([tp_[t] for t in sti])
    a_scat = sg_se[:, 0].astype(int)
    tset = {int(t): k for k, t in enumerate(sti)}
    r_const = np.zeros(len(sti)); r_coeff = np.zeros((len(sti), P))
    for row in sg_ce:
        f, t = int(row[0]), int(row[1])
        if f in tset and outdeg[t] == 0:
            r_const[tset[f]] += row[2]
    for row in sg_pe:
        f, t = int(row[0]), int(row[1])
        if f in tset and outdeg[t] == 0:
            r_coeff[tset[f]] += row[2:]

    def Q_of(n, ce, pe, th):
        Q = jnp.zeros((n, n), dtype=jnp.float64)
        if ce.shape[0]:
            Q = Q.at[ce[:, 0].astype(int), ce[:, 1].astype(int)].add(jnp.asarray(ce[:, 2]))
        if pe.shape[0]:
            Q = Q.at[pe[:, 0].astype(int), pe[:, 1].astype(int)].add(jnp.asarray(pe[:, 2:]) @ th)
        return Q - jnp.diag(jnp.sum(Q, axis=1))

    def chain(theta_flat):
        p = jnp.zeros(n_jsp, dtype=jnp.float64).at[jnp.asarray(iti)].set(jnp.asarray(ipv0))
        ipv = None
        for e in range(len(EPOCH_STARTS) - 1):
            Q = Q_of(n_jsp, jsp_ce, jsp_pe, theta_flat[e * P:(e + 1) * P])
            pt = p @ expm(Q * DT)
            vals = []
            for tgt in iti:
                v = pt[tgt]
                if tgt in tam:
                    v = v + pt[tam[tgt]]
                vals.append(v)
            ipv = jnp.stack(vals)
            p = jnp.zeros(n_jsp, dtype=jnp.float64).at[jnp.asarray(iti)].set(ipv)
        alpha = ipv[jnp.asarray(gather)]
        thf = theta_flat[-P:]
        Qs = Q_of(n_sg, sg_ce, sg_pe, thf)
        T = Qs[jnp.asarray(trans)][:, jnp.asarray(trans)]
        af = jnp.zeros(n_sg, dtype=jnp.float64).at[jnp.asarray(a_scat)].set(alpha)
        soj = jnp.linalg.solve(-T.T, af[jnp.asarray(trans)])
        r = jnp.asarray(r_const) + jnp.asarray(r_coeff) @ thf
        return (r * soj[jnp.asarray(t_pos)])[jnp.asarray(obs_pos)]

    return chain


P_DIM = 2
FINAL_LO = P_DIM  # (n_epochs-1)*P


# --------------------------------------------------------------- test 1
@pytest.mark.parametrize("theta_np", [THETA_BENIGN, THETA_MIXED],
                         ids=["benign", "mixed_scale"])
def test_composed_jacobian_matches_oracle(jpg, obs, theta_np):
    model_ex, _ = _models(jpg, obs, exact_final_grad=True)
    model_fd, _ = _models(jpg, obs)
    oracle = _oracle_per_obs(jpg, obs)
    th = jnp.asarray(theta_np)

    def out_of(m):
        return lambda t: m(t, None)[0]

    J_or = np.asarray(jax.jacobian(oracle)(th))
    J_ex = np.asarray(jax.jacrev(out_of(model_ex))(th))
    J_fd = np.asarray(jax.jacrev(out_of(model_fd))(th))
    denom = np.max(np.abs(J_or))
    err_ex = np.max(np.abs(J_ex[:, FINAL_LO:] - J_or[:, FINAL_LO:])) / denom
    err_fd = np.max(np.abs(J_fd[:, FINAL_LO:] - J_or[:, FINAL_LO:])) / denom
    assert err_ex < 1e-9, f"exact final cols off oracle: {err_ex:.3e}"
    assert err_fd / err_ex > 1e3, (
        f"improvement only {err_fd / err_ex:.1f}x (fd={err_fd:.3e}, "
        f"ex={err_ex:.3e})")
    # earlier cols identical to the FD model (unchanged path)
    assert np.array_equal(J_ex[:, :FINAL_LO], J_fd[:, :FINAL_LO])


# --------------------------------------------------------------- test 2
def test_false_path_is_plain_fd(jpg, obs):
    """exact_final_grad=False must be the untouched FD backward: (a) the
    kwarg default and explicit False produce BITWISE-equal gradients
    (same code path, same XLA program); (b) the gradient matches a
    manually-recomputed central difference of the model forward to
    float round-off (different XLA program -> 1-ULP wiggle is legitimate,
    so rtol 1e-12, NOT bitwise). The cross-install golden byte-identity
    (C default path AND the daisy False-path gradient) lives in
    experiments/dr_batchH_i1_gate.py micro-gates (a)/(a2)."""
    model_f, _ = _models(jpg, obs, exact_final_grad=False)
    model_d, _ = _models(jpg, obs)  # kwarg default
    th = jnp.asarray(THETA_BENIGN)
    cot = jnp.asarray(np.linspace(0.5, 1.5, len(obs)))

    def loss_of(m):
        return lambda t: jnp.sum(cot * m(t, None)[0])

    g_f = np.asarray(jax.grad(loss_of(model_f))(th))
    g_d = np.asarray(jax.grad(loss_of(model_d))(th))
    np.testing.assert_array_equal(g_f, g_d)

    eps = 1e-7
    ref = np.empty(th.shape[0])
    for i in range(th.shape[0]):
        tp = th.at[i].add(eps); tm = th.at[i].add(-eps)
        ref[i] = float(jnp.sum(
            cot * (model_f(tp, None)[0] - model_f(tm, None)[0]) / (2 * eps)))
    np.testing.assert_allclose(g_f, ref, rtol=1e-12, atol=0)


# --------------------------------------------------------------- test 3
def test_tied_gradient_matches_oracle_column_sums(jpg, obs):
    """tied=[(0, [0, 1])]: the master slot's gradient must equal the sum
    of the oracle's flat columns 0 and P (the _apply_tying scatter-VJP)."""
    model, _ = _models(jpg, obs, exact_final_grad=True,
                       user_tied=[(0, [0, 1])])
    oracle = _oracle_per_obs(jpg, obs)
    th = jnp.asarray(THETA_BENIGN.copy())
    th = th.at[FINAL_LO].set(th[0])  # tied: slave mirrors master
    cot = jnp.asarray(np.linspace(0.5, 1.5, len(obs)))

    def loss(t):
        return jnp.sum(cot * model(t, None)[0])

    g = np.asarray(jax.grad(loss)(th))
    J_or = np.asarray(jax.jacobian(oracle)(th))
    g_or = cot @ J_or
    tied_or = g_or[0] + g_or[FINAL_LO]
    # G4 fold: tolerances anchored to measured actuals (~8.6e-10) with
    # 100-1000x headroom -- 1e-3 could not detect a dropped earlier
    # share (its magnitude is ~4e-5 of the slave share on this fixture).
    assert abs(g[0] - tied_or) / max(abs(tied_or), 1e-30) < 1e-6, (
        f"tied master: model {g[0]:.6e} vs oracle col-sum {tied_or:.6e}")
    # the final-epoch (slave) share is exact: subtracting the FD earlier
    # share leaves the exact block's contribution. Explicit nonzero
    # precondition (G4 fold: the previous `and` idiom passed vacuously
    # at an exactly-zero oracle share).
    assert g_or[FINAL_LO] != 0.0, "fixture degenerate: zero oracle share"
    assert abs((g[0] - g_or[0]) - g_or[FINAL_LO]) / \
        abs(g_or[FINAL_LO]) < 1e-8


# --------------------------------------------------------------- test 4
def test_exposure_branch_matches_oracle(jpg, obs):
    """Duplicated exposure values (K_unique < n_obs): the exact final
    slots must match a per-obs scaled oracle; in particular the
    NON-exposure final slot must NOT be scaled by alpha_u (the
    slot-specific chain rule)."""
    expo = np.array([2.0, 2.0, 0.5, 0.5, 2.0])[:len(obs)]
    model, _ = _models(jpg, obs, exact_final_grad=True,
                       exposure_arr=jnp.asarray(expo),
                       exposure_param_index=1)
    oracle = _oracle_per_obs(jpg, obs)
    th = jnp.asarray(THETA_BENIGN)
    cot = jnp.asarray(np.linspace(0.5, 1.5, len(obs)))

    def loss(t):
        return jnp.sum(cot * model(t, None)[0])

    def loss_oracle(t):
        tot = 0.0
        for o in range(len(obs)):
            scale = jnp.ones_like(t).at[1].set(expo[o]).at[FINAL_LO + 1].set(expo[o])
            tot = tot + cot[o] * oracle(t * scale)[o]
        return tot

    g = np.asarray(jax.grad(loss)(th))
    g_or = np.asarray(jax.grad(loss_oracle)(th))
    denom = np.max(np.abs(g_or))
    err_final = np.max(np.abs(g[FINAL_LO:] - g_or[FINAL_LO:])) / denom
    assert err_final < 1e-6, (
        f"exposure exact final slots vs oracle: {err_final:.3e} "
        f"(g={g.tolist()}, oracle={g_or.tolist()})")


# --------------------------------------------------------------- test 5
def test_construction_declines_loudly(jpg, obs):
    with pytest.raises(ValueError, match="final_read='sojourn'"):
        _models(jpg, obs, exact_final_grad=True, final_read='stopprob')

    fg = _build_jpg()
    fg.weight_formula = "t0 * c0 + t1 * c1"
    with pytest.raises(ValueError, match="linear"):
        _models(fg, obs, exact_final_grad=True)


# --------------------------------------------------------------- test 6
def test_zero_mass_handoff_zero_block_no_c_call(jpg, obs, monkeypatch):
    """The mass==0 branch. NOTE a dynamically-zero handoff is IMPOSSIBLE
    for this model class (mass absorbed in earlier epochs lands at
    t-states, which are part of the handoff -- the joint-prob semantics),
    so the branch is defensive; it is forced here via the class-level
    `stop_probability` seam (per-call lookup on the private clone). The
    review-verified confound -- the C sojourn forward NaN-fills at a
    zero IPV -- is pinned directly first."""
    # (a) pin the confound: C subset-sojourn forward at a zero IPV is NaN
    jsp = jpg.joint_stop_prob_graph()
    sg = jpg.joint_sojourn_graph()
    jsp_states = jsp.states(); sg_states = sg.states()
    s2i = {tuple(int(x) for x in sg_states[v]): v
           for v in range(sg.vertices_length())}
    sti = [int(s2i[tuple(int(x) for x in jsp_states[t])])
           for t in jsp._t_vertex_indices][:3]
    sg.update_ipv(np.zeros(len(sg._ipv_target_states)))
    sg.update_weights([1e-4, 1.3e-4])
    soj0 = np.asarray(sg.expected_sojourn_time(sti))
    assert bool(np.all(np.isnan(soj0))), (
        f"expected NaN sojourn at zero IPV (0-mass normalization); got {soj0}")

    # (b) the backward's zero-mass branch: zero handoff -> exact 0.0
    # final slots, NO C adjoint call, no raise.
    model, _ = _models(jpg, obs, exact_final_grad=True)
    th = jnp.asarray(THETA_BENIGN)

    calls = []
    orig = Graph._sojourn_grad_theta_subset

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", spy)
    monkeypatch.setattr(
        Graph, "stop_probability",
        lambda self, *a, **k: np.zeros(self.vertices_length()))

    out, vjp = jax.vjp(lambda t: model(t, None)[0], th)
    (g,) = vjp(jnp.ones_like(out))  # finite synthetic cotangent
    g = np.asarray(g)
    assert np.array_equal(g[FINAL_LO:], np.zeros(P_DIM)), g
    assert not calls, "C adjoint was called despite the zero-mass branch"


# --------------------------------------------------------------- test 7
def test_flag_semantics_on_shared_c_function(jpg):
    """In-suite I1 checks: at a benign point skip_condition_gate makes no
    difference; at the H0-style declining point (realistic theta + wide-
    range ipv) the default declines while skip computes finite values.
    (Cross-install byte-identity is experiments/dr_batchH_i1_gate.py.)"""
    sg = jpg.joint_sojourn_graph()
    n_ipv = len(sg._ipv_target_states)
    rng = np.random.default_rng(0)
    idx = [int(x) for x in
           jpg.joint_stop_prob_graph()._t_vertex_indices][:3]
    jsp_states = jpg.joint_stop_prob_graph().states()
    sg_states = sg.states()
    s2i = {tuple(int(x) for x in sg_states[v]): v
           for v in range(sg.vertices_length())}
    sti = [int(s2i[tuple(int(x) for x in jsp_states[t])]) for t in idx]

    sg.update_ipv(rng.uniform(0.1, 1.0, n_ipv))
    sg.update_weights([1.0, 1.0])
    a = sg._sojourn_grad_theta_subset(sti)
    b = sg._sojourn_grad_theta_subset(sti, skip_condition_gate=True)
    assert a and b and np.array_equal(np.asarray(a), np.asarray(b))

    wide = np.full(n_ipv, 1e-40); wide[0] = 1.0
    sg.update_ipv(wide)
    sg.update_weights([0.5e-4, 1.3e-4])
    assert not sg._sojourn_grad_theta_subset(sti), (
        "expected the default conditioning gate to decline here")
    res = sg._sojourn_grad_theta_subset(sti, skip_condition_gate=True)
    assert res and np.all(np.isfinite(res))


# --------------------------------------------------------------- test 8
def test_vmap_jit_grad_runs_exact_path_and_raise_is_legible(jpg, obs,
                                                            monkeypatch):
    """(a) Under vmap(grad) tracing, the FD perturbations for the
    final-epoch slots are GONE from the emitted program. Trace-time
    call counting (the Batch-F spy mechanism -- a compiled/jitted
    re-execution calls the registered FFI handler directly, so a
    Python-level spy only sees TRACING): the exact model must emit
    exactly 4 fewer fused-FFI ops than the FD model (2 slots x 2
    perturbations), and execution under vmap(jit(grad)) stays finite.
    The spy patches phasic.ffi_wrappers BEFORE model construction (the
    builder from-imports at construction; patching later intercepts
    nothing). (b) A forced residual decline raises legibly under
    vmap(jit(grad))."""
    import phasic.ffi_wrappers as fw

    calls = []
    orig = fw.compute_daisy_chain_sojourn_ffi

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(fw, "compute_daisy_chain_sojourn_ffi", spy)
    model_ex, _ = _models(jpg, obs, exact_final_grad=True)
    model_fd, _ = _models(jpg, obs)

    def loss_of(m):
        return lambda t: jnp.sum(jnp.log(m(t, None)[0] + 1e-300))

    thetas = jnp.stack([jnp.asarray(THETA_BENIGN),
                        jnp.asarray(THETA_BENIGN) * 1.3])
    calls.clear()
    jax.make_jaxpr(jax.vmap(jax.grad(loss_of(model_ex))))(thetas)
    ex_calls = len(calls)
    calls.clear()
    jax.make_jaxpr(jax.vmap(jax.grad(loss_of(model_fd))))(thetas)
    fd_calls = len(calls)
    # Each traced _forward invocation passes through the wrapper a fixed
    # number of times under vmap+grad tracing (empirically 2; derive it
    # rather than hardcode). FD model invokes _forward (1 + 2*n_slots)
    # times; exact model (1 + 2*(n_slots - P_DIM)) times.
    # G4 fold (both refuters): a dead spy gives fd_calls == 0, which the
    # factor arithmetic would accept (0 == 0*5) -- guard it out.
    assert fd_calls > 0, "FD spy dead -- seam moved? patch-before-construction"
    n_slots = 2 * P_DIM
    assert fd_calls % (1 + 2 * n_slots) == 0, (fd_calls, ex_calls)
    factor = fd_calls // (1 + 2 * n_slots)
    assert ex_calls == factor * (1 + 2 * (n_slots - P_DIM)), (
        f"exact model traced {ex_calls} fused-FFI ops vs FD {fd_calls} "
        f"(factor {factor}); final-slot FD perturbations not removed")
    g = np.asarray(jax.vmap(jax.jit(jax.grad(loss_of(model_ex))))(thetas))
    assert np.all(np.isfinite(g))

    # (b) forced residual decline -> legible RuntimeError under the jitted
    # composition (F0 finding transfers; message must survive)
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset",
                        lambda self, *a, **k: [])
    model2, _ = _models(jpg, obs, exact_final_grad=True)
    with pytest.raises(Exception, match="exact sojourn adjoint declined"):
        jax.vmap(jax.jit(jax.grad(
            lambda t: jnp.sum(jnp.log(model2(t, None)[0] + 1e-300))
        )))(thetas)


# -------------------------------------------------------------- test 10
def test_single_epoch_all_slots_exact(jpg, obs):
    """n_epochs == 1 (G4 fold, suite-gap note): the 'final epoch' is the
    whole model -- handoff == initial IPV, the FD perturbation set is
    EMPTY (_efg_final_lo == 0), every slot gets the exact block. Checked
    against a final-read-only oracle (no intermediate epochs)."""
    model, theta_dim = _models(jpg, obs, exact_final_grad=True)
    model1, theta_dim1 = jpg._daisy_chain_svgd_model(
        observed_indices=obs, epoch_starts=[0.0], exact_final_grad=True)[:2]
    assert theta_dim1 == P_DIM
    th1 = jnp.asarray(THETA_BENIGN[:P_DIM])
    cot = jnp.asarray(np.linspace(0.5, 1.5, len(obs)))

    g = np.asarray(jax.grad(
        lambda t: jnp.sum(cot * model1(t, None)[0]))(th1))
    # oracle: the 2-epoch oracle with epoch-0 == identity is not
    # available; instead use the full-chain oracle with the SAME theta in
    # both epochs and dt->'no intermediate epoch' is not expressible --
    # so validate against tight central differences of the single-epoch
    # model forward (Richardson-free; the forward is granularity-free
    # elimination, smooth in theta).
    eps = 1e-9
    ref = np.empty(P_DIM)
    for i in range(P_DIM):
        tp = th1.at[i].add(eps); tm = th1.at[i].add(-eps)
        ref[i] = float(jnp.sum(cot * (model1(tp, None)[0] -
                                      model1(tm, None)[0]) / (2 * eps)))
    np.testing.assert_allclose(g, ref, rtol=5e-5, atol=0)
    assert np.all(np.isfinite(g))


# --------------------------------------------------------------- test 9
def test_nan_handoff_raises_loudly(jpg, obs, monkeypatch):
    model, _ = _models(jpg, obs, exact_final_grad=True)
    monkeypatch.setattr(
        Graph, "stop_probability",
        lambda self, *a, **k: np.full(self.vertices_length(), np.nan))
    with pytest.raises(Exception, match="16b item 8"):
        jax.grad(lambda t: jnp.sum(model(t, None)[0]))(
            jnp.asarray(THETA_BENIGN))
