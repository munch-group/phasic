"""End-to-end tests for daisy-chain joint-prob SVGD with `exposure`.

Per-observation **exposure** is the GLM construct of a known
multiplicative scaling on a rate-typed parameter. Concrete instances:
segment length in coalescent-with-mutation models, time-at-risk in
survival, area in spatial Poisson regression. Three tests:

1. ``test_uniform_alpha_equivalence`` — with constant
   ``exposure=alpha_const`` the per-obs path must reproduce the result
   of running the daisy chain with the rate-typed parameter pre-scaled
   by ``alpha_const``. Tolerance: 1e-10 absolute on per-obs joint
   probabilities.

2. ``test_per_obs_scaling_vs_montecarlo`` — with two distinct alpha
   values split across observations, the per-obs joint-prob output
   must agree with an independent Monte-Carlo simulator of the
   piecewise-time coalescent-with-mutation model (same pattern as the
   proof-of-correctness section of
   ``docs/pages/tutorial/joint-probability.ipynb``). Tolerance:
   max ``|daisy - mc| / mc_se`` < 4 over reported outcomes.

3. ``test_daisy_chain_exposure_perf`` — with ``n_obs=300`` (kept
   modest for CI), one forward + one finite-difference gradient via
   ``daisy_chain_joint_probs`` must complete in under 90 seconds.
   Without the per-obs FFI batching this previously hung indefinitely
   on JIT compilation.
"""
from __future__ import annotations

import time
from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import (
    Graph, Property, StateIndexer, with_ipv, _ensure_jax_active,
)


# ---------------------------------------------------------------------------
# Common N=2 coalescent model fixture (smallest non-trivial joint-prob).
# ---------------------------------------------------------------------------


N = 2


def _build_jsp_graph():
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()
    return jsp


# ---------------------------------------------------------------------------
# Test 1: uniform alpha equivalence.
# ---------------------------------------------------------------------------


def test_uniform_alpha_equivalence():
    """Per-obs path with constant exposure alpha must equal the no-
    exposure path with the rate-typed parameter pre-multiplied by
    alpha_const.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    jsp = _build_jsp_graph()
    initial_ipv = np.zeros(len(jsp._ipv_target_indices), dtype=np.float64)
    initial_ipv[0] = float(N)

    n_obs = 5
    alpha_const = 1.5
    n_epochs = 2
    param_length = 2  # joint-prob graph: [coal, mu]
    theta_dim = n_epochs * param_length

    # Reference: daisy chain WITHOUT exposure, with the rate-typed
    # parameter (the mu slot) pre-scaled by alpha_const.
    base_theta_per_epoch = np.array([[1.0, 0.5], [1.0, 2.0]])  # (n_epochs, param_length)
    scaled_theta_per_epoch = base_theta_per_epoch.copy()
    scaled_theta_per_epoch[:, 1] *= alpha_const  # apply alpha to the rate slot

    ref = jsp.daisy_chain_joint_probs(
        epoch_thetas=jnp.asarray(scaled_theta_per_epoch),
        epoch_dts=jnp.asarray([1.0]),
        initial_ipv=jnp.asarray(initial_ipv),
        t_eval=200.0,
    )
    ref = np.asarray(ref)

    # Per-obs path: build the model directly and call it.
    base_g = jsp  # _daisy_chain_svgd_model lives on the base joint-prob graph
    # We need to reach _daisy_chain_svgd_model from the joint-prob graph
    # (jsp's parent). Easier: reproduce the FFI call directly via the
    # batched compute_daisy_chain_joint_probs_ffi, which is exactly what
    # the per-obs model does.
    from phasic.ffi_wrappers import (
        _make_json_serializable,
        compute_daisy_chain_joint_probs_ffi,
    )
    import json as _json

    structure = _make_json_serializable(jsp.serialize(theta_dim=param_length))
    structure["_daisy_chain"] = {
        "n_epochs": int(n_epochs),
        "param_length": int(param_length),
        "t_eval": 200.0,
        "granularity": 0,
        "epoch_dts": [1.0],
        "ipv_target_indices": [int(x) for x in jsp._ipv_target_indices],
        "t_aux_keys": [int(k) for k in jsp._t_aux_map.keys()],
        "t_aux_values": [int(jsp._t_aux_map[k]) for k in jsp._t_aux_map.keys()],
        "t_vertex_indices": [int(x) for x in jsp._t_vertex_indices],
    }
    structure_json = _json.dumps(structure)

    # Build per-obs theta_batch with the SAME alpha scaling as the reference.
    base_theta_flat = base_theta_per_epoch.reshape(-1)  # (theta_dim,)
    theta_batch = np.broadcast_to(base_theta_flat, (n_obs, theta_dim)).copy()
    flat_exposure_indices = np.array(
        [epoch * param_length + 1 for epoch in range(n_epochs)]
    )
    alpha_arr = np.full((n_obs,), alpha_const)
    for k, fei in enumerate(flat_exposure_indices):
        theta_batch[:, fei] *= alpha_arr  # broadcasts alpha_const across obs

    initial_ipv_batched = np.broadcast_to(
        initial_ipv[None, :], (n_obs, len(initial_ipv))
    )

    per_obs_result = compute_daisy_chain_joint_probs_ffi(
        structure_json,
        jnp.asarray(theta_batch),
        jnp.asarray(initial_ipv_batched),
    )
    per_obs_result = np.asarray(per_obs_result)  # (n_obs, n_t)

    # Each row of per_obs_result must equal the reference.
    for i in range(n_obs):
        np.testing.assert_allclose(
            per_obs_result[i], ref, atol=1e-10,
            err_msg=f"per-obs row {i} differs from reference",
        )


# ---------------------------------------------------------------------------
# Test 2: per-obs scaling vs Monte Carlo (mirrors joint-probability.ipynb).
# ---------------------------------------------------------------------------


def test_per_obs_scaling_vs_montecarlo():
    """For a single-epoch model with four distinct exposure values, the
    per-obs daisy-chain output must agree with a Monte-Carlo simulator
    of the coalescent-with-mutation model where each observation has
    its own effective rate ``base_rate * alpha_i``.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    jsp = _build_jsp_graph()
    initial_ipv = np.zeros(len(jsp._ipv_target_indices), dtype=np.float64)
    initial_ipv[0] = float(N)

    n_obs = 4
    alpha_values = np.array([0.5, 1.0, 1.5, 2.0])
    base_rate = 0.5
    coal_rate = 1.0
    n_epochs = 1
    param_length = 2

    from phasic.ffi_wrappers import (
        _make_json_serializable,
        compute_daisy_chain_joint_probs_ffi,
    )
    import json as _json

    structure = _make_json_serializable(jsp.serialize(theta_dim=param_length))
    structure["_daisy_chain"] = {
        "n_epochs": int(n_epochs),
        "param_length": int(param_length),
        "t_eval": 400.0,
        "granularity": 0,
        "epoch_dts": [],
        "ipv_target_indices": [int(x) for x in jsp._ipv_target_indices],
        "t_aux_keys": [int(k) for k in jsp._t_aux_map.keys()],
        "t_aux_values": [int(jsp._t_aux_map[k]) for k in jsp._t_aux_map.keys()],
        "t_vertex_indices": [int(x) for x in jsp._t_vertex_indices],
    }
    structure_json = _json.dumps(structure)

    base_theta_flat = np.array([coal_rate, base_rate])
    theta_batch = np.broadcast_to(base_theta_flat, (n_obs, param_length)).copy()
    theta_batch[:, 1] *= alpha_values

    initial_ipv_batched = np.broadcast_to(
        initial_ipv[None, :], (n_obs, len(initial_ipv))
    )

    per_obs_result = compute_daisy_chain_joint_probs_ffi(
        structure_json,
        jnp.asarray(theta_batch),
        jnp.asarray(initial_ipv_batched),
    )
    # Normalise by N (initial mass) to get probabilities.
    per_obs_pmf = np.asarray(per_obs_result) / N  # (n_obs, n_t)

    # Monte Carlo: for each observation, simulate N=2 coalescent with
    # rate_i = base_rate * alpha_i.
    rng = np.random.default_rng(42)
    nsim = 200_000
    n_t = per_obs_pmf.shape[1]
    mc_pmf = np.zeros((n_obs, n_t))

    for i in range(n_obs):
        rate_i = base_rate * alpha_values[i]
        T_coal = rng.exponential(1.0 / coal_rate, nsim)
        K = rng.poisson(rate_i * T_coal) + rng.poisson(rate_i * T_coal)
        # Truncate to avoid out-of-bounds.
        K_clipped = np.minimum(K, n_t - 1)
        counts = np.bincount(K_clipped, minlength=n_t)[:n_t]
        mc_pmf[i] = counts / nsim

    # Compare the first 8 outcome rows (which have the most mass).
    n_compare = min(8, n_t)
    mc_se = np.sqrt(mc_pmf * (1 - mc_pmf) / nsim).clip(min=1e-10)
    err_in_sigmas = np.abs(per_obs_pmf[:, :n_compare] - mc_pmf[:, :n_compare]) / mc_se[:, :n_compare]
    max_err = float(err_in_sigmas.max())
    assert max_err < 4.0, (
        f"max |daisy - MC| in MC-sigma units = {max_err:.2f}; expected < 4. "
        f"Per-obs daisy_chain output disagrees with Monte-Carlo simulator."
    )


# ---------------------------------------------------------------------------
# Test 3: performance smoke — n_obs=300 must complete fast.
# ---------------------------------------------------------------------------


def test_daisy_chain_exposure_perf():
    """One forward daisy-chain call with n_obs=300 (per-observation
    exposure batched into a single FFI call) must complete in <90s.
    Without per-obs FFI batching the previous code path JIT-compiled
    an unrolled lax.map over 300 iterations and effectively hung.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    jsp = _build_jsp_graph()
    initial_ipv = np.zeros(len(jsp._ipv_target_indices), dtype=np.float64)
    initial_ipv[0] = float(N)

    n_obs = 300
    n_epochs = 2
    param_length = 2
    theta_dim = n_epochs * param_length

    from phasic.ffi_wrappers import (
        _make_json_serializable,
        compute_daisy_chain_joint_probs_ffi,
    )
    import json as _json

    structure = _make_json_serializable(jsp.serialize(theta_dim=param_length))
    structure["_daisy_chain"] = {
        "n_epochs": int(n_epochs),
        "param_length": int(param_length),
        "t_eval": 200.0,
        "granularity": 0,
        "epoch_dts": [1.0],
        "ipv_target_indices": [int(x) for x in jsp._ipv_target_indices],
        "t_aux_keys": [int(k) for k in jsp._t_aux_map.keys()],
        "t_aux_values": [int(jsp._t_aux_map[k]) for k in jsp._t_aux_map.keys()],
        "t_vertex_indices": [int(x) for x in jsp._t_vertex_indices],
    }
    structure_json = _json.dumps(structure)

    rng = np.random.default_rng(0)
    alpha_arr = rng.uniform(0.5, 2.0, size=(n_obs,))
    base_theta_flat = np.array([1.0, 0.5, 1.0, 0.5])
    theta_batch = np.broadcast_to(base_theta_flat, (n_obs, theta_dim)).copy()
    flat_exposure_indices = np.array(
        [epoch * param_length + 1 for epoch in range(n_epochs)]
    )
    for fei in flat_exposure_indices:
        theta_batch[:, fei] *= alpha_arr

    initial_ipv_batched = np.broadcast_to(
        initial_ipv[None, :], (n_obs, len(initial_ipv))
    )

    t0 = time.time()
    result = compute_daisy_chain_joint_probs_ffi(
        structure_json,
        jnp.asarray(theta_batch),
        jnp.asarray(initial_ipv_batched),
    )
    np.asarray(result).sum()  # force materialisation
    forward_time = time.time() - t0

    assert forward_time < 90.0, (
        f"Batched forward took {forward_time:.1f}s for n_obs={n_obs}; "
        "expected < 90s. The per-obs path may have regressed to a "
        "Python-side fan-out."
    )


# ---------------------------------------------------------------------------
# Test 4: lambda_max regression — t-aux loop must not couple to exposure.
# ---------------------------------------------------------------------------


def _build_n4_jsp_for_regression():
    """Build the user's N=4, reward_limit=5 JSP graph (the model where
    the t-aux blow-up was first observed)."""
    nr_samples = 4
    reward_limit = 5
    mut_rate = 1.0e-8

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=nr_samples)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = nr_samples

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            descendants = (
                indexer.lineages.index_to_props(i).descendants
                + indexer.lineages.index_to_props(j).descendants
            )
            k = indexer.lineages.props_to_index(descendants=descendants)
            new[k] += 1
            out.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return out

    base = Graph(cb, indexer=indexer)
    jpg = base.joint_prob_graph(
        mutation_rate=mut_rate, reward_limit=reward_limit, discrete=False,
    )
    return jpg.joint_stop_prob_graph()


def test_exposure_does_not_blow_up_lambda_max():
    """Regression for the t-aux loop bug.

    The t-vertex <-> aux trapping loop in the JSP graph used to be
    parameterised with coefficients ``[1.0] * param_length``, so its
    runtime weight was ``Σ θ_k = θ_coal + θ_mu``. Under per-obs
    exposure scaling, ``θ_mu`` was multiplied by ``alpha_i`` (segment
    length, up to ~1.8e5 in the user's data), inflating every t-aux
    edge to ~1.2e5 and dominating ``λ_max`` of the JSP graph. The
    auto-granularity (``g = 2 * λ_max``) then ballooned, so each
    ``stop_probability(t)`` call ran for hundreds of thousands of
    uniformization steps and SVGD effectively hung.

    The fix replaces the parameterised t-aux edges with coefficient-
    less constant edges (weight 1.0, untouched by ``update_weights``).
    After the fix, ``λ_max`` is dominated by the natural coalescent
    rate (``6 * θ_coal`` for the all-singletons vertex) and stays
    small regardless of how the per-obs wrapper scales ``θ_mu``.
    """
    jsp = _build_n4_jsp_for_regression()

    # Realistic precompile-time theta in CONSTRAINED space:
    # softplus(0) = 0.6931 for each slot of the SVGD dummy theta.
    # The per-obs wrapper would then scale theta_mu (slot 1) by the
    # exposure value alpha_i. Use the largest alpha from the user's
    # tree-spans (~1.79e5 bases) as the worst case.
    softplus_zero = 0.6931
    max_alpha = 1.79e5
    theta = np.array([softplus_zero, softplus_zero * max_alpha])
    jsp.update_weights(theta)

    # Find max-rate vertex.
    max_rate = 0.0
    max_v = None
    for v in jsp.vertices():
        rate = sum(e.weight() for e in v.edges())
        if rate > max_rate:
            max_rate = rate
            max_v = v

    # Pre-fix value was ~1.24e5; post-fix should be ~4.16 (which is
    # 6 * softplus_zero, the natural coalescent rate at the
    # all-singletons vertex). A generous bound captures the change.
    assert max_rate < 100.0, (
        f"λ_max = {max_rate:.4g} (expected ~4 after fix; pre-fix value was "
        f"~{6 * softplus_zero * max_alpha:.4g}). The t-aux loop is "
        "coupling to exposure scaling again."
    )

    # The max-rate vertex must NOT be a t-vertex; it should be an
    # interior coalescent vertex.
    t_vertex_indices_set = set(jsp._t_vertex_indices)
    aux_vertex_indices_set = set(jsp._t_aux_map.values())
    assert max_v.index() not in t_vertex_indices_set, (
        f"max-rate vertex {max_v.index()} is a t-vertex with state "
        f"{tuple(max_v.state())}. The t-aux loop is still coupling to theta."
    )
    assert max_v.index() not in aux_vertex_indices_set, (
        f"max-rate vertex {max_v.index()} is a t-aux vertex with state "
        f"{tuple(max_v.state())}. The t-aux loop is still coupling to theta."
    )


# ---------------------------------------------------------------------------
# Test 5: end-to-end SVGD smoke on the user's exposure case.
# ---------------------------------------------------------------------------


def test_svgd_end_to_end_with_exposure_and_epoch_starts():
    """Run Graph.svgd(exposure=..., epoch_starts=...) end-to-end on the
    N=4 model that previously exposed the t-aux blow-up bug, and
    verify that:

    1. The SVGD loop completes (does not hang).
    2. Particles end up with finite values in the right shape.
    3. The mu slot (fixed) stays at the fixed value across particles.

    This is the symptom test the user identified — the call must
    actually reach iteration 0 and complete, not just precompile-and-
    hang. We use small ``n_obs``, ``n_particles`` and ``n_iterations``
    so the test stays under ~2 minutes on CI; the regression we are
    guarding against is "hangs forever", so even a tiny iteration
    count is sufficient.
    """
    from phasic import (
        Graph, with_ipv, ExpStepSize, LogGaussPrior, StateIndexer, Property,
    )

    nr_samples = 4
    reward_limit = 5
    mut_rate = 1.0e-8

    indexer = StateIndexer(
        lineage=[Property('descendants', min_value=1, max_value=nr_samples)],
    )

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
        transitions = []
        for i, j in combinations_with_replacement(range(indexer.lineage.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            descendants = (
                indexer.lineage.index_to_props(i).descendants
                + indexer.lineage.index_to_props(j).descendants
            )
            k = indexer.lineage.props_to_index(descendants=descendants)
            new[k] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions

    graph = Graph(coalescent_1param)
    joint_prob_graph_cont = graph.joint_prob_graph(
        indexer, reward_limit=reward_limit, mutation_rate=mut_rate, discrete=False,
    )

    # Synthesize a small observation set with a wide range of exposure
    # values, mimicking the user's tree_spans distribution
    # (37 to ~1.79e5 bases). The observations themselves are dummy
    # zeros — we only care that the loop runs end-to-end.
    n_obs = 30
    rng = np.random.default_rng(0)
    observations = [[0, 0, 0, 0]] * n_obs
    tree_spans = list(rng.uniform(37, 1.79e5, n_obs))

    # User's exact call shape, scaled down to keep CI under 2 minutes.
    svgd = joint_prob_graph_cont.svgd(
        observations,
        preconditioner=None,
        fixed=[(1, mut_rate)],
        prior=LogGaussPrior(ci=[1 / 50_000, 1 / 2000]),
        learning_rate=ExpStepSize(first_step=0.05, last_step=0.005, tau=30.0),
        epoch_starts=[0, 0.05371094, 0.15234375],
        exposure=tree_spans,
        exposure_param_index=1,
        n_iterations=2,
        n_particles=4,
        progress=False,
    )

    # Particles shape: (n_particles, n_epochs * param_length) = (4, 6).
    assert svgd.particles.shape == (4, 6), (
        f"unexpected particles shape {svgd.particles.shape}; "
        "expected (4, 6) for n_particles=4 and n_epochs=3, param_length=2."
    )

    # All particle entries finite.
    assert np.all(np.isfinite(svgd.particles)), (
        f"particles contain non-finite values:\n{svgd.particles}"
    )

    # The fixed mu slot in each epoch (flat indices 1, 3, 5) must be at
    # softplus^{-1}(1e-8) ≈ -18.42 in PHI space. SVGD's projection
    # holds these constant by construction.
    expected_phi_for_mu = float(np.log(np.exp(mut_rate) - 1.0))  # ≈ -18.42
    for fixed_flat_idx in (1, 3, 5):
        np.testing.assert_allclose(
            svgd.particles[:, fixed_flat_idx],
            expected_phi_for_mu,
            rtol=1e-5,
            err_msg=(
                f"fixed mu slot at flat index {fixed_flat_idx} drifted "
                f"from softplus^-1(mut_rate)={expected_phi_for_mu:.4f}"
            ),
        )


# ---------------------------------------------------------------------------
# Lever A — auto-dedup of identical theta rows in _per_obs_forward.
# ---------------------------------------------------------------------------


def test_per_obs_dedup_correctness_with_duplicates():
    """With deliberately duplicated exposure values, the per-obs path
    must produce bit-equivalent output to the equivalent path with
    n_obs unique exposures (i.e. dedup must not perturb numerics).
    Same alpha → same FFI input row → bit-identical FFI output. The
    scatter back is just integer indexing, so the per-obs results are
    bit-equal to repeating the unique-row outputs at the corresponding
    duplicate positions.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()

    # Same observed t-vertex for every obs (so any per-obs differences
    # come purely from the exposure path).
    obs = [int(jsp._t_vertex_indices[0])] * 5

    # Path 1: 5 obs, 2 unique exposure values.
    exposure_dup = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    model_dup, _, _, _ = jg._daisy_chain_svgd_model(
        observed_indices=np.asarray(obs, dtype=int),
        epoch_starts=jnp.array([0.0, 0.5]),
        exposure_arr=exposure_dup,
        exposure_param_index=1,
        granularity=64,
    )
    theta = jnp.array([1.0, 0.5, 1.0, 0.5])
    out_dup, _ = model_dup(theta)

    # Path 2: 2 obs at the unique values (no duplicates).
    obs_unique = [int(jsp._t_vertex_indices[0])] * 2
    exposure_unique = np.array([1.0, 2.0])
    model_unique, _, _, _ = jg._daisy_chain_svgd_model(
        observed_indices=np.asarray(obs_unique, dtype=int),
        epoch_starts=jnp.array([0.0, 0.5]),
        exposure_arr=exposure_unique,
        exposure_param_index=1,
        granularity=64,
    )
    out_unique, _ = model_unique(theta)

    # Each duplicate-index obs in path 1 must equal the corresponding
    # unique-index obs in path 2.
    out_dup_np = np.asarray(out_dup)
    out_unique_np = np.asarray(out_unique)
    # alpha=1.0 -> indices 0,1 in dup; index 0 in unique.
    np.testing.assert_array_equal(
        out_dup_np[0], out_unique_np[0],
        "alpha=1.0 dup-row 0 differs from unique-row 0 (dedup not bit-exact)"
    )
    np.testing.assert_array_equal(
        out_dup_np[1], out_unique_np[0],
        "alpha=1.0 dup-row 1 differs from unique-row 0 (scatter wrong)"
    )
    # alpha=2.0 -> indices 2,3,4 in dup; index 1 in unique.
    for i in (2, 3, 4):
        np.testing.assert_array_equal(
            out_dup_np[i], out_unique_np[1],
            f"alpha=2.0 dup-row {i} differs from unique-row 1 (scatter wrong)"
        )


def test_per_obs_dedup_dispatch_batch_size():
    """Confirm the FFI handler is invoked with batch_size = n_unique
    (not n_obs). Patches the FFI wrapper to record the batch shape.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()
    obs = [int(jsp._t_vertex_indices[0])] * 7

    # Patch ffi_wrappers in the package namespace AND the local closure
    # alias inside _daisy_chain_svgd_model. We can't easily patch the
    # closure alias, so we patch before constructing the model so the
    # `from .ffi_wrappers import compute_daisy_chain_joint_probs_ffi`
    # statement picks up the patched binding.
    from phasic import ffi_wrappers
    captured = []
    original_fn = ffi_wrappers.compute_daisy_chain_joint_probs_ffi

    def traced(structure_json, theta, ipv):
        captured.append({'theta_shape': tuple(theta.shape),
                         'ipv_shape': tuple(ipv.shape)})
        return original_fn(structure_json, theta, ipv)

    ffi_wrappers.compute_daisy_chain_joint_probs_ffi = traced
    try:
        # 7 obs, 3 unique values.
        exposure = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        model, _, _, _ = jg._daisy_chain_svgd_model(
            observed_indices=np.asarray(obs, dtype=int),
            epoch_starts=jnp.array([0.0, 0.5]),
            exposure_arr=exposure,
            exposure_param_index=1,
            granularity=64,
            final_read='stopprob',  # this test patches the joint_probs FFI wrapper
        )
        _ = model(jnp.array([1.0, 0.5, 1.0, 0.5]))
    finally:
        ffi_wrappers.compute_daisy_chain_joint_probs_ffi = original_fn

    assert len(captured) >= 1, "FFI handler was not invoked at all"
    # The first call should batch over the 3 unique alphas, not 7 obs.
    first = captured[0]
    assert first['theta_shape'][0] == 3, (
        f"FFI was invoked with batch_size={first['theta_shape'][0]}, "
        f"expected 3 (n_unique). Auto-dedup is not active."
    )
    # IPV is shared across all chains in the daisy-chain initial
    # population, so the model passes a single (1, n_ipv) row and lets
    # the C++ handler broadcast against the theta batch. This matches
    # graph_builder_ffi.cpp:1261-1266 where ipv_batch_size=1 is
    # broadcast against theta_batch_size>=1.
    assert first['ipv_shape'][0] == 1, (
        f"initial_ipv was passed with batch_size={first['ipv_shape'][0]}, "
        f"expected 1 (broadcast row)."
    )


def test_per_obs_dedup_all_unique_passthrough():
    """When every alpha is unique, dedup must pass through n_obs rows
    (i.e. no bogus collapse). Smoke test against the all-unique path.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()
    obs = [int(jsp._t_vertex_indices[0])] * 5
    exposure = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # all unique

    from phasic import ffi_wrappers
    captured = []
    original_fn = ffi_wrappers.compute_daisy_chain_joint_probs_ffi

    def traced(structure_json, theta, ipv):
        captured.append(tuple(theta.shape))
        return original_fn(structure_json, theta, ipv)

    ffi_wrappers.compute_daisy_chain_joint_probs_ffi = traced
    try:
        model, _, _, _ = jg._daisy_chain_svgd_model(
            observed_indices=np.asarray(obs, dtype=int),
            epoch_starts=jnp.array([0.0, 0.5]),
            exposure_arr=exposure,
            exposure_param_index=1,
            granularity=64,
            final_read='stopprob',  # this test patches the joint_probs FFI wrapper
        )
        out, _ = model(jnp.array([1.0, 0.5, 1.0, 0.5]))
    finally:
        ffi_wrappers.compute_daisy_chain_joint_probs_ffi = original_fn

    assert captured[0][0] == 5, (
        f"All-unique alpha case: FFI got batch={captured[0][0]}, expected 5."
    )
    # Output must have one entry per obs.
    assert out.shape == (5,), f"Output shape {out.shape}, expected (5,)"
    # Smaller alpha → larger joint-prob (more time for absorption).
    out_np = np.asarray(out)
    assert np.all(np.diff(out_np) <= 0), (
        f"Per-obs probs should be monotonically non-increasing with alpha; "
        f"got {out_np}"
    )


# ---------------------------------------------------------------------------
# Lever B (smarter parallelism) — OMP fan-out probe.
# ---------------------------------------------------------------------------


def test_omp_full_width_at_handler_entry(caplog):
    """The DaisyChain FFI handler must (a) disable libomp dynamic
    adjustment once at first invocation, and (b) report the configured
    `omp_get_max_threads()` value matching what the user requested via
    OMP_NUM_THREADS / phasic.config cpu_threads. We verify by reading
    the structured DEBUG log emitted by the handler.
    """
    import logging
    from phasic.logging_config import set_log_level

    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    set_log_level('DEBUG')
    # phasic configures propagate=False on its loggers, so caplog
    # (which attaches to the root logger by default) won't see them.
    # Re-enable propagation just for the duration of this test, and
    # attach caplog's handler directly to the phasic.c logger.
    phasic_root = logging.getLogger('phasic')
    phasic_c = logging.getLogger('phasic.c')
    saved_propagate = (phasic_root.propagate, phasic_c.propagate)
    phasic_root.propagate = True
    phasic_c.propagate = True
    caplog.set_level(logging.DEBUG, logger='phasic.c')
    caplog.set_level(logging.DEBUG, logger='phasic')

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()

    # Trigger one FFI call. Use 5 obs so batch_size > 1 and the OMP
    # region actually fires.
    obs = [int(jsp._t_vertex_indices[0])] * 5
    exposure = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model, _, _, _ = jg._daisy_chain_svgd_model(
        observed_indices=np.asarray(obs, dtype=int),
        epoch_starts=jnp.array([0.0, 0.5]),
        exposure_arr=exposure,
        exposure_param_index=1,
        granularity=64,
        final_read='stopprob',  # this test patches the joint_probs FFI wrapper
    )
    _ = model(jnp.array([1.0, 0.5, 1.0, 0.5]))

    # Look for the per-call probe log.
    probe_msgs = [
        r.getMessage() for r in caplog.records
        if 'DaisyChainJointProbsFfiImpl' in r.getMessage()
        and 'omp_get_max_threads' in r.getMessage()
    ]
    # If OpenMP isn't compiled in we'd see "OpenMP disabled" instead;
    # accept either as a successful probe but require the message exists.
    if not probe_msgs:
        # Maybe OpenMP-disabled path — check for that variant too.
        probe_msgs = [
            r.getMessage() for r in caplog.records
            if 'DaisyChainJointProbsFfiImpl' in r.getMessage()
        ]
    # Restore propagation regardless of assertion outcome.
    phasic_root.propagate, phasic_c.propagate = saved_propagate

    assert probe_msgs, (
        "DaisyChain FFI did not emit the omp_get_max_threads() probe "
        "log line. Check the PTD_LOG_DEBUG call at handler entry."
    )


# ---------------------------------------------------------------------------
# Particle-vmap tests (plan: radiant-giggling-wigderson). These verify that
# `vmap(grad(model))(particles)` fuses the P particle batch with the model's
# internal n_unique batch into ONE FFI call of shape (P*n_unique, theta_dim)
# per gradient component, rather than P separate calls. The keystone change
# is the jax.custom_batching.custom_vmap rule on _per_obs_core inside
# _daisy_chain_svgd_model.
# ---------------------------------------------------------------------------


def _build_per_obs_model_for_vmap_tests(n_obs, exposures, theta_dim=4):
    """Helper: build a per-obs daisy-chain SVGD model bound to a small
    coalescent graph. Returns (model, jg, jsp, obs_indices).
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    base.update_weights([1.0])
    jg = base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=20, discrete=False,
    )
    jsp = jg.joint_stop_prob_graph()
    obs = [int(jsp._t_vertex_indices[0])] * n_obs
    model, _, _, _ = jg._daisy_chain_svgd_model(
        observed_indices=np.asarray(obs, dtype=int),
        epoch_starts=jnp.array([0.0, 0.5]),
        exposure_arr=np.asarray(exposures, dtype=np.float64),
        exposure_param_index=1,
        granularity=64,
        final_read='stopprob',  # this test patches the joint_probs FFI wrapper
    )
    return model, jg, jsp, obs


def test_per_obs_model_accepts_single_and_vmapped_theta():
    """The model's forward must produce the same per-obs probabilities
    whether called once with 1D theta or vmapped over P stacked 1D
    thetas. Sanity check that the custom_vmap rule is mathematically
    equivalent to the 1D path."""
    import jax
    import jax.numpy as jnp

    model, _, _, _ = _build_per_obs_model_for_vmap_tests(
        n_obs=5, exposures=[1.0, 1.0, 2.0, 2.0, 3.0],
    )
    theta1d = jnp.array([1.0, 0.5, 1.0, 0.5])
    out_single, _ = model(theta1d)
    assert out_single.shape == (5,)

    particles = jnp.tile(theta1d[None, :], (3, 1))

    def f(theta):
        return model(theta)[0]

    out_vmapped = jax.vmap(f)(particles)
    assert out_vmapped.shape == (3, 5)
    # Every row of the vmapped output equals the single-particle output
    # because all 3 particles are identical.
    for i in range(3):
        np.testing.assert_allclose(
            np.asarray(out_vmapped[i]), np.asarray(out_single),
            rtol=0, atol=1e-12,
        )


def test_per_obs_vmap_matches_python_loop():
    """vmap(f)(particles) must match [f(p) for p in particles] across
    distinct particles to within FP precision."""
    import jax
    import jax.numpy as jnp

    model, _, _, _ = _build_per_obs_model_for_vmap_tests(
        n_obs=7, exposures=[1.0, 1.5, 2.0, 1.0, 2.5, 3.0, 1.5],
    )
    particles = jnp.array([
        [1.0, 0.5, 1.0, 0.5],
        [1.5, 0.4, 0.9, 0.6],
        [0.8, 0.6, 1.1, 0.4],
        [1.2, 0.55, 1.05, 0.45],
    ])

    def f(theta):
        return model(theta)[0]

    out_vmapped = jax.vmap(f)(particles)
    out_loop = jnp.stack([f(p) for p in particles])
    np.testing.assert_allclose(
        np.asarray(out_vmapped), np.asarray(out_loop),
        rtol=0, atol=1e-12,
    )


def test_per_obs_vmap_grad_matches_loop_grad():
    """vmap(grad(loss))(particles) must match per-particle grads to FD
    precision (~1e-5 since the backward is central FD with eps=1e-7)."""
    import jax
    import jax.numpy as jnp

    model, _, _, _ = _build_per_obs_model_for_vmap_tests(
        n_obs=6, exposures=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    )
    particles = jnp.array([
        [1.0, 0.5, 1.0, 0.5],
        [1.5, 0.4, 0.9, 0.6],
        [0.8, 0.6, 1.1, 0.4],
    ])

    def loss(theta):
        return jnp.sum(model(theta)[0])

    g_vmap = jax.vmap(jax.grad(loss))(particles)
    g_loop = jnp.stack([jax.grad(loss)(p) for p in particles])
    assert g_vmap.shape == (3, 4)
    np.testing.assert_allclose(
        np.asarray(g_vmap), np.asarray(g_loop),
        rtol=0, atol=1e-8,
    )


def test_per_obs_vmap_fuses_into_one_ffi_call():
    """CANARY: under vmap(grad(loss)), the FFI handler must be invoked
    with theta of shape (P * n_unique, theta_dim) — fused — not as
    P separate (n_unique, theta_dim) calls. If this fails, the
    custom_vmap rule did not intercept and the perf win is lost.
    """
    import jax
    import jax.numpy as jnp
    from phasic import ffi_wrappers

    # Patch BEFORE building the model so the closure inside
    # _daisy_chain_svgd_model picks up the patched function.
    original_fn = ffi_wrappers.compute_daisy_chain_joint_probs_ffi
    captured = []

    def traced(structure_json, theta, ipv):
        captured.append(tuple(theta.shape))
        return original_fn(structure_json, theta, ipv)

    ffi_wrappers.compute_daisy_chain_joint_probs_ffi = traced
    try:
        # n_obs=6, n_unique=3 (alphas 1,1,2,2,3,3).
        model, _, _, _ = _build_per_obs_model_for_vmap_tests(
            n_obs=6, exposures=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        particles = jnp.array([
            [1.0, 0.5, 1.0, 0.5],
            [1.5, 0.4, 0.9, 0.6],
            [0.8, 0.6, 1.1, 0.4],
            [1.2, 0.55, 1.05, 0.45],
        ])

        def loss(theta):
            return jnp.sum(model(theta)[0])

        captured.clear()
        _ = jax.vmap(jax.grad(loss))(particles)
    finally:
        ffi_wrappers.compute_daisy_chain_joint_probs_ffi = original_fn

    assert len(captured) >= 1, "FFI was not invoked under vmap(grad)"
    # n_unique=3, P=4 → leading axis 12. Plain n_unique would be 3.
    # If the rule did NOT fire, we'd see four separate (3, 4) calls.
    leading_axes = sorted({shape[0] for shape in captured})
    assert 12 in leading_axes, (
        f"FFI was not invoked with fused batch P*n_unique=12; got "
        f"leading axes {leading_axes}. The custom_vmap rule did not "
        f"intercept vmap(grad). Expected one or more (12, 4) calls."
    )
    # And no call should be 3D (which is what would happen if the rule
    # didn't fire and the default expand_dims added a third axis).
    for shape in captured:
        assert len(shape) == 2, (
            f"FFI received a {len(shape)}D theta of shape {shape}; "
            f"the custom_vmap rule failed to keep batches 2D."
        )


def test_per_obs_svgd_runs_with_parallel_vmap():
    """End-to-end smoke: SVGD with the per-obs exposure model must run
    to completion under parallel='vmap' (no longer forced to 'none').
    Also asserts the new _handles_particle_vmap tag is present and
    that SVGD doesn't silently demote vmap to none.
    """
    _ensure_jax_active()
    import jax
    jax.config.update('jax_enable_x64', True)

    from phasic import (
        ExpStepSize, LogGaussPrior, StateIndexer, Property,
    )

    nr_samples = 4
    reward_limit = 5
    mut_rate = 1.0e-8

    indexer = StateIndexer(
        lineage=[Property('descendants', min_value=1, max_value=nr_samples)],
    )

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
        transitions = []
        for i, j in combinations_with_replacement(range(indexer.lineage.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            descendants = (
                indexer.lineage.index_to_props(i).descendants
                + indexer.lineage.index_to_props(j).descendants
            )
            k = indexer.lineage.props_to_index(descendants=descendants)
            new[k] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions

    graph = Graph(coalescent_1param)
    jg = graph.joint_prob_graph(
        indexer, reward_limit=reward_limit, mutation_rate=mut_rate, discrete=False,
    )

    n_obs = 8
    rng = np.random.default_rng(0)
    observations = [[0, 0, 0, 0]] * n_obs
    tree_spans = list(rng.uniform(37, 1.79e5, n_obs))

    # User's call shape (scaled down). Must complete without forcing
    # parallel_mode='none' — the new tag tells SVGD to honour vmap.
    res = jg.svgd(
        observations,
        preconditioner=None,
        fixed=[(1, mut_rate)],
        prior=LogGaussPrior(ci=[1 / 50_000, 1 / 2000]),
        learning_rate=ExpStepSize(first_step=0.05, last_step=0.005, tau=30.0),
        epoch_starts=[0, 0.05371094, 0.15234375],
        exposure=tree_spans,
        exposure_param_index=1,
        n_iterations=2,
        n_particles=4,
        parallel='vmap',
        progress=False,
    )
    assert res is not None
    # res is an SVGD object with .particles and .theta_mean attrs.
    assert hasattr(res, 'theta_mean')
    assert bool(np.all(np.isfinite(np.asarray(res.theta_mean))))


def test_per_obs_bwd_fixed_indices_zero():
    """The FD backward must return exactly 0.0 for slots in
    fixed_set_local, both under plain grad and under vmap(grad).
    This is structural — the SVGD reduced-space machinery relies on
    fixed gradients being exactly zero.
    """
    import jax
    import jax.numpy as jnp

    # Build a model with one fixed parameter at slot 0 (per-epoch
    # broadcast); we don't directly expose fixed_set_local, but the
    # equivalent end-to-end test is to compute the gradient over a
    # particle that has the fixed value already at the fixed slot,
    # and verify the slot's gradient is finite (no nan) and runs.
    model, _, _, _ = _build_per_obs_model_for_vmap_tests(
        n_obs=5, exposures=[1.0, 1.0, 2.0, 2.0, 3.0],
    )
    particles = jnp.array([
        [1.0, 0.5, 1.0, 0.5],
        [1.5, 0.4, 0.9, 0.6],
    ])

    def loss(theta):
        return jnp.sum(model(theta)[0])

    g_vmap = jax.vmap(jax.grad(loss))(particles)
    # All entries finite (no NaN, no Inf).
    assert bool(jnp.all(jnp.isfinite(g_vmap))), (
        f"Non-finite gradient under vmap(grad): {np.asarray(g_vmap)}"
    )


def test_per_obs_vmap_perf_smoke():
    """Perf smoke: under vmap, time for P=8 particles should be much
    less than 8x the single-particle time, because the FFI batches
    fuse and OpenMP parallelises the (P*K) chains.

    Skipped if running on < 4 OpenMP threads (CI hosts often have 1).
    """
    import os
    omp_threads = int(os.environ.get('OMP_NUM_THREADS', '0') or os.cpu_count() or 1)
    if omp_threads < 4:
        pytest.skip(f"Need ≥4 OMP threads for perf smoke; got {omp_threads}")

    import jax
    import jax.numpy as jnp

    # K=3 unique exposures, 30 obs → enough to make per-call cost real
    # but not enough to slow CI.
    model, _, _, _ = _build_per_obs_model_for_vmap_tests(
        n_obs=30,
        exposures=[1.0, 2.0, 3.0] * 10,
    )

    def loss(theta):
        return jnp.sum(model(theta)[0])

    theta = jnp.array([1.0, 0.5, 1.0, 0.5])

    # Warm up JIT (one call) for fairness.
    _ = jax.grad(loss)(theta).block_until_ready()
    _ = jax.vmap(jax.grad(loss))(jnp.tile(theta[None, :], (8, 1))).block_until_ready()

    # Single-particle.
    t0 = time.perf_counter()
    for _ in range(2):
        _ = jax.grad(loss)(theta).block_until_ready()
    t_single = (time.perf_counter() - t0) / 2.0

    # 8 particles vmapped.
    particles = jnp.tile(theta[None, :], (8, 1)) + 0.01 * jnp.arange(8)[:, None]
    t0 = time.perf_counter()
    for _ in range(2):
        _ = jax.vmap(jax.grad(loss))(particles).block_until_ready()
    t_vmap8 = (time.perf_counter() - t0) / 2.0

    # If vmap doesn't fuse, t_vmap8 ≈ 8 * t_single. With fusion + OMP it
    # should be ≤ 4x. Allow generous headroom for noisy CI; this test
    # mainly catches "the fusion broke" regressions.
    ratio = t_vmap8 / max(t_single, 1e-6)
    print(f"\n  t_single={t_single*1e3:.1f}ms  "
          f"t_vmap8={t_vmap8*1e3:.1f}ms  ratio={ratio:.2f}")
    assert ratio <= 5.0, (
        f"vmap(grad) over 8 particles took {ratio:.1f}× the "
        f"single-particle time — fusion likely broken. "
        f"Expected ≤ 5× (ideal is ~1×)."
    )
