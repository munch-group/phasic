"""Equivalence gate for the daisy-chained joint-probability SVGD path.

This protects the highest-value inference path the Stage-3 refactor touches, and
the one most at risk from three converging changes: the FFI-orchestration
unification (#7), the trace SSOT (Q1), and the finite-difference-gradient removal
(Q-CODEGEN). The path is:

    _daisy_chain_svgd_model / Graph.daisy_chain_joint_probs
        -> compute_daisy_chain_joint_probs_ffi  (graph_builder_ffi.cpp:1240)

whose forward is a **multi-epoch, sequential** computation — it steps the CTMC
incrementally epoch by epoch (`for epoch in 0..n_epochs-1`: update_ipv →
update_weights(theta_epoch) → stop_probability(dt) → project the surviving
distribution into the next epoch's IPV) and reads out joint probabilities at
time t only in the final epoch. Epoch N depends on epoch N-1's propagated state,
so the whole chain must stay coupled and JAX-differentiable end to end.

The gradient is currently produced by a finite-difference `custom_vjp`
(`__init__.py:11293`). Q-CODEGEN mandates replacing every finite-difference
gradient with an analytical one — for this path that means an adjoint that
backprops through the sequential epoch chain. This gate pins the current
multi-epoch **value** (tight) and **gradient through all epochs** (loose, so it
survives the finite-diff -> analytical transition while still catching a break),
so nothing may:

  * change the incremental forward computation (value golden), or
  * disconnect an epoch from the gradient (per-epoch block-norm check) — e.g. by
    flattening the epoch loop when unifying FFI orchestration, or by routing
    through a single-shot trace that assumes a static graph.

Goldens captured under x64 (the FFI path is f64); comparisons are tolerance-based
to stay robust to the ambient f32/f64 state (see test_gate_svgd_seams.py).
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, Property, StateIndexer, with_ipv

pytestmark = pytest.mark.equivalence

_N_SAMPLES = 3
_MUTATION_RATE = 0.1
_REWARD_LIMIT = 3


def _build_jsp_graph():
    """A small continuous coalescent joint-stop-prob graph (2-epoch daisy chain).
    Self-contained replica of the fixture in test_daisy_chain_c_path.py."""
    indexer = StateIndexer(
        lineages=[Property("descendants", min_value=1, max_value=_N_SAMPLES)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = _N_SAMPLES

    @with_ipv(ipv)
    def coal_callback(state, indexer=None):
        transitions = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = list(state)
            new[i] -= 1
            new[j] -= 1
            sum_idx = indexer.lineages.props_to_index(descendants=(i + 1) + (j + 1))
            new[sum_idx] += 1
            pair_count = state[i] * (state[j] - same) // (1 + same)
            transitions.append((new, [pair_count, 0]))
        return transitions

    g = Graph(coal_callback, indexer=indexer)
    jp = g.joint_prob_graph(
        indexer, mutation_rate=_MUTATION_RATE, reward_limit=_REWARD_LIMIT, discrete=False
    )
    return jp.joint_stop_prob_graph()


# 2 epochs x 3 params; epoch_dts has length n_epochs-1 = 1; t_eval fixed.
_THETA = [1.0, 1.0, 1.0, 1.5, 0.5, 1.0]
_EPOCH_DTS = [0.5]
_T_EVAL = 10.0

# --- goldens (captured under x64) ---
_PROBS_GOLDEN = [
    0.07378179487815369, 0.07884994511513092, 0.05482048003663633, 0.07931338425847392,
    0.05702717871863512, 0.05356640541796783, 0.07935173078234171, 0.05719170311375713,
    0.055520559698256276, 0.05349025530698083, 0.05720376119934417, 0.05565221481097725,
    0.05542490776242316, 0.05566078025078512, 0.055553475985514356, 0.05556165422815141,
]
_SUM_GOLDEN = 0.9779702315635292
_GRAD_GOLDEN = [0.0084950017853469, 0.0, -0.01125423917253121,
                0.006977265085139628, 0.0, -0.010465898217515424]


@pytest.fixture(scope="module")
def jsp():
    return _build_jsp_graph()


def _model(jsp, theta_flat):
    import jax.numpy as jnp

    n_ipv = jsp.starting_vertex().edges_length()
    return jsp.daisy_chain_joint_probs(
        epoch_thetas=jnp.asarray(theta_flat).reshape(2, 3),
        epoch_dts=_EPOCH_DTS,
        initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
        t_eval=_T_EVAL,
    )


def test_daisy_chain_joint_prob_value_golden(jsp):
    """The multi-epoch incremental forward computation must be preserved. Tight
    tolerance — the forward is deterministic; any change to the epoch stepping
    (or the FFI orchestration unify / trace-SSOT reroute) that alters the value
    turns this red."""
    import jax.numpy as jnp

    probs = np.asarray(_model(jsp, jnp.asarray(_THETA, dtype=jnp.float64)))
    assert probs.shape == (len(_PROBS_GOLDEN),)
    assert np.all(np.isfinite(probs))
    assert list(probs) == pytest.approx(_PROBS_GOLDEN, rel=1e-5, abs=1e-9)
    assert float(np.sum(probs)) == pytest.approx(_SUM_GOLDEN, rel=1e-5, abs=1e-9)


def test_daisy_chain_gradient_flows_through_every_epoch(jsp):
    """jax.grad must flow through the full epoch chain. The block-norm assertions
    are the load-bearing structural guard: if any refactor disconnects an epoch
    (e.g. flattens the epoch loop or drops the inter-epoch IPV coupling), that
    epoch's gradient block collapses to ~0 and this fails. Value tolerance is
    loose so it survives the mandated finite-diff -> analytical gradient swap
    (both approximate the same true gradient) while still catching a real break
    (NaN / sign flip / disconnected epoch)."""
    import jax
    import jax.numpy as jnp

    def scalar(tf):
        return jnp.sum(_model(jsp, tf))

    g = np.asarray(jax.grad(scalar)(jnp.asarray(_THETA, dtype=jnp.float64)))
    assert g.shape == (6,)
    assert np.all(np.isfinite(g)), "gradient must be finite through the epoch chain"
    epoch0, epoch1 = np.linalg.norm(g[:3]), np.linalg.norm(g[3:])
    assert epoch0 > 1e-4, f"epoch-0 gradient block collapsed ({epoch0}) — chain disconnected"
    assert epoch1 > 1e-4, f"epoch-1 gradient block collapsed ({epoch1}) — chain disconnected"
    # loose golden match (finite differences ~ analytical to a few %)
    assert list(g) == pytest.approx(_GRAD_GOLDEN, rel=5e-2, abs=1e-4)


def test_daisy_chain_jit_matches_eager(jsp):
    """The JAX-traceable path stays consistent under jit (the SVGD path is
    always jitted). Same-mode comparison, so exact to fp round-off."""
    import jax
    import jax.numpy as jnp

    theta = jnp.asarray(_THETA, dtype=jnp.float64)
    eager = np.asarray(_model(jsp, theta))
    jitted = np.asarray(jax.jit(lambda tf: _model(jsp, tf))(theta))
    np.testing.assert_allclose(jitted, eager, rtol=1e-9, atol=1e-12)
