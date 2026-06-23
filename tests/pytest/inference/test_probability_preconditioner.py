"""Gate 0 tests for ProbabilityJacobianPreconditioner (SVGD UI overhaul, Batch 0).

The new preconditioner builds its scaling from the model's FIRST output
(probabilities) instead of the SECOND (moments). This makes preconditioning
functional on joint-probability / daisy-chain models, whose second output is a
dummy ``jnp.zeros(2)`` that makes MomentJacobianPreconditioner degenerate to an
all-ones no-op.

These tests target the DORMANT class (Batch 0): the class exists and is accepted
as an instance, but ``preconditioner='auto'`` does NOT yet resolve to it (that is
Batch 1). So every test constructs the preconditioner directly.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic  # noqa: F401  (enables jax_enable_x64 before any jax array)
from phasic import (
    Graph, StateIndexer, Property, with_ipv,
    MomentJacobianPreconditioner, ProbabilityJacobianPreconditioner,
)
from phasic.svgd import _PreconditionerBase
import phasic.svgd as _svgdmod
from unittest import mock
import jax
import jax.numpy as jnp

_softplus = jax.nn.softplus


# --- standard (real-moments) fixture, mirrors test_preconditioning.py ----------

def _build_two_stage_graph():
    """S -> [3] --(theta0)--> [2] --(theta1)--> [1]. Real moments output."""
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])  # rate = theta0
    v2.add_edge(v1, [0.0, 1.0])  # rate = theta1
    return g


def _standard_2d_model_and_data(true_theta=(10.0, 1.0), n=200, seed=42):
    graph = _build_two_stage_graph()
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    np.random.seed(seed)
    data = (np.random.exponential(1.0 / true_theta[0], size=n)
            + np.random.exponential(1.0 / true_theta[1], size=n))
    return model, data


# --- joint-prob fixture (dummy 2nd output), mirrors test_svgd_config.py --------

def _make_joint_prob_graph(*, discrete=False):
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

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
    return base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=10, discrete=discrete,
    )


def _joint_observed_indices(jpg):
    """Valid joint-outcome vertex indices, the way Graph.svgd maps them
    (mirrors __init__.py:7269-7282)."""
    jpt = jpg.joint_prob_table()
    obs_cols = jpt.columns[:-1].to_list()
    obs2idx = jpt.groupby(obs_cols).groups
    indices = sorted({int(list(v)[0]) for v in obs2idx.values()})
    return np.array(indices, dtype=np.int64)


# --- tests ---------------------------------------------------------------------

def test_class_hierarchy():
    assert issubclass(ProbabilityJacobianPreconditioner, _PreconditionerBase)


def test_synthetic_dummy_moments_nontrivial_vs_degenerate():
    """Core property: with a dummy zeros 2nd output, ProbabilityJacobian yields a
    non-trivial multi-scale scaling while MomentJacobian degenerates to all-ones."""
    def synth_model(theta_c, data, rewards=None):
        # First output multi-scale in theta (theta1 ~1000x more sensitive);
        # second output is the dummy zeros that joint-prob models return.
        p0 = theta_c[0]
        p1 = 1000.0 * theta_c[1]
        probs = jnp.array([p0, p1, p0 + p1])
        return probs, jnp.zeros(2)

    data = np.array([1.0, 2.0, 3.0])
    theta_ref = jnp.zeros(2)

    pj = ProbabilityJacobianPreconditioner(
        model=synth_model, observed_data=data, theta_dim=2,
        param_transform=_softplus,
    )
    pj.compute_scaling(theta_ref)
    s = np.asarray(pj.scaling)
    assert s.shape == (2,)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
    assert abs(float(np.mean(s)) - 1.0) < 1e-6
    assert (s.max() / s.min()) > 10.0          # genuinely multi-scale
    assert not np.allclose(s, 1.0)

    mj = MomentJacobianPreconditioner(
        model=synth_model, observed_data=data, theta_dim=2,
        param_transform=_softplus,
    )
    mj.compute_scaling(theta_ref)
    sm = np.asarray(mj.scaling)
    # dummy moments -> zero Jacobian -> all-ones (no-op)
    assert np.allclose(sm, 1.0)


def test_real_joint_prob_scaling_is_sane():
    jpg = _make_joint_prob_graph(discrete=False)
    theta_dim = jpg.param_length()
    model = Graph.pmf_from_graph_joint_index(jpg, theta_dim=theta_dim)
    observed = _joint_observed_indices(jpg)
    assert observed.size >= 2  # need several outcomes for a meaningful Jacobian

    pj = ProbabilityJacobianPreconditioner(
        model=model, observed_data=observed, theta_dim=theta_dim,
        param_transform=_softplus,
    )
    pj.compute_scaling(jnp.zeros(theta_dim))
    s = np.asarray(pj.scaling)
    assert s.shape == (theta_dim,)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
    assert abs(float(np.mean(s)) - 1.0) < 1e-6


def test_momentjacobian_standard_unchanged_and_distinct():
    """Standard model has REAL moments: MomentJacobian still produces a sane,
    deterministic mean-1 scaling, and ProbabilityJacobian (different output) is a
    distinct, also-sane scaling -- proving the two are genuinely different code
    paths and the standard path is untouched."""
    model, data = _standard_2d_model_and_data()
    theta_ref = jnp.zeros(2)

    mj = MomentJacobianPreconditioner(
        model=model, observed_data=data, theta_dim=2, param_transform=_softplus,
    )
    mj.compute_scaling(theta_ref)
    sm = np.asarray(mj.scaling)
    assert sm.shape == (2,)
    assert np.all(np.isfinite(sm)) and np.all(sm > 0)
    assert abs(float(np.mean(sm)) - 1.0) < 1e-6

    # determinism: recompute gives an identical result
    mj2 = MomentJacobianPreconditioner(
        model=model, observed_data=data, theta_dim=2, param_transform=_softplus,
    )
    mj2.compute_scaling(theta_ref)
    assert np.allclose(sm, np.asarray(mj2.scaling))

    pj = ProbabilityJacobianPreconditioner(
        model=model, observed_data=data, theta_dim=2, param_transform=_softplus,
    )
    pj.compute_scaling(theta_ref)
    sp = np.asarray(pj.scaling)
    assert np.all(np.isfinite(sp)) and np.all(sp > 0)
    assert abs(float(np.mean(sp)) - 1.0) < 1e-6
    # Different output (pmf vs moments) -> different scaling.
    assert not np.allclose(sm, sp)


# --- Gate 1: dispatch wiring ---------------------------------------------------

def test_model_builder_precondition_tags():
    """Each builder tags which output the 'auto'/'jacobian' preconditioner reads."""
    std = Graph.pmf_and_moments_from_graph(
        _build_two_stage_graph(), nr_moments=2, discrete=False,
    )
    assert getattr(std, '_precondition_output', None) == 'moments'

    multi = Graph.pmf_and_moments_from_graph_multivariate(
        _build_two_stage_graph(), nr_moments=2, discrete=False,
    )
    assert getattr(multi, '_precondition_output', None) == 'moments'

    jpg = _make_joint_prob_graph(discrete=False)
    jm = Graph.pmf_from_graph_joint_index(jpg, theta_dim=jpg.param_length())
    assert getattr(jm, '_precondition_output', None) == 'probability'


def test_auto_dispatches_to_probability_jacobian_on_joint_model():
    """End-to-end: preconditioner='auto' on a joint-prob model resolves to
    ProbabilityJacobianPreconditioner (not the degenerate MomentJacobian)."""
    from phasic import SVGD
    jpg = _make_joint_prob_graph(discrete=False)
    k = jpg.param_length()
    model = Graph.pmf_from_graph_joint_index(jpg, theta_dim=k)
    observed = _joint_observed_indices(jpg)

    svgd = SVGD(
        model=model, observed_data=observed, theta_dim=k,
        n_particles=4, n_iterations=2, preconditioner='auto',
        seed=0, progress=False,
    )

    # Spy on the compute_scaling METHODS (not the classes) so the isinstance
    # type-checks in SVGD.optimize keep seeing real types.
    called = {'prob': False, 'moment': False}
    orig_prob = _svgdmod.ProbabilityJacobianPreconditioner.compute_scaling
    orig_moment = _svgdmod.MomentJacobianPreconditioner.compute_scaling

    def prob_spy(self, theta_ref):
        called['prob'] = True
        return orig_prob(self, theta_ref)

    def moment_spy(self, theta_ref):
        called['moment'] = True
        return orig_moment(self, theta_ref)

    with mock.patch.object(_svgdmod.ProbabilityJacobianPreconditioner,
                           'compute_scaling', prob_spy), \
         mock.patch.object(_svgdmod.MomentJacobianPreconditioner,
                           'compute_scaling', moment_spy):
        svgd.optimize()

    assert called['prob'], (
        "auto should resolve to ProbabilityJacobianPreconditioner on a joint model")
    assert not called['moment'], (
        "MomentJacobianPreconditioner must not be used on a joint model")
