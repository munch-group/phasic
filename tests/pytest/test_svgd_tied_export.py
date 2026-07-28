"""F-010: tied slave parameters must be resolved in EVERY exported surface.

`_daisy_chain_svgd_model` locks each tied *slave* flat index at a 0.0 sentinel
and copies master->slave only INSIDE the forward pass (a scatter on a copy that
is never persisted). So before the fix, `SVGD.particles` / `theta_mean` /
`history` kept 0.0 at the slave columns, and `softplus(0) = ln 2 = 0.693`
surfaced as the slave's rate -- ~8000x wrong for a ~1e-4 coalescent rate. Only
`summary()` resolved the slave (via slave_to_master); every other surface read
the sentinel.

The fix re-ties master->slave once, at the single result-finalization point in
`SVGD.optimize()`, so particles / theta_mean / theta_std / history all agree.
These tests pin that: the slave column must EQUAL the master column (not merely
be finite/positive), and must NOT be the 0.693 sentinel image.

Provenance: audit category-A fix A1 (see category-A-fix-plans.md).
"""
from __future__ import annotations

from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

pytest.importorskip("jax")

import phasic
from phasic import (
    Graph, LogGaussPrior, GaussPrior, BetaPrior, Adamelia, with_ipv, StateIndexer,
    Property, set_log_level,
)

_SENTINEL_IMAGE = np.log(2.0)  # softplus(0) -- the wrong value the bug produced


def _fit_tied(prior=None):
    """Short tied daisy-chain SVGD fit on a 1-param coalescent joint-prob model.

    Deterministic (seed 17). tied=[(0, [0, 1])] ties local param 0 across the two
    epochs, so the epoch-1 flat position is a slave of the epoch-0 master. The fit
    is 3 iterations x 8 particles -- enough to exercise the export path, not to
    converge (the fix is independent of convergence).

    ``prior`` overrides the default (a single broadcast LogGaussPrior, i.e. a
    homogeneous softplus transform across dims). Pass a per-dim list to exercise
    heterogeneous transforms across a tie.
    """
    set_log_level("WARNING")
    np.random.seed(17)
    all_pairs = partial(combinations_with_replacement, r=2)

    nr_samples = 4
    indexer = StateIndexer(
        lineages=[Property("descendants", min_value=1, max_value=nr_samples)]
    )

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state, indexer=None):
        transitions = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair_count = state[i] * (state[j] - same) / (1 + same)
            transitions.append([new, [pair_count]])
        return transitions

    graph = Graph(coalescent_1param, indexer=indexer)
    reward_limit = 5
    mutation_rate = 1e-4
    theta = 1 / 10_000

    jpg_disc = graph.joint_prob_graph(
        indexer, reward_limit=reward_limit, mutation_rate=mutation_rate
    )
    jpg_disc.update_weights([theta, mutation_rate])
    jpt = jpg_disc.joint_prob_table()
    p = jpt["prob"] / jpt["prob"].sum()
    sample = np.random.choice(jpt.index.values, 1000, p=p.to_numpy())
    observations = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()

    jpg_cont = graph.joint_prob_graph(
        indexer, reward_limit=reward_limit, mutation_rate=mutation_rate, discrete=False
    )
    svgd = jpg_cont.svgd(
        observations,
        fixed=[(1, mutation_rate)],
        tied=[(0, [0, 1])],
        prior=(LogGaussPrior(ci=[1 / 50_000, 1 / 5000]) if prior is None else prior),
        n_iterations=3,
        n_particles=8,
        optimizer=Adamelia(learning_rate=0.2),
        epoch_starts=[0, 0.5],
        return_history=True,
    )
    return svgd


@pytest.fixture(scope="module")
def tied_fit():
    return _fit_tied()


@pytest.fixture(scope="module")
def tied_fit_mixed_transform():
    # Master (flat 0) uses a BetaPrior -> sigmoid natural transform; the masked
    # slave (flat 2) defaults to softplus. An UNCONSTRAINED copy would export
    # softplus(phi_master) != sigmoid(phi_master) at the slave -- the F-010
    # heterogeneous-transform defect (adversarial review). The constrained-space
    # re-tie in get_results() must make the reported slave rate equal the master.
    return _fit_tied(prior=[
        BetaPrior(ci=[0.2, 0.85]),    # flat 0 = master (sigmoid)
        GaussPrior(ci=[1e-6, 1e-2]),  # flat 1 = fixed
        GaussPrior(ci=[1e-6, 1e-2]),  # flat 2 = slave  (softplus)
        GaussPrior(ci=[1e-6, 1e-2]),  # flat 3 = fixed
    ])


def test_tying_is_actually_active(tied_fit):
    # Guard against a vacuous test: the model must really have a tied slave, and
    # the slave and master must be DIFFERENT flat positions (else "slave==master"
    # is trivially true).
    s2m = tied_fit.model._tying_info["slave_to_master"]
    assert s2m, "no tied slaves -- test would be vacuous"
    for slave, master in s2m.items():
        assert slave != master


def test_particles_slave_equals_master(tied_fit):
    # The raw (unconstrained) particle tensor is what every downstream surface is
    # derived from. Each slave column must EQUAL its master column exactly.
    s2m = tied_fit.model._tying_info["slave_to_master"]
    P = np.asarray(tied_fit.particles)
    for slave, master in s2m.items():
        np.testing.assert_array_equal(P[:, slave], P[:, master])


def test_theta_mean_slave_equals_master_and_is_not_the_sentinel(tied_fit):
    # get_results() transforms to constrained (rate) space. The slave rate must
    # equal the master rate -- and must NOT be softplus(0)=0.693, the specific
    # wrong value the 0.0 sentinel produced.
    s2m = tied_fit.model._tying_info["slave_to_master"]
    res = tied_fit.get_results()
    tm = np.asarray(res["theta_mean"])
    for slave, master in s2m.items():
        assert tm[slave] == pytest.approx(tm[master], rel=1e-12)
        assert abs(tm[slave] - _SENTINEL_IMAGE) > 1e-3, (
            f"slave theta_mean {tm[slave]} is the softplus(0) sentinel image -- "
            "the tie was not applied to the export"
        )
    # And the raw-attribute path (self.theta_mean, unconstrained) must agree too.
    raw = np.asarray(tied_fit.theta_mean)
    for slave, master in s2m.items():
        assert raw[slave] == pytest.approx(raw[master], rel=1e-12)


def test_history_slave_equals_master(tied_fit):
    # History drives trace plots / diagnostics; its slave columns must be tied at
    # every recorded step, not just the final particles.
    s2m = tied_fit.model._tying_info["slave_to_master"]
    res = tied_fit.get_results()
    H = res.get("history")
    assert H is not None, "return_history=True should have produced a history"
    H = np.asarray(H)  # (n_steps, n_particles, theta_dim)
    for slave, master in s2m.items():
        np.testing.assert_array_equal(H[..., slave], H[..., master])


def test_degrees_of_freedom_unchanged_by_the_fix(tied_fit):
    # The re-tie is a pure export step: it must not change the model's DOF. Two
    # epochs x two params, minus the two fixed (idx 1 in both epochs), minus the
    # one tied slave = 1 free parameter.
    assert int(tied_fit.degrees_of_freedom) == 1


def test_mixed_transform_slave_equals_master_in_constrained_space(tied_fit_mixed_transform):
    # F-010 heterogeneous-transform case (adversarial review). get_results() is
    # the documented constrained-space API, and the model tied in constrained
    # space, so the slave's reported rate MUST equal the master's even though
    # their per-dim transforms differ. Before the constrained-space re-tie this
    # gave 0.878 (softplus) vs 0.584 (sigmoid) -- a silent ~50% disagreement.
    svgd = tied_fit_mixed_transform
    s2m = svgd.model._tying_info["slave_to_master"]
    assert s2m
    res = svgd.get_results()
    tm = np.asarray(res["theta_mean"])
    for slave, master in s2m.items():
        assert tm[slave] == pytest.approx(tm[master], rel=1e-12), (
            f"constrained slave {tm[slave]} != master {tm[master]} -- the tie did "
            "not survive differing per-dim transforms"
        )
    # particles and history must be tied in constrained space too.
    Pc = np.asarray(res["particles"])
    for slave, master in s2m.items():
        np.testing.assert_array_equal(Pc[:, slave], Pc[:, master])
    H = res.get("history")
    if H is not None:
        H = np.asarray(H)
        for slave, master in s2m.items():
            np.testing.assert_array_equal(H[..., slave], H[..., master])


def test_mixed_transform_map_estimate_ties_slave_in_constrained_space(tied_fit_mixed_transform):
    # map_estimate_from_particles() is another constrained-space export; it must
    # also report the master's value at the slave under differing transforms.
    svgd = tied_fit_mixed_transform
    s2m = svgd.model._tying_info["slave_to_master"]
    theta_map, _ = svgd.map_estimate_from_particles()  # constrained
    theta_map = np.asarray(theta_map)
    for slave, master in s2m.items():
        assert theta_map[slave] == pytest.approx(theta_map[master], rel=1e-12)


def test_summary_still_marks_the_slave_as_tied(tied_fit, capsys):
    # summary() already resolved the slave before the fix; it must keep doing so
    # (and keep labelling the row as tied), so the two paths stay consistent.
    ret = tied_fit.summary()
    captured = capsys.readouterr()
    text = (ret if isinstance(ret, str) else "") + captured.out
    assert "Tied" in text, "summary() no longer marks the tied slave row"
