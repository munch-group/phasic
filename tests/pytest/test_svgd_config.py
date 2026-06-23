"""Unit tests for ``phasic.svgd_config`` rules R1..R27.

Each rule has a dedicated test that constructs the smallest possible
config triggering the rule and asserts the right exception (or warning)
fires with a recognisable message substring.

For graph-aware rules (R1..R5, R9, R12, R13) we build a tiny coalescent
model so ``from_svgd_call`` has a real graph to introspect.
"""
from __future__ import annotations

import warnings
from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, StateIndexer, Property, with_ipv
from phasic.exceptions import SvgdConfigError
from phasic.svgd_config import (
    SvgdConfig, from_svgd_call, from_model_call, validate,
)


# ---------------------------------------------------------------------------
# Tiny model fixtures (N=2 coalescent: smallest non-trivial parameterised
# graph). Reused across rule tests where a real graph is needed.
# ---------------------------------------------------------------------------


def _make_n2_indexer() -> StateIndexer:
    return StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )


def _make_n2_base_graph() -> Graph:
    indexer = _make_n2_indexer()
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

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

    g = Graph(cb, indexer=indexer)
    return g


def _make_joint_prob_graph(*, discrete: bool = False) -> Graph:
    base = _make_n2_base_graph()
    return base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=10, discrete=discrete,
    )


# ---------------------------------------------------------------------------
# Rule tests — one per rule.
# ---------------------------------------------------------------------------


class TestR1_EpochRequiresContinuousJointProb:
    """epoch_starts requires a joint-probability graph in continuous mode."""

    def test_rejects_standard_graph(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"joint-probability graph"):
            validate(from_svgd_call(g, [1.0, 2.0], epoch_starts=[0.0, 1.0]))

    def test_rejects_discrete_joint_prob(self):
        g = _make_joint_prob_graph(discrete=True)
        with pytest.raises(SvgdConfigError, match=r"continuous-time joint-prob"):
            validate(from_svgd_call(g, [(0, 0)], epoch_starts=[0.0, 1.0]))

    def test_accepts_continuous_joint_prob(self):
        g = _make_joint_prob_graph(discrete=False)
        validate(from_svgd_call(g, [(0, 0)], epoch_starts=[0.0, 1.0]))


class TestR2_JointIndexRequiresJointProb:
    def test_rejects_standard_graph_with_joint_index(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"joint_index=True requires"):
            validate(from_svgd_call(g, [1.0], joint_index=True))


class TestR3_RewardsIncompatibleWithJointProb:
    def test_rejects_rewards_on_joint_prob(self):
        g = _make_joint_prob_graph()
        rewards = np.ones(g.vertices_length())
        with pytest.raises(SvgdConfigError, match=r"Reward transformation is not supported"):
            validate(from_svgd_call(g, [(0, 0)], rewards=rewards))

    def test_accepts_rewards_on_standard_graph(self):
        g = _make_n2_base_graph()
        rewards = np.ones(g.vertices_length())
        validate(from_svgd_call(g, [1.0], rewards=rewards))


class TestR4_RegularizationIncompatibleWithJointProb:
    def test_rejects_regularization_on_joint_prob(self):
        g = _make_joint_prob_graph()
        with pytest.raises(SvgdConfigError, match=r"Moment regularization"):
            validate(from_svgd_call(g, [(0, 0)], regularization=0.5))

    def test_accepts_zero_regularization(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(g, [(0, 0)], regularization=0.0))


class TestR5_SparseObservationsRequireRewards:
    def test_rejects_sparse_without_rewards(self):
        from phasic.svgd import SparseObservations
        sparse = SparseObservations(
            values=np.array([1.0, 2.0]),
            features=np.array([0, 0]),
            n_features=1,
            slices=((0, 1), (1, 2)),
        )
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"SparseObservations require explicit rewards"):
            validate(from_svgd_call(g, sparse, rewards=None))


class TestR6_SparseObservationsIncompatibleWithExposure:
    def test_rejects_sparse_with_exposure(self):
        from phasic.svgd import SparseObservations
        sparse = SparseObservations(
            values=np.array([1.0, 2.0]),
            features=np.array([0, 0]),
            n_features=1,
            slices=((0, 1), (1, 2)),
        )
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"exposure is not supported with SparseObservations"):
            validate(from_svgd_call(
                g, sparse, rewards=np.ones((1, g.vertices_length())),
                exposure=[1.0, 1.0], exposure_param_index=0,
            ))


class TestR7_ExposureAndParamIndexPaired:
    def test_exposure_without_param_index(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"exposure_param_index must be provided"):
            validate(from_svgd_call(g, [1.0, 2.0], exposure=[1.0, 1.0]))

    def test_param_index_without_exposure(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"exposure is None"):
            validate(from_svgd_call(g, [1.0, 2.0], exposure_param_index=0))


class TestR8_ExposureParamIndexInParamLengthRange:
    def test_param_index_too_large(self):
        g = _make_n2_base_graph()
        # Standard graph: param_length=1. exposure_param_index=5 is out of range.
        # Note R7 paired-check fires first if exposure is missing.
        with pytest.raises(SvgdConfigError, match=r"exposure_param_index=5 is out of range"):
            validate(from_svgd_call(
                g, [1.0, 2.0], exposure=[1.0, 1.0], exposure_param_index=5,
            ))

    def test_param_index_negative(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"out of range"):
            validate(from_svgd_call(
                g, [1.0, 2.0], exposure=[1.0, 1.0], exposure_param_index=-1,
            ))


class TestR9_ExposureWithVanillaJointProbUnsupported:
    def test_rejects_exposure_without_epoch_starts(self):
        g = _make_joint_prob_graph()
        # joint-prob graph param_length is 2 (coal_rate + mu);
        # exposure_param_index=1 in range.
        with pytest.raises(SvgdConfigError, match=r"vanilla joint-prob"):
            validate(from_svgd_call(
                g, [(0, 0), (0, 0)],
                exposure=[1.0, 1.0], exposure_param_index=1,
            ))

    def test_accepts_exposure_with_epoch_starts(self):
        g = _make_joint_prob_graph()
        # Should pass (R9 only fires when epoch_starts is None).
        validate(from_svgd_call(
            g, [(0, 0), (0, 0)],
            epoch_starts=[0.0, 1.0],
            exposure=[1.0, 1.0], exposure_param_index=1,
        ))


class TestR10_ExposureLengthMatchesNObservations:
    def test_rejects_length_mismatch(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"exposure length"):
            validate(from_svgd_call(
                g, [1.0, 2.0, 3.0],
                exposure=[1.0, 2.0],  # length 2 vs 3 obs
                exposure_param_index=0,
            ))

    def test_accepts_scalar_broadcast(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0, 3.0], exposure=1.5, exposure_param_index=0,
        ))

    def test_accepts_matching_length(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0, 3.0],
            exposure=[1.0, 2.0, 3.0], exposure_param_index=0,
        ))


class TestR11_ExposureWith2DRewardsWarns:
    def test_warns_for_2d_rewards(self):
        g = _make_n2_base_graph()
        rewards = np.ones((2, g.vertices_length()))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate(from_svgd_call(
                g, [[1.0, 1.0], [2.0, 2.0]],
                rewards=rewards,
                exposure=[1.0, 1.0], exposure_param_index=0,
            ))
        assert any('multivariate' in str(ww.message) for ww in w)


class TestR12_ParamTransformIncompatibleWithJointProb:
    def test_rejects_param_transform_on_joint_prob(self):
        g = _make_joint_prob_graph()
        with pytest.raises(SvgdConfigError, match=r"param_transform is not supported"):
            validate(from_svgd_call(
                g, [(0, 0)],
                param_transform=lambda x: x,
                positive_params=False,  # avoid R15 firing first
            ))


class TestR13_FisherPreconditionerWithJointProbWarns:
    def test_warns_for_fisher(self):
        # 'fisher' divides the score by the joint probability -> unstable when
        # probabilities are small; R13 still flags this specific variant.
        g = _make_joint_prob_graph()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate(from_svgd_call(
                g, [(0, 0)],
                preconditioner='fisher',
            ))
        assert any('fisher' in str(ww.message).lower() for ww in w)

    def test_does_not_warn_for_auto_or_jacobian(self):
        # The Jacobian preconditioners are now functional on joint-prob (they
        # precondition on the probability output), so they must NOT warn.
        g = _make_joint_prob_graph()
        for choice in ('auto', 'jacobian'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                validate(from_svgd_call(g, [(0, 0)], preconditioner=choice))
            assert not any('preconditioner' in str(ww.message).lower()
                           or 'fisher' in str(ww.message).lower() for ww in w), (
                f"preconditioner={choice!r} should no longer warn on joint-prob")


class TestR14_FixedWithEpochStartsLocalIndices:
    def test_rejects_out_of_range_local_index(self):
        g = _make_joint_prob_graph()
        # joint-prob param_length is 2, so local index 5 is out of range.
        with pytest.raises(SvgdConfigError, match=r"local.*per-epoch indices"):
            validate(from_svgd_call(
                g, [(0, 0), (0, 0)],
                epoch_starts=[0.0, 1.0],
                fixed=[(5, 0.5)],
            ))

    def test_accepts_in_range_local_index(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0), (0, 0)],
            epoch_starts=[0.0, 1.0],
            fixed=[(1, 1e-8)],
        ))


class TestR15_PositiveParamsXorParamTransform:
    def test_rejects_both(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"mutually exclusive"):
            validate(from_svgd_call(
                g, [1.0, 2.0],
                positive_params=True,
                param_transform=lambda x: x,
            ))

    def test_accepts_param_transform_alone(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0],
            positive_params=False,
            param_transform=lambda x: x,
        ))


class TestR16_TiedRequiresEpochStarts:
    def test_rejects_tied_without_epoch_starts(self):
        g = _make_joint_prob_graph()
        with pytest.raises(SvgdConfigError, match=r"tied.*only meaningful"):
            validate(from_svgd_call(
                g, [(0, 0), (1, 0)],
                tied=[(0, [0, 1])],
            ))

    def test_accepts_tied_with_epoch_starts(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0)] * 2,
            epoch_starts=[0.0, 1.0],
            tied=[(0, [0, 1])],
        ))


class TestR17_TiedEntryShape:
    def test_rejects_single_epoch_tie(self):
        g = _make_joint_prob_graph()
        # 1-element list is a no-op; surface as configuration error.
        with pytest.raises(SvgdConfigError, match=r"at least 2 epochs"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                tied=[(0, [0])],
            ))

    def test_accepts_two_epoch_tie(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0)] * 2,
            epoch_starts=[0.0, 1.0],
            tied=[(0, [0, 1])],
        ))


class TestR18_TiedRanges:
    def test_rejects_out_of_range_local_idx(self):
        g = _make_joint_prob_graph()
        # joint-prob param_length=2; local_idx=5 is out of range.
        with pytest.raises(SvgdConfigError, match=r"local_idx=5 out of range"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                tied=[(5, [0, 1])],
            ))

    def test_rejects_out_of_range_epoch(self):
        g = _make_joint_prob_graph()
        # epoch_starts has 2 epochs; epoch index 5 is out of range.
        with pytest.raises(SvgdConfigError, match=r"epoch indices.*out of range"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                tied=[(0, [0, 5])],
            ))


class TestR19_TiedNoDuplicateEpochs:
    def test_rejects_duplicate_epochs(self):
        g = _make_joint_prob_graph()
        with pytest.raises(SvgdConfigError, match=r"same epoch more than once"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                tied=[(0, [0, 1, 1])],
            ))

    def test_accepts_distinct_epochs(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0)] * 3,
            epoch_starts=[0.0, 1.0, 2.0],
            tied=[(0, [0, 1, 2])],
        ))


class TestR20_TiedNoOverlap:
    def test_rejects_overlap_with_fixed(self):
        g = _make_joint_prob_graph()
        # Under daisy-chain, fixed=[(0, value)] pins local_idx=0 in
        # every epoch. tied=[(0, [0, 1])] also claims those positions.
        with pytest.raises(SvgdConfigError, match=r"overlap fixed positions"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                fixed=[(0, 0.5)],
                tied=[(0, [0, 1])],
            ))

    def test_rejects_overlap_between_tied_entries(self):
        g = _make_joint_prob_graph()
        # Two tied entries that both claim local_idx=0, epoch=1.
        with pytest.raises(SvgdConfigError, match=r"claimed by two tied entries"):
            validate(from_svgd_call(
                g, [(0, 0)] * 3,
                epoch_starts=[0.0, 1.0, 2.0],
                tied=[(0, [0, 1]), (0, [1, 2])],
            ))

    def test_accepts_disjoint_tied_and_fixed(self):
        g = _make_joint_prob_graph()
        # local_idx=0 tied across epochs; local_idx=1 fixed (no overlap).
        validate(from_svgd_call(
            g, [(0, 0)] * 2,
            epoch_starts=[0.0, 1.0],
            fixed=[(1, 1e-8)],
            tied=[(0, [0, 1])],
        ))


class TestR21_CallbackIncompatibleWithEpochStarts:
    """callback= is silently ignored by the daisy-chain FFI handler
    (the C++ GraphBuilder only knows linear/log weight_mode), so the
    validator must reject the combination loudly."""

    def test_rejects_callback_with_epoch_starts(self):
        g = _make_joint_prob_graph()
        with pytest.raises(SvgdConfigError, match=r"callback.*not supported.*epoch_starts"):
            validate(from_svgd_call(
                g, [(0, 0)] * 2,
                epoch_starts=[0.0, 1.0],
                callback=lambda theta, coeffs: float(coeffs[0]) * theta[0],
            ))

    def test_accepts_callback_without_epoch_starts(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0, 3.0],
            callback=lambda theta, coeffs: float(coeffs[0]) * theta[0],
        ))


# ---------------------------------------------------------------------------
# Positive smoke tests — valid combinations should validate cleanly.
# ---------------------------------------------------------------------------


class TestValidCombinations:
    def test_standard_graph_1d_times(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(g, [1.0, 2.0, 3.0]))

    def test_standard_graph_with_exposure(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0, 3.0],
            exposure=[100.0, 200.0, 150.0], exposure_param_index=0,
        ))

    def test_joint_prob_outcomes(self):
        g = _make_joint_prob_graph()
        # joint-prob param_length=2 (coal + mu); fixed mu slot.
        validate(from_svgd_call(
            g, [(0, 0), (1, 0)],
            fixed=[(1, 1e-8)],
        ))

    def test_daisy_chain_joint_prob_with_exposure(self):
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0)] * 3,
            epoch_starts=[0.0, 1.0, 2.0],
            exposure=[100.0, 200.0, 150.0],
            exposure_param_index=1,
            fixed=[(1, 1e-8)],
        ))


# ---------------------------------------------------------------------------
# R23..R27 — added by the SVGD UI overhaul (fixed-range non-daisy, preconditioner
# value validity, optimizer/learning_rate/regularization, nr_moments on joint).
# ---------------------------------------------------------------------------


class _DummyOptimizer:
    """Stand-in for an optimizer instance (validator only checks is-not-None)."""
    pass


class TestR23_FixedIndicesInParamLengthRange:
    def test_rejects_out_of_range_on_standard_graph(self):
        g = _make_n2_base_graph()  # param_length == 1
        with pytest.raises(SvgdConfigError, match=r"out-of-range parameter indices"):
            validate(from_svgd_call(g, [1.0, 2.0], fixed=[(5, 0.0)]))

    def test_accepts_in_range(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(g, [1.0, 2.0], fixed=[(0, 0.0)]))

    def test_from_model_call_variant(self):
        with pytest.raises(SvgdConfigError, match=r"out-of-range parameter indices"):
            validate(from_model_call(
                object(), [1.0, 2.0], theta_dim=2, fixed=[(9, 0.0)]))

    def test_daisy_path_deferred_to_R14(self):
        # Under epoch_starts, R23 no-ops (R14 owns the per-epoch local check).
        g = _make_joint_prob_graph()
        validate(from_svgd_call(
            g, [(0, 0), (0, 0)],
            epoch_starts=[0.0, 1.0], fixed=[(0, 0.0)]))


class TestR24_PreconditionerValueValid:
    def test_rejects_unknown_string(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"preconditioner must be"):
            validate(from_svgd_call(g, [1.0, 2.0], preconditioner='jacobean'))

    @pytest.mark.parametrize("choice", ['auto', 'jacobian', 'fisher', 'none', None])
    def test_accepts_valid(self, choice):
        g = _make_n2_base_graph()
        validate(from_svgd_call(g, [1.0, 2.0], preconditioner=choice))


class TestR25_OptimizerXorLearningRate:
    def test_rejects_both(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"mutually exclusive"):
            validate(from_svgd_call(
                g, [1.0, 2.0],
                optimizer=_DummyOptimizer(), learning_rate=0.01))

    def test_accepts_optimizer_alone(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(g, [1.0, 2.0], optimizer=_DummyOptimizer()))

    def test_accepts_learning_rate_alone(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(g, [1.0, 2.0], learning_rate=0.01))


class TestR26_OptimizerExcludesRegularization:
    def test_rejects_optimizer_with_regularization(self):
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"requires regularization=0"):
            validate(from_svgd_call(
                g, [1.0, 2.0],
                optimizer=_DummyOptimizer(), regularization=1.0))

    def test_accepts_optimizer_with_zero_regularization(self):
        g = _make_n2_base_graph()
        validate(from_svgd_call(
            g, [1.0, 2.0], optimizer=_DummyOptimizer(), regularization=0.0))


class TestR27_NrMomentsIgnoredOnJointProb:
    def test_warns_on_non_default_nr_moments(self):
        g = _make_joint_prob_graph()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate(from_svgd_call(g, [(0, 0)], nr_moments=3))
        assert any('nr_moments' in str(ww.message) for ww in w)

    def test_no_warn_at_default(self):
        g = _make_joint_prob_graph()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validate(from_svgd_call(g, [(0, 0)], nr_moments=2))
        assert not any('nr_moments' in str(ww.message) for ww in w)


class TestEndToEndFailFast:
    def test_optimizer_plus_learning_rate_raises_before_fit(self):
        # The validator runs at the top of Graph.svgd, so an invalid combo
        # raises SvgdConfigError before any model/particle construction.
        from phasic import Adam
        g = _make_n2_base_graph()
        with pytest.raises(SvgdConfigError, match=r"mutually exclusive"):
            g.svgd([1.0, 2.0], theta_dim=1,
                   optimizer=Adam(learning_rate=0.01), learning_rate=0.01)
