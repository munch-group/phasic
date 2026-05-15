"""Centralized validator for ``Graph.svgd(...)`` argument combinations.

This module exists because ``Graph.svgd`` accepts a large product of
options (graph kind × continuous/discrete × observation shape × rewards
× epochs × fixed × exposure × …) and many combinations are either
silently broken (hang, wrong gradient) or fail deep inside model
construction with an unhelpful error.

The validator enumerates the rules in one place so that:

- A new rule is added once, in this module, with one named function.
- Each rule has a unit test that targets it specifically.
- ``Graph.svgd`` calls ``validate(from_svgd_call(...))`` once before
  any model is constructed; the existing scattered checks downstream
  remain as defensive sanity checks but rarely fire.

Public API:

- :class:`SvgdConfig` — frozen dataclass capturing the 9 constraint
  axes from a ``Graph.svgd`` call.
- :func:`from_svgd_call` — adapter that builds an ``SvgdConfig`` from
  the live ``graph`` object plus the keyword arguments to ``svgd``.
- :func:`validate` — runs every rule in order; raises
  :class:`phasic.exceptions.SvgdConfigError` on the first violation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np

from .exceptions import SvgdConfigError

GraphKind = Literal['standard', 'joint_prob', 'joint_stop_prob']
ObservationKind = Literal[
    '1d_times', '2d_times', 'sparse',
    'joint_outcomes', 'joint_indices',
]
RewardsKind = Literal['none', '1d', '2d']


@dataclass(frozen=True)
class SvgdConfig:
    """Frozen snapshot of an SVGD call's argument combination.

    Built by :func:`from_svgd_call` from the live graph plus the
    keyword arguments to ``Graph.svgd``. Consumed by :func:`validate`.

    Every field is the *user-visible* value (no auto-promotion), so the
    rules can flag misconfigurations the user wrote, not the values the
    library would silently coerce them to.
    """
    graph_kind: GraphKind
    is_discrete: bool
    param_length: int
    observation_kind: ObservationKind
    n_observations: int
    rewards_kind: RewardsKind
    has_epoch_starts: bool
    n_epochs: Optional[int]
    has_fixed: bool
    fixed_indices: Optional[Sequence[int]]
    has_exposure: bool
    exposure_length: Optional[int]
    exposure_param_index: Optional[int]
    has_param_transform: bool
    positive_params: bool
    has_preconditioner: bool
    has_regularization: bool
    nr_moments: int
    joint_index_explicit: bool


def _classify_graph_kind(graph: Any) -> GraphKind:
    if getattr(graph, '_joint_stop_prob_graph', False):
        return 'joint_stop_prob'
    if getattr(graph, '_joint_prob_base_graph_indexer', None) is not None:
        return 'joint_prob'
    return 'standard'


def _classify_observation_kind(
    observed_data: Any,
    graph_kind: GraphKind,
) -> Tuple[ObservationKind, int]:
    """Return (observation_kind, n_observations) without mutating data."""
    # Defer the import to avoid a hard dependency at module import time.
    from .svgd import is_sparse_observations

    if is_sparse_observations(observed_data):
        # Number of distinct observation rows = number of slices.
        slices = getattr(observed_data, 'slices', None)
        n_obs = len(slices) if slices is not None else 0
        return 'sparse', int(n_obs)

    # Joint-prob path takes a list of outcome tuples or vertex indices.
    if graph_kind == 'joint_prob':
        # Distinguish "list of tuples" vs "list of ints".
        try:
            length = len(observed_data)
        except TypeError:
            length = 0
        if length > 0:
            first = observed_data[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return 'joint_outcomes', int(length)
            if isinstance(first, (int, np.integer)):
                return 'joint_indices', int(length)
        return 'joint_outcomes', int(length)

    # Standard path: NumPy-coercible array.
    arr = np.asarray(observed_data)
    if arr.ndim == 1:
        return '1d_times', int(arr.shape[0])
    if arr.ndim == 2:
        return '2d_times', int(arr.shape[0])
    # ndim 0 or >2: caller will error elsewhere; report as 1D length 1
    # so the validator does not blow up before the downstream check.
    return '1d_times', int(arr.size)


def _classify_rewards_kind(rewards: Any) -> RewardsKind:
    if rewards is None:
        return 'none'
    arr = np.asarray(rewards)
    if arr.ndim == 1:
        return '1d'
    if arr.ndim == 2:
        return '2d'
    raise SvgdConfigError(
        "rewards must be None, a 1D array of shape (n_vertices,), or a "
        "2D array of shape (n_features, n_vertices); got shape "
        f"{arr.shape}."
    )


def _coerce_fixed_indices(
    fixed: Any, has_epoch_starts: bool,
) -> Tuple[bool, Optional[Sequence[int]]]:
    """Return (has_fixed, sorted-tuple-of-local-indices-or-None).

    For daisy-chain (``has_epoch_starts=True``), entries are
    ``(local_index, value_or_per_epoch_array)`` and the local index is
    in ``[0, param_length)``. For the non-epoch path, entries are
    ``(flat_index, value)``; we treat the flat index as the
    "local" index for validation purposes (the validator R8 still
    checks ``exposure_param_index < param_length`` against the same
    range, which matches the non-epoch model's flat layout).
    """
    if fixed is None:
        return False, None
    if isinstance(fixed, (list, tuple)) and len(fixed) > 0 \
            and isinstance(fixed[0], tuple):
        try:
            indices = tuple(int(idx) for idx, _v in fixed)
        except (TypeError, ValueError) as exc:
            raise SvgdConfigError(
                "fixed must be a list of (index, value) tuples; "
                f"could not parse: {exc}"
            )
        return True, indices
    if isinstance(fixed, (list, tuple, np.ndarray)):
        arr = np.asarray(fixed)
        if arr.ndim != 1:
            raise SvgdConfigError(
                "fixed mask must be 1D; got shape "
                f"{arr.shape}."
            )
        return True, tuple(int(i) for i in np.flatnonzero(arr != 0).tolist())
    raise SvgdConfigError(
        "fixed must be None, a list of (index, value) tuples, or a "
        f"1D mask array; got {type(fixed).__name__}."
    )


def from_svgd_call(
    graph: Any,
    observed_data: Any,
    *,
    rewards: Any = None,
    fixed: Any = None,
    epoch_starts: Any = None,
    exposure: Any = None,
    exposure_param_index: Optional[int] = None,
    param_transform: Any = None,
    positive_params: bool = True,
    preconditioner: Any = 'auto',
    regularization: float = 0.0,
    nr_moments: int = 2,
    joint_index: bool = False,
    **_unused: Any,
) -> SvgdConfig:
    """Build an :class:`SvgdConfig` from a live ``Graph.svgd`` call.

    Accepts every keyword that ``Graph.svgd`` accepts, but only reads
    the ones that influence validation rules. Unknown kwargs are
    ignored so this function can be called with the full ``svgd``
    kwargs dict directly.
    """
    graph_kind = _classify_graph_kind(graph)
    is_discrete = bool(getattr(graph, 'is_discrete', False))
    param_length = int(graph.param_length()) if hasattr(graph, 'param_length') else 0

    observation_kind, n_observations = _classify_observation_kind(
        observed_data, graph_kind
    )
    rewards_kind = _classify_rewards_kind(rewards)

    has_epoch_starts = epoch_starts is not None
    n_epochs: Optional[int] = None
    if has_epoch_starts:
        try:
            n_epochs = int(np.asarray(epoch_starts).ravel().size)
        except (TypeError, ValueError):
            n_epochs = None

    has_fixed, fixed_indices = _coerce_fixed_indices(fixed, has_epoch_starts)

    has_exposure = exposure is not None
    exposure_length: Optional[int] = None
    if has_exposure:
        try:
            arr = np.asarray(exposure)
            exposure_length = int(arr.size) if arr.ndim >= 1 else 1
        except (TypeError, ValueError):
            exposure_length = None

    return SvgdConfig(
        graph_kind=graph_kind,
        is_discrete=is_discrete,
        param_length=param_length,
        observation_kind=observation_kind,
        n_observations=n_observations,
        rewards_kind=rewards_kind,
        has_epoch_starts=has_epoch_starts,
        n_epochs=n_epochs,
        has_fixed=has_fixed,
        fixed_indices=fixed_indices,
        has_exposure=has_exposure,
        exposure_length=exposure_length,
        exposure_param_index=(
            int(exposure_param_index)
            if exposure_param_index is not None else None
        ),
        has_param_transform=param_transform is not None,
        positive_params=bool(positive_params),
        has_preconditioner=preconditioner is not None and preconditioner != 'none',
        has_regularization=float(regularization) > 0.0,
        nr_moments=int(nr_moments),
        joint_index_explicit=bool(joint_index),
    )


# ---------------------------------------------------------------------------
# Rules R1..R15 — each a small named function that raises on violation.
# ---------------------------------------------------------------------------

def _check_R1_epoch_requires_continuous_joint_prob(c: SvgdConfig) -> None:
    if not c.has_epoch_starts:
        return
    if c.graph_kind != 'joint_prob':
        raise SvgdConfigError(
            "epoch_starts requires a joint-probability graph "
            "(built with graph.joint_prob_graph(...)). "
            f"Got graph kind '{c.graph_kind}'. "
            "Either drop epoch_starts or build a joint-prob graph first."
        )
    if c.is_discrete:
        raise SvgdConfigError(
            "Daisy-chain SVGD (epoch_starts=...) requires a "
            "continuous-time joint-prob graph. Construct it with "
            "discrete=False:\n"
            "    graph.joint_prob_graph(indexer, ..., discrete=False)"
        )


def _check_R2_joint_index_requires_joint_prob(c: SvgdConfig) -> None:
    if not c.joint_index_explicit:
        return
    if c.graph_kind != 'joint_prob':
        raise SvgdConfigError(
            "joint_index=True requires a joint-probability graph "
            "(built with graph.joint_prob_graph(...)). "
            f"Got graph kind '{c.graph_kind}'."
        )


def _check_R3_rewards_incompatible_with_joint_prob(c: SvgdConfig) -> None:
    if c.graph_kind == 'joint_prob' and c.rewards_kind != 'none':
        raise SvgdConfigError(
            "Reward transformation is not supported with joint-probability "
            "graphs. Pass rewards=None when calling svgd() on a graph "
            "built via joint_prob_graph(...)."
        )


def _check_R4_regularization_incompatible_with_joint_prob(c: SvgdConfig) -> None:
    if c.graph_kind == 'joint_prob' and c.has_regularization:
        raise SvgdConfigError(
            "Moment regularization (regularization > 0) is not supported "
            "with joint-probability graphs. Set regularization=0 when "
            "calling svgd() on a graph built via joint_prob_graph(...)."
        )


def _check_R5_sparse_observations_require_rewards(c: SvgdConfig) -> None:
    if c.observation_kind == 'sparse' and c.rewards_kind == 'none':
        raise SvgdConfigError(
            "SparseObservations require explicit rewards. "
            "Pass rewards=<2D array of shape (n_features, n_vertices)>, "
            "or convert your data to dense format with dense_to_sparse()."
        )


def _check_R6_sparse_observations_incompatible_with_exposure(c: SvgdConfig) -> None:
    if c.observation_kind == 'sparse' and c.has_exposure:
        raise SvgdConfigError(
            "exposure is not supported with SparseObservations. "
            "Sparse observations do not carry per-observation identity "
            "(values are flattened across features), so per-observation "
            "exposure scaling has no well-defined target. Convert to "
            "dense format and pass exposure alongside it, or drop exposure."
        )


def _check_R7_exposure_and_param_index_paired(c: SvgdConfig) -> None:
    if c.has_exposure and c.exposure_param_index is None:
        raise SvgdConfigError(
            "exposure_param_index must be provided when exposure is set. "
            "Pass exposure_param_index=<integer index of the rate-typed "
            "parameter in theta that exposure scales>."
        )
    if (not c.has_exposure) and c.exposure_param_index is not None:
        raise SvgdConfigError(
            "exposure_param_index was provided but exposure is None. "
            "Pass exposure alongside exposure_param_index, or pass neither."
        )


def _check_R8_exposure_param_index_in_param_length_range(c: SvgdConfig) -> None:
    if c.exposure_param_index is None:
        return
    # exposure_param_index is the *per-epoch local* index, even under daisy-chain.
    if not (0 <= c.exposure_param_index < c.param_length):
        raise SvgdConfigError(
            f"exposure_param_index={c.exposure_param_index} is out of "
            f"range. exposure_param_index must be in "
            f"[0, param_length) = [0, {c.param_length}). "
            "Under daisy-chain (epoch_starts=...), exposure_param_index "
            "is the *local* per-epoch index; the validator broadcasts it "
            "across all epochs."
        )


def _check_R9_exposure_with_vanilla_joint_prob_unsupported(c: SvgdConfig) -> None:
    if (c.has_exposure
            and c.graph_kind == 'joint_prob'
            and not c.has_epoch_starts):
        raise SvgdConfigError(
            "exposure + vanilla joint-prob (no epoch_starts) is not "
            "supported. The per-observation theta-rescaling wrapper would "
            "fan out to O(n_obs) full graph eliminations per gradient and "
            "make JIT compilation effectively unbounded.\n"
            "  Fix: call svgd() with epoch_starts=[0.0, ...] (one entry "
            "per epoch). For a time-homogeneous model with per-observation "
            "exposure, pass epoch_starts=[0.0] (single epoch).\n"
            "  See docs/pages/tutorial/joint-probability.ipynb for the "
            "daisy-chain pattern."
        )


def _check_R10_exposure_length_matches_n_observations(c: SvgdConfig) -> None:
    if not c.has_exposure:
        return
    if c.exposure_length is None:
        return
    # Scalar broadcast is allowed.
    if c.exposure_length == 1:
        return
    if c.exposure_length != c.n_observations:
        raise SvgdConfigError(
            f"exposure length ({c.exposure_length}) does not match "
            f"the number of observations ({c.n_observations}). "
            "Pass a scalar to broadcast, or a 1D vector aligned with "
            "observed_data."
        )


def _check_R11_exposure_with_2d_rewards_warns(c: SvgdConfig) -> None:
    if c.has_exposure and c.rewards_kind == '2d':
        warnings.warn(
            "exposure + 2D rewards (multivariate) has not been "
            "benchmarked. The wrapper assumes the underlying multivariate "
            "model honours per-observation theta scaling. Verify against "
            "a Monte-Carlo cross-check before trusting posteriors.",
            UserWarning,
            stacklevel=3,
        )


def _check_R12_param_transform_incompatible_with_joint_prob(c: SvgdConfig) -> None:
    if c.has_param_transform and c.graph_kind == 'joint_prob':
        raise SvgdConfigError(
            "param_transform is not supported with joint-probability "
            "graphs. The joint-index custom-VJP backward (finite "
            "differences) runs in constrained theta space; an outer "
            "param_transform would corrupt the gradient.\n"
            "  Fix: drop param_transform, or set positive_params=True "
            "(the default softplus reparameterisation already runs "
            "inside the model's gradient pipeline)."
        )


def _check_R13_preconditioner_with_joint_prob_warns(c: SvgdConfig) -> None:
    if c.has_preconditioner and c.graph_kind == 'joint_prob':
        warnings.warn(
            "preconditioner + joint-probability graph: the interaction "
            "between the SVGD preconditioner and the joint-index custom "
            "VJP has not been validated. Consider preconditioner=None "
            "for joint-prob inference.",
            UserWarning,
            stacklevel=3,
        )


def _check_R14_fixed_with_epoch_starts_local_indices(c: SvgdConfig) -> None:
    if not (c.has_fixed and c.has_epoch_starts):
        return
    if c.fixed_indices is None:
        return
    bad = [i for i in c.fixed_indices if not (0 <= i < c.param_length)]
    if bad:
        raise SvgdConfigError(
            "Under daisy-chain (epoch_starts=...), fixed entries must "
            "use *local* per-epoch indices in "
            f"[0, param_length) = [0, {c.param_length}). "
            f"Got out-of-range indices {bad!r}. "
            "The daisy-chain model broadcasts each fixed local index "
            "across all epochs internally."
        )


def _check_R15_positive_params_xor_param_transform(c: SvgdConfig) -> None:
    if c.positive_params and c.has_param_transform:
        raise SvgdConfigError(
            "positive_params=True and param_transform=<callable> are "
            "mutually exclusive. positive_params applies softplus; "
            "param_transform is a user-provided alternative. Pass "
            "positive_params=False alongside param_transform, or drop "
            "param_transform."
        )


_RULES = (
    _check_R1_epoch_requires_continuous_joint_prob,
    _check_R2_joint_index_requires_joint_prob,
    _check_R3_rewards_incompatible_with_joint_prob,
    _check_R4_regularization_incompatible_with_joint_prob,
    _check_R5_sparse_observations_require_rewards,
    _check_R6_sparse_observations_incompatible_with_exposure,
    _check_R7_exposure_and_param_index_paired,
    _check_R8_exposure_param_index_in_param_length_range,
    _check_R9_exposure_with_vanilla_joint_prob_unsupported,
    _check_R10_exposure_length_matches_n_observations,
    _check_R11_exposure_with_2d_rewards_warns,
    _check_R12_param_transform_incompatible_with_joint_prob,
    _check_R13_preconditioner_with_joint_prob_warns,
    _check_R14_fixed_with_epoch_starts_local_indices,
    _check_R15_positive_params_xor_param_transform,
)


def validate(config: SvgdConfig) -> None:
    """Run every rule in order. First failing rule raises.

    Warnings (R11, R13) are emitted via :mod:`warnings` and do not
    interrupt validation.
    """
    for rule in _RULES:
        rule(config)


__all__ = [
    'SvgdConfig',
    'SvgdConfigError',
    'from_svgd_call',
    'validate',
]
