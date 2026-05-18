"""Zero-inflated likelihood wiring for partial-coverage rewards.

When a reward vector marks only a subset of vertices as rewarded, the
corresponding feature's distribution has an atomic mass at ``r = 0``
(trajectories that absorb without visiting any rewarded vertex). The
correct likelihood is then a mixture of a point mass and a continuous
part::

    p(obs | theta) = (1 - p_j(theta)) * 1{obs == 0}
                   + p_j(theta) * f_j(obs | theta) * 1{obs > 0}

where ``p_j(theta)`` is the probability of visiting a rewarded vertex
under feature ``j``.

``Graph.svgd`` detects partial-coverage features via
:func:`partial_coverage_features` and wires the term onto the SVGD
model via :func:`attach_zero_inflated_term`. Direct ``SVGD(...)``
callers benefit from the same machinery via the auto-attach hook in
``SVGD.__init__``.

The helpers operate on a live ``Graph`` object — they need its
``_validate_reward_coverage`` BFS, ``vertices_length()``,
``_initial_probability_vector()``, and ``serialize()`` machinery.
Direct SVGD callers whose model carries a ``_source_graph`` attribute
(stamped by the PMF builders) get auto-attachment for free; custom
user-built models without ``_source_graph`` skip the wiring silently.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def partial_coverage_features(graph: Any, rewards: Any) -> list[int]:
    """Return indices of features (rows) that fail the coverage check.

    For 1D rewards returns ``[0]`` if the single vector fails,
    otherwise ``[]``. For 2D rewards ``(n_features, n_vertices)``
    returns the list of offending feature indices. Reuses the
    ``Graph._validate_reward_coverage`` BFS — just catches the
    ``ValueError`` instead of letting it propagate.
    """
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.ndim == 1:
        try:
            graph._validate_reward_coverage(arr, context="rewards")
            return []
        except ValueError:
            return [0]
    assert arr.ndim == 2, "expected 1D or 2D rewards after shape check"
    offenders: list[int] = []
    for j in range(arr.shape[0]):
        try:
            graph._validate_reward_coverage(
                arr[j], context=f"rewards[feature {j}]"
            )
        except ValueError:
            offenders.append(j)
    return offenders


def attach_zero_inflated_term(
    graph: Any,
    model: Any,
    *,
    rewards: Any,
    offenders: Sequence[int],
    observed_data: Any,
) -> None:
    """Wire the zero-inflated likelihood term onto an SVGD model.

    Attaches two attributes to ``model``:

    - ``_zero_inflated_p_fn`` : callable ``theta -> jax.Array``,
      returning a vector of ``p_j(theta) = P(visit a rewarded
      vertex in feature j | theta)`` for the offending features
      (length ``len(offenders)``). JAX-differentiable.
    - ``_n_zero_per_feature`` : np.int64 array of shape
      ``(len(offenders),)`` counting zero-valued observations per
      offending feature. Computed once at attach time from the
      user's ``observed_data``.

    ``SVGD._log_prob_unified`` reads these to add the
    ``Σ_j n_zero_j * log(1 - p_j(theta))`` term to the
    log-likelihood.
    """
    import jax.numpy as jnp
    from .svgd import is_sparse_observations

    arr = np.asarray(rewards, dtype=np.float64)
    n_v = graph.vertices_length()
    if arr.ndim == 1:
        offender_reward_rows = [arr]
    else:
        offender_reward_rows = [arr[j] for j in offenders]

    # Per-feature zero counts from observed_data.
    n_zero_per_feature = np.zeros(len(offenders), dtype=np.int64)
    if is_sparse_observations(observed_data):
        values = np.asarray(observed_data.values)
        feat_idx = np.asarray(observed_data.features)
        for k, j in enumerate(offenders):
            mask = feat_idx == j
            if not np.any(mask):
                n_zero_per_feature[k] = 0
                continue
            vals_j = values[mask]
            n_zero_per_feature[k] = int(
                np.sum((vals_j == 0.0) & ~np.isnan(vals_j))
            )
    else:
        obs = np.asarray(observed_data)
        if obs.ndim == 1:
            # 1D path: rewards must be 1D, single offender == feature 0.
            n_zero_per_feature[0] = int(np.sum(obs == 0.0))
        elif obs.ndim == 2:
            # 2D dense (NaN-padded) observations.
            for k, j in enumerate(offenders):
                col = obs[:, j]
                n_zero_per_feature[k] = int(
                    np.sum((col == 0.0) & ~np.isnan(col))
                )
        else:
            raise ValueError(
                "attach_zero_inflated_term: observed_data must be "
                "1D, 2D, or SparseObservations; got ndim "
                f"{obs.ndim}."
            )

    # Skip the wiring entirely when there are NO zero observations
    # for any offending feature. The math then contributes nothing
    # (n_zero * log(1 - p) == 0) and we save the runtime cost of
    # the extra FFI call. The legacy path is correct in that case.
    if not np.any(n_zero_per_feature > 0):
        return

    # Precompute the (concrete) rewarded-vertex index arrays once
    # so the JAX-traced p(theta) function can use static-sized
    # int32 arrays inside the FFI call.
    rewarded_per_feature: list[np.ndarray] = []
    for row in offender_reward_rows:
        rewarded = np.where(row > 0.0)[0].astype(np.int32)
        rewarded_per_feature.append(rewarded)

    alpha = graph._initial_probability_vector()
    structure_json = graph.serialize()

    # Prefer the fused `_cdf_zero_fn` exposed by the pybind model
    # builder (the standard `Graph.svgd` path). It computes the
    # atomic mass at r = 0 of the reward-transformed distribution,
    # which is mathematically the complement of `p_j(theta)` =
    # P(visit a rewarded vertex). One call per particle, sharing
    # the reward_transform with the model's PDF pass.
    cdf_zero_fn = getattr(model, '_cdf_zero_fn', None)

    if cdf_zero_fn is not None:
        # Stack the offender rows into a 2D array (n_offenders, n_v)
        # so `_cdf_zero_fn` returns one value per offender feature
        # in the same order as `offender_reward_rows`.
        offender_stack = np.stack(
            [np.asarray(r, dtype=np.float64) for r in offender_reward_rows],
            axis=0,
        )
        offender_stack_jax = jnp.asarray(offender_stack, dtype=jnp.float64)

        def _zero_inflated_p_fn(theta):
            """Return p_j(theta) = 1 - cdf_zero_j(theta) for each
            offending feature j. Uses the fused pybind path which
            shares the reward_transform with the model's PDF
            computation; no separate backward_probabilities solve.
            """
            cdf_zeros = cdf_zero_fn(theta, offender_stack_jax)
            return 1.0 - cdf_zeros
    else:
        from .ffi_wrappers import compute_reward_visit_probability_ffi

        def _zero_inflated_p_fn(theta):
            """Return p_j(theta) for each offending feature j.

            Used when the model was built via the FFI path
            (`use_ffi=True`) which does not currently expose
            `_cdf_zero_fn`. Falls back to a separate
            `backward_probabilities` solve per offender per
            particle.
            """
            alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
            ps = []
            for rewarded in rewarded_per_feature:
                rewarded_j = jnp.asarray(rewarded, dtype=jnp.int32)
                ps.append(
                    compute_reward_visit_probability_ffi(
                        structure_json, theta, rewarded_j, alpha_j,
                    )
                )
            return jnp.stack(ps)

    model._zero_inflated_p_fn = _zero_inflated_p_fn
    model._n_zero_per_feature = jnp.asarray(
        n_zero_per_feature, dtype=jnp.float64,
    )
    # Plain-Python introspection: the list of offending feature
    # indices (which `SVGD.summary` reads to mention zero-
    # inflation) and the matching zero-count vector. Stored as
    # numpy / list so users can inspect without depending on jax.
    model._zero_inflated_features = list(offenders)
    model._n_zero_per_feature_np = np.asarray(
        n_zero_per_feature, dtype=np.int64,
    )


__all__ = [
    'partial_coverage_features',
    'attach_zero_inflated_term',
]
