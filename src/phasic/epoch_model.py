"""``FreeEpochModel`` — the reusable free daisy-chain epoch model returned by
:meth:`phasic.Graph.epoch_model`.

Build a free (untied) time-inhomogeneous joint-probability model once and fit it
many times. Two fits from the SAME object share ``.model``, so
:func:`phasic.model_selection.likelihood_ratio_test(full, nested)` takes the fast
same-model path (free-vs-fixed nested LRT on epoch models). Tying is out of
scope (use ``Graph.svgd(tied=...)`` for tied fits).
"""
from __future__ import annotations

from typing import Any


class FreeEpochModel:
    """A reusable free (untied) daisy-chain (epoch) model.

    Built by :meth:`phasic.Graph.epoch_model`; not constructed directly. The
    underlying ``model`` callable bakes **no** fixed-slot FD-skip, so an
    arbitrary ``fixed`` set may be applied per fit (no build-time/fit-time
    superset constraint).

    Attributes
    ----------
    model : callable
        JAX-differentiable free daisy-chain model callable (carries the
        ``_handles_exposure_internally`` / ``_handles_particle_vmap`` /
        ``_precondition_output`` tags and ``_tying_info={}``).
    theta_dim : int
        Flat parameter dimension ``n_epochs * param_length``.
    observed_data : list[int]
        1-D mapped joint-prob vertex indices (also baked into ``model``); passed
        to ``SVGD`` as the shape/NaN-mask carrier.
    prior : list or None
        Broadcast per-slot prior list with base-fixed slots already ``None``, or
        ``None``.
    fixed : list[tuple[int, float]] or None
        Base ``(flat_idx, value)`` fixings (broadcast across epochs), or ``None``.
    n_epochs, param_length, t_eval, epoch_starts
        Epoch metadata.
    """

    def __init__(self, *, model, theta_dim, observed_data, prior, fixed,
                 n_epochs, param_length, t_eval, epoch_starts):
        self.model = model
        self.theta_dim = int(theta_dim)
        self.observed_data = observed_data
        self.prior = prior
        self.fixed = fixed
        self.n_epochs = int(n_epochs)
        self.param_length = int(param_length)
        self.t_eval = t_eval
        self.epoch_starts = epoch_starts

    def __repr__(self) -> str:
        return (
            f"FreeEpochModel(theta_dim={self.theta_dim}, "
            f"n_epochs={self.n_epochs}, param_length={self.param_length}, "
            f"n_obs={len(self.observed_data)}, t_eval={self.t_eval})"
        )

    def fit(self, *, fixed=None, prior=None, **svgd_kwargs):
        """Fit the model — optionally with an ADDITIONAL restriction — and
        return the fitted :class:`phasic.SVGD`.

        Parameters
        ----------
        fixed : list of (flat_idx, value), optional
            Additional fixings in **flat** ``[0, n_epochs * param_length)`` index
            space — NOT the per-epoch/local space used by
            ``Graph.epoch_model(fixed=)`` / ``Graph.svgd(fixed=)``. Merged with
            the builder's base ``fixed`` (this fit's value wins on a shared
            index). Fixing ONE flat slot adds df=1 to a nested LRT; a
            per-epoch-constant restriction spans ``n_epochs`` flat slots
            (df = n_epochs).
        prior : optional
            Overrides the builder's prior for this fit. A per-slot list is
            re-masked to ``None`` at every fixed index; a scalar callable is
            auto-masked by SVGD; ``None`` disables re-masking.
        **svgd_kwargs
            Forwarded to :class:`phasic.SVGD` (``n_particles``, ``n_iterations``,
            ``optimizer``, ``seed``, ``learning_rate``, ...). ``exposure`` /
            ``exposure_param_index`` must NOT be passed — exposure is baked into
            the model at build time.

        Returns
        -------
        SVGD
            The fitted instance (``.optimize()`` already run). Reusing the SAME
            ``FreeEpochModel`` for ``full`` and ``nested`` gives ``full.model is
            nested.model`` → the same-model LRT fast path.
        """
        from .svgd import SVGD

        if 'exposure' in svgd_kwargs or 'exposure_param_index' in svgd_kwargs:
            raise TypeError(
                "exposure is baked into the FreeEpochModel at build time; do not "
                "pass exposure/exposure_param_index to fit(). Set them on "
                "Graph.epoch_model(...) instead."
            )

        # Merge base + extra fixings in FLAT index space; this fit's value wins.
        merged_map: dict[int, float] = {}
        for i, v in (self.fixed or []):
            merged_map[int(i)] = v
        for i, v in (fixed or []):
            merged_map[int(i)] = v
        merged = sorted(merged_map.items()) if merged_map else None

        # Re-mask the prior at all fixed indices (SVGD rejects a non-None prior
        # entry at a fixed slot). Only a per-slot LIST needs this; a scalar
        # callable is auto-masked by SVGD, and None is left as-is (amendment 3).
        prior_use = prior if prior is not None else self.prior
        if isinstance(prior_use, list) and merged_map:
            prior_use = [
                None if k in merged_map else p
                for k, p in enumerate(prior_use)
            ]

        svgd = SVGD(
            model=self.model,
            observed_data=self.observed_data,
            theta_dim=self.theta_dim,
            prior=prior_use,
            fixed=merged,
            **svgd_kwargs,
        )
        svgd.optimize()
        return svgd
