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
    _param_domains : tuple or None
        The flat per-slot domains the model's finite-difference gradients
        were built for (``Graph.epoch_model(param_domains=...)``, default
        every slot ``'positive'``). :meth:`fit` reads them from HERE, not from
        ``model``, and checks each fit against them.

    When built with ``prior=None``, the default data-informed prior is built
    at :meth:`fit` time from the base and fit-level ``fixed`` combined (a
    parameter counts as fixed when it is fixed in every epoch), so a model
    built with nothing fixed can still be fitted with an identifying
    ``fixed``.
    """

    def __init__(self, *, model, theta_dim, observed_data, prior, fixed,
                 n_epochs, param_length, t_eval, epoch_starts,
                 default_prior_factory=None, param_domains=None):
        self.model = model
        self.theta_dim = int(theta_dim)
        self.observed_data = observed_data
        self.prior = prior
        self.fixed = fixed
        self.n_epochs = int(n_epochs)
        self.param_length = int(param_length)
        self.t_eval = t_eval
        self.epoch_starts = epoch_starts
        # (flat_fixed, positive_params) -> (flat prior list, info); set by
        # Graph.epoch_model when it was built with prior=None.
        self._default_prior_factory = default_prior_factory
        # Plan A §3: the domains the model was built for (see fit()).
        self._param_domains = (None if param_domains is None
                               else tuple(param_domains))

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
            Values are in parameter space and reach the model exactly as
            given: a fixed parameter does not pass through any transform, so 0,
            values of any magnitude and negative values are delivered unchanged
            (a non-finite value raises).
        prior : optional
            Overrides the builder's prior for this fit. A per-slot list is
            re-masked to ``None`` at every fixed index; a scalar callable is
            auto-masked by SVGD; ``None`` disables re-masking. See Notes for
            the parameter-domain check.
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

        Notes
        -----
        **Parameter domains (Plan A).** The fit must lie inside the domains the
        model's finite-difference gradients were built for
        (``Graph.epoch_model(param_domains=...)``, default every slot
        ``'positive'``): a BetaPrior (sigmoid) slot or a softplus slot fits a
        ``'positive'`` model, but ``positive_params=False``
        (``'unconstrained'``) on a ``'positive'`` model raises; rebuild with
        ``epoch_model(..., param_domains='unconstrained')``. A model slot
        declared ``'fixed'`` must be fixed in the fit. This fit reproduces
        ``Graph.svgd(epoch_starts=...)`` bit for bit only when the model's
        domains match what that fit declares, e.g. ``'unit'`` for a
        BetaPrior slot.

        **Near-zero limit (documented, user decision D-NZ).** On an
        ``'unconstrained'`` slot the finite-difference gradient of a RATE
        within ~1e-7 of zero can be wrong, including its sign; fit rates with
        ``positive_params=True`` (the default). A forward tie (two probes
        giving identical outputs) gives a zero gradient for that slot,
        counted in ``model._fd_stats`` and logged once.
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

        # Plan A §3: this fit's domains must lie inside the model's
        # (containment, not equality). Checked before the default prior's
        # reference fit, so a mismatch fails fast. The default prior (prior
        # None here) is a GaussPrior with no transform of its own.
        if self._param_domains is not None:
            from .svgd import _check_param_domains, _resolve_param_domains
            _check_param_domains(
                self._param_domains,
                _resolve_param_domains(
                    prior if prior is not None else self.prior,
                    svgd_kwargs.get('positive_params', True),
                    svgd_kwargs.get('param_transform', None),
                    theta_dim=self.theta_dim,
                    fixed_indices=list(merged_map)),
                "FreeEpochModel.fit",
                "Rebuild the model with Graph.epoch_model(..., "
                "param_domains=...) to match this fit (e.g. "
                "param_domains='unconstrained' for positive_params=False).")

        # Re-mask the prior at all fixed indices (SVGD rejects a non-None prior
        # entry at a fixed slot). Only a per-slot LIST needs this; a scalar
        # callable is auto-masked by SVGD, and None is left as-is (amendment 3).
        prior_use = prior if prior is not None else self.prior
        from .svgd_config import assume
        if prior_use is None and self._default_prior_factory is not None:
            prior_use, _info = self._default_prior_factory(
                merged or [], svgd_kwargs.get('positive_params', True))
            if _info['exposure_binned']:
                assume(
                    "FreeEpochModel.fit: the default prior grouped the "
                    f"exposure values into {_info['exposure_groups']} quantile "
                    "bins (each at its mean exposure); this only affects the "
                    "prior's centre and width.", stacklevel=2)
        # Nothing pins the scale of the last epoch's rates unless one of its
        # slots is fixed (this model has no tying).
        _last = range((self.n_epochs - 1) * self.param_length,
                      self.n_epochs * self.param_length)
        if not any(k in merged_map for k in _last):
            assume(
                "FreeEpochModel.fit: none of the last epoch's rates is fixed, "
                "so the data cannot determine their overall scale and it "
                "follows the prior; fix a rate with fixed= to identify it.",
                stacklevel=2)
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
