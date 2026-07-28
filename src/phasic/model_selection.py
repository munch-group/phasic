"""Information-criterion and likelihood-ratio model selection for phasic SVGD fits.

These tools operate on one or more fitted :class:`~phasic.SVGD` instances and
produce small immutable result objects. They never mutate SVGD state.

Three quantities drive everything:

- ``svgd.log_likelihood(refine=...)``  — Σ log p(x_i | θ̂) at the MAP
- ``svgd.degrees_of_freedom``           — number of free parameters (``k``)
- ``svgd.n_observations``               — effective sample size (``n``)

The module wraps these into the standard model-comparison recipes:

- :func:`aic` — Akaike Information Criterion: ``2k − 2·LL``
- :func:`bic` — Bayesian Information Criterion: ``k·log(n) − 2·LL``
- :func:`likelihood_ratio_test` — Wilks LRT for nested models: statistic
  ``2·(LL_full − LL_nested)`` ∼ χ²(k_full − k_nested) under H₀
- :func:`compare` — rank a set of fits by AIC/BIC, with ΔAIC and weights

Example
-------
>>> model = Graph.pmf_and_moments_from_graph(graph)
>>> full   = SVGD(model, data, theta_dim=2).optimize()
>>> nested = SVGD(model, data, theta_dim=2, fixed=[(1, 0.0)]).optimize()
>>> import phasic.model_selection as ms
>>> print(ms.likelihood_ratio_test(full, nested))
>>> print(ms.compare(("full", full), ("nested", nested)))
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from .svgd import SVGD


__all__ = [
    "aic",
    "bic",
    "likelihood_ratio_test",
    "likelihood_ratio_test_at",
    "compare",
    "AICResult",
    "BICResult",
    "LRTResult",
    "ComparisonRow",
    "ComparisonTable",
]


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass(frozen=True)
class AICResult:
    """Result of an :func:`aic` computation."""

    aic: float
    log_likelihood: float
    k: int
    n: int
    refined: bool

    def __repr__(self) -> str:
        return (
            f"AICResult(aic={self.aic:.4f}, log_likelihood={self.log_likelihood:.4f}, "
            f"k={self.k}, n={self.n}, refined={self.refined})"
        )


@dataclass(frozen=True)
class BICResult:
    """Result of a :func:`bic` computation."""

    bic: float
    log_likelihood: float
    k: int
    n: int
    refined: bool

    def __repr__(self) -> str:
        return (
            f"BICResult(bic={self.bic:.4f}, log_likelihood={self.log_likelihood:.4f}, "
            f"k={self.k}, n={self.n}, refined={self.refined})"
        )


@dataclass(frozen=True)
class LRTResult:
    """Result of a :func:`likelihood_ratio_test` computation."""

    statistic: float
    df: int
    p_value: float
    log_likelihood_full: float
    log_likelihood_nested: float
    k_full: int
    k_nested: int
    n: int

    def __repr__(self) -> str:
        return (
            f"LRTResult(statistic={self.statistic:.4f}, df={self.df}, "
            f"p_value={self.p_value:.4g}, "
            f"ll_full={self.log_likelihood_full:.4f}, "
            f"ll_nested={self.log_likelihood_nested:.4f}, "
            f"k_full={self.k_full}, k_nested={self.k_nested}, n={self.n})"
        )


@dataclass(frozen=True)
class ComparisonRow:
    """One row of a :class:`ComparisonTable`."""

    name: str
    criterion: float
    delta: float
    weight: float
    log_likelihood: float
    k: int
    n: int


@dataclass(frozen=True)
class ComparisonTable:
    """Result of a :func:`compare` computation.

    Rows are sorted ascending by criterion (best model first). Weights
    are Akaike weights for AIC (``exp(−Δᵢ/2) / Σⱼ exp(−Δⱼ/2)``) and the
    analogous Schwarz weights for BIC.
    """

    rows: tuple
    criterion: str

    def __repr__(self) -> str:
        crit_label = self.criterion.upper()
        header = f"Model comparison by {crit_label}:\n"
        col_header = (
            f"{'name':<20} {crit_label:>12} {'Δ':>10} "
            f"{'weight':>10} {'LL':>14} {'k':>4} {'n':>6}"
        )
        sep = "-" * len(col_header)
        body_lines = [
            f"{r.name:<20} {r.criterion:>12.4f} {r.delta:>10.4f} "
            f"{r.weight:>10.4f} {r.log_likelihood:>14.4f} "
            f"{r.k:>4d} {r.n:>6d}"
            for r in self.rows
        ]
        return "\n".join([header.rstrip(), col_header, sep, *body_lines])


# ============================================================================
# Internal helpers
# ============================================================================


def _check_fitted(svgd: "SVGD") -> None:
    if not getattr(svgd, "is_fitted", False):
        raise ValueError(
            "SVGD instance has not been fitted; call optimize() first."
        )


def _violate(strict: bool, msg: str) -> None:
    """Raise in strict mode, otherwise emit a RuntimeWarning."""
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _check_nested_fixed(
    full: "SVGD", nested: "SVGD", strict: bool
) -> None:
    """Verify nested.fixed_mask is a strict superset of full.fixed_mask
    and that shared fixed values match in PHI space."""
    d = full.theta_dim
    mask_f = (
        np.zeros(d, dtype=np.int32)
        if full.fixed_mask is None
        else np.asarray(full.fixed_mask, dtype=np.int32)
    )
    mask_n = (
        np.zeros(d, dtype=np.int32)
        if nested.fixed_mask is None
        else np.asarray(nested.fixed_mask, dtype=np.int32)
    )
    if mask_f.shape != (d,) or mask_n.shape != (d,):
        _violate(strict, f"fixed_mask shape mismatch with theta_dim={d}")
        return

    diff = mask_n - mask_f
    if np.any(diff < 0):
        bad = np.where(diff < 0)[0].tolist()
        _violate(
            strict,
            f"Nested model frees parameters that full fixes: indices {bad}. "
            f"For LRT, nested.fixed_mask must include every index in "
            f"full.fixed_mask, plus at least one more.",
        )
        return
    if int(diff.sum()) == 0:
        _violate(
            strict,
            "Nested and full have identical fixed_mask — they are the same "
            "model, df = 0, LRT undefined.",
        )
        return

    shared = np.where((mask_f == 1) & (mask_n == 1))[0]
    if len(shared) > 0:
        vals_f = (
            np.asarray(full.fixed_values) if full.fixed_values is not None else None
        )
        vals_n = (
            np.asarray(nested.fixed_values) if nested.fixed_values is not None else None
        )
        if vals_f is None or vals_n is None:
            _violate(
                strict,
                "Shared fixed indices but missing fixed_values; "
                "internal inconsistency.",
            )
            return
        for i in shared:
            if not math.isclose(
                float(vals_f[i]), float(vals_n[i]), rel_tol=1e-9, abs_tol=1e-12
            ):
                _violate(
                    strict,
                    f"Parameter {int(i)} is fixed in both models but at different "
                    f"values: full={float(vals_f[i]):.6g} vs "
                    f"nested={float(vals_n[i]):.6g} (PHI space). LRT requires the "
                    f"nested restriction to lie inside the full parameter space.",
                )
                return


# ============================================================================
# Public API
# ============================================================================


def aic(svgd: "SVGD", *, refine: bool = False) -> AICResult:
    """Akaike Information Criterion. ``AIC = 2k − 2·LL``.

    Lower AIC indicates a better fit relative to model complexity.
    Use :func:`compare` to rank a set of fits.

    Parameters
    ----------
    svgd : SVGD
        A fitted SVGD instance.
    refine : bool, default=False
        If True, refine the MAP via gradient ascent on the log-posterior
        before evaluating LL. Slightly slower but closer to the MLE under
        weak priors.

    Returns
    -------
    AICResult
        Dataclass with fields ``aic``, ``log_likelihood``, ``k``, ``n``,
        and ``refined``.

    Examples
    --------
    >>> import phasic.model_selection as ms
    >>> svgd = graph.svgd(observed_data, theta_dim=2).optimize()
    >>> ms.aic(svgd)
    AICResult(aic=..., log_likelihood=..., k=2, n=200, refined=False)
    >>> ms.aic(svgd, refine=True).aic  # tighter MAP, usually smaller AIC
    ...
    """
    _check_fitted(svgd)
    ll = svgd.log_likelihood(refine=refine)
    k = svgd.degrees_of_freedom
    n = svgd.n_observations
    val = 2.0 * k - 2.0 * ll
    return AICResult(aic=val, log_likelihood=ll, k=k, n=n, refined=refine)


def bic(svgd: "SVGD", *, refine: bool = False) -> BICResult:
    """Bayesian Information Criterion. ``BIC = k·log(n) − 2·LL``.

    Penalizes complexity more aggressively than AIC for n ≥ 8, so BIC
    tends to prefer simpler models. Lower is better.

    Parameters
    ----------
    svgd : SVGD
        A fitted SVGD instance.
    refine : bool, default=False
        If True, refine the MAP via gradient ascent on the log-posterior
        before evaluating LL.

    Returns
    -------
    BICResult
        Dataclass with fields ``bic``, ``log_likelihood``, ``k``, ``n``,
        and ``refined``.

    Examples
    --------
    >>> import phasic.model_selection as ms
    >>> svgd = graph.svgd(observed_data, theta_dim=2).optimize()
    >>> ms.bic(svgd)
    BICResult(bic=..., log_likelihood=..., k=2, n=200, refined=False)
    """
    _check_fitted(svgd)
    ll = svgd.log_likelihood(refine=refine)
    k = svgd.degrees_of_freedom
    n = svgd.n_observations
    val = k * math.log(n) - 2.0 * ll
    return BICResult(bic=val, log_likelihood=ll, k=k, n=n, refined=refine)


def likelihood_ratio_test(
    full: "SVGD",
    nested: "SVGD",
    *,
    refine: bool = False,
    strict: bool = True,
) -> LRTResult:
    """Wilks likelihood-ratio test for two nested models.

    Tests H₀: nested model is correct, against H₁: full model is correct.
    Under H₀ and standard regularity conditions, the statistic
    ``2·(LL_full − LL_nested)`` is asymptotically χ² with df = k_full − k_nested.

    Parameters
    ----------
    full, nested : SVGD
        Fitted SVGD instances. Both must use the SAME model callable
        (typically a single :meth:`Graph.pmf_and_moments_from_graph`
        result), the same observed data, and the same ``theta_dim``.
        The nested fit must fix every parameter that the full fit fixes
        (at the same values), plus at least one additional parameter.

        The canonical pattern is::

            model = Graph.pmf_and_moments_from_graph(graph)
            full   = SVGD(model, data, theta_dim=k).optimize()
            nested = SVGD(model, data, theta_dim=k, fixed=[(0, 1.0)]).optimize()

    refine : bool, default=False
        If True, refine the MAP for both fits via gradient ascent before
        evaluating their log-likelihoods.

    strict : bool, default=True
        If True (recommended), all validation failures raise ``ValueError``.
        If False, they emit a ``RuntimeWarning`` instead. Loosen only
        when you understand why your fits look non-nested but are
        nevertheless on the same restriction structure.

    Returns
    -------
    LRTResult
        Dataclass with fields ``statistic`` (= 2·(LL_full − LL_nested)),
        ``df``, ``p_value``, ``log_likelihood_full``,
        ``log_likelihood_nested``, ``k_full``, ``k_nested``, ``n``.

    Raises
    ------
    ValueError
        In strict mode, on any of: mismatched ``theta_dim``,
        mismatched ``n_observations``, different model callables,
        non-superset fixed mask, mismatched fixed values, df ≤ 0,
        or LL inversion (nested > full).

    Examples
    --------
    Test whether a one-parameter restriction (rate₁ = 4.0) is consistent
    with the data:

    >>> import phasic.model_selection as ms
    >>> model = Graph.pmf_and_moments_from_graph(graph, theta_dim=2)
    >>> full   = phasic.SVGD(model, data, theta_dim=2).optimize()
    >>> nested = phasic.SVGD(model, data, theta_dim=2,
    ...                       fixed=[(1, 4.0)]).optimize()
    >>> res = ms.likelihood_ratio_test(full, nested, refine=True)
    >>> print(res)
    LRTResult(statistic=..., df=1, p_value=...)

    A small p-value (< 0.05 by convention) indicates the restriction is
    rejected; the full model is preferred. A large p-value means the
    restriction is consistent with the data and the simpler nested model
    can be retained.
    """
    _check_fitted(full)
    _check_fitted(nested)

    if full.theta_dim != nested.theta_dim:
        _violate(
            strict,
            f"theta_dim mismatch: full={full.theta_dim}, nested={nested.theta_dim}",
        )

    n_full = full.n_observations
    n_nested = nested.n_observations
    if n_full != n_nested:
        _violate(
            strict,
            f"n_observations mismatch: full={n_full}, nested={n_nested}",
        )

    if full.model is not nested.model:
        _violate(
            strict,
            "LRT requires both fits to use the SAME model callable. "
            "Build the model once and pass it to both SVGD constructors:\n"
            "    model = Graph.pmf_and_moments_from_graph(graph)\n"
            "    full   = SVGD(model, data, theta_dim=k).optimize()\n"
            "    nested = SVGD(model, data, theta_dim=k, fixed=[(0, 1.0)]).optimize()",
        )

    _check_nested_fixed(full, nested, strict)

    k_full = full.degrees_of_freedom
    k_nested = nested.degrees_of_freedom
    df = k_full - k_nested
    if df <= 0:
        _violate(
            strict,
            f"LRT requires k_full > k_nested, got k_full={k_full}, k_nested={k_nested}",
        )

    ll_full = full.log_likelihood(refine=refine)
    ll_nested = nested.log_likelihood(refine=refine)
    if ll_nested > ll_full + 1e-6:
        _violate(
            strict,
            f"LL_nested ({ll_nested:.4f}) > LL_full ({ll_full:.4f}). "
            f"A true nested restriction cannot have higher likelihood than "
            f"the full model; this likely means the fits use different data, "
            f"different graphs, or refinement has not converged.",
        )

    statistic = max(0.0, 2.0 * (ll_full - ll_nested))

    from scipy.stats import chi2
    # df may be ≤ 0 in non-strict mode; guard the p-value computation
    p_value = float(chi2.sf(statistic, df)) if df > 0 else float("nan")

    return LRTResult(
        statistic=statistic,
        df=int(df),
        p_value=p_value,
        log_likelihood_full=ll_full,
        log_likelihood_nested=ll_nested,
        k_full=int(k_full),
        k_nested=int(k_nested),
        n=int(n_full),
    )


def likelihood_ratio_test_at(
    full: "SVGD",
    nested: "SVGD",
    *,
    strict: bool = True,
    consistency_atol: float = 1e-4,
) -> LRTResult:
    """Wilks LRT for a nested pair whose restriction is baked into the model.

    Use this instead of :func:`likelihood_ratio_test` when ``full`` and
    ``nested`` are built from **different model callables** — the case for epoch
    (daisy-chain) joint-prob models where a *tied* fit and a *free* fit are
    separate callables (tying is applied inside the model, so
    ``full.model is nested.model`` can never hold and
    :func:`likelihood_ratio_test` rejects the pair).

    Both fits are scored on the SAME (full) likelihood: ``LL_full`` at the full
    fit's MAP and ``LL_nested`` at the nested fit's MAP, both evaluated through
    ``full.log_likelihood``. Because the nested restriction is a *constrained
    sub-case* of the full model (e.g. a tied fit's constrained MAP has the tied
    parameters equal across epochs — a valid point in the full model's flat
    ``n_epochs × param_length`` coordinates), scoring the nested MAP on the full
    model yields the nested model's own likelihood there.

    That equality is enforced at runtime (``consistency_atol``) and REPLACES the
    same-model identity guard: if scoring the nested MAP on the full model does
    NOT match the nested model's own likelihood, the two fits are not on the same
    likelihood (different graph/data/epochs) and the test is rejected.

    Parameters
    ----------
    full, nested : SVGD
        Fitted SVGD instances over the same data/graph/epochs, with the same
        ``theta_dim`` and ``n_observations``. ``nested`` must have fewer free
        parameters (``degrees_of_freedom``) than ``full`` and its restriction
        must be expressible in ``full``'s coordinates.
    strict : bool, default=True
        If True, validation failures raise ``ValueError``; else they warn.
    consistency_atol : float, default=1e-4
        Relative tolerance for the constrained-sub-case check
        ``|full.LL(nested_MAP) - nested.LL(nested_MAP)|``.

    Returns
    -------
    LRTResult
        Same dataclass as :func:`likelihood_ratio_test`.

    Notes
    -----
    Both log-likelihoods are evaluated at the posterior **MAP** particle of each
    fit (constrained space), consistent with each other. Unlike
    :func:`likelihood_ratio_test` there is no ``refine`` option — refining a
    tied model's MAP is not currently reliable, and the MAP particles already
    give a consistent pair.
    """
    _check_fitted(full)
    _check_fitted(nested)

    if full.theta_dim != nested.theta_dim:
        _violate(
            strict,
            f"theta_dim mismatch: full={full.theta_dim}, nested={nested.theta_dim}. "
            "Both fits must share the flat (n_epochs x param_length) layout.",
        )

    n_full = full.n_observations
    n_nested = nested.n_observations
    if n_full != n_nested:
        _violate(
            strict,
            f"n_observations mismatch: full={n_full}, nested={n_nested}",
        )

    k_full = full.degrees_of_freedom
    k_nested = nested.degrees_of_freedom
    df = k_full - k_nested
    if df <= 0:
        _violate(
            strict,
            f"LRT requires k_full > k_nested, got k_full={k_full}, k_nested={k_nested}",
        )

    # Nested fit's constrained MAP. For a tied model the tied (slave) columns
    # already equal their masters here (constrained-space re-tie at export), so
    # this vector is a valid point in the FULL model's coordinates.
    theta_nested = nested.map_estimate_from_particles()[0]

    ll_full = float(full.log_likelihood())                          # full @ its MAP
    ll_nested = float(full.log_likelihood(theta=theta_nested))      # full @ nested's MAP

    # Safety net replacing the same-model identity check: scoring the nested MAP
    # on the FULL model must match the nested model's OWN likelihood there, or
    # the two fits are not on the same likelihood and the statistic is meaningless.
    ll_nested_own = float(nested.log_likelihood(theta=theta_nested))
    if abs(ll_nested - ll_nested_own) > consistency_atol * (1.0 + abs(ll_nested_own)):
        _violate(
            strict,
            "nested fit is not a constrained sub-case of the full model: scoring the "
            f"nested MAP on the full model gives {ll_nested:.6f}, but the nested model's "
            f"own likelihood there is {ll_nested_own:.6f}. Both fits must share the same "
            "likelihood (same graph, data, epochs) and the nested restriction must be "
            "expressible in the full model's coordinates.",
        )

    if ll_nested > ll_full + 1e-6:
        _violate(
            strict,
            f"LL_nested ({ll_nested:.4f}) > LL_full ({ll_full:.4f}). A true nested "
            "restriction cannot exceed the full model's likelihood; check that the fits "
            "share data/graph and have converged.",
        )

    statistic = max(0.0, 2.0 * (ll_full - ll_nested))

    from scipy.stats import chi2
    p_value = float(chi2.sf(statistic, df)) if df > 0 else float("nan")

    return LRTResult(
        statistic=statistic,
        df=int(df),
        p_value=p_value,
        log_likelihood_full=ll_full,
        log_likelihood_nested=ll_nested,
        k_full=int(k_full),
        k_nested=int(k_nested),
        n=int(n_full),
    )


def compare(
    *models,
    criterion: str = "aic",
    refine: bool = False,
) -> ComparisonTable:
    """Rank fitted SVGD models by AIC or BIC.

    Parameters
    ----------
    *models : SVGD instances, or ``(name, SVGD)`` tuples
        At least two models. Anonymous SVGD inputs are auto-named
        ``model_0``, ``model_1``, …
    criterion : {'aic', 'bic'}, default='aic'
        Which information criterion to use for ranking.
    refine : bool, default=False
        If True, refine the MAP for every fit before evaluating its LL.

    Returns
    -------
    ComparisonTable
        Rows sorted ascending by criterion. ``delta`` is the difference
        from the best model; ``weight`` is the corresponding Akaike or
        Schwarz weight.

    Raises
    ------
    ValueError
        If fewer than 2 models are passed, the criterion is not
        recognized, or the models disagree on ``n_observations``.

    Examples
    --------
    Compare three competing models on the same data:

    >>> import phasic.model_selection as ms
    >>> table = ms.compare(
    ...     ("simple",   svgd_simple),
    ...     ("medium",   svgd_medium),
    ...     ("complex",  svgd_complex),
    ...     criterion='aic',
    ...     refine=True,
    ... )
    >>> print(table)
    Model comparison by AIC:
    name                          AIC          Δ     weight             LL    k      n
    ---------------------------------------------------------------------------------
    medium                   1234.567     0.0000     0.6321      -612.284    3    200
    simple                   1236.123     1.5560     0.2912      -616.062    2    200
    complex                  1239.412     4.8450     0.0767      -610.706    9    200

    Rows are sorted ascending by AIC (best on top). ``delta`` is the
    difference from the best model; ``weight`` is the Akaike weight.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")
    if len(models) < 2:
        raise ValueError(
            f"compare() requires at least 2 models, got {len(models)}"
        )

    # Normalize to (name, svgd) tuples
    normalized = []
    for i, m in enumerate(models):
        if (
            isinstance(m, tuple)
            and len(m) == 2
            and isinstance(m[0], str)
        ):
            normalized.append((m[0], m[1]))
        else:
            normalized.append((f"model_{i}", m))

    eval_fn = aic if criterion == "aic" else bic
    raw = []
    for name, svgd in normalized:
        res = eval_fn(svgd, refine=refine)
        crit_val = res.aic if criterion == "aic" else res.bic
        raw.append(
            (name, crit_val, res.log_likelihood, res.k, res.n)
        )

    # All models must share the same n_observations — AIC/BIC across
    # different sample sizes are not meaningful.
    ns = {r[4] for r in raw}
    if len(ns) > 1:
        raise ValueError(
            f"compare() requires all models to use the same number of "
            f"observations; got {sorted(ns)}. AIC/BIC comparisons across "
            f"different datasets are not meaningful."
        )

    min_crit = min(r[1] for r in raw)
    deltas = [r[1] - min_crit for r in raw]
    # exp(−Δ/2); Δ has min 0 so this is numerically safe (max term is 1)
    raw_w = [math.exp(-d / 2.0) for d in deltas]
    Z = sum(raw_w)
    weights = [w / Z for w in raw_w]

    rows = tuple(
        ComparisonRow(
            name=name,
            criterion=crit_val,
            delta=delta,
            weight=w,
            log_likelihood=ll,
            k=k,
            n=n,
        )
        for (name, crit_val, ll, k, n), delta, w in zip(raw, deltas, weights)
    )
    rows = tuple(sorted(rows, key=lambda r: r.criterion))
    return ComparisonTable(rows=rows, criterion=criterion)
