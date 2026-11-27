"""Domain-aware finite-difference probes (Plan A, plans-and-findings/fd-domain-steps-plan.md).

A leaf module (importing ``phasic`` here would be circular). Every
finite-difference (FD) backward in the model builders asks this module, per
theta slot, which probe points to evaluate; the site then computes TODAY's
quotient verbatim and, only where another rule can apply, the alternative
quotient, and selects between the two QUOTIENTS (never between divisors:
plan §4.7).

Per-slot domains (``param_domains``):

- ``None``          undeclared: today's probe; on a REFUSING model (one whose
                    forward rejects a negative probe, §4.4) a slot with
                    0 <= theta <= 1e-7 uses the capped positive rule (one-sided
                    [0, 1e-7] at exactly 0) instead, so the probe cannot cross
                    zero;
- ``'positive'``    (0, inf): probes theta * exp(+-u);
- ``'unit'``        (0, 1): a logit step;
- ``'unconstrained'`` plain +-1e-7 over the literal 2*eps, no clamp; on a
                    refusing model the same band rule as ``None``;
- ``'fixed'``       never probed (zero gradient).

The step is s = 3e-5 for every forward family (user decision 2026-11-27,
fd-domain-steps-findings.md §4.3). "Capped" families (forwards with jumps or
walls: Euler pdf, multi-epoch with automatic granularity, discrete) use
u = min(log1p(1e-7 / theta), s); "full" families (elimination, explicit
granularity) use u = s.

Nothing here runs unless a site has a non-today rule for some slot: on a
non-refusing model with undeclared slots ``probes`` returns the site's own
probe arrays unchanged, so the traced program is today's.
"""
from __future__ import annotations

import threading
from functools import partial
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

from .logging_config import get_logger

_log = get_logger(__name__)

EPS = 1e-7          # today's absolute FD step
S = 3e-5            # the domain-aware step (findings §4.3)
HOST_BAND = 1e-10   # joint-index host site: band below which its relative step crosses zero

POSITIVE = "positive"
UNIT = "unit"
UNCONSTRAINED = "unconstrained"
FIXED = "fixed"
DOMAINS = (POSITIVE, UNIT, UNCONSTRAINED, FIXED)

CAPPED = "capped"
FULL = "full"

# static per-slot rules
TODAY = "today"        # today's code, nothing selected
BAND = "band"          # today's probe, capped positive rule in [0, 1e-7]
PLAIN = "plain"        # +-1e-7 over the literal 2*eps, no clamp
UNC_BAND = "unc_band"  # PLAIN, capped positive rule in [0, 1e-7]
SKIP = "skip"          # fixed: not probed


# ---------------------------------------------------------------- domains
def normalize_domains(param_domains: Optional[Sequence[Any]]):
    """Validate a per-slot domain list; returns a tuple (hashable) or None."""
    if param_domains is None:
        return None
    if isinstance(param_domains, str):
        raise TypeError(
            "param_domains must be a sequence with one entry per theta slot "
            f"(None or one of {DOMAINS}), not a single string; got {param_domains!r}")
    out = tuple(None if d is None else str(d) for d in param_domains)
    bad = [d for d in out if d is not None and d not in DOMAINS]
    if bad:
        raise ValueError(
            f"param_domains entries must be None or one of {DOMAINS}; got {bad}")
    return out


def check_length(domains, n_params: int, site: str) -> None:
    """The domain list must match the traced theta length (first trace)."""
    if domains is not None and len(domains) != int(n_params):
        raise ValueError(
            f"{site}: param_domains has {len(domains)} entries but theta has "
            f"{int(n_params)} slots")


def domain_of(domains, i: int):
    return None if domains is None else domains[i]


def fixed_slots(domains):
    return set() if domains is None else {i for i, d in enumerate(domains) if d == FIXED}


# ---------------------------------------------------------------- refusal
class BuildInfo(NamedTuple):
    """Build-time facts a site needs: whether the model's forward refuses a
    negative probe regardless of rewards (§4.4), whether its weight mode is
    'log' (which also refuses exactly 0, §4.5), and the forward family's
    rule ('capped' or 'full', §4.1)."""
    refuses: bool
    log_mode: bool
    family: str


def build_info(serialized=None, *, discrete=False, family=CAPPED,
               always_refuses=False) -> BuildInfo:
    """Build-time refusal from the serialized graph: non-linear weight mode,
    or discrete (native or was_dph); ``always_refuses`` for builders that
    refuse by construction (multi-epoch, pmf_from_cpp, moments_from_graph)."""
    if family not in (CAPPED, FULL):
        raise ValueError(family)
    s = serialized or {}
    wm = s.get("weight_mode", "linear") or "linear"
    disc = bool(discrete) or bool(s.get("is_discrete", False)) or bool(s.get("was_dph", False))
    return BuildInfo(bool(always_refuses or wm != "linear" or disc), wm == "log", family)


def model_refuses(info: BuildInfo, has_rewards: bool = False) -> bool:
    """THE refusal query (§4.4): the build-time part, or called with rewards.
    The refactor (core-engine-ssot-plan.md, phase 5) replaces this body with a
    C++ query on the graph-state record."""
    return bool(info.refuses or has_rewards)


def slot_rule(domain, refuses: bool) -> str:
    """Static rule for one slot."""
    if domain == FIXED:
        return SKIP
    if domain is None:
        return BAND if refuses else TODAY
    if domain == UNCONSTRAINED:
        return UNC_BAND if refuses else PLAIN
    return domain  # POSITIVE or UNIT


def plain_divisor(rule: str) -> bool:
    """True where the base quotient divides by the literal 2*eps (no clamp)."""
    return rule in (PLAIN, UNC_BAND)


# ---------------------------------------------------------------- traced probes
def _positive_pair(t, family):
    import jax.numpy as jnp
    pos = t > 0
    ts = jnp.where(pos, t, 1.0)
    u = jnp.minimum(jnp.log1p(EPS / ts), S) if family == CAPPED else S
    up = jnp.where(pos, ts * jnp.exp(u), EPS)   # theta == 0: one-sided [0, EPS]
    dn = jnp.where(pos, ts * jnp.exp(-u), 0.0)
    return up, dn


def _unit_pair(t, family):
    import jax
    import jax.numpy as jnp
    inside = (t > 0) & (t < 1)
    ts = jnp.where(inside, t, 0.5)
    z = jnp.log(ts) - jnp.log1p(-ts)
    h = (jnp.minimum(S, EPS / (ts * (1.0 - ts))) / (1.0 + S)) if family == CAPPED else S
    return jax.nn.sigmoid(z + h), jax.nn.sigmoid(z - h)


def probes(theta, i: int, today_tp, today_tm, rule: str, info: BuildInfo):
    """Probe vectors for slot ``i``: (tp, tm, use_alt).

    ``today_tp`` / ``today_tm`` are the site's own probe vectors. ``use_alt``
    is None when only the base rule can apply (the returned arrays are then
    the inputs, or the plain +-EPS pair); otherwise a traced bool, True where
    the alternative (domain) rule's probes were selected and the site must use
    the alternative quotient over the actual probe difference."""
    import jax.numpy as jnp
    if rule == TODAY:
        return today_tp, today_tm, None
    if rule in (PLAIN, UNC_BAND):
        base_tp, base_tm = theta.at[i].add(EPS), theta.at[i].add(-EPS)
        if rule == PLAIN:
            return base_tp, base_tm, None
    else:
        base_tp, base_tm = today_tp, today_tm
    t = theta[i]
    if rule in (BAND, UNC_BAND):
        up, dn = _positive_pair(t, CAPPED)
        use = (t >= 0) & (t <= EPS)
        if info.log_mode:
            use = use & (t != 0)      # log mode refuses 0 itself: keep today's error
    elif rule == POSITIVE:
        up, dn = _positive_pair(t, info.family)
        use = t >= 0                  # theta < 0 is outside the domain: today's probe
        if info.log_mode:
            use = use & (t != 0)
    elif rule == UNIT:
        up, dn = _unit_pair(t, info.family)
        use = (t > 0) & (t < 1)       # outside (0, 1) (exposure-scaled values): today's
    else:
        raise ValueError(f"unknown FD rule {rule!r}")
    tp = jnp.where(use, theta.at[i].set(up), base_tp)
    tm = jnp.where(use, theta.at[i].set(dn), base_tm)
    return tp, tm, use


def alt_ratio(fp, fm, d):
    """(fp - fm) / d with the zero-difference guard (0 where d == 0)."""
    import jax.numpy as jnp
    ok = d != 0.0
    return jnp.where(ok, (fp - fm) / jnp.where(ok, d, 1.0), 0.0)


def flags(use, d, outs_plus, outs_minus, masks=None):
    """(zero_difference, tie) for one slot's selected alternative branch.

    A tie (§4.5): at least one entry finite in both f+ and f-, and every such
    entry equal; NaN entries (missing observations, mass-0 handoffs) are
    excluded. ``masks`` (optional, per output) restricts which entries count,
    e.g. moments only where the exact Jacobian was declined."""
    import jax.numpy as jnp
    any_f = jnp.asarray(False)
    all_eq = jnp.asarray(True)
    for k, (p, m) in enumerate(zip(outs_plus, outs_minus)):
        fin = jnp.isfinite(p) & jnp.isfinite(m)
        if masks is not None and masks[k] is not None:
            fin = fin & masks[k]
        any_f = any_f | jnp.any(fin)
        all_eq = all_eq & jnp.all(jnp.where(fin, p == m, True))
    zero = use & (d == 0.0)
    tie = use & (d != 0.0) & any_f & all_eq
    return zero, tie


# ---------------------------------------------------------------- reporting
class FDStats:
    """Per-model counters for zero probe differences and forward ties; each
    kind is logged once per model (plan §4.5: counted and logged, not silent)."""

    def __init__(self, site: str):
        self.site = site
        self.zero_difference = 0
        self.tie = 0
        self._logged = set()
        self._lock = threading.Lock()

    def add(self, n_zero: int, n_tie: int) -> None:
        with self._lock:
            self.zero_difference += n_zero
            self.tie += n_tie
            msgs = []
            if n_zero and "zero" not in self._logged:
                self._logged.add("zero")
                msgs.append(
                    f"{self.site}: zero FD gradient for a slot whose probes "
                    "collapsed onto theta (theta too large for the capped step, "
                    "above ~1.8e9); counted in model._fd_stats.zero_difference")
            if n_tie and "tie" not in self._logged:
                self._logged.add("tie")
                msgs.append(
                    f"{self.site}: zero FD gradient (forward tie; may be a genuine "
                    "zero) for a slot whose two probes gave identical outputs; "
                    "counted in model._fd_stats.tie")
        for m in msgs:
            _log.warning(m)


def _host_report(stats: FDStats, zeros, ties):
    stats.add(int(np.sum(np.asarray(zeros))), int(np.sum(np.asarray(ties))))


def report(stats: Optional[FDStats], zeros, ties) -> None:
    """One host callback per backward with every slot's flags (no-op when no
    slot had an alternative branch)."""
    if stats is None or not zeros:
        return
    import jax
    import jax.numpy as jnp
    jax.debug.callback(partial(_host_report, stats), jnp.stack(zeros), jnp.stack(ties))


# ---------------------------------------------------------------- test seam
_PROBE_HOOK = None


def set_probe_hook(fn) -> None:
    """TEST ONLY: ``fn(site, i, tp, tm)`` receives every probe vector pair a
    loop passes to its forward (traced loops via a debug callback, fed when
    the backward is traced, so install it before building the model)."""
    global _PROBE_HOOK
    _PROBE_HOOK = fn


def record(site: str, i: int, tp, tm) -> None:
    if _PROBE_HOOK is None:
        return
    import jax
    jax.debug.callback(partial(_PROBE_HOOK, site, int(i)), tp, tm)


def record_np(site: str, i: int, tp, tm) -> None:
    if _PROBE_HOOK is not None:
        _PROBE_HOOK(site, int(i), np.asarray(tp), np.asarray(tm))


# ---------------------------------------------------------------- host (numpy) variant
def host_probes_np(th, k: int, today_h: float, rule: str, info: BuildInfo):
    """Joint-index host fallback (``exact_grad_decline='fd'``): numpy probes.

    Returns (tp, tm, divisor, used_alt). Undeclared and 'unconstrained' slots
    keep today's RELATIVE step (step max(|theta|*1e-6, 1e-10), divisor 2h,
    §5), except on refusing models for 0 <= theta < 1e-10, where they use the
    capped positive rule; 'positive' / 'unit' use the full rule (elimination
    family)."""
    t = float(th[k])
    tp = th.copy(); tm = th.copy()
    if rule in (TODAY, BAND, PLAIN, UNC_BAND):
        band = rule in (BAND, UNC_BAND) and 0.0 <= t < HOST_BAND and not (info.log_mode and t == 0.0)
        if not band:
            tp[k] += today_h; tm[k] -= today_h
            return tp, tm, 2.0 * today_h, False
        up, dn = _positive_pair_np(t, CAPPED)
    elif rule == POSITIVE:
        if t < 0 or (info.log_mode and t == 0.0):
            tp[k] += today_h; tm[k] -= today_h
            return tp, tm, 2.0 * today_h, False
        up, dn = _positive_pair_np(t, info.family)
    elif rule == UNIT:
        if not (0.0 < t < 1.0):
            tp[k] += today_h; tm[k] -= today_h
            return tp, tm, 2.0 * today_h, False
        z = np.log(t) - np.log1p(-t)
        h = (min(S, EPS / (t * (1.0 - t))) / (1.0 + S)) if info.family == CAPPED else S
        up, dn = 1.0 / (1.0 + np.exp(-(z + h))), 1.0 / (1.0 + np.exp(-(z - h)))
    else:
        raise ValueError(f"unknown FD rule {rule!r}")
    tp[k] = up; tm[k] = dn
    return tp, tm, tp[k] - tm[k], True


def _positive_pair_np(t, family):
    if t == 0.0:
        return EPS, 0.0
    u = min(np.log1p(EPS / t), S) if family == CAPPED else S
    return t * np.exp(u), t * np.exp(-u)
