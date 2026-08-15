"""Deferred-4 Phase 0: the broadened conditioning sweep (design-of-record
deferred-4-mpfr-conditioning-floor-plan.md §2; oracle calibrated by
dr_d4_oracle_calibration.py at 1e-16-class before this runs).

Per swept point (fixture × mode × theta), with K=1..3 in one call:
  (a) exact returned & CORRECT (vs the exact-rational oracle)
  (b) DECLINED -> FD (the INFO line is captured -- classification (b) is
      unobservable at the default WARNING level, plan §2.3)
  (c) exact returned & WRONG beyond tolerance  == GAP  (the finding)
plus: FD's own error at the same point (context: is decline-to-FD
lossy), and the existing gate's condition number recovered by bisecting
PHASIC_CONDITION_THRESHOLD (the gate reads env per call).

Protocol requirements honored (plan §2.3): HAVE_MPFR verified up front
(a non-MPFR build's gate returns 0 unconditionally -- a CLEAN verdict
there is meaningless); PHASIC_FORCE_MPFR / PHASIC_CONDITION_THRESHOLD
unset except where the bisection sets the latter deliberately; logger
at INFO for the decline capture.

GAP tolerance: rel 1e-6 against the exact oracle (far above the exact
path's healthy 1e-10-class error, far below FD's mixed-scale 1e-1-class
failures) -- points between 1e-8 and 1e-6 are recorded as DEGRADED, not
GAP.
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
from dr_d4_exact_oracle import (build_structure,  # noqa: E402
                                continuous_moments_and_jac, to_float)

import phasic  # noqa: E402
from phasic import Graph, set_log_level  # noqa: E402

set_log_level("WARNING")
K = 3
GAP_TOL = 1e-6
DEGRADED_TOL = 1e-8

for var in ("PHASIC_FORCE_MPFR", "PHASIC_CONDITION_THRESHOLD"):
    os.environ.pop(var, None)


# --------------------------------------------------------------------------- fixtures
def chain2():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    return g


def branchy():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    v[0].add_edge(v[1], [1.0, 0.0])
    v[0].add_edge(v[2], [2.0, 0.5])
    v[1].add_edge(v[3], [1.5, 0.25])
    v[2].add_edge(v[3], [0.5, 1.0])
    v[3].add_edge(va, [1.0, 1.0])
    return g


def cyclic():
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    absb = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0, 0.0])
    b.add_edge(a, [0.0, 0.5])
    b.add_edge(absb, [1.0, 0.25])
    return g


def coalescent3():
    """Kingman-class block-counting chain for n=3: states 3->2->1
    lineages, rates binom(k,2)*theta0 with theta1 a scale nuisance on
    the last step (2 params for the mixed-scale axis)."""
    g = Graph(1)
    s = g.starting_vertex()
    k3 = g.find_or_create_vertex([3])
    k2 = g.find_or_create_vertex([2])
    k1 = g.find_or_create_vertex([1])
    s.add_edge(k3, 1.0)
    k3.add_edge(k2, [3.0, 0.0])
    k2.add_edge(k1, [1.0, 1.0])
    return g


def log2():
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_mode = 'log'
    return g


FIXTURES = {
    'chain2': (chain2, 'linear'),
    'branchy': (branchy, 'linear'),
    'cyclic': (cyclic, 'linear'),
    'coalescent3': (coalescent3, 'linear'),
    'log2': (log2, 'log'),
}

# theta paths crossing the corruption regime: component 1 sweeps 1 -> 1e-14
RATIOS = [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
# plus the inverted direction (component 0 large)
INV = [1e2, 1e4, 1e6, 1e8]


class _H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, r):
        self.msgs.append(r.getMessage())


def call_exact(g, mode, theta, rewards=None):
    """One shipped exact-Jacobian call with INFO capture."""
    lg = logging.getLogger('phasic')
    h = _H(); h.setLevel(logging.INFO)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        kw = {} if rewards is None else {'rewards': list(rewards)}
        if mode == 'linear':
            g.update_weights(list(theta))
            J = np.asarray(g._moments_grad_theta(K, **kw))
        else:
            g.update_weights(list(theta), log=True)
            J = np.asarray(g._moments_grad_theta_log(K, list(theta), **kw))
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    return J, h.msgs


def fd_jacobian(mk, mode, theta, rewards=None, rel=1e-7):
    """Central-diff of the primal (the shipped FD fallback's method
    class: absolute-ish step at tiny components, mirroring the defect)."""
    th = np.asarray(theta, float)
    J = np.empty((K, th.size))
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-10)
        for sgn in (1, -1):
            tp = th.copy(); tp[j] += sgn * h
            g = mk()
            try:
                if mode == 'linear':
                    g.update_weights(tp.tolist())
                else:
                    g.update_weights(tp.tolist(), log=True)
                m = np.asarray(g.moments(K) if rewards is None
                               else g.moments(K, list(rewards)))
            except Exception:
                m = np.full(K, np.nan)
            if sgn == 1:
                mp = m
            else:
                mm = m
        J[:, j] = (mp - mm) / (2 * h)
    return J


def bisect_condition(g, mode, theta):
    """Recover the gate's condition number by bisecting the threshold
    env var (the gate declines iff cond > threshold)."""
    def declines_at(log10_thresh):
        os.environ['PHASIC_CONDITION_THRESHOLD'] = f"1e{log10_thresh:.6f}"
        try:
            J, _ = call_exact(g, mode, theta)
            return J.size == 0
        finally:
            os.environ.pop('PHASIC_CONDITION_THRESHOLD', None)
    lo, hi = -2.0, 300.0
    if not declines_at(lo):
        return 10.0 ** lo  # cond below 1e-2 (well-conditioned floor)
    if declines_at(hi):
        return float('inf')
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if declines_at(mid):
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


# --------------------------------------------------------------------------- protocol: HAVE_MPFR
os.environ['PHASIC_FORCE_MPFR'] = '1'
g = chain2()
Jf, _ = call_exact(g, 'linear', [2.0, 1.0])
have_mpfr = (Jf.size == 0)
os.environ.pop('PHASIC_FORCE_MPFR', None)
print(f"HAVE_MPFR (force-probe): {have_mpfr}")
if not have_mpfr:
    print("!! non-MPFR build: the gate is inert; a CLEAN verdict here is "
          "meaningless (plan §2.3). Sweep continues but the report must "
          "carry this flag.")

# --------------------------------------------------------------------------- the sweep
rows = []
for fname, (mk, mode) in FIXTURES.items():
    thetas = ([[1.0, r] for r in RATIOS] + [[r, 1.0] for r in INV])
    for theta in thetas:
        g = mk()
        # oracle (exact rational; may be expensive at extreme ratios --
        # fixtures are tiny)
        try:
            m_or, J_or = continuous_moments_and_jac(g, theta, K, mode=mode)
            J_oracle = np.asarray(to_float(J_or))
        except ZeroDivisionError:
            rows.append(dict(fixture=fname, mode=mode, theta=theta,
                             cls='oracle-singular'))
            continue
        g2 = mk()
        try:
            J_ship, msgs = call_exact(g2, mode, theta)
        except Exception as exc:
            rows.append(dict(fixture=fname, mode=mode, theta=theta,
                             cls='primal-raises', exc=type(exc).__name__))
            continue
        declined = (J_ship.size == 0)
        decline_logged = any('finite differences' in m for m in msgs)
        J_fd = fd_jacobian(mk, mode, theta)
        denom = max(float(np.max(np.abs(J_oracle))), 1e-300)
        fd_err = float(np.max(np.abs(J_fd - J_oracle))) / denom \
            if np.all(np.isfinite(J_fd)) else float('inf')
        if declined:
            cls = 'declined'
            exact_err = None
        else:
            exact_err = float(np.max(np.abs(
                J_ship.reshape(K, len(theta)) - J_oracle))) / denom
            if exact_err > GAP_TOL:
                cls = 'GAP'
            elif exact_err > DEGRADED_TOL:
                cls = 'degraded'
            else:
                cls = 'correct'
        g3 = mk()
        cond = bisect_condition(g3, mode, theta) if have_mpfr else None
        rows.append(dict(fixture=fname, mode=mode, theta=theta, cls=cls,
                         exact_err=exact_err, fd_err=fd_err,
                         decline_logged=decline_logged, gate_cond=cond))
        tag = f"{fname}/{mode} theta={theta}"
        print(f"  {tag}: {cls}"
              + (f" exact_err={exact_err:.2e}" if exact_err is not None else "")
              + f" fd_err={fd_err:.2e}"
              + (f" gate_cond={cond:.2e}" if cond not in (None,) else ""))

# rewards slice (linear fixtures, one mixed-scale point each)
print("== rewards slice ==")
for fname in ('branchy', 'cyclic'):
    mk, mode = FIXTURES[fname]
    g = mk()
    n = g.vertices_length()
    rw_full = ([1.0, 0.5, 2.0, 1.0, 0.25, 1.0] + [1.0] * n)[:n]
    _, transient, _ = build_structure(g, 2)
    rw_t = [rw_full[v] for v in transient]
    for theta in ([1.0, 1e-8], [1.0, 1.0]):
        m_or, J_or = continuous_moments_and_jac(g, theta, K, mode='linear',
                                                rewards=rw_t)
        J_oracle = np.asarray(to_float(J_or))
        g2 = mk()
        J_ship, msgs = call_exact(g2, 'linear', theta, rewards=rw_full)
        denom = max(float(np.max(np.abs(J_oracle))), 1e-300)
        if J_ship.size == 0:
            cls = 'declined'
            exact_err = None
        else:
            exact_err = float(np.max(np.abs(
                J_ship.reshape(K, 2) - J_oracle))) / denom
            cls = ('GAP' if exact_err > GAP_TOL else
                   'degraded' if exact_err > DEGRADED_TOL else 'correct')
        J_fd = fd_jacobian(mk, 'linear', theta, rewards=rw_full)
        fd_err = float(np.max(np.abs(J_fd - J_oracle))) / denom \
            if np.all(np.isfinite(J_fd)) else float('inf')
        rows.append(dict(fixture=fname + "+rewards", mode='linear',
                         theta=theta, cls=cls, exact_err=exact_err,
                         fd_err=fd_err))
        print(f"  {fname}+rewards theta={theta}: {cls}"
              + (f" exact_err={exact_err:.2e}" if exact_err is not None else "")
              + f" fd_err={fd_err:.2e}")

# --------------------------------------------------------------------------- summary
n_gap = sum(1 for r in rows if r['cls'] == 'GAP')
n_deg = sum(1 for r in rows if r['cls'] == 'degraded')
n_dec = sum(1 for r in rows if r['cls'] == 'declined')
n_ok = sum(1 for r in rows if r['cls'] == 'correct')
unlogged = [r for r in rows if r['cls'] == 'declined'
            and not r.get('decline_logged', True)]
print(f"\nSUMMARY: {len(rows)} points -- correct={n_ok} declined={n_dec} "
      f"degraded={n_deg} GAP={n_gap}; unlogged declines={len(unlogged)}; "
      f"HAVE_MPFR={have_mpfr}")
with open("/private/tmp/claude-501/-Users-kmt-phasic/"
          "9a0e1098-c946-42b0-bf45-e703e9a4c09e/scratchpad/"
          "d4_sweep_rows.json", "w") as f:
    json.dump(rows, f, indent=1, default=str)
print("rows dumped")
sys.exit(0)
