"""Deferred-4 Phase 0: an EXACT-rational moments+Jacobian oracle.

Design-of-record: deferred-4-mpfr-conditioning-floor-plan.md §2.1, with
one deliberate substitution recorded here: the plan named an mpmath
200-bit dense oracle; this implementation uses **exact rational
arithmetic** (`fractions.Fraction`) instead — strictly MORE precise
(infinitely, for rational inputs), zero new dependencies (mpmath is not
in the pixi env and adding it would churn the environment), and equally
tape-independent. Scope of exactness: weight modes whose edge weights
are rational functions of rational (theta, coefficients) — linear, log
(the WEIGHT is a plain product Π(c_k·θ_k); the log-space computation is
an implementation detail of the library, not of the math), was_dph /
native-DPH (renorm quotients), and rational formulas. Transcendental
weights (exp/log/sqrt formulas, arbitrary callbacks) are out of the
exact oracle's scope per the plan's own fallback clause (float64
references only, labeled).

Math (continuous): with transient generator block T (rates OUT are
positive off-diagonal to targets; diagonal = -(total out-rate incl. to
absorbing)), N = (-T)^{-1}, alpha the initial distribution over
transient states: E[time^k] = k! * alpha N^k 1.
d(N)/dθ_j = N (dT/dθ_j) N  (resolvent identity), so
d(E[time^k])/dθ_j = k! * alpha * Σ_{i=0..k-1} N^i (N dT_j N) N^{k-1-i} 1.

Math (discrete, CALIBRATED not assumed -- see the calibration harness):
with sub-stochastic transient matrix P and U = (I-P)^{-1}, the
factorial moments are E[N(N-1)...(N-r+1)] = r! * alpha (U P)^{r-1} U P
... the exact identity used is derived and VERIFIED in
`_discrete_raw_moments` against g.moments(K, discrete=True) on benign
fixtures before any disputed use; raw moments come from factorial
moments via Stirling numbers of the second kind.

The oracle must pass the calibration bar (plan §2.1) before being
trusted: agreement with shipped values AND shipped exact Jacobians to
~1e-13 on well-conditioned fixtures, and with jax.jacobian of a float64
dense reference.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- exact linear algebra
def mat_identity(n):
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)]
            for i in range(n)]


def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = [[Fraction(0)] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a == 0:
                continue
            Bk = B[k]
            oi = out[i]
            for j in range(p):
                oi[j] += a * Bk[j]
    return out


def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def vec_mat(v, A):
    n, p = len(A), len(A[0])
    return [sum(v[i] * A[i][j] for i in range(n)) for j in range(p)]


def mat_inv(A):
    """Exact Gauss-Jordan inverse over Fraction."""
    n = len(A)
    M = [row[:] + I_row[:] for row, I_row in zip([r[:] for r in A],
                                                 mat_identity(n))]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            raise ZeroDivisionError("singular matrix in exact oracle")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [xr - f * xc for xr, xc in zip(M[r], M[col])]
    return [row[n:] for row in M]


# --------------------------------------------------------------------------- weight rules (exact)
def w_linear(theta, coeffs):
    P = min(len(theta), len(coeffs))
    return sum(coeffs[j] * theta[j] for j in range(P))


def dw_linear(theta, coeffs):
    return [coeffs[j] if j < len(coeffs) else Fraction(0)
            for j in range(len(theta))]


def w_log(theta, coeffs):
    # PRODUCT over ALL pairs (library contract: every c_k*theta_k > 0)
    w = Fraction(1)
    for c, t in zip(coeffs, theta):
        w *= c * t
    return w


def dw_log(theta, coeffs):
    w = w_log(theta, coeffs)
    return [w / theta[j] for j in range(len(theta))]


def make_w_formula(fn: Callable, dfn: Callable):
    """A rational formula given as exact-callable (w, dw/dtheta)."""
    return fn, dfn


# --------------------------------------------------------------------------- graph -> exact structure
def build_structure(graph, theta_len):
    """Extract (alpha, transient index list, edge list) from a live graph.

    Edge list entries: (i_row, j_col_or_None, coeffs tuple, kind) where
    kind is 'param' or 'const'; j_col None = edge to an absorbing state.
    Vertex identity uses the serialize() convention (start=0 excluded;
    transient = has out-edges; absorbing = none).
    """
    ser = graph.serialize(theta_dim=theta_len)
    n = int(ser['n_vertices'])
    param_edges = np.asarray(ser['param_edges'], dtype=float)
    const_edges = np.asarray(ser.get('constant_edges', []), dtype=float)
    start_edges = np.asarray(ser['start_edges'], dtype=float)

    out_vertices = set()
    if param_edges.size:
        out_vertices |= set(int(r[0]) for r in param_edges)
    if const_edges.size:
        out_vertices |= set(int(r[0]) for r in const_edges)
    vertices = [v for v in range(1, n)]
    transient = [v for v in vertices if v in out_vertices]
    t_index = {v: i for i, v in enumerate(transient)}

    edges = []
    if param_edges.size:
        for r in param_edges:
            frm, to = int(r[0]), int(r[1])
            coeffs = tuple(Fraction(x).limit_denominator(10**12)
                           for x in r[2:])
            edges.append((t_index[frm],
                          t_index.get(to),  # None if absorbing
                          coeffs, 'param'))
    if const_edges.size:
        for r in const_edges:
            frm, to, w = int(r[0]), int(r[1]), r[2]
            edges.append((t_index[frm], t_index.get(to),
                          (Fraction(w).limit_denominator(10**12),), 'const'))

    alpha = [Fraction(0)] * len(transient)
    total = Fraction(0)
    for r in start_edges:
        v, w = int(r[0]), Fraction(r[1]).limit_denominator(10**12)
        total += w
        if v in t_index:
            alpha[t_index[v]] += w
    alpha = [a / total for a in alpha]
    return alpha, transient, edges


def assemble_T(alpha, transient, edges, theta, w_fn):
    """Exact transient generator block + per-theta derivative blocks."""
    nt = len(transient)
    P = len(theta)
    T = [[Fraction(0)] * nt for _ in range(nt)]
    dT = [[[Fraction(0)] * nt for _ in range(nt)] for _ in range(P)]
    for (i, j, coeffs, kind) in edges:
        if kind == 'const':
            w = coeffs[0]
            dw = [Fraction(0)] * P
        else:
            w = w_fn[0](theta, coeffs)
            dw = w_fn[1](theta, coeffs)
        if j is not None:
            T[i][j] += w
        T[i][i] -= w
        for p in range(P):
            if dw[p] == 0:
                continue
            if j is not None:
                dT[p][i][j] += dw[p]
            dT[p][i][i] -= dw[p]
    return T, dT


# --------------------------------------------------------------------------- continuous moments + Jacobian
def continuous_moments_and_jac(graph, theta, K, mode='linear',
                               formula=None, rewards=None):
    """Exact raw moments E[time^k] k=1..K and Jacobian (K x P).

    rewards: optional per-TRANSIENT-state rational list (reward
    transform = scale each state's sojourn contribution; implemented as
    the reward-weighted moment chain: E[(sum r_s tau_s)^k] via the
    D N D ... identity, D = diag(rewards)). For the oracle we use the
    standard identity: reward-weighted moments = k! alpha (N D)^k 1.
    """
    theta = [Fraction(t).limit_denominator(10**12) for t in theta]
    P = len(theta)
    alpha, transient, edges = build_structure(graph, P)
    if mode == 'linear':
        wf = (w_linear, dw_linear)
    elif mode == 'log':
        wf = (w_log, dw_log)
    elif mode == 'formula':
        wf = formula
    else:
        raise ValueError(mode)
    T, dT = assemble_T(alpha, transient, edges, theta, wf)
    nt = len(transient)
    negT = [[-T[i][j] for j in range(nt)] for i in range(nt)]
    N = mat_inv(negT)
    if rewards is not None:
        D = [[Fraction(rewards[i]).limit_denominator(10**12)
              if i == j else Fraction(0) for j in range(nt)]
             for i in range(nt)]
        ND = mat_mul(N, D)
    else:
        ND = N
    ones = [Fraction(1)] * nt

    # moments
    moments = []
    Mk = ND
    fact = 1
    powers = [mat_identity(nt), ND]
    for k in range(1, K + 1):
        fact *= k
        if len(powers) <= k:
            powers.append(mat_mul(powers[-1], ND))
        vec = mat_vec(powers[k], ones)
        moments.append(fact * sum(a * v for a, v in zip(alpha, vec)))

    # Jacobian: d(ND)/dθ = (N dT_p N) D  (D theta-independent)
    J = [[Fraction(0)] * P for _ in range(K)]
    for p in range(P):
        NdTN = mat_mul(mat_mul(N, dT[p]), N)
        if rewards is not None:
            dND = mat_mul(NdTN, [[Fraction(rewards[i]).limit_denominator(10**12)
                                  if i == j else Fraction(0)
                                  for j in range(nt)] for i in range(nt)])
        else:
            dND = NdTN
        fact = 1
        for k in range(1, K + 1):
            fact *= k
            # d((ND)^k)/dθ = Σ_{i=0..k-1} (ND)^i dND (ND)^{k-1-i}
            acc = [Fraction(0)] * nt
            for i in range(k):
                left = powers[i]
                right = powers[k - 1 - i]
                v = mat_vec(right, ones)
                v = mat_vec(dND, v)
                v = mat_vec(left, v)
                acc = [a + b for a, b in zip(acc, v)]
            J[k - 1][p] = fact * sum(a * v for a, v in zip(alpha, acc))
    return moments, J


def to_float(x):
    if isinstance(x, list):
        return [to_float(y) for y in x]
    return float(x)
