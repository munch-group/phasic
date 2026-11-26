"""Shared helpers for review3: prototype domain-aware FD rules OUTSIDE the package."""
import numpy as np, jax, jax.numpy as jnp

def fd_today(F, th, i, cot):
    """Mimic master's backward: theta +/- 1e-7 absolute, linear in outputs."""
    tp = th.copy(); tp[i] += 1e-7
    tm = th.copy(); tm[i] -= 1e-7
    return float(np.sum(cot * (F(tp) - F(tm)) / 2e-7))

def fd_log(F, th, i, cot, s, cap=None):
    """Plan rule for a positive domain: probes theta*e^{+-s'}, divided by the
    ACTUAL probe difference. cap: if given, s' = min(cap, s*theta)/theta."""
    t = th[i]
    sp = s if cap is None else min(cap, s * t) / t
    tp = th.copy(); tp[i] = t * np.exp(sp)
    tm = th.copy(); tm[i] = t * np.exp(-sp)
    return float(np.sum(cot * (F(tp) - F(tm)) / (tp[i] - tm[i])))

def richardson_log(F, th, i, cot, s=1e-2):
    """High-accuracy reference: log-coordinate central differences at s, s/2,
    s/4 with two Richardson steps (O(s^6)); returns (value, spread)."""
    def d(ss):
        t = th[i]
        tp = th.copy(); tp[i] = t * np.exp(ss)
        tm = th.copy(); tm[i] = t * np.exp(-ss)
        return float(np.sum(cot * (F(tp) - F(tm)))) / (2 * ss) / t   # dF/dtheta, central in u
    a, b, c = d(s), d(s / 2), d(s / 4)
    r1 = (4 * b - a) / 3; r2 = (4 * c - b) / 3
    r = (16 * r2 - r1) / 15
    return r, abs(r2 - r1) / max(abs(r), 1e-300)

def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)
