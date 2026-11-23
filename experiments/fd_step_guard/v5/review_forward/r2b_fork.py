"""R2b: plan rule (relative to accumulated absorbed sum) vs componentwise rule on a vector read-out
whose components are separately meaningful (epoch-path shape: 'overflow' absorbs, an outcome trap retains mass)."""
import numpy as np
from scipy.stats import poisson
from scipy.linalg import expm
import mpmath as mp
M = 30                      # A=0 -> absorbing (rate 1) ; A -> c1 -> ... -> c_M -> T (rate 1 each) ; T retains (rate 0)
n = M + 3; A, T, ABS = 0, M + 1, M + 2
Q = np.zeros((n, n))
Q[A, ABS] = 1; Q[A, 1] = 1
for i in range(1, M + 1): Q[i, i + 1] = 1
Q[np.diag_indices(n)] = -Q.sum(1)
G = 1.02 * 2; P = np.eye(n) + Q / G
def run(t, rule, eps=1e-16):
    lam = G * t; x = np.zeros(n); x[A] = 1; new = np.zeros(n); k = 0; absorbed = 0
    while True:
        w = poisson.pmf(k, lam); Tl = poisson.sf(k, lam); new += w * x
        y = x @ P; stable = np.array_equal(y > 0, x > 0); x = y; k += 1
        if rule == 'plan' and new[ABS] > 0 and Tl <= eps * new[ABS]: break
        if rule == 'comp' and stable and Tl <= eps * new[new > 0].min(): break
        if Tl == 0 and k > lam: break
    return new, k
for t in (0.1, 1.0, 5.0):
    ex = expm(Q * t)[A].copy(); ex_expm_T = ex[T]
    # expm is NOT componentwise accurate for tiny entries: closed form 0.5 * P(Exp(2) + Gamma(M,1) <= t) via mpmath
    mp.mp.dps = 250
    ex[T] = float(mp.quad(lambda u: mp.exp(-2 * u) * mp.gammainc(M, 0, t - u, regularized=True), [0, t]))
    print(f't={t}: scipy expm gives {ex_expm_T:.3e} for the trap mass; closed form {ex[T]:.3e}')
    for rule in ('plan', 'comp'):
        s, k = run(t, rule)
        print(f't={t}: rule={rule:4s} steps {k:4d}  outcome-trap mass {s[T]:.3e} (exact {ex[T]:.3e}, rel err {abs(s[T]-ex[T])/ex[T]:.1e})  absorbed rel err {abs(s[ABS]-ex[ABS])/ex[ABS]:.1e}')
print('steps to drive the Poisson tail to a given level:')
for lam in (0.006, 0.1, 5, 1e3, 1e5):
    out = []
    for lvl in (1e-16, 1e-100, 1e-300):
        k = int(lam)
        while poisson.sf(k, lam) > lvl: k += 1 if lam < 100 else max(1, int(0.01 * lam ** 0.5))
        out.append(k + 1)
    print(f'  Lambda={lam:g}: 1e-16 -> {out[0]}, 1e-100 -> {out[1]}, 1e-300 -> {out[2]}')
