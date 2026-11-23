"""R3: Poisson read-out at large Lambda with CLOSED-FORM iterates (isolates weight error).
hypoexp(l1, l2), uniformized at G = c*l1.  Absorbed mass in step k (k>=2):
m_k = p1 p2 (b^(k-1) - a^(k-1)) / (b - a),  a = 1-p1, b = 1-p2, p_i = l_i / G."""
import math, time
import numpy as np
from scipy.special import gammaln

def exact_logf(l1, l2, t):
    return math.log(l1 * l2 / (l1 - l2)) - l2 * t + math.log1p(-math.exp(-(l1 - l2) * t))

def absorbed(l1, l2, G, n):            # m[k] for k = 1..n+1  (index 0 -> step 1)
    p1, p2 = l1 / G, l2 / G
    k = np.arange(1, n + 2, dtype=float)
    la, lb = math.log1p(-p1), math.log1p(-p2)
    return p1 * p2 * (np.exp((k - 1) * lb) - np.exp((k - 1) * la)) / (p1 - p2) * 1.0   # (b-a) = p1-p2

def w_lgamma(lam, n):
    k = np.arange(n + 1)
    return np.exp(k * math.log(lam) - lam - gammaln(k + 1))

def w_recur(lam, n):                   # recursion from the mode, normalised (Fox-Glynn idea, no left cut)
    m = int(lam); k = np.arange(n + 1, dtype=float)
    w = np.empty(n + 1); w[m] = 1.0
    if m < n: w[m + 1:] = np.cumprod(lam / k[m + 1:])
    if m > 0: w[:m] = np.cumprod((k[1:m + 1] / lam)[::-1])[::-1]
    return w / w.sum()

def logf(l1, l2, t, wfun, c=1.02):
    G = c * l1; lam = G * t; n = int(lam + 10 * math.sqrt(lam + 1) + 30)
    return math.log(G * np.dot(wfun(lam, n), absorbed(l1, l2, G, n))), n

print('columns: Lambda | steps | rel err of f (lgamma, recursion) | |FD - exact| of dlogf/dlog(l1) and dlogf/dlog(l2) at s=1e-3, 1e-4 (lgamma | recursion)')
for lam_target in (1e2, 1e4, 1e5, 1e6, 1e7):
    l1 = 1.0; t = lam_target; l2 = 0.7 / t          # slow rate with l2*t = 0.7: mixed scale, mass still unabsorbed
    ex = exact_logf(l1, l2, t)
    out = []
    h = 1e-6
    d1 = (exact_logf(l1 * (1 + h), l2, t) - exact_logf(l1 * (1 - h), l2, t)) / (2 * h)   # tiny; fine as reference to ~1e-9
    d2 = 1 + l2 / (l1 - l2) - l2 * t + (-(-t * l2) * 0)                                      # analytic below
    e = math.exp(-(l1 - l2) * t)
    d2 = 1 + l2 / (l1 - l2) - l2 * t + (l2 * t * e / (1 - e) if e > 0 else 0.0) * (-1) * (-1) * 0 + (-(l2 * t) * e / (1 - e) if e > 0 else 0.0) * 0
    d1 = 1 - l1 / (l1 - l2) + (l1 * t * e / (1 - e) if e > 0 else 0.0)
    for wf in (w_lgamma, w_recur):
        t0 = time.time(); v, n = logf(l1, l2, t, wf)
        errs = []
        for s in (1e-3, 1e-4):
            g1 = (logf(l1 * math.exp(s), l2, t, wf)[0] - logf(l1 * math.exp(-s), l2, t, wf)[0]) / (2 * s)
            g2 = (logf(l1, l2 * math.exp(s), t, wf)[0] - logf(l1, l2 * math.exp(-s), t, wf)[0]) / (2 * s)
            errs.append(f'{abs(g1 - d1):.1e}/{abs(g2 - d2):.1e}')
        out.append((abs(math.expm1(v - ex)), errs))
    print(f'  Lambda={1.02*lam_target:8.3g} n={n:9d}: f relerr lgamma {out[0][0]:.1e} recur {out[1][0]:.1e} | FD err(l1/l2) s=1e-3,1e-4: lgamma {out[0][1]} | recur {out[1][1]}', flush=True)

print('## independence of G numerically: hypo(1,3), t=0.7, G = c * max rate')
l1, l2, t = 3.0, 1.0, 0.7
for c in (1.0001, 1.02, 2, 1e2, 1e4, 1e6):
    for wf in (w_lgamma, w_recur):
        v, n = logf(l1, l2, t, wf, c=c)
        print(f'  c={c:8g} {wf.__name__:9s}: rel err {abs(math.expm1(v - exact_logf(l1, l2, t))):.1e} steps {n}')
print('## Lambda -> 0')
for t in (1e-3, 1e-8, 1e-100, 1e-300):
    v, n = logf(3.0, 1.0, t, w_lgamma)
    print(f'  t={t:g}: log f {v:.12g} exact {exact_logf(3.0, 1.0, t):.12g} steps {n}')
