"""E2: accuracy of candidate FD step rules on the real forwards (emulated
outside the custom_vjp: same forward vector, same cotangent 1/p = d sum log p).
Reference: exact gradient where one exists; else the better-self-agreeing of
(a) Richardson-extrapolated central FD with h = theta*{0.1,0.05} and
(b) Richardson one-sided 2nd-order FD with h = {1e-5, 5e-6} (positive probes)."""
import sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
import sites as S
from phasic import Graph

EPS = 1e-7
def vec_joint_index():
    J, oi = S.joint()
    m = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi)
    vi = jnp.asarray(oi)
    me = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi, exact_grad=True,
                                          exact_grad_decline='fd')
    exact = lambda th: jax.grad(lambda t: jnp.sum(jnp.log(me(t, vi)[0])))(th)
    return (lambda th: np.asarray(m(th, vi)[0])), exact
def vec_daisy():
    J, oi = S.joint()
    model, *_ = J._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=S.EPOCHS,
                                          user_prior=None, user_fixed=None)
    me, *_ = J._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=S.EPOCHS,
                                       user_prior=None, user_fixed=None, exact_final_grad=True)
    def P(th):
        o = model(th); return np.asarray(o[0] if isinstance(o, tuple) else o)
    def exact(th):   # exact for final-epoch slots only (2,3)
        def f(t):
            o = me(t); return jnp.sum(jnp.log(o[0] if isinstance(o, tuple) else o))
        return jax.grad(f)(th)
    return P, exact
def vec_pmf():
    m = Graph.pmf_from_graph(S.island(), theta_dim=2)
    return (lambda th: np.asarray(m(th, S.TIMES))), None

def safe(P, th):
    try:
        v = P(th)
        return v if np.all(np.isfinite(v)) else None
    except Exception:
        return None

def fd(P, th, i, cot, hm, hp):
    pp, pm = safe(P, th.at[i].add(hp)), safe(P, th.at[i].add(-hm))
    if pp is None or pm is None: return np.nan
    return float(np.sum(cot * (pp - pm)) / (hp + hm))

def rules(P, th, i, cot):
    t = float(th[i]); out = {}
    out['cur'] = fd(P, th, i, cot, EPS, EPS)
    tm = max(t - EPS, 1e-9); out['clamp'] = fd(P, th, i, cot, t - tm, EPS)
    for c in (0.5, 0.1, 1e-2, 1e-3):
        h = min(EPS, c * t); out[f'c={c:g}'] = fd(P, th, i, cot, h, h)
    return out

def reference(P, th, i, cot):
    t = float(th[i])
    a1, a2 = fd(P, th, i, cot, t*0.1, t*0.1), fd(P, th, i, cot, t*0.05, t*0.05)
    ra = (4*a2 - a1)/3
    def os2(h):
        p0, p1, p2 = P(th), safe(P, th.at[i].add(h)), safe(P, th.at[i].add(2*h))
        return float(np.sum(cot*(-3*p0 + 4*p1 - p2))/(2*h))
    b1, b2 = os2(1e-5), os2(5e-6)
    rb = (4*b2 - b1)/3
    da, db = abs(a1-a2)/abs(ra), abs(b1-b2)/abs(rb)
    return (ra, 'centralR', da) if da < db else (rb, 'onesideR', db)

CASES = [('joint_index', vec_joint_index, [2.0, None], 1, 'mu'),
         ('joint_index', vec_joint_index, [None, 0.5], 0, 'coal'),
         ('daisy',       vec_daisy, [None, 0.5, 2.0, 0.5], 0, 'epoch1 coal'),
         ('daisy',       vec_daisy, [2.0, 0.5, None, 0.5], 2, 'final coal'),
         ('daisy',       vec_daisy, [2.0, None, 2.0, 0.5], 1, 'epoch1 mu'),
         ('daisy',       vec_daisy, [2.0, 0.5, 2.0, None], 3, 'final mu'),
         ('pmf_from_graph', vec_pmf, [1.0, None], 1, 'mig'),
         ('pmf_from_graph', vec_pmf, [None, 0.5], 0, 'coal')]
GRID = [1e-9, 1e-8, 5e-8, 9.99e-8, 1.001e-7, 2e-7, 1e-6, 1e-5, 1e-4, 1e-3]
for site, mk, base, i, lab in CASES:
    P, exact = mk()
    print(f'\n### {site}: {lab} small (slot {i})', flush=True)
    names = None
    for v in GRID:
        th = jnp.array([v if b is None else b for b in base], dtype=jnp.float64)
        cot = 1.0 / P(th)
        r = rules(P, th, i, cot)
        ex = None
        if exact is not None and (site != 'daisy' or i >= 2):
            try: ex = float(exact(th)[i])
            except Exception as e: ex = None
        rr, how, agree = reference(P, th, i, cot)
        ref = ex if ex is not None and np.isfinite(ex) else rr
        how = 'exact' if ref is ex else f'{how}({agree:.0e})'
        if names is None:
            names = list(r); print('  theta      ref[how]                ' + ''.join(f'{n:>11}' for n in names))
        errs = ''.join(f'{(abs(r[n]-ref)/abs(ref) if np.isfinite(r[n]) else np.nan):11.1e}' for n in names)
        extra = f'  (exact vs numref {abs(ex-rr)/abs(ex):.1e})' if ex is not None and np.isfinite(ex) else ''
        print(f'  {v:8.3g}  {ref: .6e}[{how:>16}] {errs}{extra}', flush=True)
