"""E1: what does each theta-space FD site return when a free component is
below the 1e-7 step?  Reference = central FD on the same forward with
h = theta_i*1e-3 and theta_i*1e-4 (Richardson-agreement reported)."""
import sys, time, traceback
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
import sites as S

def ref_grad(f, th, i):
    out = []
    for r in (1e-3, 1e-4):
        h = th[i] * r
        out.append((float(f(th.at[i].add(h))) - float(f(th.at[i].add(-h)))) / (2 * h))
    return out

CASES = {
  'joint_index_baked (mu small)':  (lambda: S.site_joint_index(True),  [2.0, None], 1),
  'joint_index_baked (coal small)':(lambda: S.site_joint_index(True),  [None, 0.5], 0),
  'joint_index_unbaked (mu small)':(lambda: S.site_joint_index(False), [2.0, None], 1),
  'daisy (epoch-1 coal small)':    (lambda: S.site_daisy(),            [None, 0.5, 2.0, 0.5], 0),
  'daisy (final coal small)':      (lambda: S.site_daisy(),            [2.0, 0.5, None, 0.5], 2),
  'daisy exp (epoch-1 coal small)':(lambda: S.site_daisy(exposure=True), [None, 0.5, 2.0, 0.5], 0),
  'daisy efg (epoch-1 coal small)':(lambda: S.site_daisy(efg=True),  [None, 0.5, 2.0, 0.5], 0),
  'daisy efg (final coal small)':  (lambda: S.site_daisy(efg=True),  [2.0, 0.5, None, 0.5], 2),
  'pmf_from_graph (mig small)':    (S.site_pmf_from_graph,             [1.0, None], 1),
  'pmf_and_moments pmf (mig small)':(S.site_pmf_and_moments,           [1.0, None], 1),
  'moments_from_graph (mig small)':(S.site_moments_from_graph,         [1.0, None], 1),
  'reward_visit_prob (mig small)': (S.site_rvp,                        [1.0, None], 1),
}
SMALL = [1e-5, 1e-6, 2e-7, 1.5e-7, 1e-7, 5e-8, 1e-8, 1e-9]
only = sys.argv[1:]
for name, (build, base, i) in CASES.items():
    if only and not any(o in name for o in only):
        continue
    print(f'\n### {name}', flush=True)
    try:
        f = build()
    except Exception as e:
        print('  BUILD FAILED', type(e).__name__, str(e)[:200]); continue
    for v in SMALL:
        th = jnp.array([v if b is None else b for b in base], dtype=jnp.float64)
        t0 = time.time()
        try:
            g = float(jax.grad(f)(th)[i]); gerr = None
        except Exception as e:
            g = np.nan; gerr = (type(e).__name__ + ': ' + str(e).split('\n')[0])[:140]
        try:
            r1, r2 = ref_grad(f, th, i)
        except Exception as e:
            r1 = r2 = np.nan
        try:
            fneg = float(f(th.at[i].add(-1e-7)))
        except Exception as e:
            fneg = type(e).__name__
        rel = abs(g - r2) / max(abs(r2), 1e-300) if np.isfinite(g) else np.nan
        print(f'  theta_i={v:8.1e}  fd={g: .6e}  ref={r2: .6e} (ref agree {abs(r1-r2)/max(abs(r2),1e-300):.1e})  '
              f'relerr={rel:.2e}  f(theta-1e-7)={fneg}  {gerr or ""}  [{time.time()-t0:.1f}s]', flush=True)
