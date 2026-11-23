"""Plan test 7 (bit-identity of free slots) and fixed-at-0 on a real model, using the runtime prototype of the fix."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from r2_lib import *
print('## C. free-slot particles, unmodified vs prototype fix (default preconditioner, real model, 15 iterations, 8 particles)')
for v in (2.0, 50.0, 1.0, 0.5, 0.3, 0.7, 5.0, 1e-4, 1e-8):
    try:
        _, p0 = fit(v, False); _, p1 = fit(v, True)
        d = np.abs(p0[:, 0] - p1[:, 0])
        print(f'  v={v:g}: bit-identical={bool(np.all(p0[:, 0] == p1[:, 0]))}  max abs diff={d.max():.3e}  max rel={np.max(d / np.abs(p0[:, 0])):.3e}  fixed col before={sorted(set(p0[:, 1].tolist()))} after={sorted(set(p1[:, 1].tolist()))}')
    except Exception as e:
        print(f'  v={v:g}: RAISED {type(e).__name__}: {str(e)[:300]}')
print('## C1b. same with a full-vector callable prior (non-separable) -- does the placeholder phi leak into free slots?')
def corr_prior(phi): return -0.5 * (phi[0] - 1.0) ** 2 - 0.4 * phi[0] * phi[1]
for v in (2.0,):
    _, p0 = fit(v, False, prior=corr_prior); _, p1 = fit(v, True, prior=corr_prior)
    d = np.abs(p0[:, 0] - p1[:, 0])
    print(f'  v={v:g}: bit-identical={bool(np.all(p0[:, 0] == p1[:, 0]))}  max abs diff={d.max():.3e}  means {p0[:,0].mean():.5f} -> {p1[:,0].mean():.5f}')
print('## C3. fixed at exactly 0 with the prototype fix, real model; reference = reduced 1-parameter model')
for v in (0.0, 1e-12):
    try:
        s0, p = fit(v, True)
        print(f'  prototype fixed=(1,{v:g}): free', np.round(p[:, 0], 6), ' fixed col', sorted(set(p[:, 1].tolist())), 'finite:', bool(np.all(np.isfinite(p))))
        print('    log_likelihood():', s0.log_likelihood(), ' refine:', s0.log_likelihood(refine=True))
        print('    theta_mean attr:', np.asarray(s0.theta_mean), ' map_estimate:', s0.map_estimate_from_particles()[0])
    except Exception as e:
        import traceback; print(f'  prototype fixed=(1,{v:g}) RAISED {type(e).__name__}: {str(e)[:400]}')
sr, pr = fit(None, False, model=red_model(), theta_dim=1, prior=[GaussPrior(ci=[0.5, 5], lower=0.0)])
print('  reduced model            : free', np.round(pr[:, 0], 6)); print('    log_likelihood():', sr.log_likelihood())
