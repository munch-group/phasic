"""Review experiments (run on prototype; PHASIC_FD_RULE=old|new via env).
R2: large-k power law on C3 (pmf_and_moments) coalescent pdf, n in {5,12,20}.
R4: daisy NON-final slots (smooth through zero) accuracy, one-sided Richardson ref.
R3: mixed vmap batch: normal particles bitwise identical to an all-normal batch.
"""
import sys, os, random, logging
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
logging.disable(logging.WARNING)
from test_notebook_model_recovery import coalescent_graph
import sites as S
from phasic import Graph
rule = os.environ.get('PHASIC_FD_RULE', 'new'); print('rule', rule, flush=True)
which = sys.argv[1]

def rich_central(f, th, i, r=1e-3):
    v = float(th[i]); d = lambda h: (float(f(th.at[i].add(h))) - float(f(th.at[i].add(-h))))/(2*h)
    return (4*d(v*r/2) - d(v*r))/3

if which == 'R2':
    for n in (5, 12, 20):
        g = coalescent_graph(n)
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1)
        T = jnp.array([0.5, 1.0, 2.0])
        f = lambda th: jnp.sum(jnp.log(m(th, T)[0]))
        for v in (1e-9, 1e-7, 1e-6, 1e-5, 3e-5):
            th = jnp.array([v]); gr = float(jax.grad(f)(th)[0]); ref = rich_central(f, th, 0)
            print(f'  n={n:2d} V={g.vertices_length():4d} theta={v:6.0e} grad={gr:.6e} ref={ref:.6e} relerr={abs(gr-ref)/abs(ref):.2e}  predicted (k-1)(k-2)c^2/6={(n-2)*(n-3)/6*9e-6:.1e}', flush=True)

if which == 'R4':
    f = S.site_daisy()
    def os2(th, i, h):
        f0, f1, f2 = float(f(th)), float(f(th.at[i].add(h))), float(f(th.at[i].add(2*h)))
        return (-3*f0 + 4*f1 - f2)/(2*h)
    for i, base in ((0, [None, 0.5, 2.0, 0.5]), (1, [2.0, None, 2.0, 0.5])):
        for v in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
            th = jnp.array([v if b is None else b for b in base], dtype=jnp.float64)
            gr = float(jax.grad(f)(th)[i])
            b1, b2 = os2(th, i, 1e-5), os2(th, i, 5e-6); ref = (4*b2-b1)/3
            print(f'  slot {i} theta={v:6.0e} grad={gr:.9e} ref={ref:.9e} (ref self-agree {abs(b1-b2)/abs(ref):.0e}) relerr={abs(gr-ref)/abs(ref):.2e}  abs phi-space err={v*abs(gr-ref):.1e}', flush=True)

if which == 'R3':
    rng = np.random.default_rng(3)
    out = {}
    for name, mk, d in (('daisy', S.site_daisy, 4), ('joint_index', S.site_joint_index, 2),
                        ('pmf_and_moments', S.site_pmf_and_moments, 2)):
        vg = jax.jit(jax.vmap(jax.grad(mk())))
        normal = jnp.asarray(rng.uniform(3.4e-5, 5.0, size=(8, d)))
        mixed = normal.at[:4, 0].set(jnp.asarray(10.0**rng.uniform(-9, -6, size=4)))
        gn = np.asarray(vg(normal)); gm = np.asarray(vg(mixed))
        single = np.stack([np.asarray(jax.jit(jax.grad(mk()))(normal[j])) for j in range(8)])
        print(f'{name}: normal rows of mixed batch bit-identical to all-normal batch: {np.array_equal(gn[4:], gm[4:])}; '
              f'small rows finite: {np.isfinite(gm[:4]).all()}; vmap vs unbatched bit-identical: {np.array_equal(gn, single)} '
              f'(max rel {np.max(np.abs(gn-single)/np.abs(single)):.1e})', flush=True)
        out[name] = gn
    np.savez(f'/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/review_fd/r3_{rule}.npz', **out)
