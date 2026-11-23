"""R2: per-component error behind test_loglik_gradient_degrades_only_below_1e8,
plus bit-identity at values just above the 3.33e-5 boundary (saved to npz)."""
import sys, os
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.getcwd() + '/tests/pytest')
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
import test_likelihood_correctness as T
import phasic
print('phasic from', phasic.__file__)
for l2 in (1e-8, 1e-9):
    gs, gt = T._shipped_and_true_grad(l2)
    print(f'l2={l2:g}: per-component rel err', np.abs(gs - gt) / np.abs(gt))

import sites as S
out = {}
vals = np.array([1e-7 / 3e-3 * (1 + 1e-12), 3.34e-5, 4e-5, 1e-4, 1e-3])
for name, mk, d in (('daisy', S.site_daisy, 4), ('joint_index', S.site_joint_index, 2),
                    ('pmf_and_moments', S.site_pmf_and_moments, 2)):
    f = mk(); vg = jax.jit(jax.vmap(jax.grad(f)))
    base = np.tile(np.array([2.0, 0.5, 2.0, 0.5][:d]), (len(vals) * d, 1))
    for k, v in enumerate(vals):
        for i in range(d):
            base[k * d + i, i] = v
    try:
        out[name] = np.asarray(vg(jnp.asarray(base)))
    except Exception as e:
        print(name, 'ERROR', str(e).split('\n')[0][:150]); out[name] = np.full(base.shape, np.nan)
np.savez(sys.argv[1], **out)
print('saved', sys.argv[1])
