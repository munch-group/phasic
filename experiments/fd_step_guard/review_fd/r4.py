"""C3 (pmf_and_moments, uniformization forward) smooth-through-zero slot: migration small."""
import sys, os, logging
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
logging.disable(logging.WARNING)
import sites as S
print('rule', os.environ.get('PHASIC_FD_RULE', 'new'))
f = S.site_pmf_and_moments()
def os2(th, i, h):
    return (-3*float(f(th)) + 4*float(f(th.at[i].add(h))) - float(f(th.at[i].add(2*h))))/(2*h)
for v in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
    th = jnp.array([0.5, v]); gr = float(jax.grad(f)(th)[1])
    b1, b2 = os2(th, 1, 1e-5), os2(th, 1, 5e-6); ref = (4*b2-b1)/3
    print(f'  mig={v:6.0e} grad={gr:.9e} ref={ref:.9e} self-agree {abs(b1-b2)/abs(ref):.0e} relerr={abs(gr-ref)/abs(ref):.2e}', flush=True)
