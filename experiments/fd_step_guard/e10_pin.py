import sys, os
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest/inference')
import numpy as np, jax, jax.numpy as jnp
import test_fd_gradient_mixed_scale as T
print('rule', os.environ.get('PHASIC_FD_RULE', 'new'))
for th in ([1.0, 1e-8], [1.0, 1e-4], [1e-8, 1e-8]):
    th = np.array(th)
    fd = np.array(T._grad_A(jnp.array(th))); o = T._oracle(th)
    m = np.array(T._MODEL_A(jnp.array(th), T._TIMES)[1])
    print(th, 'fd', fd, 'oracle', o, 'rel', np.abs(fd-o)/np.abs(o), 'moments', m)
