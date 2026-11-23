"""Review 1: tied-slave placeholder and per-epoch fixed values on an epoch fit
(fixture copied by import from tests/pytest/test_svgd_tied_export.py)."""
import sys, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/tests/pytest')
import test_svgd_tied_export as T
s = T._fit_tied()
print('theta_dim', s.theta_dim, 'fixed_mask', np.asarray(s.fixed_mask), 'slave_to_master', s.model._tying_info)
print('fixed_values (phi):', np.asarray(s.fixed_values))
print('param_transform(fixed_values) = model INPUT at fixed/slave slots:', np.asarray(s.param_transform(jnp.asarray(s.fixed_values))))
r = s.get_results()
print('theta_mean (exported):', np.asarray(r['theta_mean']))
print('mutation given 1e-4; exported theta_mean[1] - 1e-4 =', float(r['theta_mean'][1]) - 1e-4, ' [3]-1e-4 =', float(r['theta_mean'][3]) - 1e-4)
# does the model output depend on the slave INPUT value? (placeholder must be inert)
th = np.asarray(s.param_transform(jnp.asarray(s.particles[0])))
outs = []
for ph in (0.0, 0.6931471805599453, 123.0, -5.0, np.nan):
    t2 = th.copy(); t2[2] = ph
    outs.append(np.asarray(s.model(jnp.asarray(t2), s.observed_data)[0]))
print('model output identical for slave input in {0, 0.693, 123, -5, nan}:',
      all(np.array_equal(outs[0], o) for o in outs[1:]))
g = jax.grad(lambda t: jnp.sum(jnp.log(s.model(t, s.observed_data)[0])))(jnp.asarray(th))
print('d loglik / d theta (slave slot 2 must be 0):', np.asarray(g))
# map_estimate_with_optimization(transforms on) already raises today (list passed to softplus) -> not a surface.
try:
    s.map_estimate_with_optimization(n_steps=0)
except Exception as e:
    print('map_estimate_with_optimization today:', type(e).__name__, str(e)[:80])
# emulate the plan's wrapper (slave slot 2 -> placeholder 0.0) on the fitted object
base = s.param_transform
mask = np.asarray(s.fixed_mask) == 1
fixed_theta = np.zeros(s.theta_dim); fixed_theta[1] = 1e-4; fixed_theta[3] = 1e-4
s.param_transform = lambda phi: jnp.where(jnp.asarray(mask), jnp.asarray(fixed_theta), base(phi))
r = s.get_results()
print('EMULATED get_results theta_mean:', np.asarray(r['theta_mean']), ' [1]==1e-4:', float(r['theta_mean'][1]) == 1e-4,
      ' std[1]:', float(r['theta_std'][1]))
print('EMULATED map_estimate_from_particles:', s.map_estimate_from_particles()[0])
print('EMULATED raw param_transform(particles[0]) (what a user / new code calling the transform directly gets):',
      np.asarray(s.param_transform(jnp.asarray(s.particles[0]))))
print('EMULATED log_likelihood():', s.log_likelihood())
