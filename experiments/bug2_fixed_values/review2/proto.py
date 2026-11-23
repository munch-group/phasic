"""Prototype of the plan's design by monkeypatching (no source edits).

install(mode):
  'plan'   - wrapper as specified; emulates the plan's "use _free_transform on
             the learnable subset" by dispatching on length (the inline
             preconditioner code in optimize() cannot be patched otherwise)
  'strict' - wrapper raises on any input whose last dim != theta_dim
  'naive'  - no shape check at all
"""
import numpy as np, jax, jax.numpy as jnp
import phasic
from phasic import svgd as _svgd_mod, mcmc as _mcmc_mod
SVGD = _svgd_mod.SVGD
MCMC = _mcmc_mod.MCMC
_orig_svgd_init = SVGD.__init__
_orig_mcmc_init = MCMC.__init__
CALLS = []   # (shape, kind) log of wrapper calls
PLACEHOLDER = 0.0

def _make_wrapper(base, mask_bool, fixed_theta, theta_dim, mode):
    mask_j = jnp.asarray(mask_bool); ft = jnp.asarray(fixed_theta, dtype=jnp.float64)
    n_learn = int((~mask_bool).sum())
    def with_fixed(phi):
        shp = tuple(np.shape(phi))
        CALLS.append(shp)
        if mode != 'naive' and (len(shp) == 0 or shp[-1] != theta_dim):
            if mode == 'plan' and len(shp) and shp[-1] == n_learn:
                # emulate "preconditioner gets _free_transform"
                fns = getattr(base, '_per_dim_fns', None)
                if fns is not None:
                    li = np.where(~mask_bool)[0]
                    return jnp.concatenate([fns[int(j)](phi[..., k:k+1]) for k, j in enumerate(li)], axis=-1)
                return base(phi)
            raise ValueError(f'with_fixed: wrong-length input {shp}, theta_dim={theta_dim}')
        theta = base(phi) if base is not None else phi
        return jnp.where(mask_j, ft, theta)
    if hasattr(base, '_per_dim_fns'):
        with_fixed._per_dim_fns = base._per_dim_fns
    return with_fixed

def _split_fixed(fixed):
    if fixed is None: return None, None
    if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
        true = [(int(i), float(v)) for i, v in fixed]
        return [(i, 1.0) for i, _ in true], true     # construct with harmless 1.0
    arr = np.asarray(fixed)
    return fixed, [(int(i), 1.0) for i in np.where(arr == 1)[0]]

def _apply(self, true, mode, init_attr='theta_init'):
    if true is None: return
    base = self.param_transform
    if base is None:
        # plan: nothing changes without a transform; restore true values
        fv = np.array(self.fixed_values, dtype=float)
        for i, v in true: fv[i] = v
        self.fixed_values = jnp.asarray(fv)
        ti = jnp.asarray(getattr(self, init_attr))
        for i, v in true: ti = ti.at[..., i].set(v)
        setattr(self, init_attr, ti)
        return
    mask = np.zeros(self.theta_dim, bool); ft = np.zeros(self.theta_dim)
    for i, v in true: mask[i] = True; ft[i] = v
    self._free_transform = base
    self.fixed_theta = ft
    self.param_transform = _make_wrapper(base, mask, ft, self.theta_dim, mode)
    self.fixed_values = jnp.where(jnp.asarray(mask), PLACEHOLDER, 0.0) + jnp.zeros(self.theta_dim)
    ti = jnp.asarray(getattr(self, init_attr))
    for i, _ in true: ti = ti.at[..., i].set(PLACEHOLDER)
    setattr(self, init_attr, ti)

def install(mode='plan'):
    def svgd_init(self, *a, **kw):
        f2, true = _split_fixed(kw.get('fixed'))
        if 'fixed' in kw: kw['fixed'] = f2
        _orig_svgd_init(self, *a, **kw)
        _apply(self, true, mode)
    def mcmc_init(self, *a, **kw):
        f2, true = _split_fixed(kw.get('fixed'))
        if 'fixed' in kw: kw['fixed'] = f2
        _orig_mcmc_init(self, *a, **kw)
        _apply(self, true, mode)
    SVGD.__init__ = svgd_init
    MCMC.__init__ = mcmc_init

def uninstall():
    SVGD.__init__ = _orig_svgd_init
    MCMC.__init__ = _orig_mcmc_init

def clear():
    SVGD._compiled_cache.clear(); CALLS.clear()

def graph3():
    """theta0: a->c and b->c ; theta1: a->b (rate that may be switched off); 2 params"""
    from phasic import Graph
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(c, [1.0, 0.0]); a.add_edge(b, [0.0, 1.0]); b.add_edge(c, [3.0, 0.0])
    return g

def spy_model(base_model):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.asarray(t).copy()), theta)
        return base_model(theta, times, rewards=rewards) if rewards is not None else base_model(theta, times)
    for attr in ('_precondition_output', '_tying_info'):
        if hasattr(base_model, attr): setattr(spy, attr, getattr(base_model, attr))
    return spy, seen

def slot_values(seen, slot):
    return sorted({float(np.atleast_2d(x)[r, slot]) for x in seen for r in range(np.atleast_2d(x).shape[0])})
