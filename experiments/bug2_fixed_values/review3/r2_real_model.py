"""Review 3: (A) which fixed values round-trip exactly today; (B) real model at a
rate of exactly 0 / tiny; (C) prototype of the planned fix by runtime patching
(no source edit) to test plan test 7's bit-identity claim and the fixed-at-0 use case."""
import warnings, random
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph, GaussPrior, SVGD

print('## A. round trip today: maximum(softplus(log(exp(v)-1)), 1e-9) == v ?')
tr = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
def rt(v):
    return float(tr(jnp.asarray(float(jnp.log(jnp.exp(v) - 1)))))
for v in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 50.0, 0.3, 0.25, 0.1, 0.01, 0.0123456789,
          1e-2, 1e-3, 1e-4, 1.2e-4, 1e-5, 1e-6, 1e-7, 1e-8, 2.5e-8, 1e-9, 37.0, 40.0, 100.0, 700.0, 710.0):
    r = rt(v)
    print(f'  v={v!r:>16}  ->  {r!r:<24} {"exact" if r == v else f"rel err {abs(r - v) / v:.2e}"}')

def full_graph():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([2]); z = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0)
    a.add_edge(z, [1.0, 0.0])      # theta0
    a.add_edge(c, [0.0, 1.0])      # theta1  (the rate switched off)
    c.add_edge(z, [2.0, 0.0])      # 2*theta0
    return g
def reduced_graph():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); z = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(z, [1.0])
    return g
def full_model(): return Graph.pmf_and_moments_from_graph(full_graph(), nr_moments=2, theta_dim=2)
def red_model(): return Graph.pmf_and_moments_from_graph(reduced_graph(), nr_moments=2, theta_dim=1)

t = jnp.array([0.2, 0.7, 1.3])
print('## B. real model at theta1 in {0, 1e-12, 1e-9, 0.693}: pdf vs closed form 1.5*exp(-1.5 t)')
ref = 1.5 * np.exp(-1.5 * np.asarray(t))
for v in (0.0, 1e-12, 1e-9, 0.6931471805599453):
    try:
        p, m = full_model()(jnp.array([1.5, v]), t)
        print(f'  theta1={v:g}: pdf={np.asarray(p)}  rel err vs reduced closed form={np.max(np.abs(np.asarray(p) - ref) / ref):.2e}  moments={np.asarray(m)}')
    except Exception as e:
        print(f'  theta1={v:g}: RAISED {type(e).__name__}: {str(e)[:200]}')
    try:
        mdl = full_model()
        g = jax.grad(lambda th: jnp.sum(jnp.log(mdl(th, t)[0])))(jnp.array([1.5, v]))
        print(f'      grad of sum log pdf = {np.asarray(g)}   (closed form d/dtheta0 = {float(np.sum(1/1.5 - np.asarray(t))):.6f})')
    except Exception as e:
        print(f'      grad RAISED {type(e).__name__}: {str(e)[:200]}')

def apply_fix(s, fixed):
    """Runtime prototype of plan section 2 (wrapper + 0.0 placeholder)."""
    mask = np.zeros(s.theta_dim, bool); ft = np.zeros(s.theta_dim)
    for i, v in fixed: mask[i] = True; ft[i] = v
    base = s.param_transform; m = jnp.asarray(mask); f = jnp.asarray(ft)
    def with_fixed(phi):
        th = base(phi) if base is not None else phi
        return jnp.where(m, f, th)
    if getattr(base, '_per_dim_fns', None) is not None: with_fixed._per_dim_fns = base._per_dim_fns
    s._free_transform = base; s.param_transform = with_fixed
    s.fixed_values = jnp.where(m, 0.0, jnp.asarray(s.fixed_values))
    s.theta_init = jnp.where(m[None, :], 0.0, s.theta_init)
    return s

random.seed(3); np.random.seed(3)
data = np.random.exponential(1 / 1.5, size=40)
def fit(v, patched, n_iter=15, n_particles=8, model=None, theta_dim=2, **kw):
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0), None])
    fx = [(1, v)] if theta_dim == 2 else None
    s = SVGD(model=model or full_model(), observed_data=data, theta_dim=theta_dim, fixed=fx,
             n_particles=n_particles, n_iterations=n_iter, progress=False, seed=1, **kw)
    if patched: apply_fix(s, fx)
    s.optimize()
    return s, np.asarray(s.get_results()['particles'])

print('## C. plan test 7: free-slot particles, unmodified vs prototype fix')
for v in (2.0, 50.0, 0.3, 1e-4, 1e-8):
    try:
        _, p0 = fit(v, False); _, p1 = fit(v, True)
        d = np.abs(p0[:, 0] - p1[:, 0])
        print(f'  v={v:g}: free slot bit-identical={bool(np.all(p0[:, 0] == p1[:, 0]))}  max abs diff={d.max():.3e}  max rel={np.max(d / np.abs(p0[:, 0])):.3e}  fixed col after fix={sorted(set(p1[:, 1].tolist()))}')
    except Exception as e:
        print(f'  v={v:g}: RAISED {type(e).__name__}: {str(e)[:300]}')
print('## C2. run-to-run determinism of the UNMODIFIED build (same seed, fresh model)')
_, a = fit(2.0, False); _, b = fit(2.0, False)
print('  identical:', bool(np.all(a == b)))
print('## C3. fixed at exactly 0 with the prototype fix, real model; reference = reduced 1-parameter model')
try:
    s0, p = fit(0.0, True)
    print('  prototype fixed=(1,0): free particles', np.round(p[:, 0], 6), ' fixed col', sorted(set(p[:, 1].tolist())), 'finite:', bool(np.all(np.isfinite(p))))
    try: print('  log_likelihood():', s0.log_likelihood())
    except Exception as e: print('  log_likelihood RAISED', type(e).__name__, str(e)[:200])
except Exception as e:
    print(f'  prototype fixed=(1,0) RAISED {type(e).__name__}: {str(e)[:400]}')
try:
    sr, pr = fit(None, False, model=red_model(), theta_dim=1, prior=[GaussPrior(ci=[0.5, 5], lower=0.0)])
    print('  reduced model         : free particles', np.round(pr[:, 0], 6))
    try: print('  log_likelihood():', sr.log_likelihood())
    except Exception as e: print('  log_likelihood RAISED', type(e).__name__, str(e)[:200])
except Exception as e:
    print(f'  reduced RAISED {type(e).__name__}: {str(e)[:300]}')
su, pu = fit(0.0, False)
print('  UNMODIFIED fixed=(1,0): free particles', np.round(pu[:, 0], 6), ' fixed col', sorted(set(pu[:, 1].tolist())))
try: print('  log_likelihood():', su.log_likelihood())
except Exception as e: print('  log_likelihood RAISED', type(e).__name__, str(e)[:200])
print('## D. real-model check without a spy: log_likelihood of the fit vs the model evaluated by hand at the exact fixed value')
for v in (0.3, 1e-10, 800.0):
    try:
        s, p = fit(v, False, n_iter=2)
        mdl = full_model()
        best = None
        for row in p:
            ll = float(jnp.sum(jnp.log(mdl(jnp.array([row[0], v]), jnp.asarray(data))[0])))
            best = ll if best is None else max(best, ll)
        print(f'  v={v:g}: fit.log_likelihood()={s.log_likelihood()!r}  best by-hand at exact v={best!r}')
    except Exception as e:
        print(f'  v={v:g}: RAISED {type(e).__name__}: {str(e)[:200]}')
