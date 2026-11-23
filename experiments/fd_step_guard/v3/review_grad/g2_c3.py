"""G2: pmf_and_moments FD backward (site C3) under a relative step (prototype
PHASIC_FD_RULE=rel ~ log-step) vs today, closed-form oracles.
 A: v2's E10 fixture  S->[3]-t0->[2]-t1->[1]:  E[T]=1/t0+1/t1 (output dominated by the small parameter)
 B: ADDITIVE parameter: first edge weight = t0 + t1 (coefficients [1,1]): E[T]=1/(t0+t1)+1/t0
Errors: theta-space relative error of dE/dtheta_i, and absolute phi-space error theta_i*dE/dtheta_i / |E|."""
import os, sys, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import Graph
T = jnp.array([1.0])
def model(c_first):
    g = Graph(1); s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0); v3.add_edge(v2, c_first); v2.add_edge(v1, [1.0, 0.0] if c_first == [1.0, 1.0] else [0.0, 1.0])
    return Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
MA = model([1.0, 0.0]); MB = model([1.0, 1.0])
def grads(M, th, k):
    out = {}
    for name, env in (('today', ('old', '1')), ('rel1e-3', ('rel', '1e-3')), ('rel1e-4', ('rel', '1e-4')), ('rel1e-5', ('rel', '1e-5'))):
        os.environ['PHASIC_FD_RULE'], os.environ['PHASIC_FD_REL'] = env
        out[name] = np.asarray(jax.grad(lambda t: M(t, T)[1][k])(jnp.asarray(th)))
    return out
print('### A: E[T] = 1/t0 + 1/t1 (moment 1) and E[T^2] (moment 2)')
for th in ([1.0, 1e-4], [1.0, 1e-8], [1.0, 1e-11], [1e3, 1e-8]):
    t0, t1 = th
    for k, orc in ((0, np.array([-1/t0**2, -1/t1**2])), (1, np.array([-4/t0**3 - 2/(t0**2*t1), -4/t1**3 - 2/(t0*t1**2)]))):
        G = grads(MA, th, k)
        print(f' theta={th} moment{k+1}: ' + '; '.join(f'{n}: rel err {np.abs(v-orc)/np.abs(orc)}' for n, v in G.items()), flush=True)
print('### B: additive parameter, first rate = t0 + t1; d E[T]/d t1 = -1/(t0+t1)^2')
for th in ([1.0, 1e-3], [1.0, 1e-8], [1.0, 1e-11], [1.0, 1e-13], [1.0, 1e-15]):
    t0, t1 = th
    orc = np.array([-1/(t0+t1)**2 - 1/t0**2, -1/(t0+t1)**2])
    G = grads(MB, th, 0)
    print(f' theta={th}: ' + '; '.join(f'{n}: d/dt1 = {v[1]:+.6e} (rel err {abs(v[1]-orc[1])/abs(orc[1]):.1e})' for n, v in G.items()), flush=True)
print('### B, pdf part: d log pdf(t=1)/d t1 (reference: today rule at t1=1e-3-equivalent closed form)')
def dlogpdf_exact(t0, t1, t=1.0):
    a, b = t0 + t1, t0   # hypoexp rates a (stage 1), b (stage 2)
    import mpmath as mp
    mp.mp.dps = 40
    f = lambda a_: mp.log(mp.mpf(a_) * b / (b - mp.mpf(a_)) * (mp.exp(-mp.mpf(a_) * t) - mp.exp(-b * t)))
    return float(mp.diff(f, a))
for th in ([1.0, 1e-3], [1.0, 1e-8], [1.0, 1e-11], [1.0, 1e-13]):
    row = []
    for name, env in (('today', ('old', '1')), ('rel1e-3', ('rel', '1e-3')), ('rel1e-4', ('rel', '1e-4')), ('rel1e-5', ('rel', '1e-5'))):
        os.environ['PHASIC_FD_RULE'], os.environ['PHASIC_FD_REL'] = env
        v = np.asarray(jax.grad(lambda t: jnp.sum(jnp.log(MB(t, T)[0])))(jnp.asarray(th)))
        row.append(f'{name}: {v[1]:+.6e}')
    try: ex = dlogpdf_exact(*th)
    except Exception as e: ex = float('nan')
    print(f' theta={th} exact(continuous) {ex:+.6e}: ' + '; '.join(row), flush=True)
