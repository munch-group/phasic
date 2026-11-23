"""G3: formula-mode parameters that are NOT positive scale parameters, at the
pmf_and_moments FD backward (C3), prototype relative step (~log-step) vs today.
Loss = E[T] (moment 1) of a 3-stage chain, all three rates = formula weight w."""
import os, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import Graph
T = jnp.array([1.0])
def chain(formula, coeffs, P):
    g = Graph(1); g.set_param_length(P); s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    s.add_edge(v[0], 1.0)
    for i in range(3): v[i].add_edge(v[i + 1], list(coeffs))
    g.weight_formula = formula
    return Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=P, exact_moment_grad=False)
def show(tag, M, th, exact, which):
    row = []
    for name, env in (('today', ('old', '1')), ('rel1e-3', ('rel', '1e-3')), ('rel1e-4', ('rel', '1e-4'))):
        os.environ['PHASIC_FD_RULE'], os.environ['PHASIC_FD_REL'] = env
        try:
            v = float(np.asarray(jax.grad(lambda t: M(t, T)[1][0])(jnp.asarray(th)))[which])
            row.append(f'{name}: {v:+.6e} (rel err {abs(v-exact)/abs(exact):.1e})')
        except Exception as e:
            row.append(f'{name}: RAISES {type(e).__name__}: {str(e)[:90]}')
    print(f' {tag} theta={th} exact={exact:+.6e}\n    ' + '\n    '.join(row), flush=True)
print('### (a) upper-bounded parameter: w = t0*(1 - t1); E[T] = 3/w; dE/dt1 = 3/(t0 (1-t1)^2)')
M = chain("t0*c0*(1 - t1)", (1.0, 0.0), 2)
for t1 in (0.9, 1 - 5e-4, 1 - 5e-5, 1 - 5e-6):
    show('1-t1=%.0e' % (1 - t1), M, [1.0, t1], 3 / (1 - t1) ** 2, 1)
print('### (b) offset/location parameter: w = (t0 - 1000)*c0; E[T] = 3/w; dE/dt0 = -3/w^2')
M = chain("(t0 - 1000.0)*c0", (1.0,), 1)
for d in (50.0, 0.5, 0.05):
    show(f'w={d}', M, [1000.0 + d], -3 / d ** 2, 0)
print('### (c) log-rate parameter with a multiplier: w = exp(c0*t0), c0=10; E[T]=3 exp(-10 t0); dE/dt0 = -10 E')
M = chain("exp(t0*c0)", (10.0,), 1)
for t0 in (0.1, 1.0, 2.0):
    show('', M, [t0], -30 * np.exp(-10 * t0), 0)
