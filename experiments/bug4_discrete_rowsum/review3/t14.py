"""Plan v2 section 3 on the UNMODIFIED build: which new tests fail / which guards pass today."""
import subprocess, sys, warnings, random, time
import numpy as np
warnings.filterwarnings('ignore')
PRE = '''
import random, warnings, numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
def geom(p, ipv=1.0):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, ipv); a.add_edge(b, p); g.is_discrete = True; return g
def pgeom():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); return g
random.seed(1)
try: print('OK', %s)
except Exception as e: print('RAISES', type(e).__name__, str(e).splitlines()[0][:70])
'''
def sub(label, expr):
    r = subprocess.run([sys.executable, '-c', PRE % expr], capture_output=True, text=True)
    print(f'  rc={r.returncode:4d} {label[:62]:62s} -> {(r.stdout.strip().splitlines() or [r.stderr.strip()[-90:]])[-1][:120]!r}')
print('## tests 1/2: read-outs named in v2, p = 1+1e-6 / 1+1e-5 / 1+2e-4 / 1.5 (expect: no raise today for 1,2-elimination)')
for p in ('1+1e-6', '1+1e-5', '1+2e-4', '1.5'):
    for c in ('g.pdf(np.arange(1,4))', 'g.cdf(np.arange(1,4))', 'g.stop_probability(2)', 'g.accumulated_visits(3)', 'g.sample(3)',
              'g.reward_transform_discrete(np.array([0,1,0])).pdf(np.arange(1,4))', 'g.expectation()', 'g.variance()', 'g.covariance()',
              'g.covariance(np.array([0.,1,0]), np.array([0.,1,0]))', 'g.moments(2, discrete=True)'):
        sub(f'p={p}: {c}', f'(lambda g: {c})(geom({p}))')
print('## guard 10 extra: covariance() with default arguments on a VALID chain')
sub('p=0.5 covariance()', 'geom(0.5).covariance()'); sub('p=0.5 variance()', 'geom(0.5).variance()')
print('## test 5: starting vertex')
for ipv in ('1.5', '1+1e-5', '1+1e-12'):
    sub(f'ipv={ipv}: cdf(200)', f'float(geom(0.5, {ipv}).cdf(200))'); sub(f'ipv={ipv}: expectation()', f'geom(0.5, {ipv}).expectation()')
print('## test 7: crashes')
sub('p=1.5 distribution_context()', 'geom(1.5).distribution_context()')
sub('p=1.5 distribution_context().step()', 'geom(1.5).distribution_context().step()')
sub('p=1.5 distribution_context().pmf()', '(lambda c: (c.step(), c.pmf()))(geom(1.5).distribution_context())')
sub('p=1.5 sample_multivariate_discrete', 'geom(1.5).sample_multivariate_discrete(3, np.array([[0.],[1.],[0.]]))')
sub('p=1.5 sample_multivariate (plan 2.3 name)', 'geom(1.5).sample_multivariate(3, np.array([[0.],[1.],[0.]]))')
sub('p=1+1e-5 distribution_context().step()', 'geom(1+1e-5).distribution_context().step()')
print('## test 9 / guard 12: stale contexts after update_weights')
sub('accumulated_visits: valid(0.5) -> read -> update 1.5 -> read', '(lambda g: (g.update_weights([0.5]), setattr(g, "is_discrete", True), g.accumulated_visits(3), g.update_weights([1.5]), g.accumulated_visits(3))[-1])(pgeom())')
sub('accumulated_visits: same but second read at MORE jumps', '(lambda g: (g.update_weights([0.5]), setattr(g, "is_discrete", True), g.accumulated_visits(3), g.update_weights([1.5]), g.accumulated_visits(6))[-1])(pgeom())')
sub('stop_probability: valid -> read -> update 1.5 -> read', '(lambda g: (g.update_weights([0.5]), setattr(g, "is_discrete", True), g.stop_probability(3), g.update_weights([1.5]), g.stop_probability(3))[-1])(pgeom())')
sub('pdf: valid -> read -> update 1.5 -> read (guard 12)', '(lambda g: (g.update_weights([0.5]), setattr(g, "is_discrete", True), g.pdf(np.arange(1,4)), g.update_weights([1.5]), g.pdf(np.arange(1,4)))[-1])(pgeom())')
sub('cdf: valid -> read -> update 1.5 -> read', '(lambda g: (g.update_weights([0.5]), setattr(g, "is_discrete", True), g.cdf(np.arange(1,4)), g.update_weights([1.5]), g.cdf(np.arange(1,4)))[-1])(pgeom())')
print('## guard 14 on master: is there an exception to classify at 1+1e-5?')
sub('pmf_and_moments at 1+1e-5', 'np.asarray(Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True)(np.array([1+1e-5]), np.array([1.,2.]))[0])')
print('## guard 13 on master: grad at gap 5e-8 / 2e-7')
for gap in ('5e-8', '2e-7'):
    sub(f'jax.grad pmf_from_graph at 1-{gap}', f'__import__("jax").grad(lambda x: Graph.pmf_from_graph(pgeom(), discrete=True)(x, np.array([1.,2.]))[0])(__import__("jax").numpy.asarray([1-{gap}]))')
print('## test 8 on master: vmap pmf_and_moments use_ffi=True at 1+1e-5')
sub('vmap', 'np.asarray(__import__("jax").vmap(lambda q: Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi=True)(q, __import__("jax").numpy.asarray([1.,2.]))[0])(__import__("jax").numpy.asarray([[1+1e-5],[1+1e-5]])))')
print('## 2.3: pmf_and_moments with EMPTY times at 1.5 (is the pmf half reached, so that it validates?)')
for ffi in (False, True):
    sub(f'pmf_and_moments(use_ffi={ffi}) empty times at 1.5', f'[np.asarray(x) for x in Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi={ffi})(np.array([1.5]), np.array([]))]')
    sub(f'pmf_and_moments(use_ffi={ffi}) times [1,2] at 1.5', f'[np.asarray(x) for x in Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi={ffi})(np.array([1.5]), np.array([1.,2.]))]')
