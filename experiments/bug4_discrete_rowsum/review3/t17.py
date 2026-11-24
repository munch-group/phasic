"""Plan v3 on the UNMODIFIED build: classification of tests 9, 9b, 9c, 14 and weight signs on legitimate discrete graphs."""
import subprocess, sys, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
PRE = '''
import random, warnings, numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
def pgeom():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); return g
def two():   # a -> b (theta0), a -> c (theta1), b -> c (0.5)
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); b.add_edge(c, [0.5, 0.5]); return g
def mk(g, th):
    g.update_weights(th); g.is_discrete = True; return g
random.seed(1)
try: print('OK', %s)
except Exception as e: print('RAISES', type(e).__name__, str(e).splitlines()[0][:80])
'''
def sub(label, expr):
    r = subprocess.run([sys.executable, '-c', PRE % expr], capture_output=True, text=True, timeout=120)
    print(f'  rc={r.returncode:4d} {label[:58]:58s} -> {(r.stdout.strip().splitlines() or [r.stderr.strip()[-90:]])[-1][:130]!r}')
print('## test 9: user-held context')
sub('ctx at 0.5, update 1.5, 3 steps: pmf', '(lambda g: (lambda c: [(g.update_weights([1.5]) if i == 0 else None, c.step(), c.pmf())[-1] for i in range(3)])(g.distribution_context()))(mk(pgeom(), [0.5]))')
sub('ctx at 0.5, update 0.25, 3 steps: pmf', '(lambda g: (lambda c: [(g.update_weights([0.25]) if i == 0 else None, c.step(), c.pmf())[-1] for i in range(3)])(g.distribution_context()))(mk(pgeom(), [0.5]))')
print('## test 9b: negative / NaN weight on a native parameterized graph')
for th in ('[-0.5, 0.9]', '[float("nan"), 0.3]'):
    for c in ('np.asarray(g.pdf(np.arange(1,4)))', 'g.stop_probability(1)', 'g.sample(3)', 'g.expectation()', 'g.variance()', '[e.weight() for e in g.vertex_at(1).edges()]'):
        try: sub(f'theta={th}: {c}', f'(lambda g: {c})(mk(two(), {th}))')
        except subprocess.TimeoutExpired: print(f'  TIMEOUT(120s) theta={th}: {c}')
print('## test 9c: empty times')
for ffi in (False, True):
    for th in ('1+1e-5', '1.5'):
        sub(f'pmf_and_moments(use_ffi={ffi}) empty times theta={th}', f'[np.asarray(x) for x in Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi={ffi})(np.array([{th}]), np.array([]))]')
