"""Each discrete read-out in its own subprocess at an invalid row sum (1.5): raise, value, or crash?"""
import subprocess, sys
PRE = '''
import random, numpy as np, phasic
from phasic import Graph
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
random.seed(1); g = geom(%r)
try: print('OK', %s)
except Exception as e: print('RAISES', type(e).__name__, str(e)[:60])
'''
CALLS = ["g.sample_multivariate_discrete(3, np.array([[0.],[1.],[0.]]))", "g.sample_discrete(3, [0.,1.,0.])", "g.sample_discrete(3, [])",
         "g.random_sample_discrete_stop_vertex(1)", "g.stop_probability_discrete(2)", "g.accumulated_visits(3)", "g.cdf_discrete(1)",
         "g.sample_path(1)", "g.sample(3)", "g.sample(3, rewards=np.array([0.,1.,0.]))", "g.expected_sojourn_time()", "g.reward_transform_discrete(np.array([0,2,0])).pdf(np.arange(1,4))",
         "g.reward_transform_discrete(np.array([0,2,0])).vertices_length()", "g.covariance_discrete(np.array([0.,1,0]),np.array([0.,1,0]))", "g.expectation_discrete(np.array([0,1,0]))", "g.variance_discrete(np.array([0,1,0]))",
         "g.expected_waiting_time()", "g.moments(2, discrete=True)"]
for p in (1.5, 1 + 1e-5):
    print('## p =', p)
    for c in CALLS:
        r = subprocess.run([sys.executable, '-c', PRE % (p, c)], capture_output=True, text=True)
        print(f'  rc={r.returncode:4d} {c[:70]:70s} -> {(r.stdout.strip() or r.stderr.strip()[-80:])[:110]!r}')
