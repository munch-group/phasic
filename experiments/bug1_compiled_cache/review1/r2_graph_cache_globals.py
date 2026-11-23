"""Review 1, question 2: does the on-disk GRAPH cache (on by default) have the same class of
defect -- a key (callback AST + kwargs) narrower than what the callback depends on?
HOME is redirected into this directory; the real ~/.phasic_cache is not touched."""
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_HOME = os.path.join(HERE, 'fakehome2'); shutil.rmtree(FAKE_HOME, ignore_errors=True); os.makedirs(FAKE_HOME)
os.environ['HOME'] = FAKE_HOME
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import phasic
from phasic import Graph, with_ipv
from phasic.graph_cache import DEFAULT_CACHE_DIR
print('graph cache dir:', DEFAULT_CACHE_DIR)

SCALE = 1.0
@with_ipv([3])
def cb_global(state):
    if state[0] <= 1:
        return []
    new = state.copy(); new[0] -= 1
    return [[new, [SCALE * state[0]]]]

def expect(g):
    g.update_weights([1.0]); return float(g.expectation())

print('## (a) with_ipv callback reading a MODULE GLOBAL')
SCALE = 1.0; e1 = expect(Graph(cb_global))
SCALE = 10.0; e2 = expect(Graph(cb_global)); e2n = expect(Graph(cb_global, graph_cache=False))
print(f'   SCALE=1: {e1:.6f}   SCALE=10 (cache on): {e2:.6f}   SCALE=10 (graph_cache=False): {e2n:.6f}')

def make(scale):
    @with_ipv([3])
    def cb_closure(state):
        if state[0] <= 1:
            return []
        new = state.copy(); new[0] -= 1
        return [[new, [scale * state[0]]]]
    return cb_closure
print('## (b) with_ipv callback CLOSING OVER a local (closure check is skipped for with_ipv wrappers)')
try:
    e1 = expect(Graph(make(1.0))); e2 = expect(Graph(make(10.0))); e2n = expect(Graph(make(10.0), graph_cache=False))
    print(f'   scale=1: {e1:.6f}   scale=10 (cache on): {e2:.6f}   scale=10 (graph_cache=False): {e2n:.6f}')
except Exception as e:
    print('   raised', type(e).__name__, str(e)[:200])

def helper(k): return 1.0 * k
@with_ipv([3])
def cb_helper(state):
    if state[0] <= 1:
        return []
    new = state.copy(); new[0] -= 1
    return [[new, [helper(state[0])]]]
print('## (c) with_ipv callback calling a module-level HELPER that is then redefined')
e1 = expect(Graph(cb_helper))
def helper(k): return 10.0 * k
e2 = expect(Graph(cb_helper)); e2n = expect(Graph(cb_helper, graph_cache=False))
print(f'   helper x1: {e1:.6f}   helper x10 (cache on): {e2:.6f}   helper x10 (graph_cache=False): {e2n:.6f}')
