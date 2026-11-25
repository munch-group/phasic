"""Bug 5 on unmodified code: the on-disk graph cache key (callback source +
ipv + kwargs) misses what the callback depends on. HOME is redirected into
this directory; the real ~/.phasic_cache is never touched."""
import os, sys, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, 'fakehome'); shutil.rmtree(FAKE, ignore_errors=True); os.makedirs(FAKE)
os.environ['HOME'] = FAKE
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import phasic
from phasic import Graph, with_ipv, StateIndexer, Property
from phasic.graph_cache import DEFAULT_CACHE_DIR
print('phasic from', phasic.__file__); print('cache dir', DEFAULT_CACHE_DIR)
def E(g): g.update_weights([1.0]); return float(g.expectation())

SCALE = 1.0
@with_ipv([3])
def cb_global(state):
    if state[0] <= 1: return []
    new = state.copy(); new[0] -= 1
    return [[new, [SCALE * state[0]]]]
print('## (a) module global changed')
SCALE = 1.0; a = E(Graph(cb_global)); SCALE = 10.0; b = E(Graph(cb_global)); c = E(Graph(cb_global, graph_cache=False))
print(f'   before {a:.6f}; after (cache on) {b:.6f}; after (cache off, correct) {c:.6f}')

def make(scale):
    @with_ipv([3])
    def cb(state):
        if state[0] <= 1: return []
        new = state.copy(); new[0] -= 1
        return [[new, [scale * state[0]]]]
    return cb
print('## (b) closure over a local')
a = E(Graph(make(1.0))); b = E(Graph(make(10.0))); c = E(Graph(make(10.0), graph_cache=False))
print(f'   before {a:.6f}; after (cache on) {b:.6f}; after (cache off, correct) {c:.6f}')

def helper(k): return 1.0 * k
@with_ipv([3])
def cb_helper(state):
    if state[0] <= 1: return []
    new = state.copy(); new[0] -= 1
    return [[new, [helper(state[0])]]]
print('## (c) helper function redefined')
a = E(Graph(cb_helper))
def helper(k): return 10.0 * k
b = E(Graph(cb_helper)); c = E(Graph(cb_helper, graph_cache=False))
print(f'   before {a:.6f}; after (cache on) {b:.6f}; after (cache off, correct) {c:.6f}')

print('## (d) plain callback (ipv= argument) with a closure: what happens today?')
def make_plain(scale):
    def cb(state):
        if state[0] <= 1: return []
        new = state.copy(); new[0] -= 1
        return [[new, [scale * state[0]]]]
    return cb
try:
    a = E(Graph(make_plain(1.0), ipv=[3])); b = E(Graph(make_plain(10.0), ipv=[3]))
    print(f'   scale 1: {a:.6f}; scale 10: {b:.6f}')
except Exception as e:
    print('   raised', type(e).__name__, str(e).splitlines()[0][:150])

print('## (e) the same model in a NEW PROCESS-like state: files written')
files = [f for f in os.listdir(DEFAULT_CACHE_DIR)] if os.path.isdir(DEFAULT_CACHE_DIR) else []
print('   cache files:', len(files))
