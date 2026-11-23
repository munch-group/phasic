"""pytest plugin (review 3, read-only instrumentation): log every elimination-based read-out and every
discrete parameterized-model call that runs on a discrete graph whose largest non-starting row sum
exceeds 1 + 1e-9 -- i.e. the calls that the plan's item 3 ("invalid model raises on every read-out")
would turn into errors.  Unmodified library.  Log: $ROWSUM_LOG (jsonl)."""
import os, json, functools
import numpy as np
import pytest
TOL = 1e-9
LOG = os.environ.get('ROWSUM_LOG', '/tmp/rowsum.jsonl')
CUR = {'id': None}
def _rows_max(g):
    n = g.vertices_length()
    if n > 6000: return None
    mx = 0.0
    for i in range(1, n):
        es = g.vertex_at(i).edges()
        if es: mx = max(mx, sum(e.weight() for e in es))
    return mx
def _log(kind, what, mx, extra=''):
    with open(LOG, 'a') as f:
        f.write(json.dumps({'test': CUR['id'], 'kind': kind, 'what': what, 'max_row': mx, 'excess': None if mx is None else mx - 1, 'extra': extra}) + '\n')
def pytest_runtest_setup(item): CUR['id'] = item.nodeid
def pytest_configure(config):
    import phasic
    G = phasic.Graph
    def wrap_method(name, always_discrete=False):
        orig = getattr(G, name, None)
        if orig is None: return
        @functools.wraps(orig)
        def w(self, *a, **k):
            try:
                disc = always_discrete or bool(getattr(self, 'is_discrete', False)) or bool(k.get('discrete', False))
                if disc:
                    mx = _rows_max(self)
                    if mx is not None and mx > 1 + TOL: _log('readout', name, mx, f'is_discrete={getattr(self, "is_discrete", None)} was_dph={self.get_was_dph()}')
                    elif mx is not None and mx > 1: _log('rounding', name, mx)
            except Exception as e: _log('plugin-error', name, None, repr(e)[:100])
            return orig(self, *a, **k)
        setattr(G, name, w)
    for m in ('expectation', 'variance', 'covariance', 'moments', 'expected_waiting_time', 'expected_sojourn_time'):
        wrap_method(m)
    for m in ('expectation_discrete', 'variance_discrete', 'covariance_discrete'):
        wrap_method(m, always_discrete=True)
    def wrap_factory(name):
        orig = getattr(G, name).__func__
        def fac(cls, graph, *a, **k):
            model = orig(cls, graph, *a, **k)
            disc = bool(k.get('discrete', False)) or bool(getattr(graph, 'is_discrete', False))
            if not disc: return model
            def wm(theta, *b, **kk):
                try:
                    th = np.asarray(theta, dtype=float)
                    if th.ndim == 1:
                        g2 = graph.clone(); g2.update_weights(list(th)); mx = _rows_max(g2)
                        if mx is not None and mx > 1 + TOL: _log('model', name, mx, f'theta={th.tolist()} was_dph={graph.get_was_dph()}')
                except Exception as e:
                    if 'Tracer' in type(e).__name__ or 'Tracer' in repr(e): _log('traced', name, None)
                    else: _log('plugin-error', name, None, repr(e)[:100])
                return model(theta, *b, **kk)
            return wm
        setattr(G, name, classmethod(fac))
    for m in ('pmf_and_moments_from_graph', 'pmf_from_graph', 'moments_from_graph'):
        wrap_factory(m)
