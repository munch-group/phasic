"""pytest plugin (review 3, plan v3): log every discrete read-out / discrete model call on a graph that holds a
NEGATIVE or NON-FINITE edge weight -- the calls plan v3's per-vertex weight check would turn into errors.
Unmodified library. Log: $NEGW_LOG (jsonl)."""
import os, json, functools, math
import numpy as np
LOG = os.environ.get('NEGW_LOG', '/tmp/negw.jsonl')
CUR = {'id': None}
def _scan(g):
    n = g.vertices_length()
    if n > 6000: return None
    mn = 0.0; bad = 0
    for i in range(n):
        for e in g.vertex_at(i).edges():
            w = e.weight()
            if not math.isfinite(w): bad += 1
            elif w < mn: mn = w
    return mn, bad
def _log(kind, what, mn, bad, extra=''):
    with open(LOG, 'a') as f: f.write(json.dumps({'test': CUR['id'], 'kind': kind, 'what': what, 'min_w': mn, 'nonfinite': bad, 'extra': extra}) + '\n')
def pytest_runtest_setup(item): CUR['id'] = item.nodeid
def pytest_configure(config):
    import phasic
    G = phasic.Graph
    def wrap_method(name, always=False, elim=False):
        orig = getattr(G, name, None)
        if orig is None: return
        @functools.wraps(orig)
        def w(self, *a, **k):
            try:
                isd = bool(getattr(self, 'is_discrete', False)); wd = bool(self.get_was_dph())
                if always or isd or bool(k.get('discrete', False)):
                    if not (elim and wd):          # elimination read-outs validate native graphs only
                        r = _scan(self)
                        if r and (r[0] < 0 or r[1]): _log('readout', name, r[0], r[1], f'is_discrete={isd} was_dph={wd}')
                        else: _log('clean', name, 0, 0)
            except Exception as e: _log('plugin-error', name, None, None, repr(e)[:100])
            return orig(self, *a, **k)
        setattr(G, name, w)
    for m in ('pdf', 'cdf', 'stop_probability', 'accumulated_visits', 'sample', 'distribution_context'): wrap_method(m)
    for m in ('pdf_discrete', 'cdf_discrete', 'sample_discrete', 'sample_multivariate_discrete', 'reward_transform_discrete', 'stop_probability_discrete', 'distribution_context_discrete'): wrap_method(m, always=True)
    for m in ('expectation', 'variance', 'covariance', 'moments'): wrap_method(m, elim=True)
    for m in ('expectation_discrete', 'variance_discrete', 'covariance_discrete'): wrap_method(m, always=True, elim=True)
    def wrap_factory(name):
        orig = getattr(G, name).__func__
        def fac(cls, graph, *a, **k):
            model = orig(cls, graph, *a, **k)
            if not (bool(k.get('discrete', False)) or bool(getattr(graph, 'is_discrete', False))): return model
            def wm(theta, *b, **kk):
                try:
                    th = np.asarray(theta, dtype=float)
                    if th.ndim == 1:
                        g2 = graph.clone(); g2.update_weights(list(th)); r = _scan(g2)
                        if r and (r[0] < 0 or r[1]): _log('model', name, r[0], r[1], f'theta={th.tolist()} was_dph={graph.get_was_dph()}')
                        else: _log('clean', name, 0, 0)
                except Exception as e:
                    _log('traced' if 'Tracer' in (type(e).__name__ + repr(e)) else 'plugin-error', name, None, None, repr(e)[:100])
                return model(theta, *b, **kk)
            return wm
        setattr(G, name, classmethod(fac))
    for m in ('pmf_and_moments_from_graph', 'pmf_from_graph'): wrap_factory(m)
