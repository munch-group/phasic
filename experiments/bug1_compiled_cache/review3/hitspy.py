# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""pytest plugin (-p hitspy): on UNMODIFIED code, log every SVGD compiled-gradient
cache hit, and whether the instance that hits differs from the one that compiled
(data / prior / transform / sample moments).  Read-only spy; forwards unchanged."""
import os, hashlib
import numpy as np
LOG = os.environ.get('HITSPY_LOG', '/tmp/hitspy.log')
_fp = {}
def _h(a):
    try: return hashlib.sha1(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()[:10]
    except Exception: return 'id%d' % id(a)
def _prior(p):
    if p is None: return None
    try:
        v = {k: (float(x) if isinstance(x, (int, float)) else _h(x)) for k, x in sorted(vars(p).items())
             if isinstance(x, (int, float, np.ndarray)) or hasattr(x, 'shape')}
        return (type(p).__name__, tuple(v.items())) if v else (type(p).__name__, id(p))
    except TypeError:
        return (getattr(p, '__qualname__', type(p).__name__), id(p))
def fingerprint(s):
    od = s.observed_data
    data = (_h(od.values), _h(getattr(od, 'feature_indices', 0)), _h(getattr(od, 'slice_indices', 0))) if s._sparse_format else _h(od)
    pl = s.prior_list if s.prior_list is not None else [s.prior]
    try: pr = tuple(_prior(p) for p in pl)
    except Exception: pr = ('?', id(s.prior))
    tr = getattr(s.param_transform, '__qualname__', repr(type(s.param_transform)))
    return dict(data=data, prior=pr, transform=tr,
                moments=None if s.sample_moments is None else _h(s.sample_moments), inst=id(s))
def pytest_configure(config):
    from phasic.svgd import SVGD
    orig = SVGD._precompile_unified
    def spy(self, nr_moments, sample_moments, regularization, rewards=None, value_and_grad=False):
        rt = tuple(np.asarray(rewards).flatten()) if rewards is not None else None
        shp = (len(self.observed_data.values), self.observed_data.n_features) if self._sparse_format else self.observed_data.shape
        key = (id(self.model), self.theta_dim, shp, nr_moments, regularization, rt) + (('value_and_grad',) if value_and_grad else ())
        hit = key in SVGD._compiled_cache
        fp = fingerprint(self)
        if hit and key in _fp:
            old = _fp[key]
            diff = [k for k in ('data', 'prior', 'transform', 'moments') if old[k] != fp[k]]
            kind = ('STALE:' + ','.join(diff)) if diff else ('benign-same-instance' if old['inst'] == fp['inst'] else 'benign-other-instance')
            with open(LOG, 'a') as f:
                f.write(f"{kind}\t{os.environ.get('PYTEST_CURRENT_TEST', '?')}\n")
        elif not hit:
            _fp[key] = fp
        return orig(self, nr_moments, sample_moments, regularization, rewards, value_and_grad=value_and_grad)
    SVGD._precompile_unified = spy
