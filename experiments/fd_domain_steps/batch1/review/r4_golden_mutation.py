"""Review probe r4: are the bit-identity goldens sensitive to a wrong band
selection? Mutate _fd_steps.probes so the BAND rule applies on a wider range
and re-run every golden cell; report which cells still pass (i.e. would NOT
detect the defect). Mutations:
  M_all : BAND applies for every t >= 0 (band wrongly unbounded)
  M_2x  : BAND applies for 0 <= t <= 2e-7 (band edge off by 2x)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
import jax, jax.numpy as jnp
import numpy as np
from phasic import _fd_steps
import _fd_golden_cells as C

GOLD = json.load(open(C.GOLDEN_PATH))
_orig_probes = _fd_steps.probes


def make(upper):
    def probes(theta, i, today_tp, today_tm, rule, info):
        if rule != _fd_steps.BAND:
            return _orig_probes(theta, i, today_tp, today_tm, rule, info)
        t = theta[i]
        up, dn = _fd_steps._positive_pair(t, _fd_steps.CAPPED)
        use = (t >= 0) if upper is None else ((t >= 0) & (t <= upper))
        tp = jnp.where(use, theta.at[i].set(up), today_tp)
        tm = jnp.where(use, theta.at[i].set(dn), today_tm)
        return tp, tm, use
    return probes


for label, upper in (("M_all", None), ("M_2x", 2.0000001e-7)):
    _fd_steps.probes = make(upper)
    undetected = []
    for name, theta in C.CELLS:
        for mode in C.modes_for(name):
            want = np.array([float.fromhex(h) for h in GOLD["grads"][C.key(name, theta, mode)]])
            got = np.ravel(C.grad_in_mode(name, list(theta), mode))
            if name in C.NON_REPRODUCIBLE:
                ok = np.allclose(got, want, rtol=C.NON_REPRODUCIBLE_RTOL, atol=0)
                maxrel = float(np.max(np.abs(got - want) / np.abs(want)))
            else:
                ok = got.tobytes() == want.tobytes()
                maxrel = float(np.max(np.abs(got - want) / np.abs(want)))
            if ok:
                undetected.append((name, theta, mode, maxrel))
            print(f"{label} {name:14s} {str(theta):28s} {mode:5s} golden-test-passes={ok} maxrel={maxrel:.3g}", flush=True)
    print(f"{label}: cells that would NOT detect the mutation: {len(undetected)}", flush=True)
_fd_steps.probes = _orig_probes
