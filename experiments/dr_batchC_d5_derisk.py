"""Batch C pre-implementation de-risk D-C5 (plan v2): discrete x callback
exclusion evidence -- silently computes or fails loudly? (The log/formula
D1 discipline: never assert an exclusion by analogy.)

Also re-runs the two decoupled-contract probes the plan review resolved
(v2 SS-A) as a durable branch artifact: update_weights(callback=) accepts
any theta length; the lazy aux-coeff idiom forward works.
"""
import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
FAILS = []


def check(label, ok, detail=""):
    print(f"  {label}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(label)


print("== D-C5a: was_dph (discretize) x callback ==")


def _erlang():
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    return g


# NOTE (first-run lesson): a callback mapping the discretize()-created
# aux edge's coefficients to weight 0 raises the positivity check -- a
# FIXTURE artifact, not mode evidence. Use a strictly-positive callback.
cb = lambda theta, coefficients: float(coefficients[0] * theta[0] + 0.1)  # noqa: E731
try:
    gd = _erlang().discretize(0.5)
    gd.weight_callback = cb
    gd.update_weights([0.3], callback=cb)
    m1 = np.asarray(gd.moments(2))
    gd.update_weights([0.6], callback=cb)
    m2 = np.asarray(gd.moments(2))
    print(f"  was_dph+callback: SILENTLY COMPUTES (moments {m1.tolist()} / "
          f"{m2.tolist()}) -- the exit wrapper's was_dph decline is "
          f"LOAD-BEARING, matching log/formula")
    dc5a = "silently-computes"
except Exception as exc:
    print(f"  was_dph+callback: raises {type(exc).__name__}: {exc}")
    dc5a = f"raises {type(exc).__name__}"

print("== D-C5b: native DPH (is_discrete attr) x callback ==")
gn = _erlang()
gn.weight_callback = cb
gn.is_discrete = True
try:
    gn.update_weights([0.3], callback=cb)
    print("  native-DPH+callback update_weights: ACCEPTED (C cannot see "
          "is_discrete -- the Python _effective_discrete gate is the only "
          "defense, as with formula)")
    dc5b = "silently-computes"
except Exception as exc:
    print(f"  native-DPH+callback: raises {type(exc).__name__}")
    dc5b = f"raises {type(exc).__name__}"

check("D-C5 evidence recorded", True, f"was_dph={dc5a}; native={dc5b}")

print("== v2 SS-A durable re-probe: theta-length freedom + lazy idiom ==")
gl = Graph(1)
s = gl.starting_vertex()
v2 = gl.find_or_create_vertex([2])
v1 = gl.find_or_create_vertex([1])
s.add_edge(v2, 1.0)
v2.add_edge(v1, [2.0, 0.5, 7.7])  # 3 coeffs (aux data) -> param_length 3
gl.weight_callback = lambda t, c: float(c[0] * t[0] + c[1] * t[1])
ok = True
for th in ([1.0, 2.0], [1.0], [0.5, 0.5, 0.5, 0.5]):
    try:
        gl.update_weights(th, callback=gl.weight_callback) \
            if len(th) >= 2 else gl.update_weights(
                th, callback=lambda t, c: float(c[0] * t[0]))
    except Exception as exc:
        ok = False
        print(f"    theta len {len(th)}: RAISED {type(exc).__name__}")
check("update_weights(callback=) accepts any theta length "
      "(phasiccpp.cpp:1879-1884 by design)", ok)
check("lazy aux-coeff idiom: param_length locked to coeff length",
      gl.param_length() == 3, f"param_length={gl.param_length()}")
clone = gl.clone()
clone.update_weights([1.5, 0.25], callback=gl.weight_callback)
gl.update_weights([1.5, 0.25], callback=gl.weight_callback)
w_c = np.asarray([e[2] for e in clone.serialize()['edges']], dtype=float) \
    if 'edges' in clone.serialize() else None
check("clone accepts short theta via callback (the B crash class is "
      "ABSENT here)", True)

print()
print("ALL DE-RISKS RECORDED" if not FAILS else f"FAILURES: {FAILS}")
import sys
sys.exit(0 if not FAILS else 1)
