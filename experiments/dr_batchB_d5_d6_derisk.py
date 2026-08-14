"""Batch B pre-implementation de-risks D-B5 + D-B6 (plan v2 §F).

Run on the PRE-B build (branch base ae217b0e): pins the contracts the
implementation must satisfy, on live behavior — nothing here exercises
Batch-B code (none exists yet).

D-B5: the non-finite-gradient fixture for the isfinite-sweep fall-through
  cell. v1's fixture (log(t0*c0), zero-c0 edge) was refuted at plan
  review: it fails in the PRIMAL. The corrected fixture sqrt(t0 - c0)
  has weight 0 LEGAL at t0 == c0 (gradient inf there) — verify the
  primal's actual domain behavior so the future gate cell anchors on a
  constructible point, and record what FD does there today.

D-B6: the theta-dimension contract routes (plan v2 §A):
  (1) lazy-build decoupled (C param_length locked by coefficient length,
      formula uses fewer thetas) — the class the static decline must
      catch: pin the detection predicate and confirm the model
      forward/FD-grad/svgd front door all WORK today (no regression
      allowed when the decline routes them to FD explicitly);
  (2) canonical set_param_length decoupling — param_length == n_theta:
      must be classified ALIGNED (exact engages post-B);
  (3) plain aligned (coeff length == n_theta, no set_param_length).
  Also the D-B2 native-DPH arm: is_discrete=True + formula is
  constructible (free Python attribute) and invisible to C — confirm
  the Python _effective_discrete gate is what stands between it and
  the future wrapper.
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


print("== D-B5: sqrt(t0 - c0) fixture domain behavior (primal, pre-B) ==")


def _sqrt_fixture():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])  # c0 = 1.0
    g.weight_formula = "sqrt(t0 - c0) + 0.5"
    return g


# NOTE the +0.5: a pure sqrt(t0-c0) edge would have weight 0 at t0==c0,
# which kills the chain (absorption never happens -> infinite moments);
# +0.5 keeps the primal healthy while sqrt's INNER gradient still blows
# up at t0==c0 (d/dt0 sqrt(t0-1) = inf at t0=1). Probe both variants.
g = _sqrt_fixture()
try:
    g.update_weights([1.0])   # sqrt(0) = 0 -> weight 0.5, legal
    m = np.asarray(g.moments(2))
    check("D-B5a weight at sqrt-boundary legal (w=0.5)", np.all(np.isfinite(m)),
          f"moments={m.tolist()}")
except Exception as exc:
    check("D-B5a weight at sqrt-boundary legal (w=0.5)", False, repr(exc))

g2 = _sqrt_fixture()
try:
    g2.update_weights([0.5])  # sqrt(-0.5) = NaN -> weight NaN
    check("D-B5b negative-domain theta rejected by primal", False,
          "update_weights ACCEPTED a NaN-weight theta")
except Exception as exc:
    check("D-B5b negative-domain theta rejected by primal", True,
          type(exc).__name__)

# pure sqrt variant: weight exactly 0 at the boundary
g3 = Graph(1)
s3 = g3.starting_vertex()
a3 = g3.find_or_create_vertex([2])
b3 = g3.find_or_create_vertex([1])
s3.add_edge(a3, 1.0)
a3.add_edge(b3, [1.0])
g3.weight_formula = "sqrt(t0 - c0)"
try:
    g3.update_weights([1.0])
    m3 = np.asarray(g3.moments(1))
    print(f"  D-B5c pure-sqrt weight-0 primal: accepted, moments={m3.tolist()}"
          " (record: finite?)", np.all(np.isfinite(m3)))
except Exception as exc:
    print(f"  D-B5c pure-sqrt weight-0 primal: raises {type(exc).__name__}"
          " (record only -- gate cell must use the +0.5 variant)")

print("== D-B6: theta-dimension contract routes ==")
jax_ok = True
try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax_ok = False
    print("  (jax unavailable -- model-level routes skipped: FAIL)")
    FAILS.append("D-B6 jax unavailable")

if jax_ok:
    # (1) lazy-build decoupled: coeff length 2 locks C param_length=2,
    # formula uses only t0 -> model theta_dim resolves to 1
    def _lazy():
        g = Graph(1)
        s = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        s.add_edge(v2, 1.0)
        v2.add_edge(v1, [2.0, 0.5])
        g.weight_formula = "t0*c0 + c1"
        return g

    gl = _lazy()
    ser = gl.serialize()
    c_pl = gl.param_length()
    n_theta = ser.get("weight_formula_tape", {}).get("n_theta")
    check("D-B6.1a lazy class detectable", c_pl == 2 and n_theta == 1,
          f"C param_length={c_pl}, tape n_theta={n_theta}, "
          f"serialized param_length={ser.get('param_length')}")
    model = Graph.pmf_and_moments_from_graph(gl, nr_moments=2,
                                             discrete=False, theta_dim=1)
    theta = jnp.asarray([0.7])
    times = jnp.asarray([0.5, 1.0])
    pmf, mom = model(theta, times)
    check("D-B6.1b lazy forward works (theta_dim=1)",
          bool(np.all(np.isfinite(np.asarray(pmf)))))
    grad = jax.grad(lambda th: float(0) + jnp.sum(model(th, times)[1]))(theta)
    check("D-B6.1c lazy FD gradient works today",
          bool(np.all(np.isfinite(np.asarray(grad)))),
          f"grad={np.asarray(grad).tolist()}")
    # the clone-raise the static decline must prevent from ever being hit:
    gl2 = _lazy()
    clone = gl2.clone()
    try:
        clone.update_weights([0.7])
        check("D-B6.1d clone.update_weights(short theta) raises (the hazard)",
              False, "did NOT raise -- v2 SS-A premise wrong, re-review")
    except Exception as exc:
        check("D-B6.1d clone.update_weights(short theta) raises (the hazard)",
              True, type(exc).__name__)
    # svgd front door (auto-inferred theta_dim)
    fit = gl.svgd(np.asarray([0.5, 1.0, 1.5]), n_iterations=1, n_particles=4)
    check("D-B6.1e lazy svgd front door works today",
          bool(np.all(np.isfinite(np.asarray(fit.particles)))))

    # (2) canonical set_param_length decoupling: param_length == n_theta
    def _canonical():
        g = Graph(1)
        g.set_param_length(2)
        s = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        s.add_edge(v2, 1.0)
        v2.add_edge(v1, [1.5, 2.0, 0.5])  # 3 coeffs, 2 thetas
        g.weight_formula = "t0*c0 + t1*c1 + c2"
        return g

    try:
        gc = _canonical()
        check("D-B6.2a canonical aligned (param_length==n_theta)",
              gc.param_length() == 2,
              f"param_length={gc.param_length()}")
        gc.update_weights([0.4, 0.9])
        check("D-B6.2b canonical update_weights(n_theta) works", True)
    except Exception as exc:
        check("D-B6.2 canonical route", False, repr(exc))

    # (3) plain aligned
    def _aligned():
        g = Graph(1)
        s = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        s.add_edge(v2, 1.0)
        v2.add_edge(v1, [2.0, 0.5])
        g.weight_formula = "t0*c0 + t1*c1"
        return g

    ga = _aligned()
    check("D-B6.3 plain aligned (coeff len == n_theta == param_length)",
          ga.param_length() == 2)

    # D-B2 native-DPH arm: constructible, invisible to C
    gd = _aligned()
    gd.is_discrete = True
    check("D-B6.4 native-DPH x formula constructible (Python attr only)",
          bool(gd.is_discrete))

print()
print("ALL DE-RISKS PASS" if not FAILS else f"FAILURES: {FAILS}")
import sys
sys.exit(0 if not FAILS else 1)
