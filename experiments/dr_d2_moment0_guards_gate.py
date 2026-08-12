"""Batch D Tier-1 / D.2 gate: the two guards backported to ptd_moment0_grad_theta
(validators build only). The guards have OPPOSITE expected behaviors (adversarial
plan review, finding 3):

  guard 1 (coefficients_length==0 skip): a graph with a CONSTANT tape-input edge
      must SUCCEED at benign theta (constant edge contributes zero gradient) and
      match a theta central difference — pre-fix this crashed (NULL deref).
  guard 2 (MPFR decline): the same fixtures must DECLINE (empty grad) under
      PHASIC_FORCE_MPFR=1 and at the naturally ill-conditioned theta=[1,1e-13].

Exits nonzero on any failure.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def aux_constant_edge():
    # Parameterized chain + a constant-weight aux loop: the guard-1 crash
    # class (coefficient-less constant tape-input edges), the exact building
    # block joint_stop_prob_graph() uses — see test_aux_vertex_constant.py.
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2])
    va = g.find_or_create_vertex([0])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])   # t0 (array syntax auto-parameterizes)
    v2.add_edge(va, [0.0, 1.0])   # t1
    v1.add_aux_vertex_constant(1.0)
    return g


def two_stage():
    g = Graph(1); s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0]); v2.add_edge(v1, [0.0, 1.0])
    return g


def theta_cd(build, theta, j, eps=1e-6):
    tp = list(theta); tm = list(theta); tp[j] += eps; tm[j] -= eps
    gp = build(); gp.update_weights(tp); ep = float(np.asarray(gp.expected_waiting_time())[0])
    gm = build(); gm.update_weights(tm); em = float(np.asarray(gm.expected_waiting_time())[0])
    return (ep - em) / (2 * eps)


ok_all = True

print("=== guard 1: constant tape-input edge at benign theta -> SUCCEEDS, matches CD ===")
for theta in ([1.0, 2.0], [0.5, 3.0]):
    g = aux_constant_edge(); g.update_weights(theta)
    ewt, dth = g._moment0_grad_theta()
    dth = np.asarray(dth)
    applicable = dth.size > 0
    if not applicable:
        print(f" theta={theta}: [FAIL] declined — guard 1 must SUCCEED at benign theta")
        ok_all = False
        continue
    native = float(np.asarray(g.expected_waiting_time())[0])
    cd = np.array([theta_cd(aux_constant_edge, theta, j) for j in range(2)])
    ok_ewt = abs(ewt - native) / max(1.0, abs(native)) < 1e-9
    ok_grad = np.allclose(dth, cd, rtol=1e-5, atol=1e-10)
    ok_all &= ok_ewt and ok_grad
    print(f" theta={theta}: E[T]={ewt:.10g} dtheta={np.array2string(dth, precision=8)} "
          f"CD={np.array2string(cd, precision=8)} "
          f"[{'OK' if (ok_ewt and ok_grad) else 'FAIL'}]")

# MPFR detectability probe (plan requirement): without HAVE_MPFR compiled in,
# ptd_dbg_tape_needs_mpfr returns 0 unconditionally (even under FORCE_MPFR),
# so the decline assertions would mis-report the guards as regressed on a
# non-MPFR build. Probe: FORCE_MPFR on a benign graph must decline iff MPFR
# is compiled.
os.environ['PHASIC_FORCE_MPFR'] = '1'
try:
    _g = two_stage(); _g.update_weights([1.0, 2.0])
    _, _dth = _g._moment0_grad_theta()
    HAVE_MPFR = np.asarray(_dth).size == 0
finally:
    del os.environ['PHASIC_FORCE_MPFR']

if not HAVE_MPFR:
    print("=== guards 2a/2b: SKIPPED (build lacks HAVE_MPFR; the gate is "
          "inert by design — decline assertions not applicable) ===")
else:
    print("=== guard 2a: PHASIC_FORCE_MPFR=1 -> DECLINES (empty grad) ===")
    os.environ['PHASIC_FORCE_MPFR'] = '1'
    try:
        for build in (two_stage, aux_constant_edge):
            g = build(); g.update_weights([1.0, 2.0])
            ewt, dth = g._moment0_grad_theta()
            declined = np.asarray(dth).size == 0
            ok_all &= declined
            print(f" {build.__name__}: [{'OK' if declined else 'FAIL'}] "
                  f"{'declined' if declined else 'DID NOT decline under FORCE_MPFR'}")
    finally:
        del os.environ['PHASIC_FORCE_MPFR']

    print("=== guard 2b: naturally ill-conditioned theta=[1,1e-13] -> DECLINES ===")
    g = two_stage(); g.update_weights([1.0, 1e-13])
    ewt, dth = g._moment0_grad_theta()
    declined = np.asarray(dth).size == 0
    ok_all &= declined
    print(f" two_stage: [{'OK' if declined else 'FAIL'}] "
          f"{'declined' if declined else 'DID NOT decline at cond~1e13'}")

print("=== guard 2c: benign theta WITHOUT force -> still succeeds (no false positive) ===")
g = two_stage(); g.update_weights([1.0, 2.0])
ewt, dth = g._moment0_grad_theta()
applicable = np.asarray(dth).size > 0
ok_all &= applicable
print(f" two_stage: [{'OK' if applicable else 'FAIL'}] "
      f"{'applicable' if applicable else 'falsely declined'}")

print("ALL PASS" if ok_all else "FAILURES")
sys.exit(0 if ok_all else 1)
