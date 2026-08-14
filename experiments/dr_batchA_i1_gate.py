"""Batch A I1 micro-gates (plan v2, G4-fold corrected 2026-08-14): rewards
in the moments adjoint.

Modes:
  dump <npz>   -- PRE-A install (main checkout): rewardless Jacobians on
                  the fixtures (byte-identity goldens; the old pybind has
                  no rewards kwarg).
  check <npz>  -- NEW build:
                  (a) rewardless byte-identity vs the dump;
                  (b) LINEAR rewards vs FD-of-the-PRIMAL (the independent
                      anchor; the plan's dense-JAX-oracle leg was dropped
                      as redundant -- deviation recorded in
                      b3-batchA-findings.md) at nr_moments 1..3, incl.
                      mixed/all-ones/extreme-scale rewards (all-ones ==
                      rewardless bitwise-class);
                  (b2) LOG-mode rewards vs FD-of-the-log-PRIMAL (G4-fold
                      addition: the review found this wrapper's rewards
                      arm had ZERO empirical coverage -- the plan claimed
                      it, the original gate tested linear only);
                  (c) the dph rewards CONTRACT on both discrete sub-kinds
                      (was_dph via discretize(), native DPH): rewardless
                      COMPUTES, rewards DECLINE -- discriminating proof
                      the decline is rewards-caused, not graph-kind-
                      caused. The numeric REFUTATION evidence itself
                      (c2d correction invalid under reward weighting)
                      lives in the plan review's direct computation
                      (b3-batchA-plan.md), not here;
                  (d) rewards_len mismatch declines in all three wrappers.
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
MODE, NPZ = sys.argv[1], sys.argv[2]


def _chain4():
    # the atlas headline fixture: 4-vertex chain, one parameter
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    s.add_edge(v[0], 1.0)
    for i in range(3):
        v[i].add_edge_parameterized(v[i + 1], 0.0, [float(i + 1)])
    va = g.find_or_create_vertex([9])
    v[3].add_edge_parameterized(va, 0.0, [2.0])
    return g


def _branchy():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    v[0].add_edge_parameterized(v[1], 0.0, [1.0])
    v[0].add_edge_parameterized(v[2], 0.0, [2.0])
    v[1].add_edge_parameterized(v[3], 0.0, [1.5])
    v[2].add_edge_parameterized(v[3], 0.0, [0.5])
    v[3].add_edge_parameterized(va, 0.0, [1.0])
    return g


THETA = [2.0]
K = 3


def _moments_primal(g, theta, rewards, k):
    g.update_weights(list(theta))
    if rewards is None:
        return np.asarray(g.moments(k))
    return np.asarray(g.moments(k, list(rewards)))


def _fd_primal_jac(g, theta, rewards, k, rel=1e-6):
    th = np.asarray(theta, float)
    J = np.empty((k, th.size))
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-10)
        tp = th.copy(); tp[j] += h
        tm = th.copy(); tm[j] -= h
        J[:, j] = (_moments_primal(g, tp, rewards, k)
                   - _moments_primal(g, tm, rewards, k)) / (2 * h)
    return J


fixtures = {"chain4": _chain4, "branchy": _branchy}

if MODE == "dump":
    out = {}
    for name, mk in fixtures.items():
        g = mk()
        g.update_weights(THETA)
        out[f"{name}_linear"] = np.asarray(g._moments_grad_theta(K))
    np.savez(NPZ, **out)
    print("dumped:", {k: v.shape for k, v in out.items()})
    sys.exit(0)

assert MODE == "check"
old = np.load(NPZ)
fails = []

print("== (a) rewardless byte-identity vs pre-A install ==")
for name, mk in fixtures.items():
    g = mk()
    g.update_weights(THETA)
    new = np.asarray(g._moments_grad_theta(K))
    same = np.array_equal(new, old[f"{name}_linear"])
    print(f"  {name}: identical={same}")
    if not same:
        fails.append(f"(a) {name}")

print("== (b) linear rewards vs primal-FD (incl. headline + ones + extreme) ==")
for name, mk in fixtures.items():
    g = mk()
    n = g.vertices_length()
    _mixed = np.ones(n); _mixed[1] = 0.0; _mixed[2] = 2.0
    _mixed[3] = 3.0; _mixed[-1] = 0.5
    _extreme = np.ones(n); _extreme[1] = 1e6; _extreme[2] = 1e-6
    for rlabel, rw in (("mixed", _mixed),
                       ("ones", np.ones(n)),
                       ("extreme", _extreme)):
        g.update_weights(THETA)
        J = np.asarray(g._moments_grad_theta(K, rewards=rw.tolist()))
        if J.size != K:
            print(f"  {name}/{rlabel}: DECLINED (size {J.size})")
            if rlabel != "extreme":
                fails.append(f"(b) {name}/{rlabel} declined")
            continue
        J = J.reshape(K, 1)
        J_fd = _fd_primal_jac(g, THETA, rw, K)
        rel = np.max(np.abs(J - J_fd)) / max(np.max(np.abs(J_fd)), 1e-300)
        status = "PASS" if rel < 1e-5 else "FAIL"
        print(f"  {name}/{rlabel}: exact vs primal-FD rel={rel:.2e} {status}")
        if rel >= 1e-5:
            fails.append(f"(b) {name}/{rlabel}")
        if rlabel == "ones":
            g.update_weights(THETA)
            J0 = np.asarray(g._moments_grad_theta(K)).reshape(K, 1)
            same = np.array_equal(J, J0)
            print(f"  {name}/ones == rewardless: {same}")
            if not same:
                # x*1.0 bit-stability: report, tolerate 1-ulp
                d = np.max(np.abs(J - J0))
                print(f"    (max abs diff {d:.3e})")
                if d > 1e-15 * np.max(np.abs(J0)):
                    fails.append(f"(b) {name} ones!=rewardless")

# the headline value check: seed-only would give the WRONG 2nd moment.
g = _chain4()
_n = g.vertices_length()
rw = np.ones(_n); rw[1] = 0.0; rw[2] = 2.0; rw[3] = 3.0; rw[-1] = 0.5
m = _moments_primal(g, THETA, rw, 2)
print(f"  headline primal moments (rewarded): {m.tolist()}")

print("== (b2) LOG-mode rewards vs primal-FD (G4-fold addition) ==")


def _log2():
    # 2-param log fixture: every edge's full product c0*t0*c1*t1 > 0 at
    # THETA_LOG (log mode multiplies ALL coefficient*theta pairs)
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_mode = 'log'
    return g


THETA_LOG = [1.0, 2.0]


def _moments_primal_log(g, theta, rewards, k):
    g.update_weights(list(theta), log=True)
    if rewards is None:
        return np.asarray(g.moments(k))
    return np.asarray(g.moments(k, list(rewards)))


def _fd_primal_jac_log(g, theta, rewards, k, rel=1e-6):
    th = np.asarray(theta, float)
    J = np.empty((k, th.size))
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-10)
        tp = th.copy(); tp[j] += h
        tm = th.copy(); tm[j] -= h
        J[:, j] = (_moments_primal_log(g, tp, rewards, k)
                   - _moments_primal_log(g, tm, rewards, k)) / (2 * h)
    return J


gl = _log2()
_nl = gl.vertices_length()
rw_log = np.ones(_nl); rw_log[1] = 0.0; rw_log[2] = 2.0; rw_log[-1] = 0.5
# NOTE (G4-fold lesson): the private wrapper reads the graph's CURRENT
# weight state for the tape -- without update_weights(theta, log=True)
# first it silently computes a Jacobian for the stale weights (the model
# callback handles this internally; direct callers must do it themselves)
gl.update_weights(THETA_LOG, log=True)
J_log = np.asarray(gl._moments_grad_theta_log(K, THETA_LOG,
                                              rewards=rw_log.tolist()))
if J_log.size != K * len(THETA_LOG):
    print(f"  log/mixed: DECLINED (size {J_log.size})")
    fails.append("(b2) log/mixed declined")
else:
    J_log = J_log.reshape(K, len(THETA_LOG))
    J_log_fd = _fd_primal_jac_log(gl, THETA_LOG, rw_log, K)
    rel = (np.max(np.abs(J_log - J_log_fd))
           / max(np.max(np.abs(J_log_fd)), 1e-300))
    status = "PASS" if rel < 1e-5 else "FAIL"
    print(f"  log/mixed: exact vs primal-FD rel={rel:.2e} {status}")
    if rel >= 1e-5:
        fails.append("(b2) log/mixed")
# all-ones == rewardless, log wrapper
gl.update_weights(THETA_LOG, log=True)
J_lo = np.asarray(gl._moments_grad_theta_log(K, THETA_LOG,
                                             rewards=[1.0] * _nl))
J_l0 = np.asarray(gl._moments_grad_theta_log(K, THETA_LOG))
same = np.array_equal(J_lo, J_l0)
print(f"  log ones == rewardless: {same}")
if not same:
    fails.append("(b2) log ones!=rewardless")

print("== (c) dph rewards CONTRACT on both discrete sub-kinds ==")


def _wasdph():
    # 2-stage Erlang (P=1), discretize()'d: was_dph=True
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    return g.discretize(0.5)


def _dphnative():
    # native DPH (P=2): is_discrete=True, was_dph=False
    g = Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([3 - i]) for i in range(3)]
    s.add_edge(vs[0], 1.0)
    for i in range(2):
        coeff = [0.0] * 2
        coeff[i] = 1.0
        vs[i].add_edge(vs[i + 1], coeff)
    g.is_discrete = True
    return g


K_C = 2
# theta length = param_length() of the DISCRETIZED graph (discretize()
# ADDS a parameter slot: the P=1 erlang becomes P=2 -- G4-fold lesson,
# a hardcoded [0.3] was a length mismatch that correctly declined)
for label, mk in (("was_dph(discretize)", _wasdph),
                  ("native_dph", _dphnative)):
    gd = mk()
    th = [0.3] * gd.param_length()
    base = np.asarray(gd._moments_grad_theta_dph(K_C, th)).size
    withrw = np.asarray(gd._moments_grad_theta_dph(
        K_C, th, rewards=[1.0] * gd.vertices_length())).size
    ok = base == K_C * len(th) and withrw == 0
    print(f"  {label}: rewardless size={base} (expect {K_C * len(th)}), "
          f"rewards size={withrw} (expect 0) {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"(c) {label}")

print("== (d) rewards_len mismatch declines ==")
g = _chain4()
g.update_weights(THETA)
bad = [1.0, 2.0]  # wrong length
r1 = np.asarray(g._moments_grad_theta(K, rewards=bad)).size
r2 = np.asarray(g._moments_grad_theta_log(K, THETA, rewards=bad)).size
r3 = np.asarray(g._moments_grad_theta_dph(K, THETA, rewards=bad)).size
print(f"  linear/log/dph mismatch result sizes: {r1}/{r2}/{r3} (0 = declined)")
if r1 or r2 or r3:
    fails.append("(d) mismatch not declined")

print("\n" + ("ALL MICRO-GATES PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(0 if not fails else 1)
