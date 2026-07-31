"""Batch-2a gate: the PRODUCTION exact d(E[T])/dtheta (Graph._moment0_grad_theta)
-- env-free / thread-safe reverse tape adjoint + edge->theta linear contraction.
Validated against:
  - the CLOSED-FORM theta-gradient on the pin's OWN two-stage model
    (S->[3]--t0-->[2]--t1-->[1]): E[T]=1/t0+1/t1, grad=[-1/t0^2,-1/t1^2]
    -- INCLUDING the mixed scale theta=[1,1e-8] where FD is 4-9% wrong;
  - a theta-perturbation central difference (through update_weights) on cyclic
    fixtures, at benign scales.
Exits nonzero on any real failure.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def two_stage():  # the pin's model: acyclic chain, E[T]=1/t0+1/t1
    g = Graph(1); s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])   # t0
    v2.add_edge(v1, [0.0, 1.0])   # t1
    return g


def two_cycle():
    g = Graph(1); s = g.starting_vertex()
    A = g.find_or_create_vertex([2]); B = g.find_or_create_vertex([1]); g.find_or_create_vertex([0])
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0]); B.add_edge(A, [1.0, 0.0])
    B.add_edge(g.find_or_create_vertex([0]), [0.0, 1.0])
    return g


def theta_cd(build, theta, j, eps):
    tp = list(theta); tm = list(theta); tp[j] += eps; tm[j] -= eps
    gp = build(); gp.update_weights(tp); ep = float(np.asarray(gp.expected_waiting_time())[0])
    gm = build(); gm.update_weights(tm); em = float(np.asarray(gm.expected_waiting_time())[0])
    return (ep - em) / (2 * eps)


ok_all = True

print("=== two-stage (pin model): exact dtheta vs closed form [-1/t0^2, -1/t1^2] ===")
for theta in ([1.0, 1e-4], [1.0, 1e-8], [2.0, 0.5], [0.3, 3.0]):
    g = two_stage(); g.update_weights(theta)
    ewt, dth = g._moment0_grad_theta()
    dth = np.asarray(dth)
    native = float(np.asarray(g.expected_waiting_time())[0])
    closed_ET = 1.0/theta[0] + 1.0/theta[1]
    oracle = np.array([-1.0/theta[0]**2, -1.0/theta[1]**2])
    applicable = dth.size > 0
    ok_ewt = applicable and abs(ewt - native)/max(1.0, abs(native)) < 1e-9
    ok_grad = applicable and np.allclose(dth, oracle, rtol=1e-9, atol=1e-12)
    ok_all &= ok_ewt and ok_grad
    print(f" theta={theta}: E[T]={ewt:.10g} (closed {closed_ET:.10g}); "
          f"dtheta={np.array2string(dth, precision=6)} oracle={np.array2string(oracle, precision=6)}")
    print(f"   [{'OK' if ok_ewt else 'FAIL'}] E[T]==native  [{'OK' if ok_grad else 'FAIL'}] dtheta==closed-form"
          + ("" if applicable else "  [FAIL] not applicable (empty grad)"))

print("\n=== 2-cycle: exact dtheta vs theta-perturbation central difference ===")
for theta in ([1.0, 1.0], [1.5, 0.7], [2.0, 0.5]):
    g = two_cycle(); g.update_weights(theta)
    ewt, dth = g._moment0_grad_theta(); dth = np.asarray(dth)
    native = float(np.asarray(g.expected_waiting_time())[0])
    cd = np.array([theta_cd(two_cycle, theta, j, 1e-6*theta[j]) for j in range(len(theta))])
    applicable = dth.size > 0
    ok_ewt = applicable and abs(ewt - native)/max(1.0, abs(native)) < 1e-9
    ok_cd = applicable and np.allclose(dth, cd, rtol=1e-5, atol=1e-8)
    ok_all &= ok_ewt and ok_cd
    print(f" theta={theta}: E[T]={ewt:.8g}; dtheta={np.array2string(dth, precision=6)} "
          f"theta-CD={np.array2string(cd, precision=6)}")
    print(f"   [{'OK' if ok_ewt else 'FAIL'}] E[T]==native  [{'OK' if ok_cd else 'FAIL'}] dtheta==theta-CD")

print(f"\n{'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
sys.exit(0 if ok_all else 1)
