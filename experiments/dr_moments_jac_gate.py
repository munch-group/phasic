"""Batch-3 gate: the C moment-vector Jacobian (Graph._moments_grad_theta) vs
theta-perturbation central difference of native moments, on the two-stage model
(acyclic) and a cyclic fixture, across scales incl. mixed.
For the two-stage model (E[T]=1/t0+1/t1) the SECOND raw moment has a closed form
m1 = E[T^2] = 2(1/t0^2 + 1/t1^2 + 1/(t0 t1)), so d(m1)/dt0 = -4/t0^3 - 2/(t0^2 t1)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph

def two_stage():
    g = Graph(1); s = g.starting_vertex()
    v3=g.find_or_create_vertex([3]); v2=g.find_or_create_vertex([2]); v1=g.find_or_create_vertex([1])
    s.add_edge(v3,1.0); v3.add_edge(v2,[1.0,0.0]); v2.add_edge(v1,[0.0,1.0])
    return g

def two_cycle():
    g = Graph(1); s=g.starting_vertex()
    A=g.find_or_create_vertex([2]); B=g.find_or_create_vertex([1]); g.find_or_create_vertex([0])
    s.add_edge(A,1.0); A.add_edge(B,[1.0,0.0]); B.add_edge(A,[1.0,0.0]); B.add_edge(g.find_or_create_vertex([0]),[0.0,1.0])
    return g

def native_moments(build, theta, K):
    g = build(); g.update_weights(theta)
    return np.asarray(g.moments(K, discrete=False))[:K]

def moments_jac_cd(build, theta, K, j, eps):
    tp=list(theta); tm=list(theta); tp[j]+=eps; tm[j]-=eps
    return (native_moments(build,tp,K) - native_moments(build,tm,K))/(2*eps)  # d[m]/dtheta_j

def m1_closed_two_stage(t0,t1):  # d(E[T^2])/d[t0,t1]
    return np.array([-4/t0**3 - 2/(t0**2*t1), -4/t1**3 - 2/(t0*t1**2)])

ok_all=True; K=3
for name, build, thetas, closed in [
    ("two-stage", two_stage, [[1.0,1e-4],[1.0,1e-8],[2.0,0.5]], True),
    ("2-cycle",   two_cycle, [[1.0,1.0],[1.5,0.7],[2.0,0.5]], False)]:
    print(f"\n=== {name}: exact d[m_0..m_{K-1}]/dtheta (flat) vs native theta-CD ===")
    for theta in thetas:
        g=build(); g.update_weights(theta)
        Jflat = np.asarray(g._moments_grad_theta(K))
        P=len(theta)
        applicable = Jflat.size == K*P
        J = Jflat.reshape(K,P) if applicable else None
        # native theta-CD Jacobian (K x P)
        spread = max(theta)/min(theta); cd_reliable = spread <= 1e4
        line = f" theta={theta}: applicable={applicable}  dm1/dt (exact)={np.array2string(J[1], precision=5) if applicable else 'NA'}"
        print(line)
        if cd_reliable:
            cd = np.stack([moments_jac_cd(build, theta, K, j, 1e-6*max(theta[j],1e-9)) for j in range(P)], axis=1)
            ok_cd = applicable and np.allclose(J, cd, rtol=1e-4, atol=1e-6)
            ok_all &= ok_cd
            print(f"   native theta-CD dm1/dt = {np.array2string(cd[1], precision=5)}   [{'OK' if ok_cd else 'FAIL'}] exact==theta-CD (all moments)")
        else:
            print(f"   [FD-defect regime spread={spread:.0e}] native theta-CD unreliable; validating vs closed-form")
            ok_all &= applicable
        if closed and applicable:
            orc = m1_closed_two_stage(theta[0], theta[1])
            ok_closed = np.allclose(J[1], orc, rtol=1e-8)
            ok_all &= ok_closed
            print(f"   closed-form dm1/dt     = {np.array2string(orc, precision=5)}   [{'OK' if ok_closed else 'FAIL'}] exact==closed-form (moment 1)")

print(f"\n{'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
sys.exit(0 if ok_all else 1)
