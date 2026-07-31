"""B3 discrete/was_dph gate: Graph._moments_grad_theta_dph vs native
theta-perturbation central difference of graph.moments(K, discrete=True), on
was_dph graphs (Graph.discretize(), renormalised sibling-coupled edges) and
native DPH graphs (is_discrete only, was_dph=False, plain linear contraction).
Also gates the MPFR-conditioning decline (same gate as the continuous path)
and confirms no regression on the continuous gates.

Run: pixi run python experiments/dr_dph_moments_jac_gate.py
"""
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def erlang():
    """2-stage parameterized Erlang (continuous), to feed discretize()."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    return g


def chain(n):
    """n-stage parameterized chain (continuous), to feed discretize()."""
    g = Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        vs[i].add_edge(vs[i + 1], [1.0 if j == i else 0.0 for j in range(n)])
    return g


def dph_native(probs):
    """Native DPH: is_discrete=True, was_dph=False (edge weight IS c.theta
    directly -- plain linear contraction, no renormalisation)."""
    g = Graph(1)
    s = g.starting_vertex()
    n = len(probs)
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        coeff = [0.0] * n
        coeff[i] = 1.0
        vs[i].add_edge(vs[i + 1], coeff)
    g.is_discrete = True
    return g


def native_moments(build, theta, K):
    g = build()
    g.update_weights(list(theta))
    return np.asarray(g.moments(K, discrete=True))[:K]


def moments_jac_cd(build, theta, K, j, eps):
    tp = list(theta); tm = list(theta)
    tp[j] += eps; tm[j] -= eps
    return (native_moments(build, tp, K) - native_moments(build, tm, K)) / (2 * eps)


def exact_jac(build, theta, K):
    g = build()
    g.update_weights(list(theta))
    raw = g._moments_grad_theta_dph(K, list(theta))
    if not raw:
        return None
    return np.asarray(raw).reshape(K, len(theta))


def check_case(name, build, theta, K, eps=1e-6, rtol=1e-4, atol=1e-6):
    P = len(theta)
    J = exact_jac(build, theta, K)
    applicable = J is not None
    print(f"  {name}: theta={theta} K={K} applicable={applicable}")
    if not applicable:
        return False
    cd = np.stack([moments_jac_cd(build, theta, K, j, eps * max(abs(theta[j]), 1e-9)) for j in range(P)], axis=1)
    ok = np.allclose(J, cd, rtol=rtol, atol=atol)
    if not ok:
        print(f"    exact={J}")
        print(f"    cd   ={cd}")
        print(f"    max abs diff={np.max(np.abs(J-cd)):.3e}")
    print(f"    exact==native-CD: {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ok_all = True

    print("=== was_dph (Graph.discretize()): renorm quotient-rule (sibling coupling) ===")
    ok_all &= check_case("erlang.discretize(0.5)", lambda: erlang().discretize(0.5), [0.3, 0.4], 2)
    ok_all &= check_case("erlang.discretize(0.5)", lambda: erlang().discretize(0.5), [0.3, 0.4], 3)
    ok_all &= check_case("erlang.discretize(0.5)", lambda: erlang().discretize(0.5), [0.8, 1.2], 2)
    ok_all &= check_case("chain(2).discretize(0.3)", lambda: chain(2).discretize(0.3), [0.4, 0.6, 1.0], 2)
    ok_all &= check_case("chain(2).discretize(0.3)", lambda: chain(2).discretize(0.3), [0.4, 0.6, 1.0], 3)
    ok_all &= check_case("chain(3).discretize(0.2)", lambda: chain(3).discretize(0.2), [0.3, 0.5, 0.7, 0.9], 2)

    print("\n=== moderately mixed scale (was_dph) ===")
    ok_all &= check_case("erlang.discretize(0.5) mixed", lambda: erlang().discretize(0.5), [1.0, 1e-3], 2,
                          eps=1e-6, rtol=1e-3, atol=1e-4)

    print("\n=== native DPH (is_discrete only, was_dph=False): plain linear contraction ===")
    ok_all &= check_case("dph_native(2-stage)", lambda: dph_native((1.0, 1.0)), [0.3, 0.4], 2)
    ok_all &= check_case("dph_native(2-stage)", lambda: dph_native((1.0, 1.0)), [0.2, 0.6], 3)
    ok_all &= check_case("dph_native(3-stage)", lambda: dph_native((1.0, 1.0, 1.0)), [0.3, 0.4, 0.5], 3)

    print("\n=== MPFR conditioning decline (mirrors the continuous gate) ===")
    g = erlang().discretize(0.5)
    g.update_weights([1.0, 1e-13])
    n_gate = len(g._moments_grad_theta_dph(2, [1.0, 1e-13]))
    print(f"  ill-conditioned theta=[1,1e-13]: exact J size={n_gate} (expect 0 -> FD fallback)")
    ok_all &= (n_gate == 0)

    g2 = erlang().discretize(0.5)
    g2.update_weights([1.0, 0.5])
    n_ok = len(g2._moments_grad_theta_dph(2, [1.0, 0.5]))
    print(f"  well-conditioned theta=[1,0.5]: exact J size={n_ok} (expect 4)")
    ok_all &= (n_ok == 4)

    print("\n=== no-regression: continuous gates (unmodified ptd_moments_grad_theta) ===")
    for script in ["experiments/dr_moments_jac_gate.py", "experiments/dr_mpfr_gate_test.py"]:
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        passed = r.returncode == 0 and "ALL PASS" in r.stdout
        print(f"  {script}: {'ALL PASS' if passed else 'FAIL'}")
        if not passed:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
        ok_all &= passed

    print("\n" + ("ALL PASS" if ok_all else "FAILURES PRESENT"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
