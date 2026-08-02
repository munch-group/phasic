"""B3 log-weight-mode gate: Graph._moments_grad_theta_log vs native
theta-perturbation central difference of graph.update_weights(theta,
log=True) + graph.moments(K), on continuous weight_mode='log' graphs.
Also gates the was_dph decline (mandatory, not defensive -- see
b3-log-weight-mode-plan.md), the MPFR conditioning decline (mirrors the
continuous/discrete gates), and quantifies (does not gate on) how readily
MPFR declines for a spread-theta log-mode graph vs the equivalent linear
graph. Re-runs the three existing gates as a no-regression check.

Run: pixi run python experiments/dr_log_mode_moments_jac_gate.py
"""
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def two_param_dense():
    """3-vertex chain, 2 params, EVERY edge has both coefficients nonzero
    (required for weight_mode='log')."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])
    g.weight_mode = 'log'
    return g


def three_param_two_edge():
    """4-vertex chain, 3 params, two edges each using all 3 params.

    NOTE (found via adversarial review): both edges' coefficient PRODUCTS
    are equal (1.0*0.5*2.0 == 2.0*1.0*0.5 == 1.0), so in log mode the two
    edges are numerically indistinguishable at every theta -- this fixture
    alone cannot catch an edge-index-mapping bug in the C contraction. See
    four_param_branching below for one that can."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.5, 2.0])
    v2.add_edge_parameterized(v1, 0.0, [2.0, 1.0, 0.5])
    g.weight_mode = 'log'
    return g


def four_param_branching():
    """5-vertex graph with a BRANCHING vertex (v3 has two out-edges) and 4
    params, where every edge's coefficient product is genuinely different
    (1.0, 10.5, 1.5, 0.5) -- the four edges have distinguishable weight
    values at every theta, so an edge-index-mapping bug would be caught."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [2.0, 0.1, 5.0, 1.0])     # prod c = 1.0
    v3.add_edge_parameterized(v1, 0.0, [0.5, 3.0, 1.0, 7.0])     # prod c = 10.5 (branch)
    v2.add_edge_parameterized(v0, 0.0, [1.0, 2.0, 0.25, 3.0])    # prod c = 1.5
    v1.add_edge_parameterized(v0, 0.0, [4.0, 0.5, 2.0, 0.125])   # prod c = 0.5
    g.weight_mode = 'log'
    return g


def closed_form_log_mode_jacobian(theta, m):
    """Exact oracle for ANY log-mode graph: in log mode w_e =
    prod_i(c_e[i]*theta_i) = (prod_i theta_i) * (prod_i c_e[i]) -- the
    theta-product factors out identically across every edge, so the raw
    moments are homogeneous of degree -r in theta: m_r(theta) =
    m_r(1)*(prod_i theta_i)^(-r), giving dm_r/dtheta_j = -r*m_r/theta_j
    exactly, independent of the specific graph topology/coefficients."""
    theta = np.asarray(theta, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    K, P = len(m), len(theta)
    return np.array([[-(k + 1) * m[k] / theta[j] for j in range(P)] for k in range(K)])


def native_moments(build, theta, K):
    g = build()
    g.update_weights(list(theta), log=True)
    return np.asarray(g.moments(K))[:K]


def moments_jac_cd(build, theta, K, j, eps):
    tp = list(theta); tm = list(theta)
    tp[j] += eps; tm[j] -= eps
    return (native_moments(build, tp, K) - native_moments(build, tm, K)) / (2 * eps)


def exact_jac(build, theta, K):
    g = build()
    g.update_weights(list(theta), log=True)
    raw = g._moments_grad_theta_log(K, list(theta))
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

    print("=== continuous weight_mode='log' ===")
    ok_all &= check_case("two_param_dense", two_param_dense, [2.0, 3.0], 2)
    ok_all &= check_case("two_param_dense", two_param_dense, [2.0, 3.0], 3)
    ok_all &= check_case("two_param_dense", two_param_dense, [0.5, 5.0], 2)
    ok_all &= check_case("three_param_two_edge", three_param_two_edge, [1.0, 2.0, 0.5], 2)
    ok_all &= check_case("three_param_two_edge", three_param_two_edge, [1.0, 2.0, 0.5], 3)

    print("\n=== branching, distinguishable-edge fixture vs EXACT closed form (not CD) ===")
    theta_b = [1e3, 1e-2, 5.0, 0.2]
    K_b = 3
    g_native = four_param_branching()
    g_native.update_weights(theta_b, log=True)
    m_b = np.asarray(g_native.moments(K_b))
    J_closed = closed_form_log_mode_jacobian(theta_b, m_b)
    g_exact = four_param_branching()
    g_exact.update_weights(theta_b, log=True)
    J_exact = np.asarray(g_exact._moments_grad_theta_log(K_b, theta_b)).reshape(K_b, 4)
    err_b = np.max(np.abs(J_closed - J_exact) / np.maximum(np.abs(J_closed), 1e-300))
    ok_b = err_b < 1e-9
    print(f"  four_param_branching theta={theta_b}: max_rel_err(closed_form, exact)={err_b:.3e} "
          f"{'OK' if ok_b else 'FAIL'}")
    ok_all &= ok_b

    print("\n=== mixed-scale theta (log mode) ===")
    ok_all &= check_case("two_param_dense mixed", two_param_dense, [1.0, 1e-3], 2,
                          eps=1e-6, rtol=1e-3, atol=1e-4)
    ok_all &= check_case("two_param_dense mixed", two_param_dense, [1.0, 1e3], 2,
                          eps=1e-6, rtol=1e-3, atol=1e-4)

    print("\n=== was_dph decline (mandatory exclusion, see plan D1.2) ===")
    g = two_param_dense()
    gd = g.discretize(lambda state, **kw: [0.5, 0.5])
    gd.weight_mode = 'log'
    gd.update_weights([1.0, 2.0], log=True)
    n_decline = len(gd._moments_grad_theta_log(2, [1.0, 2.0]))
    print(f"  discretize()+log (was_dph=True): exact J size={n_decline} (expect 0 -> FD fallback)")
    ok_all &= (n_decline == 0)

    print("\n=== MPFR conditioning decline ===")
    g2 = two_param_dense()
    g2.update_weights([1.0, 1e-13], log=True)
    n_gate = len(g2._moments_grad_theta_log(2, [1.0, 1e-13]))
    print(f"  ill-conditioned theta=[1,1e-13]: exact J size={n_gate} (expect 0 -> FD fallback)")
    ok_all &= (n_gate == 0)

    g3 = two_param_dense()
    g3.update_weights([1.0, 0.5], log=True)
    n_ok = len(g3._moments_grad_theta_log(2, [1.0, 0.5]))
    print(f"  well-conditioned theta=[1,0.5]: exact J size={n_ok} (expect 4)")
    ok_all &= (n_ok == 4)

    print("\n=== MPFR decline rate under log mode (quantify, don't gate) ===")
    # A log-mode edge's effective tape multiplier scales MULTIPLICATIVELY
    # with each theta component (unlike linear's additive dot product), so a
    # spread-theta log graph's condition number can grow faster than the
    # equivalent linear graph's. Report where the decline threshold sits;
    # this is informational (per the plan review), not a pass/fail gate.
    for spread_exp in [2, 4, 6, 8, 10, 12]:
        theta = [1.0, 10.0 ** (-spread_exp)]
        g4 = two_param_dense()
        g4.update_weights(theta, log=True)
        n = len(g4._moments_grad_theta_log(2, theta))
        print(f"  spread=1e{spread_exp}: theta={theta} exact_applicable={n == 4}")

    print("\n=== no-regression: existing gates (unmodified functions) ===")
    for script in ["experiments/dr_moments_jac_gate.py", "experiments/dr_mpfr_gate_test.py",
                   "experiments/dr_dph_moments_jac_gate.py"]:
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
