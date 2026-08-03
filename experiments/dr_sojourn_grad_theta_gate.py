"""B3 joint-index gate: Graph._sojourn_grad_theta_subset vs native
theta-perturbation central difference of Graph.expected_sojourn_time(indices),
on continuous and native-DPH (is_discrete=True, was_dph=False) fixtures,
including realistic joint-probability graphs built the same way
tests/pytest/test_sojourn_subset_adjoint.py does. Also gates the was_dph
decline (mandatory, not defensive -- was_dph needs a different, deferred
quotient-rule contraction, see b3-joint-index-plan.md) and the MPFR
conditioning decline (mirrors the continuous moments gate), re-runs the
existing B3 gates as a no-regression check, and benchmarks wall-clock
(single-call exact vs 2xP-call FD) on a realistic joint-prob graph to
inform the D4 default decision.

Run: pixi run python experiments/dr_sojourn_grad_theta_gate.py
"""
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph

sys.path.insert(0, "tests/pytest")
from test_sojourn_subset_adjoint import _make_joint, _t_vertices  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def two_param_chain():
    """3-vertex chain, 2 params, s -> v2 -> v1 (v1 absorbing)."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])
    return g


def four_param_branching():
    """5-vertex graph with a branching vertex (v3 has two out-edges), 4
    params, every path terminating at the absorbing vertex v0."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [2.0, 0.1, 5.0, 1.0])
    v3.add_edge_parameterized(v1, 0.0, [0.5, 3.0, 1.0, 7.0])
    v2.add_edge_parameterized(v0, 0.0, [1.0, 2.0, 0.25, 3.0])
    v1.add_edge_parameterized(v0, 0.0, [4.0, 0.5, 2.0, 0.125])
    return g


def dph_native(probs):
    """Native DPH: is_discrete=True, was_dph=False (edge weight IS c.theta
    directly, no renormalisation) -- confirmed needing zero special-casing
    for sojourn (neither ComputeSojournTimesFfiImpl nor
    ptd_expected_sojourn_time_subset branch on is_discrete)."""
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


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------
def native_sojourn(build, theta, indices):
    g = build()
    g.update_weights(list(theta))
    return np.asarray(g.expected_sojourn_time(list(indices)))


def sojourn_jac_cd(build, theta, indices, j, eps):
    tp = list(theta); tm = list(theta)
    tp[j] += eps; tm[j] -= eps
    return (native_sojourn(build, tp, indices) - native_sojourn(build, tm, indices)) / (2 * eps)


def exact_jac(build, theta, indices):
    g = build()
    g.update_weights(list(theta))
    raw = g._sojourn_grad_theta_subset(list(indices))
    if not raw:
        return None
    return np.asarray(raw).reshape(len(indices), len(theta))


def check_case(name, build, theta, indices, eps=1e-6, rtol=1e-4, atol=1e-6):
    P = len(theta)
    J = exact_jac(build, theta, indices)
    applicable = J is not None
    print(f"  {name}: theta={theta} k={len(indices)} applicable={applicable}")
    if not applicable:
        return False
    cd = np.stack(
        [sojourn_jac_cd(build, theta, indices, j, eps * max(abs(theta[j]), 1e-9)) for j in range(P)],
        axis=1,
    )
    ok = np.allclose(J, cd, rtol=rtol, atol=atol)
    if not ok:
        print(f"    exact=\n{J}")
        print(f"    cd   =\n{cd}")
        print(f"    max abs diff={np.max(np.abs(J - cd)):.3e}")
    print(f"    exact==native-CD: {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ok_all = True

    print("=== continuous, weight_mode='linear' ===")
    ok_all &= check_case("two_param_chain", two_param_chain, [2.0, 3.0], [0, 1, 2])
    ok_all &= check_case("two_param_chain", two_param_chain, [0.5, 5.0], [0, 1, 2])
    ok_all &= check_case("four_param_branching", four_param_branching,
                          [1.0, 2.0, 0.5, 1.5], [0, 1, 2, 3, 4])
    ok_all &= check_case("four_param_branching mixed", four_param_branching,
                          [10.0, 1e-2, 0.5, 2.0], [0, 1, 2, 3, 4],
                          eps=1e-6, rtol=1e-3, atol=1e-4)

    print("\n=== native DPH (is_discrete=True, was_dph=False) ===")
    ok_all &= check_case("dph_native(2-stage)", lambda: dph_native((1.0, 1.0)),
                          [0.3, 0.4], [0, 1, 2])
    ok_all &= check_case("dph_native(3-stage)", lambda: dph_native((1.0, 1.0, 1.0)),
                          [0.3, 0.4, 0.5], [0, 1, 2, 3])

    print("\n=== realistic joint-probability graphs (StateIndexer/joint_prob_graph) ===")
    for n_samples, reward_limit in [(4, 3), (5, 2)]:
        def build(_ns=n_samples, _rl=reward_limit):
            return _make_joint(_ns, _rl, False)
        g0 = build()
        t_idx = _t_vertices(g0)[:12]  # a modest subset -- k doesn't matter for correctness
        name = f"joint(n_samples={n_samples},reward_limit={reward_limit},discrete=False)"
        ok_all &= check_case(name, build, [0.5, 1.0], t_idx, eps=1e-6, rtol=1e-3, atol=1e-6)

    print("\n=== joint_prob_graph(discrete=True) is was_dph, NOT native DPH -- must decline ===")
    # Confirmed empirically: joint_prob_graph's discrete=True path renormalises
    # (was_dph=True), unlike the hand-built dph_native() fixture above
    # (is_discrete=True, was_dph=False). This is the correct, expected
    # decline -- not a bug -- exercising the SAME was_dph gate as the
    # dedicated decline check below, on a realistic graph shape.
    gj = _make_joint(4, 3, True)
    assert gj.is_discrete and gj.get_was_dph()
    gj.update_weights([0.5, 1.0])
    n_decline_joint = len(gj._sojourn_grad_theta_subset(list(_t_vertices(gj)[:12])))
    print(f"  joint(discrete=True) was_dph=True: exact J size={n_decline_joint} (expect 0 -> FD fallback)")
    ok_all &= (n_decline_joint == 0)

    print("\n=== was_dph decline (mandatory exclusion, see plan scope section) ===")
    g = two_param_chain()
    gd = g.discretize(lambda state, **kw: [0.5, 0.5])
    gd.update_weights([1.0, 2.0])
    n_decline = len(gd._sojourn_grad_theta_subset([0, 1, 2]))
    print(f"  discretize() (was_dph=True): exact J size={n_decline} (expect 0 -> FD fallback)")
    ok_all &= (n_decline == 0)

    print("\n=== MPFR conditioning decline ===")
    g2 = two_param_chain()
    g2.update_weights([1.0, 1e-13])
    n_gate = len(g2._sojourn_grad_theta_subset([0, 1, 2]))
    print(f"  ill-conditioned theta=[1,1e-13]: exact J size={n_gate} (expect 0 -> FD fallback)")
    ok_all &= (n_gate == 0)

    g3 = two_param_chain()
    g3.update_weights([1.0, 0.5])
    n_ok = len(g3._sojourn_grad_theta_subset([0, 1, 2]))
    print(f"  well-conditioned theta=[1,0.5]: exact J size={n_ok} (expect 6)")
    ok_all &= (n_ok == 6)

    print("\n=== edge cases ===")
    g4 = two_param_chain()
    g4.update_weights([1.0, 0.5])
    n_empty = len(g4._sojourn_grad_theta_subset([]))
    print(f"  empty indices: exact J size={n_empty} (expect 0)")
    ok_all &= (n_empty == 0)

    print("\n=== no-regression: existing gates (unmodified functions) ===")
    for script in ["experiments/dr_moments_jac_gate.py", "experiments/dr_mpfr_gate_test.py",
                   "experiments/dr_dph_moments_jac_gate.py", "experiments/dr_log_mode_moments_jac_gate.py"]:
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        passed = r.returncode == 0 and "ALL PASS" in r.stdout
        print(f"  {script}: {'ALL PASS' if passed else 'FAIL'}")
        if not passed:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
        ok_all &= passed

    print("\n" + ("ALL PASS" if ok_all else "FAILURES PRESENT"))
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# D3 benchmark: single-call exact (cache-reused) vs FD (2xP calls). Both
# scale ~linearly in P (exact: convert + stage-0 fixed cost + P tangent
# passes; FD: 2*P plain adjoint calls, no fixed cost) -- so the comparison
# is run across a RANGE of P, not just the joint-prob fixture's native P=2,
# to show the trend that actually determines the D4 default decision.
# ---------------------------------------------------------------------------
def _big_chain(n_stages, n_params, seed=0):
    g = Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([n_stages + 1 - i]) for i in range(n_stages + 1)]
    s.add_edge(vs[0], 1.0)
    rng = np.random.default_rng(seed)
    for i in range(n_stages):
        coeff = list(rng.uniform(0.1, 1.0, n_params))
        vs[i].add_edge(vs[i + 1], coeff)
    return g


def _time_exact(g, idx, n_rep):
    g._sojourn_grad_theta_subset(idx)  # warm
    t0 = time.time()
    for _ in range(n_rep):
        g._sojourn_grad_theta_subset(idx)
    return (time.time() - t0) / n_rep


def _time_fd(g, idx, theta, n_rep, eps=1e-6):
    P = len(theta)
    g.expected_sojourn_time(idx)  # warm
    t0 = time.time()
    for _ in range(n_rep):
        for j in range(P):
            tp = list(theta); tm = list(theta)
            tp[j] += eps; tm[j] -= eps
            g.update_weights(tp); g.expected_sojourn_time(idx)
            g.update_weights(tm); g.expected_sojourn_time(idx)
        g.update_weights(theta)
    return (time.time() - t0) / n_rep


def benchmark():
    print("\n=== D3 benchmark: exact (1 call) vs FD (2xP calls) ===")
    print("  (1) realistic joint-prob graph, native P=2 (this model's actual param count):")
    n_samples, reward_limit = 6, 6
    g = _make_joint(n_samples, reward_limit, False)
    n, t_idx, theta = g.vertices_length(), _t_vertices(g), [0.5, 1.0]
    print(f"      n_samples={n_samples} reward_limit={reward_limit}: n={n} k={len(t_idx)} P={len(theta)}")
    g.update_weights(theta)
    idx_list = list(t_idx)
    t_exact = _time_exact(g, idx_list, 5)
    t_fd = _time_fd(g, idx_list, theta, 5)
    print(f"      exact={t_exact:.4f}s  FD={t_fd:.4f}s  speedup(FD/exact)={t_fd / t_exact:.2f}x")

    print("\n  (2) synthetic large chain, P swept (n_stages=2000, k=51):")
    for n_params in [2, 5, 10, 20, 50]:
        g2 = _big_chain(2000, n_params)
        theta2 = [1.0] * n_params
        g2.update_weights(theta2)
        idx2 = list(range(0, 2001, 40))
        t_exact2 = _time_exact(g2, idx2, 10)
        t_fd2 = _time_fd(g2, idx2, theta2, 10)
        print(f"      P={n_params:3d}: exact={t_exact2:.4f}s  FD={t_fd2:.4f}s  "
              f"speedup(FD/exact)={t_fd2 / t_exact2:.2f}x")

    print("\n  Both paths scale ~linearly in P (exact has a fixed convert+stage-0 cost "
          "amortized over P tangent passes; FD has no fixed cost but pays 2 calls per "
          "P with no amortization) -- exact's relative advantage GROWS with P, is at or "
          "below parity for this model's small native P=2. This benchmark exercises the "
          "Python/pybind call path (not the FFI host-callback path SVGD actually uses), "
          "so it is informative for the D4 default decision, not a final production number.")


if __name__ == "__main__":
    rc = main()
    benchmark()
    raise SystemExit(rc)
