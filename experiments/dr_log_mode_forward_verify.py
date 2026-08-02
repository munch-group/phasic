"""De-risk step D0 for the log-weight-mode B3 batch: verify that
pmf_and_moments_from_graph's FORWARD is actually correct for weight_mode='log'
before building any gradient extension on top of it. This directly checks a
premise the 2025-07 audit only partially covered (it fixed moments_from_graph
and pmf_from_graph_joint_index to RAISE on 'log', but did not claim to have
exhaustively verified pmf_and_moments_from_graph's 'log' forward values --
only that "GraphBuilder honours the mode" by code inspection).

Independent oracle: graph.update_weights(theta, log=True) (native C, NOT
GraphBuilder) followed by graph.moments(K) / graph.pdf(t).

log mode requires EVERY edge to have ALL param_length coefficients nonzero
(weight = prod_i(c_i*theta_i) over ALL i, and the C layer raises if any
product <= 0), so the fixture graph must be dense-coefficient.

Run: pixi run python experiments/dr_log_mode_forward_verify.py
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph


def build_log_graph():
    """3-vertex chain, 2 params, EVERY edge has both coefficients nonzero
    (required for weight_mode='log')."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])  # BOTH coefficients nonzero
    g.weight_mode = 'log'
    return g


def native_oracle_moments(theta, K):
    g = build_log_graph()
    g.update_weights(list(theta), log=True)
    return np.asarray(g.moments(K))


def native_oracle_pdf(theta, times):
    g = build_log_graph()
    g.update_weights(list(theta), log=True)
    return np.asarray([g.pdf(float(t)) for t in times])


def check(theta, K=3, tag=""):
    times = np.array([0.5, 1.0, 2.0])
    model = Graph.pmf_and_moments_from_graph(build_log_graph(), nr_moments=K, discrete=False, theta_dim=2)
    pmf, moments = model(jnp.asarray(theta), jnp.asarray(times))
    moments = np.asarray(moments)
    pmf = np.asarray(pmf)

    ref_moments = native_oracle_moments(theta, K)
    ref_pdf = native_oracle_pdf(theta, times)

    err_m = np.max(np.abs(moments - ref_moments) / np.maximum(np.abs(ref_moments), 1e-300))
    err_p = np.max(np.abs(pmf - ref_pdf) / np.maximum(np.abs(ref_pdf), 1e-300))
    ok = err_m < 1e-9 and err_p < 1e-9
    print(f"  {tag} theta={theta}: moments_err={err_m:.3e} pdf_err={err_p:.3e} {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    got moments={moments} ref={ref_moments}")
        print(f"    got pdf={pmf} ref={ref_pdf}")
    return ok


def main():
    all_ok = True
    print("Forward parity: pmf_and_moments_from_graph vs native update_weights(log=True) oracle")
    all_ok &= check([1.0, 1.0], tag="uniform")
    all_ok &= check([0.5, 2.0], tag="asymmetric")
    all_ok &= check([1.0, 1e-3], tag="mixed-scale-mild")
    all_ok &= check([1.0, 1e6], tag="mixed-scale-large")
    all_ok &= check([2.0, 3.0], K=2, tag="K=2")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
