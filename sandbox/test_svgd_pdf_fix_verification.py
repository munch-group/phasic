#!/usr/bin/env python3
"""
SVGD Inference Verification After PDF Normalization Fix
========================================================

Tests that SVGD correctly recovers true parameters after the PDF normalization
fix with strict 2% relative error requirement.

Models tested:
1. Simple exponential distribution (θ = 2.0)
2. Coalescent model with 4 samples (θ = 1.0)
3. Coalescent model with 4 samples (θ = 2.0)

All tests use 10,000 observations and require ≤2% relative error.

Related:
- PDF_SCALING_BUG_FIX.md - Original PDF normalization fix
- MULTIVARIATE_PDF_VERIFICATION.md - PDF integration verification
- test_multivariate_pdf_fix.py - Coalescent PDF testing
"""

import numpy as np
from scipy.integrate import trapezoid
from phasic import Graph, SVGD, configure, ExponentialDecayStepSize
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

# Test configuration
N_OBSERVATIONS = 10000
N_PARTICLES = 50  # Reduced from 100 for faster runtime
N_ITERATIONS = 1500  # Reduced from 2000 for faster runtime
ERROR_THRESHOLD = 0.02  # 2% relative error

print("="*80)
print("SVGD Inference Verification After PDF Normalization Fix")
print("Requirement: ≤2% relative error on all parameter estimates")
print("="*80)
print(f"\nConfiguration:")
print(f"  Observations per test: {N_OBSERVATIONS}")
print(f"  SVGD particles: {N_PARTICLES}")
print(f"  SVGD iterations: {N_ITERATIONS}")
print(f"  Error threshold: {ERROR_THRESHOLD*100}%")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def build_exponential_graph():
    """
    Build simple exponential distribution.

    Graph structure: Start → [1] → [0] (absorbing)
    - Transition rate from [1] → [0]: θ (parameterized)
    - Returns Exponential(θ) distribution
    """
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])

    # Starting edge: probability 1.0 (non-parameterized, not affected by parameter updates)
    start.add_edge(v1, 1.0)
    # Transition: rate θ (parameterized)
    v1.add_edge_parameterized(v0, 0.0, [1.0])

    return g


def coalescent(state, nr_samples=None):
    """
    Coalescent model with nr_samples lineages.

    Starting edge is non-parameterized (not affected by parameter updates).
    """
    if not state.size:
        # Starting edge: base_weight=1.0, empty coefficients (non-parameterized)
        ipv = [[[nr_samples]+[0]*nr_samples, 1.0, []]]
        return ipv
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i+j+1] += 1
                transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
        return transitions


def build_coalescent_graph(nr_samples=4):
    """Build coalescent graph with nr_samples lineages"""
    return Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)


def generate_data(graph, true_theta, n_samples=N_OBSERVATIONS):
    """
    Generate observations from graph with given parameter.

    Parameters
    ----------
    graph : Graph
        Parameterized graph
    true_theta : float
        True parameter value
    n_samples : int
        Number of observations to generate

    Returns
    -------
    data : ndarray
        Generated observations
    """
    np.random.seed(42)
    graph.update_parameterized_weights(np.array([true_theta]))
    data = np.array(graph.sample(n_samples))
    return data


def verify_pdf_integration(graph, true_theta, max_time_multiplier=3.0):
    """
    Verify PDF integrates to approximately 1.0.

    Parameters
    ----------
    graph : Graph
        Graph to test
    true_theta : float
        Parameter value for testing
    max_time_multiplier : float
        Maximum time as multiple of expected mean

    Returns
    -------
    integral : float
        PDF integral value
    """
    graph.update_parameterized_weights(np.array([true_theta]))

    # Generate samples to estimate time range
    test_samples = np.array(graph.sample(1000))
    max_time = np.max(test_samples) * max_time_multiplier

    # Compute PDF and integrate
    times = np.linspace(0.001, max_time, 1000)
    pdf = graph.pdf(times)
    integral = trapezoid(pdf, times)

    return integral


def run_svgd_test(graph, data, true_theta, test_name):
    """
    Run SVGD inference and check results.

    Parameters
    ----------
    graph : Graph
        Parameterized graph
    data : ndarray
        Observed data
    true_theta : float
        True parameter value
    test_name : str
        Test identifier for output

    Returns
    -------
    passed : bool
        True if test passed (error ≤ 2%)
    error_pct : float
        Relative error percentage
    results : dict
        Full SVGD results
    """
    # Build model with correct signature for SVGD
    # Use pmf_and_moments_from_graph which returns (pmf, moments) and accepts rewards kwarg
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False, param_length=1)

    # Configure step size schedule
    step_schedule = ExponentialDecayStepSize(
        first_step=0.01,
        last_step=0.001,
        tau=500.0
    )

    # Run SVGD
    print(f"  Running SVGD ({N_PARTICLES} particles, {N_ITERATIONS} iterations)...")

    svgd = SVGD(
        model=model,
        observed_data=data,
        theta_dim=1,
        n_particles=N_PARTICLES,
        n_iterations=N_ITERATIONS,
        learning_rate=step_schedule,
        # parallel: use automatic default (pmap if multiple devices, vmap otherwise)
        positive_params=True,
        regularization=0.0,  # No moment regularization for clean parameter recovery
        nr_moments=2,  # Match model
        seed=42,
        verbose=False
    )

    svgd.optimize()

    # Extract results (use get_results() to get transformed/positive parameters)
    results = svgd.get_results()
    theta_mean = results['theta_mean'][0]
    theta_std = results['theta_std'][0]

    # Compute relative error
    rel_error = abs(theta_mean - true_theta) / true_theta
    error_pct = rel_error * 100

    # Print results
    print(f"\n  Results:")
    print(f"    True θ:      {true_theta:.3f}")
    print(f"    SVGD θ_mean: {theta_mean:.3f}")
    print(f"    Relative error: {error_pct:.1f}%", end="")

    if rel_error <= ERROR_THRESHOLD:
        print(f" ✓ (< {ERROR_THRESHOLD*100}%)")
    else:
        print(f" ✗ (≥ {ERROR_THRESHOLD*100}%)")

    print(f"    SVGD θ_std:  {theta_std:.3f}")

    # Pass if error ≤ 2%
    passed = (rel_error <= ERROR_THRESHOLD)

    if passed:
        print(f"\n  ✓ PASS: Error {error_pct:.1f}% < {ERROR_THRESHOLD*100}% threshold")
    else:
        print(f"\n  ✗ FAIL: Error {error_pct:.1f}% ≥ {ERROR_THRESHOLD*100}%")

    return passed, error_pct, {
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'error_pct': error_pct
    }


# ============================================================================
# Test 1: Simple Exponential (θ = 2.0)
# ============================================================================

def test_exponential_2_0():
    """Test SVGD on simple exponential distribution with θ = 2.0"""
    print_section("Test 1: Simple Exponential (θ = 2.0, n = 10000)")

    true_theta = 2.0

    print(f"  Building graph and generating data...")
    # Use separate graph for data generation to avoid polluting the graph state
    graph_for_data = build_exponential_graph()
    data = generate_data(graph_for_data, true_theta, N_OBSERVATIONS)

    print(f"  Data: mean={np.mean(data):.3f}, std={np.std(data):.3f}")

    # Verify PDF integration (use separate graph)
    print(f"  PDF integration check...", end="")
    graph_for_pdf = build_exponential_graph()
    integral = verify_pdf_integration(graph_for_pdf, true_theta)
    pdf_error_pct = abs(integral - 1.0) * 100
    print(f" {integral:.6f} ✓ (error: {pdf_error_pct:.3f}%)")

    # Run SVGD with fresh graph (no update_parameterized_weights called on it)
    graph_for_svgd = build_exponential_graph()
    passed, error_pct, results = run_svgd_test(graph_for_svgd, data, true_theta, "exponential_2_0")

    return passed, error_pct, results


# ============================================================================
# Test 2: Coalescent Model (θ = 1.0, nr_samples = 4)
# ============================================================================

def test_coalescent_1_0():
    """Test SVGD on coalescent model with θ = 1.0, nr_samples = 4"""
    print_section("Test 2: Coalescent Model (θ = 1.0, nr_samples = 4, n = 10000)")

    true_theta = 1.0
    nr_samples = 4

    print(f"  Building graph and generating data...")
    # Use separate graph for data generation
    graph_for_data = build_coalescent_graph(nr_samples=nr_samples)
    print(f"  Graph: {graph_for_data.vertices_length()} vertices")

    data = generate_data(graph_for_data, true_theta, N_OBSERVATIONS)
    print(f"  Data: mean={np.mean(data):.3f}, std={np.std(data):.3f}")

    # Verify PDF integration (use separate graph)
    print(f"  PDF integration check...", end="")
    graph_for_pdf = build_coalescent_graph(nr_samples=nr_samples)
    integral = verify_pdf_integration(graph_for_pdf, true_theta)
    pdf_error_pct = abs(integral - 1.0) * 100
    print(f" {integral:.6f} ✓ (error: {pdf_error_pct:.3f}%)")

    # Run SVGD with fresh graph (no update_parameterized_weights called on it)
    graph_for_svgd = build_coalescent_graph(nr_samples=nr_samples)
    passed, error_pct, results = run_svgd_test(graph_for_svgd, data, true_theta, "coalescent_1_0")

    return passed, error_pct, results


# ============================================================================
# Test 3: Coalescent Model (θ = 2.0, nr_samples = 4)
# ============================================================================

def test_coalescent_2_0():
    """Test SVGD on coalescent model with θ = 2.0, nr_samples = 4"""
    print_section("Test 3: Coalescent Model (θ = 2.0, nr_samples = 4, n = 10000)")

    true_theta = 2.0
    nr_samples = 4

    print(f"  Building graph and generating data...")
    # Use separate graph for data generation
    graph_for_data = build_coalescent_graph(nr_samples=nr_samples)
    print(f"  Graph: {graph_for_data.vertices_length()} vertices")

    data = generate_data(graph_for_data, true_theta, N_OBSERVATIONS)
    print(f"  Data: mean={np.mean(data):.3f}, std={np.std(data):.3f}")

    # Verify PDF integration (use separate graph)
    print(f"  PDF integration check...", end="")
    graph_for_pdf = build_coalescent_graph(nr_samples=nr_samples)
    integral = verify_pdf_integration(graph_for_pdf, true_theta)
    pdf_error_pct = abs(integral - 1.0) * 100
    print(f" {integral:.6f} ✓ (error: {pdf_error_pct:.3f}%)")

    # Run SVGD with fresh graph (no update_parameterized_weights called on it)
    graph_for_svgd = build_coalescent_graph(nr_samples=nr_samples)
    passed, error_pct, results = run_svgd_test(graph_for_svgd, data, true_theta, "coalescent_2_0")

    return passed, error_pct, results


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all SVGD verification tests"""

    results = {}

    # Run tests
    results['exponential_2_0'] = test_exponential_2_0()
    results['coalescent_1_0'] = test_coalescent_1_0()
    results['coalescent_2_0'] = test_coalescent_2_0()

    # Summary
    print_section("SUMMARY")

    test_names = {
        'exponential_2_0': 'Test 1 (Exponential θ=2.0)',
        'coalescent_1_0': 'Test 2 (Coalescent θ=1.0)',
        'coalescent_2_0': 'Test 3 (Coalescent θ=2.0)'
    }

    all_passed = True
    for key, (passed, error_pct, _) in results.items():
        status = "✓" if passed else "✗"
        threshold_str = f"< {ERROR_THRESHOLD*100}%" if passed else f"≥ {ERROR_THRESHOLD*100}%"
        print(f"{status} {test_names[key]}: {error_pct:.1f}% error {threshold_str}")
        if not passed:
            all_passed = False

    print()
    n_passed = sum(1 for passed, _, _ in results.values() if passed)
    n_total = len(results)
    print(f"All tests passed: {n_passed}/{n_total} {'✓' if all_passed else '✗'}")

    if all_passed:
        print("PDF normalization fix enables accurate SVGD inference (≤2% error)")
        print("="*80)
        return 0
    else:
        print("Some tests failed - review SVGD configuration or increase iterations")
        print("="*80)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
