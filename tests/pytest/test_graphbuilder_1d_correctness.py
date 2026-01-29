#!/usr/bin/env python3
"""
Phase 1: GraphBuilder 1D Model Correctness Tests
================================================

Tests whether GraphBuilder-compiled models produce correct PMF and moments
for 1D observations. This will help diagnose the θ=10 SVGD convergence issue.

Tests:
- Test 1.1: First moment vs empirical mean (100k samples)
- Test 1.2: PMF normalization (∫ PMF dt = 1.0)
- Test 1.3: PMF vs empirical distribution (chi-square test)

For θ ∈ {1.0, 2.0, 5.0, 10.0}
"""

import numpy as np
from phasic import Graph, configure
import scipy.stats as stats
from scipy.integrate import trapezoid

configure(ffi=True, openmp=True, jit=True)

print("=" * 80)
print("Phase 1: GraphBuilder 1D Model Correctness Tests")
print("=" * 80)

# ============================================================================
# Coalescent Model Callback
# ============================================================================

def coalescent_callback(state, nr_samples=None):
    """
    Coalescent model: n lineages → n-1 lineages at rate n*(n-1)/2
    """
    if state.size == 0:
        # Initial state: nr_samples lineages
        return [([nr_samples], 0.0, [1.0])]

    n = state[0]
    if n <= 1:
        return []

    coalescence_rate = n * (n - 1) / 2
    return [([n - 1], 0.0, [coalescence_rate])]

# ============================================================================
# Test Functions
# ============================================================================

def test_first_moment_vs_empirical(graph, theta, nr_samples=3, n_samples=100000):
    """
    Test 1.1: First moment vs empirical mean
    """
    print(f"\n{'─' * 80}")
    print(f"Test 1.1: First Moment vs Empirical Mean (θ={theta})")
    print(f"{'─' * 80}")

    # Set parameter
    graph.update_parameterized_weights(np.array([theta]))

    # Generate 100k samples using graph.sample()
    np.random.seed(42)
    samples = graph.sample(n_samples)
    empirical_mean = np.mean(samples)
    empirical_std = np.std(samples)

    print(f"Empirical statistics from {n_samples:,} samples:")
    print(f"  Mean: {empirical_mean:.6f}")
    print(f"  Std:  {empirical_std:.6f}")

    # Get GraphBuilder first moment
    graphbuilder_moment = graph.moments(power=1)[0]

    print(f"\nGraphBuilder first moment: {graphbuilder_moment:.6f}")

    # Compute error
    error = abs(graphbuilder_moment - empirical_mean)
    rel_error = error / empirical_mean

    print(f"\nComparison:")
    print(f"  Absolute error: {error:.6e}")
    print(f"  Relative error: {rel_error*100:.4f}%")

    # Success criterion: < 0.1% relative error
    success = rel_error < 0.001
    print(f"\n{'PASS' if success else '❌ FAIL'}: Relative error {'<' if success else '>='} 0.1%")

    return success, rel_error


def test_pmf_normalization(graph, theta, nr_samples=3):
    """
    Test 1.2: PMF normalization (∫ PMF dt = 1.0)
    """
    print(f"\n{'─' * 80}")
    print(f"Test 1.2: PMF Normalization (θ={theta})")
    print(f"{'─' * 80}")

    # Set parameter
    graph.update_parameterized_weights(np.array([theta]))

    # Compute PMF over dense time grid
    times = np.linspace(0.001, 5.0, 1000)
    pmf_values = graph.pdf(times, granularity=0)  # granularity=0 → auto

    print(f"PMF computed at {len(times)} points")
    print(f"  Time range: [{times[0]:.3f}, {times[-1]:.3f}]")
    print(f"  PMF range:  [{np.min(pmf_values):.6e}, {np.max(pmf_values):.6e}]")

    # Integrate using trapezoid rule
    integral = trapezoid(pmf_values, times)

    print(f"\n∫ PMF dt = {integral:.6f}")

    # Success criterion: ∫ PMF dt = 1.0 ± 0.001
    error = abs(integral - 1.0)
    success = error < 0.001

    print(f"Error from 1.0: {error:.6e}")
    print(f"\n{'PASS' if success else '❌ FAIL'}: |∫ PMF dt - 1.0| {'<' if success else '>='} 0.001")

    return success, error


def test_pmf_vs_empirical(graph, theta, nr_samples=3, n_samples=100000, n_bins=50):
    """
    Test 1.3: PMF vs empirical distribution (chi-square test)
    """
    print(f"\n{'─' * 80}")
    print(f"Test 1.3: PMF vs Empirical Distribution (θ={theta})")
    print(f"{'─' * 80}")

    # Set parameter
    graph.update_parameterized_weights(np.array([theta]))

    # Generate samples
    np.random.seed(42)
    samples = graph.sample(n_samples)

    print(f"Generated {n_samples:,} samples")
    print(f"  Sample range: [{np.min(samples):.4f}, {np.max(samples):.4f}]")

    # Create histogram
    hist_counts, bin_edges = np.histogram(samples, bins=n_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    print(f"\nHistogram: {n_bins} bins of width {bin_width:.4f}")

    # Compute expected counts from PMF
    pmf_values = graph.pdf(bin_centers, granularity=0)
    expected_counts = pmf_values * bin_width * n_samples

    # Filter bins with sufficient expected counts (chi-square requirement)
    valid_mask = expected_counts >= 5
    hist_counts_valid = hist_counts[valid_mask]
    expected_counts_valid = expected_counts[valid_mask]

    print(f"Valid bins (expected ≥ 5): {np.sum(valid_mask)} / {n_bins}")

    if np.sum(valid_mask) < 10:
        print(f"\n⚠️  WARNING: Too few valid bins ({np.sum(valid_mask)}) for chi-square test")
        return False, 0.0

    # Chi-square test
    chi2_stat = np.sum((hist_counts_valid - expected_counts_valid)**2 / expected_counts_valid)
    dof = np.sum(valid_mask) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)

    print(f"\nChi-square test:")
    print(f"  χ² statistic: {chi2_stat:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.4f}")

    # Success criterion: p-value > 0.05
    success = p_value > 0.05
    print(f"\n{'PASS' if success else '❌ FAIL'}: p-value {'>' if success else '<='} 0.05")

    return success, p_value


def print_graph_diagnostics(graph, theta):
    """
    Print diagnostic information about the graph structure
    """
    print(f"\n{'─' * 80}")
    print(f"Graph Diagnostics (θ={theta})")
    print(f"{'─' * 80}")

    graph.update_parameterized_weights(np.array([theta]))

    print(f"\nGraph structure:")
    print(f"  Number of vertices: {graph.vertices_length()}")

    # Get states
    states = graph.states()
    print(f"  States shape: {states.shape}")
    print(f"  State values:\n{states}")

    # Serialize graph
    structure_json = graph.serialize()
    if isinstance(structure_json, dict):
        n_vertices_serialized = len(structure_json.get('vertices', []))
        print(f"\nSerialized graph:")
        print(f"  Type: dict")
        print(f"  Number of vertices in serialized: {n_vertices_serialized}")
    else:
        print(f"\nSerialized graph:")
        print(f"  Type: string")
        print(f"  Length: {len(structure_json)} characters")


# ============================================================================
# Main Test Execution
# ============================================================================

def run_all_tests():
    """
    Run all Phase 1 tests for θ ∈ {1.0, 2.0, 5.0, 10.0}
    """
    theta_values = [1.0, 2.0, 5.0, 10.0]
    nr_samples = 3  # Number of lineages in coalescent

    all_results = {}

    for theta in theta_values:
        print(f"\n{'=' * 80}")
        print(f"Testing θ = {theta}")
        print(f"{'=' * 80}")

        # Build graph
        graph = Graph(
            callback=coalescent_callback,
            parameterized=True,
            nr_samples=nr_samples
        )

        # Run tests
        results = {}

        # Test 1.1: First moment
        success_1_1, error_1_1 = test_first_moment_vs_empirical(graph, theta, nr_samples)
        results['moment'] = {'success': success_1_1, 'rel_error': error_1_1}

        # Test 1.2: PMF normalization
        success_1_2, error_1_2 = test_pmf_normalization(graph, theta, nr_samples)
        results['normalization'] = {'success': success_1_2, 'error': error_1_2}

        # Test 1.3: PMF vs empirical
        success_1_3, p_value_1_3 = test_pmf_vs_empirical(graph, theta, nr_samples)
        results['distribution'] = {'success': success_1_3, 'p_value': p_value_1_3}

        # Diagnostics
        print_graph_diagnostics(graph, theta)

        all_results[theta] = results

    return all_results


def print_summary(all_results):
    """
    Print summary of all test results
    """
    print(f"\n{'=' * 80}")
    print("PHASE 1 SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'θ':<8} {'Test 1.1':<12} {'Test 1.2':<12} {'Test 1.3':<12} {'Overall':<10}")
    print(f"{'':─<8} {'':─<12} {'':─<12} {'':─<12} {'':─<10}")

    for theta, results in all_results.items():
        moment_pass = 'PASS' if results['moment']['success'] else '❌ FAIL'
        norm_pass = 'PASS' if results['normalization']['success'] else '❌ FAIL'
        dist_pass = 'PASS' if results['distribution']['success'] else '❌ FAIL'

        all_pass = (results['moment']['success'] and
                   results['normalization']['success'] and
                   results['distribution']['success'])
        overall = 'PASS' if all_pass else '❌ FAIL'

        print(f"{theta:<8.1f} {moment_pass:<12} {norm_pass:<12} {dist_pass:<12} {overall:<10}")

    print(f"\n{'─' * 80}")
    print("Detailed Metrics:")
    print(f"{'─' * 80}")

    for theta, results in all_results.items():
        print(f"\nθ = {theta}:")
        print(f"  Test 1.1 (Moment):        {results['moment']['rel_error']*100:.4f}% error")
        print(f"  Test 1.2 (Normalization): {results['normalization']['error']:.6e} error")
        print(f"  Test 1.3 (Distribution):  p = {results['distribution']['p_value']:.4f}")

    # Determine overall pass/fail
    all_tests_passed = all(
        results['moment']['success'] and
        results['normalization']['success'] and
        results['distribution']['success']
        for results in all_results.values()
    )

    print(f"\n{'=' * 80}")
    if all_tests_passed:
        print("ALL PHASE 1 TESTS PASSED")
        print("=" * 80)
        print("\nConclusion: GraphBuilder 1D model produces correct PMF and moments")
        print("The θ=10 SVGD issue is NOT due to incorrect GraphBuilder outputs")
        print("→ Next: Investigate SVGD gradients, kernel function, or likelihood computation")
        return 0
    else:
        print("❌ SOME PHASE 1 TESTS FAILED")
        print("=" * 80)
        print("\nConclusion: GraphBuilder 1D model has issues")
        print("→ Next: Debug C++ forward algorithm, parameter application, or serialization")

        # Check which θ values failed
        failed_thetas = [theta for theta, results in all_results.items()
                        if not (results['moment']['success'] and
                               results['normalization']['success'] and
                               results['distribution']['success'])]

        if failed_thetas:
            print(f"\nFailed for θ ∈ {failed_thetas}")
            print("This suggests issues with specific parameter values or ranges")

        return 1


# ============================================================================
# Execute Tests
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_all_tests()
        exit_code = print_summary(results)
        exit(exit_code)
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"❌ TEST EXECUTION FAILED")
        print(f"{'=' * 80}")
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
