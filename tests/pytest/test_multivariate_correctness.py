"""
Comprehensive correctness tests for multivariate phase-type distributions.
the pybind11 implementation (always available) with various scenarios:
- Length-1 vectors (must match existing compute_pmf)
- Sparse observations (zeros produce zero PDF)
- Multiple features (independent computation)
- Different reward vectors
- Analytical validation (exponential distribution)
"""

import numpy as np
import json
import phasic
phasic.configure(ffi=True) 

from phasic import Graph

# Access GraphBuilder from phasic.parameterized
GraphBuilder = phasic.parameterized.GraphBuilder


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def create_test_graph():
    """Create a simple parameterized graph for testing."""
    def callback(state, **kwargs):
        n = state[0]
        if n <= 0:
            return []
        rate = n
        return [(np.array([n - 1]), [rate])]

    graph = Graph(callback, ipv=[3], parameterized=True)
    return graph


def test_pybind_length1_vs_existing():
    """Test multivariate with length-1 vs existing compute_pmf."""
    print("\n" + "=" * 70)
    print("TEST: Multivariate (length-1) vs existing compute_pmf")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    theta = np.array([2.0])
    times_1d = np.array([0.5, 1.0, 1.5, 2.0])
    times_2d = times_1d.reshape(-1, 1)

    n_vertices = builder.vertices_length
    rewards_2d = np.ones((n_vertices, 1))

    # 1. Existing compute_pmf (1D)
    pdf_existing = builder.compute_pmf(theta, times_1d, discrete=False, granularity=100)

    # 2. New multivariate (2D with length-1)
    pdf_multivariate = builder.compute_pmf_multivariate(
        theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    # 3. Compare
    pdf_multivariate_1d = pdf_multivariate[:, 0]

    print(f"\nExisting PDF:     {pdf_existing}")
    print(f"Multivariate PDF: {pdf_multivariate_1d}")
    print(f"Difference:       {pdf_existing - pdf_multivariate_1d}")

    max_diff = np.max(np.abs(pdf_existing - pdf_multivariate_1d))
    print(f"\nMax absolute difference: {max_diff:.2e}")

    assert max_diff < 1e-12, f"Multivariate differs from existing: {max_diff}"
    print("✓ PASS: Multivariate matches existing compute_pmf")


def test_sparse_observations():
    """Test that zero observations produce zero PDF."""
    print("\n" + "=" * 70)
    print("TEST: Sparse observations (zeros produce zero PDF)")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    theta = np.array([2.0])

    # Mix of zero and non-zero observations
    times_2d = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [1.5, 0.5],
        [0.0, 0.0]
    ])

    n_vertices = builder.vertices_length
    rewards_2d = np.ones((n_vertices, 2))

    pdf = builder.compute_pmf_multivariate(
        theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    print(f"\nTimes:\n{times_2d}")
    print(f"\nPDF values:\n{pdf}")

    # Check zeros
    print("\nChecking zero entries:")
    assert pdf[0, 1] == 0.0, "times[0,1]=0 should give PDF=0"
    print(f"  PDF[0,1] = {pdf[0, 1]} ✓ (times[0,1] = 0.0)")

    assert pdf[1, 0] == 0.0, "times[1,0]=0 should give PDF=0"
    print(f"  PDF[1,0] = {pdf[1, 0]} ✓ (times[1,0] = 0.0)")

    assert pdf[3, 0] == 0.0, "times[3,0]=0 should give PDF=0"
    print(f"  PDF[3,0] = {pdf[3, 0]} ✓ (times[3,0] = 0.0)")

    assert pdf[3, 1] == 0.0, "times[3,1]=0 should give PDF=0"
    print(f"  PDF[3,1] = {pdf[3, 1]} ✓ (times[3,1] = 0.0)")

    # Check non-zeros
    print("\nChecking non-zero entries:")
    assert pdf[0, 0] > 0.0, "times[0,0]=1.0 should give PDF>0"
    print(f"  PDF[0,0] = {pdf[0, 0]:.6f} ✓ (times[0,0] = 1.0)")

    assert pdf[1, 1] > 0.0, "times[1,1]=2.0 should give PDF>0"
    print(f"  PDF[1,1] = {pdf[1, 1]:.6f} ✓ (times[1,1] = 2.0)")

    assert pdf[2, 0] > 0.0, "times[2,0]=1.5 should give PDF>0"
    print(f"  PDF[2,0] = {pdf[2, 0]:.6f} ✓ (times[2,0] = 1.5)")

    assert pdf[2, 1] > 0.0, "times[2,1]=0.5 should give PDF>0"
    print(f"  PDF[2,1] = {pdf[2, 1]:.6f} ✓ (times[2,1] = 0.5)")

    print("\n✓ PASS: Sparse observations work correctly")


def test_independent_features():
    """Test that features are computed independently with different rewards."""
    print("\n" + "=" * 70)
    print("TEST: Independent feature computation with different rewards")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    theta = np.array([2.0])
    times = np.array([[1.0], [2.0]])

    n_vertices = builder.vertices_length

    # Two different reward vectors
    rewards_1 = np.ones((n_vertices, 1))
    rewards_2 = np.ones((n_vertices, 1)) * 2.0

    # Compute separately
    pdf_1 = builder.compute_pmf_multivariate(
        theta, times, rewards_1,
        discrete=False, granularity=100, compute_joint=False
    )

    pdf_2 = builder.compute_pmf_multivariate(
        theta, times, rewards_2,
        discrete=False, granularity=100, compute_joint=False
    )

    # Compute together (2 features)
    times_2feat = np.array([[1.0, 1.0], [2.0, 2.0]])
    rewards_2feat = np.concatenate([rewards_1, rewards_2], axis=1)

    pdf_combined = builder.compute_pmf_multivariate(
        theta, times_2feat, rewards_2feat,
        discrete=False, granularity=100, compute_joint=False
    )

    # Check that combined result matches separate computations
    pdf_1_1d = pdf_1[:, 0]
    pdf_2_1d = pdf_2[:, 0]

    print(f"\nFeature 0 (rewards=1.0):")
    print(f"  Separate:  {pdf_1_1d}")
    print(f"  Combined:  {pdf_combined[:, 0]}")
    print(f"  Diff:      {pdf_1_1d - pdf_combined[:, 0]}")

    print(f"\nFeature 1 (rewards=2.0):")
    print(f"  Separate:  {pdf_2_1d}")
    print(f"  Combined:  {pdf_combined[:, 1]}")
    print(f"  Diff:      {pdf_2_1d - pdf_combined[:, 1]}")

    max_diff_0 = np.max(np.abs(pdf_1_1d - pdf_combined[:, 0]))
    max_diff_1 = np.max(np.abs(pdf_2_1d - pdf_combined[:, 1]))

    print(f"\nMax difference feature 0: {max_diff_0:.2e}")
    print(f"\nMax difference feature 1: {max_diff_1:.2e}")

    assert max_diff_0 < 1e-12, f"Feature 0 differs: {max_diff_0}"
    assert max_diff_1 < 1e-12, f"Feature 1 differs: {max_diff_1}"

    # Also check they're different (rewards are different!)
    assert not np.allclose(pdf_1_1d, pdf_2_1d), "Different rewards should give different PDFs"
    print(f"\n✓ Different rewards produce different PDFs (as expected)")
    print("✓ PASS: Features computed independently")


def test_correctness_analytical():
    """Test correctness against analytical solution for exponential distribution."""
    print("\n" + "=" * 70)
    print("TEST: Correctness against analytical exponential PDF")
    print("=" * 70)

    # Create exponential(rate) graph with single state
    def exp_callback(state, **kwargs):
        # Single transient state with rate parameter
        if state[0] == 1:
            return [(np.array([0]), [1.0])]  # Coefficient = 1.0
        return []

    graph = Graph(exp_callback, ipv=[1], parameterized=True)
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    n_vertices = builder.vertices_length
    print(f"\nGraph has {n_vertices} vertices")

    # Test with different rates
    rates = np.array([0.5, 1.0, 2.0, 5.0])
    times = np.array([[0.5], [1.0], [2.0]])
    rewards = np.ones((n_vertices, 1))

    for rate in rates:
        theta = np.array([rate])

        # Compute using multivariate
        pdf_computed = builder.compute_pmf_multivariate(
            theta, times, rewards,
            discrete=False, granularity=100, compute_joint=False
        )

        # Analytical exponential PDF: f(t) = rate * exp(-rate * t)
        times_1d = times[:, 0]
        pdf_analytical = rate * np.exp(-rate * times_1d)
        pdf_computed_1d = pdf_computed[:, 0]

        diff = np.abs(pdf_computed_1d - pdf_analytical)
        rel_error = diff / pdf_analytical

        print(f"\nRate = {rate}:")
        print(f"  Times:       {times_1d}")
        print(f"  Computed:    {pdf_computed_1d}")
        print(f"  Analytical:  {pdf_analytical}")
        print(f"  Abs error:   {diff}")
        print(f"  Rel error:   {rel_error}")

        # Allow small relative error due to discretization
        max_rel_error = np.max(rel_error)
        assert max_rel_error < 0.01, f"Relative error too large: {max_rel_error}"
        print(f"  ✓ Max relative error: {max_rel_error:.2e} < 1%")

    print("\n✓ PASS: Results match analytical exponential distribution")


def test_multiple_features_correctness():
    """Test correctness with multiple features against individual feature computations."""
    print("\n" + "=" * 70)
    print("TEST: Multiple features correctness")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    theta = np.array([2.0])
    n_vertices = builder.vertices_length

    # 3 features with different observations
    times_multi = np.array([
        [1.0, 2.0, 1.5],
        [0.5, 1.0, 2.5],
        [2.0, 0.0, 1.0]  # Feature 1 has no observation in row 2
    ])

    # 3 different reward vectors
    rewards_multi = np.array([
        [1.0, 2.0, 0.5],  # Vertex 0
        [1.0, 2.0, 0.5],  # Vertex 1
        [1.0, 2.0, 0.5],  # Vertex 2
        [1.0, 2.0, 0.5]   # Vertex 3
    ])

    # Compute all features together
    pdf_all = builder.compute_pmf_multivariate(
        theta, times_multi, rewards_multi,
        discrete=False, granularity=100, compute_joint=False
    )

    print(f"\nMulti-feature PDF shape: {pdf_all.shape}")
    print(f"PDF values:\n{pdf_all}")

    # Verify each feature independently
    for j in range(3):
        times_j = times_multi[:, j:j+1]  # Keep 2D
        rewards_j = rewards_multi[:, j:j+1]

        pdf_j = builder.compute_pmf_multivariate(
            theta, times_j, rewards_j,
            discrete=False, granularity=100, compute_joint=False
        )

        diff = np.abs(pdf_all[:, j] - pdf_j[:, 0])
        max_diff = np.max(diff)

        print(f"\nFeature {j}:")
        print(f"  Combined:   {pdf_all[:, j]}")
        print(f"  Individual: {pdf_j[:, 0]}")
        print(f"  Max diff:   {max_diff:.2e}")

        assert max_diff < 1e-12, f"Feature {j} differs: {max_diff}"

    # Check that zero observation in row 2, feature 1 gives zero PDF
    assert pdf_all[2, 1] == 0.0, "Zero observation should give zero PDF"
    print(f"\n✓ Zero observation correctly handled: PDF[2,1] = {pdf_all[2, 1]}")

    print("\n✓ PASS: Multiple features computed correctly")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTIVARIATE CORRECTNESS TESTS (pybind11)")
    print("=" * 70)

    test_pybind_length1_vs_existing()
    test_sparse_observations()
    test_independent_features()
    test_correctness_analytical()
    test_multiple_features_correctness()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
