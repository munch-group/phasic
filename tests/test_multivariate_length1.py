"""
Test compute_pmf_multivariate with length-1 vectors.

This validates that the new multivariate function produces identical results
to the existing compute_pmf when given vectors of length 1.
"""

import numpy as np
import json
import phasic
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


def test_length1_validation():
    """Test that length-1 vectors match existing compute_pmf()."""

    # Create a simple parameterized graph: exponential distribution
    # State [n], rate = theta * n
    def callback(state, **kwargs):
        n = state[0]
        if n <= 0:
            return []
        rate = n
        return [(np.array([n - 1]), [rate])]

    graph = Graph(callback, ipv=[3], parameterized=True)

    # Serialize and create GraphBuilder
    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    # Test parameters
    theta = np.array([2.0])
    times = np.array([0.5, 1.0, 1.5, 2.0])

    # 1. Compute using existing compute_pmf (known correct)
    pdf_original = builder.compute_pmf(theta, times, discrete=False, granularity=100)

    # 2. Compute using new compute_pmf_multivariate with length-1 vectors
    times_2d = times.reshape(-1, 1)  # Shape: (n_times, 1)

    # Need rewards vector (all ones for no transformation)
    n_vertices = builder.vertices_length
    rewards_2d = np.ones((n_vertices, 1))  # Shape: (n_vertices, 1)

    pdf_multivariate = builder.compute_pmf_multivariate(
        theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    # 3. Extract column from multivariate result
    pdf_multivariate_1d = pdf_multivariate[:, 0]

    # 4. Compare
    print("Original PDF shape:", pdf_original.shape)
    print("Multivariate PDF shape:", pdf_multivariate.shape)
    print("\nOriginal PDF values:")
    print(pdf_original)
    print("\nMultivariate PDF values (column 0):")
    print(pdf_multivariate_1d)
    print("\nDifference:")
    print(pdf_original - pdf_multivariate_1d)

    # Check if they match
    max_diff = np.max(np.abs(pdf_original - pdf_multivariate_1d))
    print(f"\nMax absolute difference: {max_diff:.2e}")

    # They should be identical (or very close due to floating point)
    assert max_diff < 1e-12, f"Length-1 validation failed: max diff = {max_diff}"

    print("\n✓ Length-1 validation passed!")
    return True


def test_zero_observation():
    """Test that zero time produces zero PDF in sparse mode."""

    def callback(state, **kwargs):
        n = state[0]
        if n <= 0:
            return []
        rate = n
        return [(np.array([n - 1]), [rate])]

    graph = Graph(callback, ipv=[3], parameterized=True)

    structure_dict = graph.serialize()
    structure_json = json.dumps(structure_dict, cls=NumpyEncoder)
    builder = GraphBuilder(structure_json)

    theta = np.array([2.0])

    # Mix of zero and non-zero times
    times_2d = np.array([[1.0], [0.0], [2.0], [0.0]])  # Shape: (4, 1)

    n_vertices = builder.vertices_length
    rewards_2d = np.ones((n_vertices, 1))

    pdf = builder.compute_pmf_multivariate(
        theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    print("\nZero observation test:")
    print("Times:", times_2d.ravel())
    print("PDF values:", pdf.ravel())

    # Check that zeros produce zero PDF
    assert pdf[1, 0] == 0.0, "Zero time should produce zero PDF"
    assert pdf[3, 0] == 0.0, "Zero time should produce zero PDF"

    # Check that non-zeros produce non-zero PDF
    assert pdf[0, 0] > 0.0, "Non-zero time should produce non-zero PDF"
    assert pdf[2, 0] > 0.0, "Non-zero time should produce non-zero PDF"

    print("✓ Zero observation test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing compute_pmf_multivariate with length-1 vectors")
    print("=" * 60)

    test_length1_validation()
    print()
    test_zero_observation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
