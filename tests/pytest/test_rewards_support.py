"""
Tests for reward vector support in pmf_and_moments_from_graph()

Tests both backward compatibility (rewards=None) and reward transformation.
"""

import numpy as np
from phasic import Graph
import jax
import jax.numpy as jnp


def test_rewards_none_backward_compat():
    """Test that rewards=None preserves backward compatibility."""
    # Build simple exponential graph: single state, rate=theta
    graph = Graph(state_length=1, parameterized=True)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])

    # Parameterized edge: weight = theta[0]
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    # Create model factory
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    # Test without rewards (default)
    theta = jnp.array([2.0])
    times = jnp.array([0.5, 1.0, 1.5])
    pmf1, moments1 = model(theta, times)

    # Test with explicit rewards=None
    pmf2, moments2 = model(theta, times, rewards=None)

    # Should be identical
    assert jnp.allclose(pmf1, pmf2), "PMF should be identical with and without rewards=None"
    assert jnp.allclose(moments1, moments2), "Moments should be identical with and without rewards=None"

    print("✓ Backward compatibility test passed")
    print(f"  PMF shape: {pmf1.shape}")
    print(f"  Moments shape: {moments1.shape}")
    print(f"  Moments (E[T], E[T^2]): {moments1}")


def test_rewards_transformation():
    """Test that rewards parameter transforms moments correctly."""
    # Build two-state graph
    graph = Graph(state_length=1, parameterized=True)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v2 = graph.find_or_create_vertex([2])

    # Two parameterized edges
    v0.add_edge_parameterized(v1, 0.0, [1.0])  # rate = theta[0]
    v1.add_edge_parameterized(v2, 0.0, [1.0])  # rate = theta[0]

    # Create model factory
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    theta = jnp.array([2.0])
    times = jnp.array([0.5, 1.0, 1.5])

    # Compute without rewards (standard moments)
    pmf_standard, moments_standard = model(theta, times, rewards=None)

    # Compute with uniform rewards (should equal standard moments)
    rewards_uniform = jnp.array([1.0, 1.0])  # One per vertex (excluding start)
    pmf_uniform, moments_uniform = model(theta, times, rewards=rewards_uniform)

    # Compute with non-uniform rewards
    rewards_custom = jnp.array([2.0, 0.5])
    pmf_custom, moments_custom = model(theta, times, rewards=rewards_custom)

    # PMF should be identical (rewards don't affect PMF, only moments)
    assert jnp.allclose(pmf_standard, pmf_uniform), "PMF should be same for uniform rewards"
    assert jnp.allclose(pmf_standard, pmf_custom), "PMF should be same for custom rewards"

    # Moments with uniform rewards should match standard moments
    assert jnp.allclose(moments_standard, moments_uniform, rtol=1e-5), \
        "Uniform rewards should give standard moments"

    # Moments with custom rewards should differ
    assert not jnp.allclose(moments_standard, moments_custom), \
        "Custom rewards should transform moments"

    print("✓ Reward transformation test passed")
    print(f"  Standard moments: {moments_standard}")
    print(f"  Uniform rewards moments: {moments_uniform}")
    print(f"  Custom rewards moments: {moments_custom}")


def test_vmap_with_rewards():
    """Test that rewards work with JAX vmap (batched theta)."""
    # Simple exponential graph
    graph = Graph(state_length=1, parameterized=True)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    # Batch of theta values
    theta_batch = jnp.array([[1.0], [2.0], [3.0]])
    times = jnp.array([0.5, 1.0])
    rewards = jnp.array([1.0])

    # vmap over theta batch
    vmapped_model = jax.vmap(lambda t: model(t, times, rewards))
    pmf_batch, moments_batch = vmapped_model(theta_batch)

    assert pmf_batch.shape == (3, 2), f"Expected shape (3, 2), got {pmf_batch.shape}"
    assert moments_batch.shape == (3, 2), f"Expected shape (3, 2), got {moments_batch.shape}"

    print("✓ vmap with rewards test passed")
    print(f"  Batched PMF shape: {pmf_batch.shape}")
    print(f"  Batched moments shape: {moments_batch.shape}")


if __name__ == "__main__":
    print("\nTesting reward vector support...")
    print("=" * 60)

    test_rewards_none_backward_compat()
    print()

    test_rewards_transformation()
    print()

    test_vmap_with_rewards()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
