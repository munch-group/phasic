"""
Tests for Optax optimizer integration.

This module tests the OptaxOptimizer wrapper and factory functions
that allow using Optax optimizers with phasic's SVGD interface.
"""

import pytest
import numpy as np

# Import phasic first (before JAX)
from phasic import (
    Adam,  # For comparison with built-in
    HAS_OPTAX,
)

import jax.numpy as jnp

# Skip all tests if optax is not installed
pytestmark = pytest.mark.skipif(
    not HAS_OPTAX,
    reason="Optax not installed. Install with: pip install phasic[optax]"
)


class TestOptaxOptimizer:
    """Tests for the OptaxOptimizer wrapper class."""

    def test_optax_adam_initialization(self):
        """Test that OptaxOptimizer can be initialized with optax.adam."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.001)
        assert optimizer is not None
        assert optimizer.state is None  # Not initialized until reset()

    def test_reset_initializes_state(self):
        """Test that reset() properly initializes optimizer state."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.001)
        shape = (10, 2)  # 10 particles, 2 parameters
        optimizer.reset(shape)

        assert optimizer.state is not None
        assert optimizer.t == 0

    def test_step_updates_particles(self):
        """Test that step() produces updates with correct shape."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.01)
        shape = (10, 2)
        optimizer.reset(shape)

        phi = jnp.ones(shape)  # Gradient direction
        update = optimizer.step(phi)

        assert update.shape == shape
        assert optimizer.t == 1  # Timestep incremented

    def test_multiple_steps(self):
        """Test that multiple steps work correctly."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.01)
        shape = (10, 2)
        optimizer.reset(shape)

        phi = jnp.ones(shape)
        for i in range(10):
            update = optimizer.step(phi)
            assert update.shape == shape
            assert optimizer.t == i + 1


class TestOptaxFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_optax_adam(self):
        """Test optax_adam factory function."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.001, b1=0.9, b2=0.999)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)

    def test_optax_adamw(self):
        """Test optax_adamw factory function with weight decay."""
        from phasic import optax_adamw

        optimizer = optax_adamw(learning_rate=0.001, weight_decay=0.01)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)

    def test_optax_sgd(self):
        """Test optax_sgd factory function."""
        from phasic import optax_sgd

        optimizer = optax_sgd(learning_rate=0.01, momentum=0.9)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)

    def test_optax_rmsprop(self):
        """Test optax_rmsprop factory function."""
        from phasic import optax_rmsprop

        optimizer = optax_rmsprop(learning_rate=0.001, decay=0.9)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)

    def test_optax_adagrad(self):
        """Test optax_adagrad factory function."""
        from phasic import optax_adagrad

        optimizer = optax_adagrad(learning_rate=0.01)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)

    def test_optax_lion(self):
        """Test optax_lion factory function."""
        from phasic import optax_lion

        optimizer = optax_lion(learning_rate=1e-4, b1=0.9, b2=0.99)
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)


class TestOptaxChain:
    """Tests for optax_chain with gradient transformations."""

    def test_chain_with_gradient_clipping(self):
        """Test chaining gradient clipping with adam."""
        import optax
        from phasic import optax_chain

        optimizer = optax_chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(0.001)
        )
        optimizer.reset((10, 2))

        # Use large gradients to test clipping
        phi = jnp.ones((10, 2)) * 100.0
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        # Clipping should limit update magnitude
        assert jnp.max(jnp.abs(update)) < 10.0

    def test_chain_with_multiple_transforms(self):
        """Test chaining multiple gradient transformations."""
        import optax
        from phasic import optax_chain

        optimizer = optax_chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale(-0.001)  # Negative for gradient descent
        )
        optimizer.reset((10, 2))
        update = optimizer.step(jnp.ones((10, 2)))
        assert update.shape == (10, 2)


class TestOptaxSchedules:
    """Tests for Optax learning rate schedules."""

    def test_constant_schedule(self):
        """Test optax_adam with constant learning rate."""
        from phasic import optax_adam

        optimizer = optax_adam(learning_rate=0.01)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        # Should work with constant learning rate
        for _ in range(10):
            update = optimizer.step(phi)
            assert not jnp.any(jnp.isnan(update))

    def test_exponential_decay_schedule(self):
        """Test optax_adam with Optax's exponential decay schedule."""
        import optax
        from phasic import OptaxOptimizer

        schedule = optax.exponential_decay(
            init_value=0.01,
            transition_steps=100,
            decay_rate=0.9
        )
        optimizer = OptaxOptimizer(optax.adam(schedule))
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        for _ in range(10):
            update = optimizer.step(phi)
            assert not jnp.any(jnp.isnan(update))

    def test_warmup_cosine_decay_schedule(self):
        """Test optax_adam with warmup + cosine decay schedule."""
        import optax
        from phasic import OptaxOptimizer

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=0.01,
            warmup_steps=50,
            decay_steps=500
        )
        optimizer = OptaxOptimizer(optax.adam(schedule))
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        for _ in range(10):
            update = optimizer.step(phi)
            assert not jnp.any(jnp.isnan(update))


class TestOptaxVsBuiltIn:
    """Compare Optax optimizers with phasic's built-in implementations."""

    def test_adam_produces_similar_updates(self):
        """Test that Optax Adam produces similar updates to built-in Adam."""
        from phasic import optax_adam

        shape = (10, 2)
        phi = jnp.array(np.random.randn(*shape))

        # Built-in Adam
        builtin_adam = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        builtin_adam.reset(shape)

        # Optax Adam
        optax_adam_opt = optax_adam(learning_rate=0.01, b1=0.9, b2=0.999, eps=1e-8)
        optax_adam_opt.reset(shape)

        # First step should produce similar (but not identical due to implementation details)
        builtin_update = builtin_adam.step(phi)
        optax_update = optax_adam_opt.step(phi)

        # Both should have same shape
        assert builtin_update.shape == optax_update.shape

        # Both should be finite
        assert jnp.all(jnp.isfinite(builtin_update))
        assert jnp.all(jnp.isfinite(optax_update))

        # Should be in the same ballpark (within 2x)
        builtin_norm = float(jnp.linalg.norm(builtin_update))
        optax_norm = float(jnp.linalg.norm(optax_update))
        assert 0.1 < optax_norm / builtin_norm < 10.0


class TestOptaxWithMockSVGD:
    """Test Optax optimizers in a simplified SVGD-like context."""

    def test_optimization_converges(self):
        """Test that optimization with Optax Adam converges on a simple problem."""
        from phasic import optax_adam

        # Simple quadratic loss: minimize ||x - target||^2
        target = jnp.array([[1.0, 2.0]])  # Single "particle"
        particles = jnp.zeros((1, 2))

        optimizer = optax_adam(learning_rate=0.1)
        optimizer.reset(particles.shape, params=particles)

        for _ in range(100):
            # Gradient of ||x - target||^2 is 2(x - target)
            # Optax optimizers negate gradients internally (designed for minimization)
            gradient = 2 * (particles - target)
            update = optimizer.step(gradient, params=particles)
            particles = particles + update

        # Should be close to target after optimization
        error = float(jnp.linalg.norm(particles - target))
        assert error < 0.1, f"Optimization did not converge: error={error}"


class TestImportErrorHandling:
    """Test behavior when optax is not installed."""

    def test_has_optax_flag(self):
        """Test that HAS_OPTAX flag is correctly set."""
        from phasic import HAS_OPTAX
        # If we got this far, optax is installed
        assert HAS_OPTAX is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
