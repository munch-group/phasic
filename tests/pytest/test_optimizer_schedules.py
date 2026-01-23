"""
Tests for optimizer schedule support.

This module tests that all optimizers (Adam, SGDMomentum, RMSprop, Adagrad)
correctly support StepSizeSchedule objects for their hyperparameters.
"""

import pytest
import numpy as np

# Import phasic first (before JAX)
from phasic import (
    Adam, Adamelia, SGDMomentum, RMSprop, Adagrad,
    ExpStepSize, ConstantStepSize, WarmupExpStepSize
)

import jax.numpy as jnp


class TestWarmupExpStepSize:
    """Tests for WarmupExpStepSize schedule."""

    def test_warmup_phase(self):
        """Test that warmup phase linearly increases learning rate."""
        schedule = WarmupExpStepSize(peak_lr=0.1, warmup_steps=100, last_lr=0.01, tau=500)

        # At iteration 0, should be close to 0 (first step is peak_lr/warmup_steps)
        assert schedule(0) == pytest.approx(0.001, rel=0.01)  # 0.1 * 1/100

        # At iteration 49, should be halfway
        assert schedule(49) == pytest.approx(0.05, rel=0.01)  # 0.1 * 50/100

        # At iteration 99, should be near peak
        assert schedule(99) == pytest.approx(0.1, rel=0.01)  # 0.1 * 100/100

    def test_decay_phase(self):
        """Test that decay phase exponentially decreases learning rate."""
        schedule = WarmupExpStepSize(peak_lr=0.1, warmup_steps=100, last_lr=0.01, tau=100)

        # At warmup end (iteration 100), should be at peak
        lr_at_100 = float(schedule(100))

        # After decay (iteration 200, which is 100 steps into decay)
        # Should be approximately 0.1 * exp(-1) + 0.01 * (1 - exp(-1))
        lr_at_200 = float(schedule(200))
        expected = 0.1 * np.exp(-1) + 0.01 * (1 - np.exp(-1))
        assert lr_at_200 == pytest.approx(expected, rel=0.01)

    def test_plot_compatibility(self):
        """Test that WarmupExpStepSize has attributes for plot()."""
        schedule = WarmupExpStepSize(peak_lr=0.01, warmup_steps=100, last_lr=0.001, tau=500)

        # These attributes are needed by StepSizeSchedule.plot()
        assert hasattr(schedule, 'first_step')
        assert hasattr(schedule, 'last_step')
        assert schedule.first_step == 0.0
        assert schedule.last_step == 0.001


class TestAdamWithSchedules:
    """Tests for Adam optimizer with schedule support."""

    def test_constant_learning_rate_backward_compatibility(self):
        """Test that Adam still works with scalar learning_rate."""
        optimizer = Adam(learning_rate=0.01)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        # Learning rate property should return the constant value
        assert float(optimizer.lr) == pytest.approx(0.01, rel=0.01)

    def test_exp_step_size_learning_rate(self):
        """Test Adam with ExpStepSize for learning rate."""
        schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=10)
        optimizer = Adam(learning_rate=schedule)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # First update should use higher learning rate
        update1 = optimizer.step(phi)
        initial_magnitude = float(jnp.mean(jnp.abs(update1)))

        # Run many iterations
        for _ in range(50):
            optimizer.step(phi)

        # Later update should use lower learning rate
        update_late = optimizer.step(phi)
        late_magnitude = float(jnp.mean(jnp.abs(update_late)))

        # The magnitude should decrease (though Adam's moment correction
        # affects this, overall trend should be downward)
        assert late_magnitude < initial_magnitude

    def test_warmup_exp_step_size_learning_rate(self):
        """Test Adam with WarmupExpStepSize for learning rate."""
        schedule = WarmupExpStepSize(peak_lr=0.1, warmup_steps=10, last_lr=0.01, tau=50)
        optimizer = Adam(learning_rate=schedule)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # During warmup, learning rate should increase
        lr_values = []
        for i in range(15):
            optimizer.step(phi)
            lr_values.append(float(optimizer.lr))

        # First few lr values should be increasing (warmup)
        assert lr_values[4] > lr_values[0]

        # After warmup (iteration 10+), lr should start decreasing
        assert lr_values[14] < lr_values[10]

    def test_beta_schedules(self):
        """Test Adam with schedules for beta1 and beta2."""
        # Use constant schedules to verify they work
        lr_schedule = ConstantStepSize(0.01)
        beta1_schedule = ConstantStepSize(0.9)
        beta2_schedule = ConstantStepSize(0.999)

        optimizer = Adam(
            learning_rate=lr_schedule,
            beta1=beta1_schedule,
            beta2=beta2_schedule
        )
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        assert float(optimizer.beta1) == pytest.approx(0.9)
        assert float(optimizer.beta2) == pytest.approx(0.999)


class TestSGDMomentumWithSchedules:
    """Tests for SGDMomentum optimizer with schedule support."""

    def test_constant_values_backward_compatibility(self):
        """Test that SGDMomentum still works with scalar values."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        assert float(optimizer.lr) == pytest.approx(0.01)
        assert float(optimizer.momentum) == pytest.approx(0.9)

    def test_learning_rate_schedule(self):
        """Test SGDMomentum with learning rate schedule."""
        schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=10)
        optimizer = SGDMomentum(learning_rate=schedule, momentum=0.9)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # Run iterations and check lr decreases
        initial_lr = float(optimizer.lr)
        for _ in range(50):
            optimizer.step(phi)
        final_lr = float(optimizer.lr)

        assert final_lr < initial_lr


class TestRMSpropWithSchedules:
    """Tests for RMSprop optimizer with schedule support."""

    def test_constant_values_backward_compatibility(self):
        """Test that RMSprop still works with scalar values."""
        optimizer = RMSprop(learning_rate=0.001, decay=0.99)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        assert float(optimizer.lr) == pytest.approx(0.001)
        assert float(optimizer.decay) == pytest.approx(0.99)

    def test_learning_rate_schedule(self):
        """Test RMSprop with learning rate schedule."""
        schedule = ExpStepSize(first_step=0.01, last_step=0.001, tau=10)
        optimizer = RMSprop(learning_rate=schedule, decay=0.99)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # Run iterations and check lr decreases
        initial_lr = float(optimizer.lr)
        for _ in range(50):
            optimizer.step(phi)
        final_lr = float(optimizer.lr)

        assert final_lr < initial_lr


class TestAdagradWithSchedules:
    """Tests for Adagrad optimizer with schedule support."""

    def test_constant_learning_rate_backward_compatibility(self):
        """Test that Adagrad still works with scalar learning_rate."""
        optimizer = Adagrad(learning_rate=0.01)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))
        update = optimizer.step(phi)

        assert update.shape == (10, 2)
        assert float(optimizer.lr) == pytest.approx(0.01)

    def test_learning_rate_schedule(self):
        """Test Adagrad with learning rate schedule."""
        schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=10)
        optimizer = Adagrad(learning_rate=schedule)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # Run iterations and check lr decreases
        initial_lr = float(optimizer.lr)
        for _ in range(50):
            optimizer.step(phi)
        final_lr = float(optimizer.lr)

        assert final_lr < initial_lr


class TestOptimizerReset:
    """Tests for optimizer reset behavior with schedules."""

    def test_adam_reset_clears_timestep(self):
        """Test that Adam.reset() resets the timestep counter."""
        schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=10)
        optimizer = Adam(learning_rate=schedule)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # Run some iterations
        for _ in range(20):
            optimizer.step(phi)

        assert optimizer.t == 20

        # Reset and verify timestep is cleared
        optimizer.reset((10, 2))
        assert optimizer.t == 0

    def test_sgd_momentum_reset_clears_timestep(self):
        """Test that SGDMomentum.reset() resets the timestep counter."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        for _ in range(20):
            optimizer.step(phi)

        assert optimizer.t == 20

        optimizer.reset((10, 2))
        assert optimizer.t == 0

    def test_rmsprop_reset_clears_timestep(self):
        """Test that RMSprop.reset() resets the timestep counter."""
        optimizer = RMSprop(learning_rate=0.001, decay=0.99)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        for _ in range(20):
            optimizer.step(phi)

        assert optimizer.t == 20

        optimizer.reset((10, 2))
        assert optimizer.t == 0

    def test_adagrad_reset_clears_timestep(self):
        """Test that Adagrad.reset() resets the timestep counter."""
        optimizer = Adagrad(learning_rate=0.01)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        for _ in range(20):
            optimizer.step(phi)

        assert optimizer.t == 20

        optimizer.reset((10, 2))
        assert optimizer.t == 0


class TestAdamelia:
    """Tests for Adamelia optimizer with oscillation detection."""

    def test_adamelia_detects_oscillation(self):
        """Test that Adamelia detects oscillation and reduces LR."""
        optimizer = Adamelia(learning_rate=0.1, patience=2, verbose=False)
        optimizer.reset((10, 2))

        # Simulate oscillating gradients (flip sign each iteration)
        phi_positive = jnp.ones((10, 2))
        phi_negative = -jnp.ones((10, 2))

        initial_lr = optimizer.lr

        # Alternate gradients to trigger oscillation
        for i in range(10):
            phi = phi_positive if i % 2 == 0 else phi_negative
            optimizer.step(phi)

        # LR should have been reduced
        assert optimizer.lr < initial_lr
        assert optimizer.lr_reductions > 0

    def test_adamelia_no_false_positives(self):
        """Test that Adamelia doesn't reduce LR on smooth convergence."""
        optimizer = Adamelia(learning_rate=0.01, patience=3, verbose=False)
        optimizer.reset((10, 2))

        initial_lr = optimizer.lr

        # Simulate smooth gradients (all same direction)
        for i in range(20):
            phi = jnp.ones((10, 2)) * (1.0 - 0.01 * i)  # Gradually decreasing
            optimizer.step(phi)

        # LR should NOT have been reduced
        assert optimizer.lr == initial_lr
        assert optimizer.lr_reductions == 0

    def test_adamelia_respects_min_lr(self):
        """Test that Adamelia doesn't go below min_lr."""
        optimizer = Adamelia(learning_rate=0.001, min_lr=1e-5,
                                 patience=1, lr_reduction_factor=0.1)
        optimizer.reset((10, 2))

        phi_positive = jnp.ones((10, 2))
        phi_negative = -jnp.ones((10, 2))

        # Trigger many oscillations
        for i in range(50):
            phi = phi_positive if i % 2 == 0 else phi_negative
            optimizer.step(phi)

        # LR should be at or above min_lr
        assert optimizer.lr >= 1e-5

    def test_adamelia_reset_clears_state(self):
        """Test that reset clears oscillation detection state."""
        optimizer = Adamelia(learning_rate=0.1, patience=2)
        optimizer.reset((10, 2))

        # Trigger some oscillations
        phi_positive = jnp.ones((10, 2))
        phi_negative = -jnp.ones((10, 2))
        for i in range(10):
            phi = phi_positive if i % 2 == 0 else phi_negative
            optimizer.step(phi)

        assert optimizer.lr_reductions > 0
        assert optimizer.lr_multiplier < 1.0

        # Reset should clear everything
        optimizer.reset((10, 2))
        assert optimizer.lr_reductions == 0
        assert optimizer.lr_multiplier == 1.0
        assert optimizer.phi_prev is None
        assert optimizer.oscillation_count == 0

    def test_adamelia_with_schedule(self):
        """Test Adamelia works with learning rate schedules."""
        schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=10)
        optimizer = Adamelia(learning_rate=schedule, patience=2)
        optimizer.reset((10, 2))

        phi = jnp.ones((10, 2))

        # Run some iterations
        for _ in range(20):
            optimizer.step(phi)

        # lr property should combine schedule and multiplier
        base_lr = float(schedule(optimizer.t))
        assert optimizer.lr == pytest.approx(base_lr * optimizer.lr_multiplier)

    def test_adamelia_threshold_sensitivity(self):
        """Test that oscillation_threshold affects detection."""
        # Low threshold - more sensitive
        optimizer_sensitive = Adamelia(learning_rate=0.1, patience=1,
                                           oscillation_threshold=0.1)
        optimizer_sensitive.reset((10, 2))

        # High threshold - less sensitive
        optimizer_tolerant = Adamelia(learning_rate=0.1, patience=1,
                                          oscillation_threshold=0.9)
        optimizer_tolerant.reset((10, 2))

        # Partial oscillation: only 50% of components flip
        phi1 = jnp.concatenate([jnp.ones((10, 1)), jnp.ones((10, 1))], axis=1)
        phi2 = jnp.concatenate([jnp.ones((10, 1)), -jnp.ones((10, 1))], axis=1)

        for i in range(10):
            phi = phi1 if i % 2 == 0 else phi2
            optimizer_sensitive.step(phi)
            optimizer_tolerant.step(phi)

        # Sensitive should detect oscillation, tolerant should not
        assert optimizer_sensitive.lr_reductions > 0
        assert optimizer_tolerant.lr_reductions == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
