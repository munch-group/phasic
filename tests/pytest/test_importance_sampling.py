"""
Tests for importance sampling density functions.

Verifies:
1. Homogeneous density matches scipy exponential
2. Inhomogeneous reduces to homogeneous for constant N
3. Piecewise-constant integral correctness
4. Importance weight = 0 when target = proposal
5. Multi-epoch density against manual calculation
6. JAX compatibility (jit, grad)
"""

from phasic import (
    piecewise_constant_integral,
    homogeneous_log_density,
    inhomogeneous_log_density,
    importance_log_weight,
    make_demographic_log_likelihood,
)
from phasic.importance_sampling import _choose2

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import expon


class TestChoose2:
    def test_basic(self):
        assert _choose2(2) == 1.0
        assert _choose2(3) == 3.0
        assert _choose2(4) == 6.0
        assert _choose2(5) == 10.0

    def test_one(self):
        assert _choose2(1) == 0.0


class TestPiecewiseConstantIntegral:

    def test_single_epoch(self):
        """Integral over single epoch: (b-a)/N."""
        result = piecewise_constant_integral(
            0.0, 1.0,
            epoch_boundaries=jnp.array([]),
            epoch_sizes=jnp.array([2.0]),
        )
        assert jnp.isclose(result, 0.5)

    def test_two_epochs_within_first(self):
        """Integral entirely within first epoch."""
        result = piecewise_constant_integral(
            0.0, 0.5,
            epoch_boundaries=jnp.array([1.0]),
            epoch_sizes=jnp.array([2.0, 10.0]),
        )
        assert jnp.isclose(result, 0.25)

    def test_two_epochs_spanning_boundary(self):
        """Integral spanning an epoch boundary."""
        # [0, 2] with boundary at 1, sizes [2, 4]
        # integral = 1/2 * 1 + 1/4 * 1 = 0.75
        result = piecewise_constant_integral(
            0.0, 2.0,
            epoch_boundaries=jnp.array([1.0]),
            epoch_sizes=jnp.array([2.0, 4.0]),
        )
        assert jnp.isclose(result, 0.75)

    def test_three_epochs(self):
        """Integral spanning two epoch boundaries."""
        # [0, 3] with boundaries at [1, 2], sizes [1, 5, 10]
        # integral = 1/1 * 1 + 1/5 * 1 + 1/10 * 1 = 1.3
        result = piecewise_constant_integral(
            0.0, 3.0,
            epoch_boundaries=jnp.array([1.0, 2.0]),
            epoch_sizes=jnp.array([1.0, 5.0, 10.0]),
        )
        assert jnp.isclose(result, 1.3)

    def test_zero_width(self):
        """Integral with a=b should be 0."""
        result = piecewise_constant_integral(
            1.0, 1.0,
            epoch_boundaries=jnp.array([0.5]),
            epoch_sizes=jnp.array([2.0, 3.0]),
        )
        assert jnp.isclose(result, 0.0)

    def test_offset_start(self):
        """Integral not starting at 0."""
        # [0.5, 1.5] with boundary at 1, sizes [2, 4]
        # integral = 1/2 * 0.5 + 1/4 * 0.5 = 0.375
        result = piecewise_constant_integral(
            0.5, 1.5,
            epoch_boundaries=jnp.array([1.0]),
            epoch_sizes=jnp.array([2.0, 4.0]),
        )
        assert jnp.isclose(result, 0.375)


class TestHomogeneousLogDensity:

    def test_two_lineages(self):
        """n=2: single exponential waiting time."""
        N = 3.0
        w = jnp.array([1.5])  # Single waiting time
        rate = 1.0 / N  # C(2,2)/N = 1/N

        result = float(homogeneous_log_density(w, 2, N))
        expected = float(expon.logpdf(1.5, scale=N))
        assert np.isclose(result, expected, atol=1e-10)

    def test_three_lineages(self):
        """n=3: two exponential waiting times at different rates."""
        N = 2.0
        w = jnp.array([0.5, 1.0])  # w_3, w_2

        # k=3: rate = C(3,2)/N = 3/2, w=0.5
        # k=2: rate = C(2,2)/N = 1/2, w=1.0
        rate3 = 3.0 / N
        rate2 = 1.0 / N
        expected = (np.log(rate3) - rate3 * 0.5) + (np.log(rate2) - rate2 * 1.0)

        result = float(homogeneous_log_density(w, 3, N))
        assert np.isclose(result, expected, atol=1e-10)

    def test_positive_density(self):
        """Density should be positive (log-density finite) for positive times."""
        w = jnp.array([0.1, 0.2, 0.3])
        result = float(homogeneous_log_density(w, 4, 1.0))
        assert np.isfinite(result)


class TestInhomogeneousLogDensity:

    def test_constant_matches_homogeneous(self):
        """Single epoch (constant N) should match homogeneous density."""
        N = 2.0
        w = jnp.array([0.5, 1.0])

        result_inhom = float(inhomogeneous_log_density(
            w, 3,
            epoch_boundaries=jnp.array([]),
            epoch_sizes=jnp.array([N]),
        ))
        result_hom = float(homogeneous_log_density(w, 3, N))
        assert np.isclose(result_inhom, result_hom, atol=1e-10)

    def test_two_lineages_two_epochs(self):
        """n=2, two epochs: verify against manual calculation."""
        # Boundary at t=1, sizes [1, 5]
        # Single waiting time w=1.5
        # Cumulative: starts at 0, ends at 1.5
        # Rate at event (t=1.5): N(1.5) = 5, rate = 1/5
        # Integrated hazard: C(2,2) * integral_0^1.5 1/N(s) ds
        #   = 1 * (1/1 * 1 + 1/5 * 0.5) = 1.1
        w = jnp.array([1.5])
        result = float(inhomogeneous_log_density(
            w, 2,
            epoch_boundaries=jnp.array([1.0]),
            epoch_sizes=jnp.array([1.0, 5.0]),
        ))
        expected = np.log(1.0 / 5.0) - 1.1
        assert np.isclose(result, expected, atol=1e-10)

    def test_event_in_first_epoch(self):
        """Event entirely within first epoch should match homogeneous."""
        w = jnp.array([0.3])  # Short waiting time, boundary at 1.0
        N1 = 2.0

        result = float(inhomogeneous_log_density(
            w, 2,
            epoch_boundaries=jnp.array([1.0]),
            epoch_sizes=jnp.array([N1, 10.0]),
        ))
        expected = float(homogeneous_log_density(w, 2, N1))
        assert np.isclose(result, expected, atol=1e-10)

    def test_three_lineages_two_epochs(self):
        """n=3, two epochs: first event in epoch 1, second spanning boundary."""
        # Boundary at t=0.5, sizes [1, 4]
        # w_3 = 0.3 (k=3, starts at 0, ends at 0.3 — in epoch 1)
        # w_2 = 0.5 (k=2, starts at 0.3, ends at 0.8 — spans boundary)
        w = jnp.array([0.3, 0.5])

        # Event 1 (k=3): rate at t=0.3 is C(3,2)/N(0.3) = 3/1 = 3
        #   integrated hazard = 3 * integral_0^0.3 1/1 ds = 3 * 0.3 = 0.9
        #   contribution: log(3) - 0.9

        # Event 2 (k=2): rate at t=0.8 is C(2,2)/N(0.8) = 1/4 = 0.25
        #   integrated hazard = 1 * integral_0.3^0.8 1/N(s) ds
        #     = 1 * (1/1 * 0.2 + 1/4 * 0.3) = 0.275
        #   contribution: log(0.25) - 0.275

        expected = (np.log(3.0) - 0.9) + (np.log(0.25) - 0.275)
        result = float(inhomogeneous_log_density(
            w, 3,
            epoch_boundaries=jnp.array([0.5]),
            epoch_sizes=jnp.array([1.0, 4.0]),
        ))
        assert np.isclose(result, expected, atol=1e-10)


class TestImportanceLogWeight:

    def test_zero_when_equal(self):
        """Weight should be 0 when target = proposal."""
        N = 2.0
        w = jnp.array([0.5, 1.0])

        result = float(importance_log_weight(
            w, 3,
            epoch_boundaries=jnp.array([]),
            epoch_sizes=jnp.array([N]),
            proposal_pop_size=N,
        ))
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_nonzero_when_different(self):
        """Weight should be nonzero when target != proposal."""
        w = jnp.array([0.5])

        result = float(importance_log_weight(
            w, 2,
            epoch_boundaries=jnp.array([]),
            epoch_sizes=jnp.array([3.0]),
            proposal_pop_size=1.0,
        ))
        assert not np.isclose(result, 0.0)

    def test_sign_direction(self):
        """Larger target N (slower coalescence) should downweight short waiting times."""
        w = jnp.array([0.1])  # Short waiting time

        # Target has larger N (slower coalescence) — short times are less likely
        result = float(importance_log_weight(
            w, 2,
            epoch_boundaries=jnp.array([]),
            epoch_sizes=jnp.array([10.0]),
            proposal_pop_size=1.0,
        ))
        assert result < 0  # Target density < proposal density for short times


class TestMakeDemographicLogLikelihood:

    def test_returns_callable(self):
        """make_demographic_log_likelihood returns a callable."""
        w = jnp.array([0.5, 1.0])
        log_lik = make_demographic_log_likelihood(
            w, 3, epoch_boundaries=jnp.array([1.0])
        )
        assert callable(log_lik)

    def test_evaluates(self):
        """Log-likelihood evaluates to a finite scalar."""
        w = jnp.array([0.5, 1.0])
        log_lik = make_demographic_log_likelihood(
            w, 3, epoch_boundaries=jnp.array([1.0])
        )
        result = float(log_lik(jnp.array([2.0, 5.0])))
        assert np.isfinite(result)

    def test_matches_direct(self):
        """Log-likelihood matches direct call to inhomogeneous_log_density."""
        w = jnp.array([0.5, 1.0])
        boundaries = jnp.array([0.8])
        sizes = jnp.array([2.0, 5.0])

        log_lik = make_demographic_log_likelihood(w, 3, boundaries)
        result = float(log_lik(sizes))

        expected = float(inhomogeneous_log_density(w, 3, boundaries, sizes))
        assert np.isclose(result, expected, atol=1e-10)


class TestJAXCompatibility:

    def test_jit_homogeneous(self):
        """homogeneous_log_density works with jax.jit."""
        w = jnp.array([0.5])
        jitted = jax.jit(homogeneous_log_density, static_argnums=(1,))
        result = float(jitted(w, 2, 2.0))
        expected = float(homogeneous_log_density(w, 2, 2.0))
        assert np.isclose(result, expected, atol=1e-10)

    def test_grad_homogeneous(self):
        """Gradient of homogeneous_log_density w.r.t. pop_size is finite."""
        w = jnp.array([0.5])

        def f(pop_size):
            return homogeneous_log_density(w, 2, pop_size)

        grad_val = float(jax.grad(f)(2.0))
        assert np.isfinite(grad_val)

    def test_grad_inhomogeneous(self):
        """Gradient of inhomogeneous_log_density w.r.t. epoch_sizes is finite."""
        w = jnp.array([0.5, 1.0])
        boundaries = jnp.array([0.8])

        def f(sizes):
            return inhomogeneous_log_density(w, 3, boundaries, sizes)

        grad_val = jax.grad(f)(jnp.array([2.0, 5.0]))
        assert np.all(np.isfinite(grad_val))

    def test_grad_log_likelihood(self):
        """Gradient of make_demographic_log_likelihood works."""
        w = jnp.array([0.5, 1.0])
        boundaries = jnp.array([0.8])
        log_lik = make_demographic_log_likelihood(w, 3, boundaries)

        grad_val = jax.grad(log_lik)(jnp.array([2.0, 5.0]))
        assert np.all(np.isfinite(grad_val))
