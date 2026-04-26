"""Tests for _compute_hpd() and HPD integration in SVGD."""
import numpy as np
import pytest

from phasic.svgd import _compute_hpd


# ---------------------------------------------------------------------------
# Unit tests for _compute_hpd
# ---------------------------------------------------------------------------

class TestComputeHPD:
    """Tests for the _compute_hpd helper function."""

    def test_symmetric_matches_equal_tailed(self):
        """For a symmetric distribution, HPD should approximate equal-tailed CI."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=5.0, scale=1.0, size=10_000)

        hpd_lo, hpd_hi = _compute_hpd(samples, alpha=0.95)

        et_lo = np.percentile(samples, 2.5)
        et_hi = np.percentile(samples, 97.5)

        # For a symmetric distribution, HPD and equal-tailed should be close
        assert abs(hpd_lo - et_lo) < 0.15, f"Lower bounds differ: HPD={hpd_lo}, ET={et_lo}"
        assert abs(hpd_hi - et_hi) < 0.15, f"Upper bounds differ: HPD={hpd_hi}, ET={et_hi}"

    def test_skewed_hpd_shorter_than_equal_tailed(self):
        """For a skewed distribution, HPD should be shorter than equal-tailed CI."""
        rng = np.random.default_rng(123)
        # Log-normal is right-skewed
        samples = rng.lognormal(mean=0.0, sigma=1.0, size=10_000)

        hpd_lo, hpd_hi = _compute_hpd(samples, alpha=0.95)
        hpd_width = hpd_hi - hpd_lo

        et_lo = np.percentile(samples, 2.5)
        et_hi = np.percentile(samples, 97.5)
        et_width = et_hi - et_lo

        assert hpd_width < et_width, (
            f"HPD width ({hpd_width:.4f}) should be shorter than "
            f"equal-tailed width ({et_width:.4f}) for skewed distributions"
        )

    def test_skewed_hpd_contains_mode(self):
        """HPD should contain the mode for a unimodal skewed distribution."""
        rng = np.random.default_rng(456)
        # Log-normal mode = exp(mu - sigma^2)
        mu, sigma = 0.0, 1.0
        samples = rng.lognormal(mean=mu, sigma=sigma, size=50_000)
        mode = np.exp(mu - sigma**2)

        hpd_lo, hpd_hi = _compute_hpd(samples, alpha=0.95)

        assert hpd_lo <= mode <= hpd_hi, (
            f"HPD [{hpd_lo:.4f}, {hpd_hi:.4f}] should contain mode {mode:.4f}"
        )

    def test_minimum_width_property(self):
        """Brute-force: HPD should have the minimum width among all intervals
        containing at least alpha fraction of mass."""
        rng = np.random.default_rng(789)
        samples = rng.lognormal(mean=0.0, sigma=0.5, size=200)
        alpha = 0.90

        hpd_lo, hpd_hi = _compute_hpd(samples, alpha=alpha)
        hpd_width = hpd_hi - hpd_lo

        # Brute-force: check all contiguous windows
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        n_included = int(np.ceil(n * alpha))
        for start in range(n - n_included + 1):
            width = sorted_s[start + n_included - 1] - sorted_s[start]
            assert hpd_width <= width + 1e-12, (
                f"HPD width {hpd_width} exceeds window [{start}] width {width}"
            )

    def test_alpha_1_returns_full_range(self):
        """alpha=1.0 should return (min, max)."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = _compute_hpd(samples, alpha=1.0)
        assert lo == 1.0
        assert hi == 5.0

    def test_single_sample(self):
        """n=1 should return (value, value) with a warning."""
        with pytest.warns(UserWarning, match="at least 2 samples"):
            lo, hi = _compute_hpd(np.array([3.14]), alpha=0.95)
        assert lo == hi == 3.14

    def test_two_samples(self):
        """n=2 with alpha=0.95 should include both samples."""
        with pytest.warns(UserWarning, match="only 2 samples"):
            lo, hi = _compute_hpd(np.array([1.0, 5.0]), alpha=0.95)
        assert lo == 1.0
        assert hi == 5.0

    def test_identical_values(self):
        """All identical samples should return a zero-width interval."""
        samples = np.full(50, 2.5)
        lo, hi = _compute_hpd(samples, alpha=0.95)
        assert lo == hi == 2.5

    def test_returns_plain_floats(self):
        """Output should be plain Python floats, not numpy scalars."""
        samples = np.random.default_rng(0).normal(size=100)
        lo, hi = _compute_hpd(samples)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_different_alpha_levels(self):
        """Higher alpha should produce wider (or equal) intervals."""
        rng = np.random.default_rng(11)
        samples = rng.normal(size=5000)

        _, hi_50 = _compute_hpd(samples, alpha=0.50)
        lo_50, _ = _compute_hpd(samples, alpha=0.50)
        width_50 = hi_50 - lo_50

        _, hi_95 = _compute_hpd(samples, alpha=0.95)
        lo_95, _ = _compute_hpd(samples, alpha=0.95)
        width_95 = hi_95 - lo_95

        assert width_95 >= width_50, (
            f"95% HPD width ({width_95}) should be >= 50% HPD width ({width_50})"
        )

    def test_small_sample_warning(self):
        """Fewer than 10 samples should produce a warning."""
        with pytest.warns(UserWarning, match="only 5 samples"):
            _compute_hpd(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), alpha=0.95)
