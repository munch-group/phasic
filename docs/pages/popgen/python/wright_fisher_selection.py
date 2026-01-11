"""
Wright-Fisher with Selection: Enhanced Beta-Binomial Approximation

This module implements a mathematically rigorous Beta-Binomial approximation
for the Wright-Fisher process WITH SELECTION, including:

1. Transformed Beta-Binomial that accounts for the nonlinear selection map
2. Delta method approximation for moment transformation
3. Theoretical predictions for validation (Kimura fixation probability, etc.)

Mathematical Framework
----------------------

Standard Wright-Fisher with selection:
- Transition: X_{t+1} | X_t = i ~ Binomial(N, p_tilde)
- Effective frequency: p_tilde = p(1+s) / (1 + s*p)
  where s is the selection coefficient (s > 0 = beneficial, s < 0 = deleterious)

The challenge: When p is distributed across a bin, p_tilde has a complicated
distribution due to the nonlinear transformation.

References
----------
- Kimura (1962): Probability of fixation
- Ewens (2004): Mathematical Population Genetics
- Fisher (1930): The Genetical Theory of Natural Selection
"""

import numpy as np
from scipy.stats import binom, betabinom, beta
from scipy.special import comb, hyp2f1
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
import warnings


# =============================================================================
# Core Mathematical Functions
# =============================================================================

def selection_transform(p: np.ndarray, s: float) -> np.ndarray:
    """
    Transform allele frequency under selection.

    p_tilde = p * (1 + s) / (1 + s * p)

    This represents the expected frequency in the next generation
    before genetic drift (binomial sampling).

    Parameters
    ----------
    p : array-like
        Current allele frequency (can be array)
    s : float
        Selection coefficient. s > 0 means allele A is beneficial.

    Returns
    -------
    p_tilde : array-like
        Effective frequency for binomial sampling
    """
    p = np.asarray(p)
    if s == 0:
        return p
    return p * (1 + s) / (1 + s * p)


def selection_transform_derivative(p: float, s: float) -> float:
    """
    First derivative of selection transform: d(p_tilde)/dp

    g'(p) = (1 + s) / (1 + s*p)^2
    """
    if s == 0:
        return 1.0
    return (1 + s) / (1 + s * p) ** 2


def selection_transform_second_derivative(p: float, s: float) -> float:
    """
    Second derivative of selection transform: d^2(p_tilde)/dp^2

    g''(p) = -2s(1 + s) / (1 + s*p)^3
    """
    if s == 0:
        return 0.0
    return -2 * s * (1 + s) / (1 + s * p) ** 3


# =============================================================================
# Moment Computation for Transformed Beta
# =============================================================================

def transformed_beta_moments_delta(
    alpha: float,
    beta_param: float,
    s: float
) -> Tuple[float, float]:
    """
    Compute mean and variance of transformed frequency using delta method.

    If p ~ Beta(alpha, beta), compute E[g(p)] and Var(g(p))
    where g(p) = p(1+s)/(1+sp) using Taylor expansion.

    Delta method approximation:
    E[g(p)] ≈ g(μ) + (1/2) g''(μ) σ²
    Var(g(p)) ≈ [g'(μ)]² σ²

    Parameters
    ----------
    alpha, beta_param : float
        Beta distribution parameters
    s : float
        Selection coefficient

    Returns
    -------
    mean_tilde, var_tilde : float
        Approximate mean and variance of transformed frequency
    """
    # Original Beta moments
    mu_p = alpha / (alpha + beta_param)
    var_p = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))

    if s == 0:
        return mu_p, var_p

    # Transform using delta method
    g_mu = selection_transform(mu_p, s)
    g_prime = selection_transform_derivative(mu_p, s)
    g_double_prime = selection_transform_second_derivative(mu_p, s)

    mean_tilde = g_mu + 0.5 * g_double_prime * var_p
    var_tilde = g_prime**2 * var_p

    return float(mean_tilde), float(var_tilde)


def transformed_beta_moments_exact(
    alpha: float,
    beta_param: float,
    s: float,
    n_quad: int = 1000
) -> Tuple[float, float]:
    """
    Compute exact mean and variance of transformed frequency via numerical integration.

    E[g(p)] = integral_0^1 g(p) * Beta_pdf(p; alpha, beta) dp

    Parameters
    ----------
    alpha, beta_param : float
        Beta distribution parameters
    s : float
        Selection coefficient
    n_quad : int
        Number of quadrature points

    Returns
    -------
    mean_tilde, var_tilde : float
        Exact mean and variance of transformed frequency
    """
    if s == 0:
        mu_p = alpha / (alpha + beta_param)
        var_p = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
        return mu_p, var_p

    beta_dist = beta(alpha, beta_param)

    def integrand_mean(p):
        return selection_transform(p, s) * beta_dist.pdf(p)

    def integrand_sq(p):
        return selection_transform(p, s)**2 * beta_dist.pdf(p)

    mean_tilde, _ = quad(integrand_mean, 0, 1, limit=100)
    mean_sq_tilde, _ = quad(integrand_sq, 0, 1, limit=100)
    var_tilde = mean_sq_tilde - mean_tilde**2

    return float(mean_tilde), float(max(0, var_tilde))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SelectionBin:
    """
    A bin for Wright-Fisher with selection.

    Stores both the original Beta parameters (for frequency distribution)
    and transformed parameters (for effective frequency under selection).
    """
    bin_id: int
    states: np.ndarray
    N: int
    s: float  # Selection coefficient

    # Original Beta parameters (fitted to bin)
    alpha: Optional[float] = None
    beta_param: Optional[float] = None

    # Transformed Beta parameters (accounting for selection)
    alpha_tilde: Optional[float] = None
    beta_tilde: Optional[float] = None

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=int)
        self.min_state = int(self.states[0])
        self.max_state = int(self.states[-1])
        self.width = self.max_state - self.min_state + 1

    @property
    def mean(self) -> float:
        """Mean allele count in bin."""
        return float(np.mean(self.states))

    @property
    def variance(self) -> float:
        """Variance of allele count in bin."""
        return float(np.var(self.states))

    @property
    def frequency_mean(self) -> float:
        """Mean allele frequency in bin."""
        return self.mean / self.N

    @property
    def frequency_variance(self) -> float:
        """Variance of allele frequency in bin."""
        return self.variance / (self.N ** 2)

    def fit_beta_parameters(self) -> Tuple[float, float]:
        """
        Fit Beta parameters to match bin mean and variance.
        """
        p = self.frequency_mean
        tau2 = self.frequency_variance

        if tau2 <= 0:
            raise ValueError(f"Bin variance must be positive, got {tau2}")

        max_var = p * (1 - p)
        if tau2 > max_var:
            raise ValueError(f"Bin variance {tau2:.6f} exceeds maximum {max_var:.6f}")

        # Method of moments
        s_param = p * (1 - p) / tau2 - 1
        self.alpha = p * s_param
        self.beta_param = (1 - p) * s_param

        return self.alpha, self.beta_param

    def compute_transformed_beta(self, method: str = 'delta') -> Tuple[float, float]:
        """
        Compute Beta parameters for the selection-transformed frequency.

        Parameters
        ----------
        method : str
            'delta' for delta method approximation, 'exact' for numerical integration

        Returns
        -------
        alpha_tilde, beta_tilde : float
            Parameters of Beta distribution for effective frequency
        """
        if self.alpha is None:
            self.fit_beta_parameters()

        if self.s == 0:
            self.alpha_tilde = self.alpha
            self.beta_tilde = self.beta_param
            return self.alpha_tilde, self.beta_tilde

        # Get moments of transformed distribution
        if method == 'delta':
            mean_tilde, var_tilde = transformed_beta_moments_delta(
                self.alpha, self.beta_param, self.s
            )
        else:
            mean_tilde, var_tilde = transformed_beta_moments_exact(
                self.alpha, self.beta_param, self.s
            )

        # Fit new Beta to these moments
        # From method of moments for Beta:
        # mean = alpha / (alpha + beta)
        # var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))

        if var_tilde <= 0 or mean_tilde <= 0 or mean_tilde >= 1:
            # Fallback: use simple transformation of center
            p_center = self.frequency_mean
            p_tilde_center = selection_transform(p_center, self.s)
            # Very concentrated Beta
            self.alpha_tilde = p_tilde_center * 1000
            self.beta_tilde = (1 - p_tilde_center) * 1000
        else:
            # Standard method of moments
            max_var = mean_tilde * (1 - mean_tilde)
            if var_tilde >= max_var:
                var_tilde = 0.99 * max_var  # Numerical safety

            s_param = mean_tilde * (1 - mean_tilde) / var_tilde - 1
            self.alpha_tilde = mean_tilde * s_param
            self.beta_tilde = (1 - mean_tilde) * s_param

        return self.alpha_tilde, self.beta_tilde


# =============================================================================
# Main Model Class
# =============================================================================

class WrightFisherSelection:
    """
    Wright-Fisher model with selection using transformed Beta-Binomial approximation.

    Key Innovation: Instead of naively applying Beta-Binomial to the original
    frequency, we:
    1. Fit Beta(α, β) to the frequency distribution within each bin
    2. Compute the distribution of the TRANSFORMED frequency p_tilde = p(1+s)/(1+sp)
    3. Fit a new Beta(α̃, β̃) to this transformed distribution
    4. Use Beta-Binomial(N, α̃, β̃) for transitions

    This properly accounts for the nonlinearity of the selection transformation.
    """

    def __init__(
        self,
        N: int,
        n_bins: int,
        s: float,
        transformation_method: str = 'delta'
    ):
        """
        Initialize Wright-Fisher model with selection.

        Parameters
        ----------
        N : int
            Population size
        n_bins : int
            Number of bins
        s : float
            Selection coefficient (s > 0 = beneficial)
        transformation_method : str
            'delta' or 'exact' for computing transformed moments
        """
        self.N = N
        self.n_bins = n_bins
        self.s = s
        self.transformation_method = transformation_method

        self.bins: List[SelectionBin] = []
        self.state_to_bin: dict = {}

        self._create_bins()
        self._fit_all_bins()

    def _create_bins(self):
        """Create uniform bins across state space."""
        bin_width = (self.N + 1) / self.n_bins

        for b in range(self.n_bins):
            bin_start = int(np.floor(b * bin_width))
            bin_end = int(np.floor((b + 1) * bin_width)) - 1

            if b == self.n_bins - 1:
                bin_end = self.N

            bin_states = np.arange(bin_start, bin_end + 1)

            bin_obj = SelectionBin(
                bin_id=b,
                states=bin_states,
                N=self.N,
                s=self.s
            )

            self.bins.append(bin_obj)

            for state in bin_states:
                self.state_to_bin[int(state)] = b

    def _fit_all_bins(self):
        """Fit Beta parameters for all bins."""
        for bin_obj in self.bins:
            try:
                bin_obj.fit_beta_parameters()
                bin_obj.compute_transformed_beta(method=self.transformation_method)
            except ValueError as e:
                warnings.warn(f"Failed to fit bin {bin_obj.bin_id}: {e}")

    def transition_distribution(self, bin_id: int) -> np.ndarray:
        """
        Compute transition distribution from a bin using transformed Beta-Binomial.

        Returns
        -------
        dist : np.ndarray of shape (N+1,)
            Probability distribution over next states
        """
        bin_obj = self.bins[bin_id]

        if bin_obj.alpha_tilde is None:
            raise RuntimeError(f"Bin {bin_id} not fitted")

        dist = betabinom.pmf(
            np.arange(self.N + 1),
            self.N,
            bin_obj.alpha_tilde,
            bin_obj.beta_tilde
        )

        return dist

    def lumped_transition_matrix(self) -> np.ndarray:
        """Compute bin-to-bin transition matrix."""
        P = np.zeros((self.n_bins, self.n_bins))

        for i in range(self.n_bins):
            dist = self.transition_distribution(i)
            for j in range(self.n_bins):
                bin_j = self.bins[j]
                P[i, j] = np.sum(dist[bin_j.min_state:bin_j.max_state + 1])

        return P

    def simulate_trajectory(
        self,
        initial_state: int,
        n_generations: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate trajectory using transformed Beta-Binomial.
        """
        rng = np.random.default_rng(seed)
        trajectory = np.zeros(n_generations + 1, dtype=int)
        trajectory[0] = initial_state

        current_state = initial_state

        for t in range(n_generations):
            current_bin = self.state_to_bin[current_state]
            bin_obj = self.bins[current_bin]

            # Sample from transformed Beta
            p_tilde = rng.beta(bin_obj.alpha_tilde, bin_obj.beta_tilde)
            p_tilde = np.clip(p_tilde, 0, 1)

            # Binomial sampling
            next_state = rng.binomial(self.N, p_tilde)
            trajectory[t + 1] = next_state
            current_state = next_state

        return trajectory

    def simulate_ensemble(
        self,
        initial_state: int,
        n_generations: int,
        n_replicates: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate multiple trajectories."""
        trajectories = np.zeros((n_replicates, n_generations + 1), dtype=int)

        for rep in range(n_replicates):
            rep_seed = None if seed is None else seed + rep
            trajectories[rep] = self.simulate_trajectory(
                initial_state, n_generations, seed=rep_seed
            )

        return trajectories


# =============================================================================
# Theoretical Predictions
# =============================================================================

class TheoreticalPredictions:
    """
    Theoretical predictions for Wright-Fisher with selection.

    All formulas from classical population genetics theory.
    """

    @staticmethod
    def fixation_probability(p0: float, N: int, s: float) -> float:
        """
        Kimura (1962) fixation probability formula.

        P_fix = (1 - exp(-2Ns*p0)) / (1 - exp(-2Ns))

        For neutral: P_fix = p0
        For strong selection (2Ns >> 1): P_fix ≈ 1 - exp(-2Ns*p0)
        """
        if s == 0:
            return p0

        two_Ns = 2 * N * s

        # Numerical stability for small s
        if abs(two_Ns) < 1e-6:
            return p0

        numerator = 1 - np.exp(-two_Ns * p0)
        denominator = 1 - np.exp(-two_Ns)

        # Handle edge cases
        if np.abs(denominator) < 1e-10:
            return p0

        return numerator / denominator

    @staticmethod
    def fixation_time_given_fixation(p0: float, N: int, s: float) -> float:
        """
        Expected fixation time conditional on fixation (approximate).

        For beneficial mutations (s > 0):
        E[T | fix] ≈ (2/s) * (ln(2Ns) + γ - ln(p0))

        where γ ≈ 0.5772 is Euler's constant.
        """
        if s <= 0:
            # For neutral or deleterious, use different formula
            return 4 * N * p0 * np.log(1/p0)  # Approximate

        gamma = 0.5772156649  # Euler's constant
        return (2/s) * (np.log(2*N*s) + gamma - np.log(p0))

    @staticmethod
    def establishment_probability_new_mutation(N: int, s: float) -> float:
        """
        Haldane (1927) probability of establishment for a single new mutation.

        P_establish ≈ 2s for s << 1

        More accurate: P_establish = 1 - exp(-2s) ≈ 2s - 2s² + ...
        """
        if s <= 0:
            return 0.0

        # Use Kimura formula with p0 = 1/N
        return TheoreticalPredictions.fixation_probability(1/N, N, s)

    @staticmethod
    def expected_frequency_change(p: float, s: float) -> float:
        """
        Expected change in frequency per generation (deterministic component).

        Δp = s * p * (1-p) / (1 + s*p)

        For weak selection: Δp ≈ s * p * (1-p)
        """
        if s == 0:
            return 0.0
        return s * p * (1 - p) / (1 + s * p)

    @staticmethod
    def stationary_frequency_mean(N: int, s: float, mu: float = 0, nu: float = 0) -> float:
        """
        Expected frequency at stationarity with mutation.

        mu = mutation rate A -> a
        nu = mutation rate a -> A

        For s > 0 and no mutation: fixation at 1
        With symmetric mutation (mu = nu): mean shifts toward 0.5 + O(Ns)
        """
        if mu == 0 and nu == 0:
            if s > 0:
                return 1.0  # Fixation
            elif s < 0:
                return 0.0  # Loss
            else:
                return 0.5  # Neutral, depends on initial

        # With mutation, approximate
        theta_nu = 4 * N * nu
        theta_mu = 4 * N * mu

        # Neutral expectation
        p_neutral = theta_nu / (theta_mu + theta_nu)

        # Selection shifts this
        return p_neutral + 2 * N * s * p_neutral * (1 - p_neutral) / (theta_mu + theta_nu + 1)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_fixation_probability(
    model: WrightFisherSelection,
    initial_count: int,
    n_generations: int,
    n_replicates: int,
    seed: Optional[int] = None
) -> dict:
    """
    Validate model against Kimura fixation probability.
    """
    trajectories = model.simulate_ensemble(
        initial_count, n_generations, n_replicates, seed
    )

    final_counts = trajectories[:, -1]
    n_fixed = np.sum(final_counts == model.N)
    n_lost = np.sum(final_counts == 0)
    resolved = n_fixed + n_lost

    if resolved == 0:
        p_fix_empirical = np.nan
    else:
        p_fix_empirical = n_fixed / resolved

    p0 = initial_count / model.N
    p_fix_theory = TheoreticalPredictions.fixation_probability(p0, model.N, model.s)

    return {
        'p0': p0,
        's': model.s,
        'N': model.N,
        'n_replicates': n_replicates,
        'n_fixed': n_fixed,
        'n_lost': n_lost,
        'n_segregating': n_replicates - resolved,
        'p_fix_empirical': p_fix_empirical,
        'p_fix_theory': p_fix_theory,
        'error': abs(p_fix_empirical - p_fix_theory) if not np.isnan(p_fix_empirical) else np.nan,
        'relative_error': abs(p_fix_empirical - p_fix_theory) / p_fix_theory if p_fix_theory > 0 and not np.isnan(p_fix_empirical) else np.nan
    }


def validate_frequency_trajectory(
    model: WrightFisherSelection,
    initial_count: int,
    n_generations: int,
    n_replicates: int,
    seed: Optional[int] = None
) -> dict:
    """
    Validate expected frequency trajectory against deterministic prediction.
    """
    trajectories = model.simulate_ensemble(
        initial_count, n_generations, n_replicates, seed
    )

    # Empirical mean trajectory
    mean_freq = np.mean(trajectories, axis=0) / model.N

    # Deterministic trajectory (no drift)
    det_freq = np.zeros(n_generations + 1)
    det_freq[0] = initial_count / model.N
    for t in range(n_generations):
        p = det_freq[t]
        dp = TheoreticalPredictions.expected_frequency_change(p, model.s)
        det_freq[t + 1] = np.clip(p + dp, 0, 1)

    return {
        'times': np.arange(n_generations + 1),
        'mean_freq_empirical': mean_freq,
        'mean_freq_deterministic': det_freq,
        'max_deviation': np.max(np.abs(mean_freq - det_freq))
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_validation_results(
    model: WrightFisherSelection,
    initial_count: int,
    n_generations: int = 500,
    n_replicates: int = 1000,
    seed: int = 42
):
    """
    Comprehensive validation plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Simulate
    trajectories = model.simulate_ensemble(
        initial_count, n_generations, n_replicates, seed
    )

    # 1. Example trajectories
    ax = axes[0, 0]
    for i in range(min(20, n_replicates)):
        ax.plot(trajectories[i], alpha=0.3, linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax.axhline(model.N, color='blue', linestyle='--', alpha=0.3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Allele count')
    ax.set_title(f'Sample Trajectories (s={model.s})')

    # 2. Mean trajectory vs deterministic
    ax = axes[0, 1]
    result = validate_frequency_trajectory(
        model, initial_count, n_generations, n_replicates, seed
    )
    ax.plot(result['times'], result['mean_freq_empirical'], 'b-',
            linewidth=2, label='Empirical mean')
    ax.plot(result['times'], result['mean_freq_deterministic'], 'r--',
            linewidth=2, label='Deterministic')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Allele frequency')
    ax.set_title('Mean Trajectory')
    ax.legend()

    # 3. Fixation probability vs theory
    ax = axes[0, 2]
    # Test across different initial frequencies
    initial_freqs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    p_fix_emp = []
    p_fix_thy = []

    for p0 in initial_freqs:
        init_count = int(p0 * model.N)
        result = validate_fixation_probability(
            model, init_count, n_generations, 200, seed
        )
        p_fix_emp.append(result['p_fix_empirical'])
        p_fix_thy.append(result['p_fix_theory'])

    ax.scatter(initial_freqs, p_fix_emp, s=100, label='Empirical', zorder=3)
    ax.plot(initial_freqs, p_fix_thy, 'r-', linewidth=2, label='Kimura theory')
    ax.set_xlabel('Initial frequency')
    ax.set_ylabel('Fixation probability')
    ax.set_title('Fixation Probability')
    ax.legend()

    # 4. Final state distribution
    ax = axes[1, 0]
    final_counts = trajectories[:, -1]
    ax.hist(final_counts, bins=50, density=True, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Lost')
    ax.axvline(model.N, color='blue', linestyle='--', label='Fixed')
    ax.set_xlabel('Final allele count')
    ax.set_ylabel('Density')
    ax.set_title('Final State Distribution')
    ax.legend()

    # 5. Transformed vs Original Beta
    ax = axes[1, 1]
    mid_bin = model.bins[model.n_bins // 2]
    p_range = np.linspace(0.001, 0.999, 1000)

    # Original Beta
    original_pdf = beta.pdf(p_range, mid_bin.alpha, mid_bin.beta_param)
    ax.plot(p_range, original_pdf, 'b-', linewidth=2, label=f'Original β({mid_bin.alpha:.1f}, {mid_bin.beta_param:.1f})')

    # Transformed Beta
    transformed_pdf = beta.pdf(p_range, mid_bin.alpha_tilde, mid_bin.beta_tilde)
    ax.plot(p_range, transformed_pdf, 'r--', linewidth=2,
            label=f'Transformed β({mid_bin.alpha_tilde:.1f}, {mid_bin.beta_tilde:.1f})')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Density')
    ax.set_title(f'Beta Distribution (Bin {mid_bin.bin_id}, s={model.s})')
    ax.legend()

    # 6. Variance growth
    ax = axes[1, 2]
    var_freq = np.var(trajectories, axis=0) / model.N**2
    ax.plot(var_freq, 'g-', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Frequency variance')
    ax.set_title('Variance Growth')

    plt.tight_layout()
    plt.suptitle(f'Wright-Fisher Validation: N={model.N}, s={model.s}',
                 fontsize=14, y=1.02)

    return fig


def compare_selection_strengths(
    N: int = 5000,
    n_bins: int = 10,
    initial_freq: float = 0.1,
    n_generations: int = 500,
    n_replicates: int = 500,
    seed: int = 42
):
    """
    Compare dynamics across different selection strengths.
    """
    selection_coefficients = [0, 0.001, 0.005, 0.01, 0.02]
    initial_count = int(initial_freq * N)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selection_coefficients)))

    # Store results
    all_trajectories = {}

    for s, color in zip(selection_coefficients, colors):
        model = WrightFisherSelection(N, n_bins, s)
        trajectories = model.simulate_ensemble(
            initial_count, n_generations, n_replicates, seed
        )
        all_trajectories[s] = trajectories

        # Mean trajectory
        mean_freq = np.mean(trajectories, axis=0) / N
        axes[0, 0].plot(mean_freq, color=color, linewidth=2, label=f's={s}')

        # Variance
        var_freq = np.var(trajectories, axis=0) / N**2
        axes[0, 1].plot(var_freq, color=color, linewidth=2, label=f's={s}')

    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Mean frequency')
    axes[0, 0].set_title('Mean Allele Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].set_title('Frequency Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Fixation probability
    ax = axes[1, 0]
    p_fix_emp = []
    p_fix_thy = []

    for s in selection_coefficients:
        trajs = all_trajectories[s]
        final = trajs[:, -1]
        n_fix = np.sum(final == N)
        n_lost = np.sum(final == 0)
        if n_fix + n_lost > 0:
            p_fix_emp.append(n_fix / (n_fix + n_lost))
        else:
            p_fix_emp.append(np.nan)
        p_fix_thy.append(TheoreticalPredictions.fixation_probability(initial_freq, N, s))

    ax.scatter(selection_coefficients, p_fix_emp, s=100, label='Empirical', zorder=3)
    ax.plot(selection_coefficients, p_fix_thy, 'r-', linewidth=2, label='Kimura theory')
    ax.set_xlabel('Selection coefficient (s)')
    ax.set_ylabel('Fixation probability')
    ax.set_title(f'Fixation Probability (p₀={initial_freq})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final distributions
    ax = axes[1, 1]
    for s, color in zip(selection_coefficients, colors):
        final = all_trajectories[s][:, -1] / N
        ax.hist(final, bins=30, alpha=0.3, color=color, label=f's={s}', density=True)
    ax.set_xlabel('Final frequency')
    ax.set_ylabel('Density')
    ax.set_title('Final Frequency Distribution')
    ax.legend()

    plt.tight_layout()
    plt.suptitle(f'Selection Strength Comparison: N={N}, p₀={initial_freq}',
                 fontsize=14, y=1.02)

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("Wright-Fisher with Selection: Validation")
    print("=" * 60)

    # Parameters
    N = 5000
    n_bins = 10
    s = 0.01  # 1% selection advantage
    initial_count = 500  # 10% initial frequency

    print(f"\nModel parameters:")
    print(f"  Population size (N): {N}")
    print(f"  Number of bins: {n_bins}")
    print(f"  Selection coefficient (s): {s}")
    print(f"  Initial count: {initial_count} (freq = {initial_count/N})")

    # Create model
    model = WrightFisherSelection(N, n_bins, s)

    # Validate fixation probability
    print("\n" + "-" * 60)
    result = validate_fixation_probability(
        model, initial_count, n_generations=1000, n_replicates=500, seed=42
    )

    print(f"\nFixation Probability Validation:")
    print(f"  Theoretical (Kimura): {result['p_fix_theory']:.4f}")
    print(f"  Empirical: {result['p_fix_empirical']:.4f}")
    print(f"  Error: {result['error']:.4f}")
    print(f"  Fixed: {result['n_fixed']}, Lost: {result['n_lost']}, Segregating: {result['n_segregating']}")

    # Plot
    fig = plot_validation_results(model, initial_count)
    plt.savefig('/Users/kmt/phasic/docs/pages/tutorials/popgen/python/wf_selection_validation.png',
                dpi=150, bbox_inches='tight')
    print("\nValidation plot saved.")

    # Compare selection strengths
    fig2 = compare_selection_strengths()
    plt.savefig('/Users/kmt/phasic/docs/pages/tutorials/popgen/python/wf_selection_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Selection comparison plot saved.")
