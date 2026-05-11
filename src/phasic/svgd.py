from __future__ import annotations

import logging
import os
import platform
from time import time, sleep
from typing import Callable, NamedTuple
import warnings
import matplotlib
import numpy as np
import pickle
import hashlib
import pathlib
from itertools import zip_longest
# Note: JAX environment (XLA_FLAGS, device count) is configured by
# phasic.__init__.py before this module is imported.
# Users should configure via:
#   1. phasic.configure() before import, OR
#   2. PTDALG_CPUS environment variable
# See: src/phasic/__init__.py lines 101-133
import jax
# print(jax.devices())
import jax.numpy as jnp
from jax import grad, vmap, jit, pmap
from jax.scipy.stats import norm
import jax.nn as jnn
import jax.sharding as jsh
from jax.experimental import checkify
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.scipy.stats import norm

from phasic.optax_wrapper import OptaxOptimizer
from scipy.stats import gaussian_kde, gengamma

from functools import partial

from .logging_config import get_logger

logger = get_logger(__name__)

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection

# "iridis" color map (viridis without the deep purple)
def truncate_colormap(cmap: matplotlib.colors.Colormap, minval: float = 0.0, maxval: float = 1.0, n: int = 100) -> matplotlib.colors.LinearSegmentedColormap:
    """Truncate a colormap to a subset of its range."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2g},{b:.2g})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)


# Import configuration system
from .config import get_config
from .exceptions import PTDConfigError

#from .vscode_theme import black_white, phasic_theme

# from . import svgd_plots

## requires equinox dependency
# from .decoders import VariableDimPTDDecoder, LessThanOneDecoder, 
#     SumToOneDecoder, IndependentProbDecoder

# def string_to_class(s, suffix=''):
#     class_name = ''.join(x.capitalize() for x in s.split('_')) + suffix
#     if class_name not in globals():
#         raise ValueError(f"Cannot translate string to class name: {s}")
#     return globals()[class_name]


try:
    from vscodenb import pqdm as tqdm
    from vscodenb import prange as trange
except ImportError:
    from tqdm import trange, tqdm
    trange = partial(trange, leave=False)
    tqdm = partial(tqdm, leave=False)

#from jax import random, vmap, grad, jit

black_or_white = matplotlib.rcParams['text.color']

# ============================================================================
# Sparse Observation Format for Multivariate SVGD
# ============================================================================

class SparseObservations(NamedTuple):
    """Sparse representation of multivariate observations.

    Replaces dense NaN-padded format with parallel arrays containing only
    valid observations. This avoids NaN propagation through JAX callbacks
    during gradient computation.

    Attributes
    ----------
    values : jnp.ndarray
        Observation values (n_obs,) - no NaN values
    features : jnp.ndarray
        Feature index for each observation (n_obs,) - integers
    n_features : int
        Total number of features (for rewards indexing)
    slices : tuple of tuples, optional
        Pre-computed (start, end) slices for each feature to avoid
        dynamic boolean indexing in JIT-compiled code. If provided,
        observations must be sorted by feature index.

    Examples
    --------
    >>> # 10 observations for feature 0, 10 for feature 1, 10 for feature 2
    >>> sparse = SparseObservations(
    ...     values=jnp.array([1.1, 1.2, ..., 2.1, 2.2, ..., 3.1, 3.2, ...]),
    ...     features=jnp.array([0, 0, ..., 1, 1, ..., 2, 2, ...]),
    ...     n_features=3
    ... )

    See Also
    --------
    dense_to_sparse : Convert dense NaN-padded array to sparse format
    """
    values: jnp.ndarray
    features: jnp.ndarray
    n_features: int
    slices: tuple = None  # Optional: pre-computed (start, end) per feature

    def get_feature_values(self, feature_idx: int) -> jnp.ndarray:
        """Get observation values for a specific feature.

        Uses pre-computed slices if available (JAX JIT compatible),
        otherwise falls back to boolean indexing (not JIT compatible).

        Parameters
        ----------
        feature_idx : int
            Feature index

        Returns
        -------
        jnp.ndarray
            Observation values for this feature
        """
        if self.slices is not None:
            start, end = self.slices[feature_idx]
            return self.values[start:end]
        else:
            # Fallback (not JIT compatible)
            mask = self.features == feature_idx
            return self.values[mask]

    def __repr__(self) -> str:
        return ("SparseObservations("
                f"values=<{len(self.values)} values>, "
                f"n_features=<{self.n_features}, "
                f"features=<{len(self.values)} values>, "
                f"slices=<{len(self.values)} values>)")
    

def dense_to_sparse(data: jnp.ndarray) -> SparseObservations:
    """Convert dense NaN-padded array to sparse observation format.

    Parameters
    ----------
    data : jnp.ndarray
        Dense 2D array of shape (n_times, n_features) where NaN indicates
        missing observations.

    Returns
    -------
    SparseObservations
        Sparse representation with only valid observations, sorted by feature
        with pre-computed slices for JAX JIT compatibility.

    Examples
    --------
    >>> dense = jnp.array([
    ...     [1.0, np.nan, 3.0],
    ...     [np.nan, 2.0, np.nan],
    ...     [1.5, 2.5, 3.5]
    ... ])
    >>> sparse = dense_to_sparse(dense)
    >>> print(sparse.values)   # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    >>> print(sparse.features) # [0, 0, 1, 1, 2, 2]
    >>> print(sparse.n_features)  # 3
    """
    data = jnp.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"dense_to_sparse requires 2D array (n_times, n_features). "
            f"Got shape: {data.shape}"
        )

    n_times, n_features = data.shape

    # Collect values and feature indices, keeping track of counts for slices
    values_list = []
    features_list = []
    slices = []
    current_pos = 0

    for j in range(n_features):
        col = data[:, j]
        valid_mask = ~jnp.isnan(col)
        valid_values = col[valid_mask]
        count = len(valid_values)

        values_list.append(valid_values)
        features_list.append(jnp.full(valid_values.shape, j, dtype=jnp.int32))
        slices.append((current_pos, current_pos + count))
        current_pos += count

    # Concatenate all features
    if values_list:
        values = jnp.concatenate(values_list)
        features = jnp.concatenate(features_list)
    else:
        values = jnp.array([], dtype=jnp.float64)
        features = jnp.array([], dtype=jnp.int32)

    return SparseObservations(
        values=values,
        features=features,
        n_features=n_features,
        slices=tuple(slices)
    )


def is_sparse_observations(data: object) -> bool:
    """Check if data is in sparse observation format.

    Parameters
    ----------
    data : object
        Data to check.

    Returns
    -------
    bool
        True if data is a ``SparseObservations`` instance.
    """
    return isinstance(data, SparseObservations)


# ============================================================================
# Helper Functions
# ============================================================================

def _inverse_softplus(theta: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: phi = log(exp(theta) - 1).

    Numerically stable implementation that handles large values.

    Parameters
    ----------
    theta : array
        Values in constrained space (positive)

    Returns
    -------
    array
        Values in unconstrained space
    """
    # For large theta, softplus(phi) ≈ phi, so inverse is identity
    # For smaller values, use log(exp(theta) - 1) = log(expm1(theta))
    # Use 1e-6 as minimum to match param_transform minimum
    return jnp.where(
        theta > 20,
        theta,
        jnp.log(jnp.expm1(jnp.maximum(theta, 1e-6)))
    )

def _compute_hpd(samples: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Compute the Highest Posterior Density (HPD) interval.

    Finds the shortest contiguous interval containing at least `alpha` fraction
    of the samples. For unimodal posteriors, this naturally centers on the mode.

    The algorithm sorts the samples and slides a window of size
    ``ceil(n * alpha)`` across them, selecting the window whose endpoints
    have the smallest difference.  This guarantees the minimum-width interval
    that contains at least the requested fraction of mass.  Unlike equal-tailed
    percentile intervals, the HPD interval is invariant to monotone
    reparametrisation and, for skewed posteriors, is shorter and better
    centred on the high-density region.

    Parameters
    ----------
    samples : np.ndarray
        1D array of posterior samples.
    alpha : float, default=0.95
        Credible level (fraction of mass to include).

    Returns
    -------
    lower : float
        Lower bound of the HPD interval.
    upper : float
        Upper bound of the HPD interval.
    """
    samples = np.asarray(samples).ravel()
    n = len(samples)

    if n < 2:
        warnings.warn("HPD interval requires at least 2 samples; returning (min, max).")
        return float(samples.min()), float(samples.max())

    if n < 10:
        warnings.warn(
            f"HPD interval computed from only {n} samples; result may be unreliable."
        )

    sorted_samples = np.sort(samples)

    # Number of samples that must be inside the interval
    n_included = int(np.ceil(n * alpha))
    # Clamp to n so alpha=1.0 works
    n_included = min(n_included, n)

    # All contiguous windows of size n_included
    # Width of each window: sorted[i + n_included - 1] - sorted[i]
    widths = sorted_samples[n_included - 1:] - sorted_samples[:n - n_included + 1]

    best = int(np.argmin(widths))
    lower = float(sorted_samples[best])
    upper = float(sorted_samples[best + n_included - 1])

    return lower, upper


def _hex_grid(x_min: float, x_max: float, y_min: float, y_max: float, size: float, flat_topped: bool = False) -> jnp.ndarray:
    """Generate hex grid midpoints.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal bounding box limits.
    y_min, y_max : float
        Vertical bounding box limits.
    size : float
        Distance from hex center to vertex.
    flat_topped : bool, default=False
        If False, pointy-topped hexagons. If True, flat-topped hexagons.

    Returns
    -------
    jnp.ndarray
        Array of midpoint coordinates with shape ``(n_points, 2)``.
    """
    if flat_topped:
        dx = 1.5 * size
        dy = np.sqrt(3) * size
    else:
        dx = np.sqrt(3) * size
        dy = 1.5 * size
    
    x = jnp.arange(x_min, x_max + dx, dx)
    y = jnp.arange(y_min, y_max + dy, dy)
    
    xx, yy = np.meshgrid(x, y)
    
    if flat_topped:
        yy[:, 1::2] += dy / 2  # offset odd columns
        mask = yy.ravel() <= y_max
    else:
        xx[1::2] += dx / 2  # offset odd rows
        mask = xx.ravel() <= x_max
    
    points = jnp.column_stack((xx.ravel(), yy.ravel()))

    return points[mask]

# ============================================================================
# Prior Distribution Classes
# ============================================================================

class Prior:
    """Base class for prior distributions.

    Subclasses must implement __call__ (log-probability) and sample methods.
    The sample method enables SVGD to initialize particles from the prior.
    """

    def __call__(self, phi: jnp.ndarray) -> float:
        """Compute log-probability of phi.

        Parameters
        ----------
        phi : jnp.ndarray
            Parameter vector (in unconstrained space)

        Returns
        -------
        float
            Log-probability of phi under the prior
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def sample(self, key: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Sample from the prior distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for JAX
        shape : tuple
            Shape of samples, typically (n_particles, theta_dim)

        Returns
        -------
        jnp.ndarray
            Samples from prior with given shape
        """
        raise NotImplementedError("Subclasses must implement sample")

    def plot(self, ax: matplotlib.axes.Axes | None = None, **kwargs) -> matplotlib.axes.Axes:
        """Plot the prior distribution.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to plot function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        raise NotImplementedError("Subclasses must implement plot")


class GaussPrior(Prior):
    """Gaussian prior distribution.

    The prior is defined in THETA space (the natural parameter space).
    When used with positive_params=True, SVGD automatically handles
    the transformation to PHI space with proper Jacobian correction.

    Can be specified via mean/std or credible interval.

    Parameters
    ----------
    mean : float, optional
        Prior mean in THETA space. Required if std is provided.
    std : float, optional
        Prior standard deviation in THETA space. Required if mean is provided.
    ci : tuple of (float, float), optional
        Credible interval (low, high) in THETA space. Alternative to mean/std.
    prob : float, default=0.95
        Probability mass in the credible interval (only used with ci).

    Examples
    --------
    >>> # Specify via mean and std
    >>> prior = GaussPrior(mean=5.0, std=2.0)
    >>>
    >>> # Specify via 95% credible interval
    >>> prior = GaussPrior(ci=(2.0, 8.0))
    >>>
    >>> # Plot to verify prior matches your beliefs
    >>> prior.plot()  # Shows Gaussian centered at 5
    >>>
    >>> # Use in SVGD - transformations handled automatically
    >>> svgd = graph.svgd(data, theta_dim=1, prior=prior)
    """

    def __init__(self, mean: float | None = None, std: float | None = None, ci: tuple[float, float] | None = None, prob: float = 0.95) -> None:
        if mean is not None and std is not None:
            self.mu = mean
            self.sigma = std
        elif ci is not None:
            low, high = ci
            mu = (low + high) / 2
            z = norm.ppf((1 + prob) / 2)
            sigma = (high - low) / (2 * z)
            self.mu = mu
            self.sigma = sigma
        else:
            raise ValueError(
                "Invalid prior specification. Provide either (mean, std) or ci."
            )
        # Transform function set by SVGD when positive_params=True
        self._transform = None

    def __call__(self, phi: jnp.ndarray) -> float:
        """Compute log-probability.

        When _transform is set (by SVGD with positive_params=True), evaluates
        the prior in THETA space with proper Jacobian correction. Otherwise,
        evaluates directly on the input.

        Parameters
        ----------
        phi : array
            Parameter values (in PHI space if transform is set)

        Returns
        -------
        float
            Log-probability
        """
        if self._transform is not None:
            # Prior is defined in THETA space, input is in PHI space
            theta = self._transform(phi)
            # Gaussian log-probability in THETA space
            log_prior_theta = -0.5 * jnp.sum(((theta - self.mu) / self.sigma)**2)
            # Jacobian correction: log|dθ/dφ| = log(sigmoid(φ)) = -softplus(-φ)
            log_jacobian = -jnp.sum(jax.nn.softplus(-phi))
            return log_prior_theta + log_jacobian
        else:
            # No transform: evaluate directly (backward compatible)
            return -0.5 * jnp.sum(((phi - self.mu) / self.sigma)**2)

    def sample(self, key: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Sample from the prior.

        When _transform is set, samples in THETA space and converts to PHI space.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key
        shape : tuple
            Shape of samples (n_particles, theta_dim)

        Returns
        -------
        array
            Samples (in PHI space if transform is set)
        """
        # Sample in THETA space (what user understands)
        theta_samples = jax.random.normal(key, shape) * self.sigma + self.mu

        if self._transform is not None:
            # Convert THETA → PHI using inverse softplus
            # Clip to ensure positive values before inverse transform
            theta_samples = jnp.maximum(theta_samples, 1e-6)
            return _inverse_softplus(theta_samples)
        else:
            return theta_samples

    def plot(self, log: bool = False, ax: matplotlib.axes.Axes | None = None, return_ax: bool = False, **kwargs) -> matplotlib.axes.Axes | None:
        """Plot the Gaussian prior distribution in THETA space.

        Parameters
        ----------
        log : bool, default=False
            If True, plot log-probability instead of probability density.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        return_ax : bool, default=True
            If True, return ax. If False, call plt.show() instead.
        **kwargs
            Additional arguments passed to plot function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot (only if return_ax=False)
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        # Always plot in THETA space (what user understands)
        x = np.linspace(max(0, self.mu - 4*self.sigma), self.mu + 4*self.sigma, 200)
        if log:
            # Log-probability in THETA space
            log_prob = -0.5 * ((x - self.mu) / self.sigma)**2
            ax.plot(x, log_prob, **kwargs)
            ax.set_title(f'Log Gaussian({self.mu:.2g}, {self.sigma:.2g})')
            ax.set_ylabel('Log density')
        else:
            ax.plot(x, norm.pdf(x, loc=self.mu, scale=self.sigma), **kwargs)
            ax.set_title(f'Gaussian({self.mu:.2g}, {self.sigma:.2g})')
            ax.set_ylabel('Density')
        ax.set_xlabel('Parameter value (θ)')
        if ax or return_ax:
            return ax
        else:
            plt.show()


class LogGaussPrior(Prior):
    """Log-normal prior distribution (positive support only).

    The prior is defined in THETA space: log(θ) ~ Normal(mu, sigma), so θ > 0.
    This is the natural prior for scale-like parameters (e.g. rates, inverse
    population sizes) because it is closed under reciprocation: if θ is
    log-normal, so is 1/θ.

    When used with positive_params=True, SVGD automatically handles the
    transformation to PHI space with proper Jacobian correction.

    Can be specified via mean/std on the log scale, or via a credible interval
    in THETA space (the bounds are converted to the log scale internally).

    Parameters
    ----------
    mean : float, optional
        Prior mean of log(θ). Required if std is provided.
    std : float, optional
        Prior standard deviation of log(θ). Required if mean is provided.
    ci : tuple of (float, float), optional
        Credible interval (low, high) on θ (both > 0). Internally converted
        to a Gaussian on log(θ) whose CI matches (log low, log high).
    prob : float, default=0.95
        Probability mass in the credible interval (only used with ci).

    Examples
    --------
    >>> # 95% CI on θ ∈ [1/50_000, 1/5_000]
    >>> prior = LogGaussPrior(ci=(1/50_000, 1/5_000))
    >>>
    >>> # Or directly via log-scale parameters
    >>> prior = LogGaussPrior(mean=-9.2, std=0.59)
    >>>
    >>> prior.plot()  # log-scaled x axis
    """

    def __init__(self, mean: float | None = None, std: float | None = None, ci: tuple[float, float] | None = None, prob: float = 0.95) -> None:
        if mean is not None and std is not None:
            self.mu = mean
            self.sigma = std
        elif ci is not None:
            low, high = ci
            if low <= 0 or high <= 0:
                raise ValueError("LogGaussPrior CI bounds must be strictly positive.")
            log_low = np.log(low)
            log_high = np.log(high)
            mu = (log_low + log_high) / 2
            z = norm.ppf((1 + prob) / 2)
            sigma = (log_high - log_low) / (2 * z)
            self.mu = float(mu)
            self.sigma = float(sigma)
        else:
            raise ValueError(
                "Invalid prior specification. Provide either (mean, std) or ci."
            )
        self._transform = None

    def __call__(self, phi: jnp.ndarray) -> float:
        """Compute log-probability.

        When _transform is set (by SVGD with positive_params=True), evaluates
        the log-normal prior in THETA space with proper Jacobian correction.
        Otherwise, evaluates directly on the input (which must be > 0).
        """
        if self._transform is not None:
            theta = self._transform(phi)
            log_theta = jnp.log(theta)
            # Log-normal log-density in THETA space (drop -log θ since it
            # is a θ-only constant once transform is fixed; keep it for
            # correctness so plot/sample/log-prob all agree)
            log_prior_theta = (
                -0.5 * jnp.sum(((log_theta - self.mu) / self.sigma)**2)
                - jnp.sum(log_theta)
            )
            log_jacobian = -jnp.sum(jax.nn.softplus(-phi))
            return log_prior_theta + log_jacobian
        else:
            log_phi = jnp.log(phi)
            return (
                -0.5 * jnp.sum(((log_phi - self.mu) / self.sigma)**2)
                - jnp.sum(log_phi)
            )

    def sample(self, key: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Sample from the log-normal prior.

        When _transform is set, samples in THETA space and converts to PHI space.
        """
        log_theta_samples = jax.random.normal(key, shape) * self.sigma + self.mu
        theta_samples = jnp.exp(log_theta_samples)

        if self._transform is not None:
            theta_samples = jnp.maximum(theta_samples, 1e-12)
            return _inverse_softplus(theta_samples)
        else:
            return theta_samples

    def plot(self, log: bool = False, ax: matplotlib.axes.Axes | None = None, return_ax: bool = False, **kwargs) -> matplotlib.axes.Axes | None:
        """Plot the log-normal prior in THETA space with a log-scaled x axis.

        Parameters
        ----------
        log : bool, default=False
            If True, plot log-probability instead of probability density.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        return_ax : bool, default=False
            If True, return ax. If False, call plt.show().
        **kwargs
            Additional arguments passed to plot function.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        # Span ±4σ on the log scale, then exponentiate
        log_x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 400)
        x = np.exp(log_x)
        if log:
            # Log of the log-normal density:  -log(x) - log(σ√2π) - (log x - μ)² / (2σ²)
            log_prob = (
                -np.log(x)
                - np.log(self.sigma * np.sqrt(2 * np.pi))
                - 0.5 * ((log_x - self.mu) / self.sigma)**2
            )
            ax.plot(x, log_prob, **kwargs)
            ax.set_title(f'Log LogNormal({self.mu:.2g}, {self.sigma:.2g})')
            ax.set_ylabel('Log density')
        else:
            pdf = (
                1.0 / (x * self.sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((log_x - self.mu) / self.sigma)**2)
            )
            ax.plot(x, pdf, **kwargs)
            ax.set_title(f'LogNormal({self.mu:.2g}, {self.sigma:.2g})')
            ax.set_ylabel('Density')
        ax.set_xscale('log')
        ax.set_xlabel('Parameter value (θ)')
        if ax or return_ax:
            return ax
        else:
            plt.show()


class HalfCauchyPrior(Prior):
    """Half-Cauchy prior distribution (positive support only).

    The prior is defined in THETA space (the natural parameter space).
    When used with positive_params=True, SVGD automatically handles
    the transformation to PHI space with proper Jacobian correction.

    Useful for scale parameters due to heavy tails.
    PDF: f(θ) = 2 / (π × scale × (1 + (θ/scale)²)) for θ > 0

    Parameters
    ----------
    scale : float, optional
        Scale parameter of the half-Cauchy distribution. Mutually exclusive with `ci`.
    ci : float, optional
        Upper bound of the credible interval. The scale is computed such that
        P(θ < ci) = prob. Mutually exclusive with `scale`.
    prob : float, default=0.95
        Coverage probability for the credible interval (only used with `ci`).
        E.g., prob=0.95 means 95% of the prior mass is below `ci`.

    Examples
    --------
    >>> # Specify scale directly
    >>> prior = HalfCauchyPrior(scale=2.0)
    >>>
    >>> # Specify via 95% CI upper bound
    >>> prior = HalfCauchyPrior(ci=10.0)  # 95% of mass below 10
    >>>
    >>> # Specify via 90% CI upper bound
    >>> prior = HalfCauchyPrior(ci=10.0, prob=0.90)
    >>>
    >>> # Plot to verify prior matches your beliefs
    >>> prior.plot()
    >>>
    >>> # Use in SVGD
    >>> svgd = graph.svgd(data, theta_dim=1, prior=prior)
    """

    def __init__(self, scale: float | None = None, ci: float | None = None, prob: float = 0.95) -> None:
        # Validate parameters
        if scale is not None and ci is not None:
            raise ValueError("Cannot specify both 'scale' and 'ci'. Use one or the other.")
        if scale is None and ci is None:
            scale = 1.0  # Default

        if ci is not None:
            # Compute scale from CI upper bound and prob
            # For Half-Cauchy: CDF(x) = (2/π) * arctan(x/scale)
            # So: prob = (2/π) * arctan(ci/scale)
            # Solving: scale = ci / tan(π * prob / 2)
            if not 0 < prob < 1:
                raise ValueError(f"prob must be in (0, 1), got {prob}")
            self.scale = ci / np.tan(np.pi * prob / 2)
            self.ci = ci
            self.prob = prob
        else:
            self.scale = scale
            self.ci = None
            self.prob = None

        # Transform function set by SVGD when positive_params=True
        self._transform = None

    def __call__(self, phi: jnp.ndarray) -> float:
        """Compute log-probability.

        When _transform is set (by SVGD with positive_params=True), evaluates
        the prior in THETA space with proper Jacobian correction. Otherwise,
        evaluates directly on the input.

        Parameters
        ----------
        phi : array
            Parameter values (in PHI space if transform is set)

        Returns
        -------
        float
            Log-probability
        """
        if self._transform is not None:
            # Prior is defined in THETA space, input is in PHI space
            theta = self._transform(phi)
            # Half-Cauchy log-probability in THETA space (always positive after softplus)
            log_prior_theta = jnp.sum(
                jnp.log(2) - jnp.log(jnp.pi) - jnp.log(self.scale)
                - jnp.log(1 + (theta/self.scale)**2)
            )
            # Jacobian correction: log|dθ/dφ| = log(sigmoid(φ)) = -softplus(-φ)
            log_jacobian = -jnp.sum(jax.nn.softplus(-phi))
            return log_prior_theta + log_jacobian
        else:
            # No transform: evaluate directly (backward compatible)
            # Returns -inf for phi <= 0
            log_prob = jnp.where(
                phi > 0,
                jnp.log(2) - jnp.log(jnp.pi) - jnp.log(self.scale) - jnp.log(1 + (phi/self.scale)**2),
                -jnp.inf
            )
            return jnp.sum(log_prob)

    def sample(self, key: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Sample from the prior.

        When _transform is set, samples in THETA space and converts to PHI space.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key
        shape : tuple
            Shape of samples (n_particles, theta_dim)

        Returns
        -------
        array
            Samples (in PHI space if transform is set)
        """
        u = jax.random.uniform(key, shape)
        # Sample in THETA space using inverse CDF
        theta_samples = self.scale * jnp.tan(jnp.pi * u / 2)

        if self._transform is not None:
            # Convert THETA → PHI using inverse softplus
            # Half-Cauchy samples are always positive, but clip for safety
            theta_samples = jnp.maximum(theta_samples, 1e-6)
            return _inverse_softplus(theta_samples)
        else:
            return theta_samples

    def plot(self, ax: matplotlib.axes.Axes | None = None, show_ci: bool = True, return_ax: bool = False, **kwargs) -> matplotlib.axes.Axes | None:
        """Plot the half-Cauchy prior distribution in THETA space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_ci : bool, default=True
            If True and ci was specified, show vertical line at CI bound.
        return_ax : bool, default=False
            If True, return ax. If False, call plt.show() instead.
        **kwargs
            Additional arguments passed to plot function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot (only if return_ax=False)
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        # Plot in THETA space (what user understands)
        x_max = 5 * self.scale if self.ci is None else max(5 * self.scale, 1.5 * self.ci)
        x = np.linspace(0.001, x_max, 200)
        pdf = 2 / (np.pi * self.scale * (1 + (x/self.scale)**2))
        ax.plot(x, pdf, **kwargs)

        # # Show CI bound if specified
        # if show_ci and self.ci is not None:
        #     ax.axvline(self.ci, color='magenta', linestyle='--', alpha=0.7,
        #               label=f'{int(self.prob*100)}% CI bound')
        #     ax.fill_between(x[x <= self.ci], pdf[x <= self.ci], alpha=0.2, color='blue')
        #     ax.legend()

        # title = f'Half-Cauchy({self.scale:.2g})'
        # if self.ci is not None:
        #     title += f'\n{int(self.prob*100)}% CI: (0, {self.ci:.3g}]'
        ax.set_title(f'HalfCauchy({self.scale:.2g})')
        ax.set_xlabel('Parameter value (θ)')
        ax.set_ylabel('Density')
        if ax or return_ax:
            return ax
        else:
            plt.show()


class DataPrior(Prior):
    """Data-informed prior estimated from observed data.

    Auto-detects the graph type and uses method-of-moments (standard graphs)
    or probability matching (joint probability graphs) to estimate parameter
    prior means from the data.  The result is a list of per-parameter
    ``GaussPrior`` objects (or ``None`` for fixed parameters) that can be
    passed directly to :meth:`Graph.svgd`.

    Implements the list-like interface so SVGD can iterate over per-parameter
    priors, and the ``Prior`` interface (``__call__``, ``sample``, ``plot``)
    for use as a single prior object.

    Parameters
    ----------
    graph : Graph
        Parameterized graph (standard or joint probability).
    observed_data : np.ndarray
        Observed data appropriate for the graph type.
    sd : float, default=2.0
        Multiplier applied to the asymptotic standard error to obtain the
        prior standard deviation (passed as ``std_multiplier``).
    fixed : list, optional
        List of ``(index, value)`` tuples pinning specific parameters.
    nr_moments : int, optional
        Number of moments (standard graphs only).
    rewards : np.ndarray, optional
        Reward vectors (standard graphs only).
    theta_dim : int, optional
        Number of model parameters.  Inferred from the graph when ``None``.
    theta_init : np.ndarray, optional
        Initial guess for the free parameters.
    discrete : bool, optional
        ``True`` for discrete models, ``False`` for continuous (standard graphs only).
    verbose : bool, default=False
        Print progress information.

    Examples
    --------
    >>> from phasic import Graph, DataPrior
    >>> g = Graph(...)
    >>> data = g.sample(1000)
    >>> svgd = g.svgd(data, prior=DataPrior(g, data))
    """

    def __init__(
        self,
        graph,
        observed_data,
        sd=2.0,
        fixed=None,
        nr_moments=None,
        rewards=None,
        theta_dim=None,
        theta_init=None,
        discrete=None,
        verbose=False,
    ):
        is_joint = graph._joint_prob_base_graph_indexer is not None

        if is_joint:
            self._method = "probability_matching"
            self._result = graph.probability_matching(
                observed_data,
                fixed=fixed,
                std_multiplier=sd,
                verbose=verbose,
                theta_dim=theta_dim,
                theta_init=theta_init,
            )
        else:
            self._method = "method_of_moments"
            self._result = graph.method_of_moments(
                observed_data,
                fixed=fixed,
                std_multiplier=sd,
                verbose=verbose,
                nr_moments=nr_moments,
                rewards=rewards,
                theta_dim=theta_dim,
                theta_init=theta_init,
                discrete=discrete,
            )

        self._priors = self._result.prior
        # Transform function set by SVGD when positive_params=True
        self._transform = None

        if not self._result.success:
            warnings.warn(
                f"DataPrior: {self._method} did not converge. "
                f"Message: {self._result.message}"
            )

    # ----- properties --------------------------------------------------------

    @property
    def result(self) -> object:
        """The underlying ``MoMResult`` or ``ProbMatchResult``."""
        return self._result

    @property
    def theta(self) -> np.ndarray:
        """Parameter estimate from the underlying method."""
        return self._result.theta

    @property
    def std(self) -> np.ndarray:
        """Standard errors of the parameter estimates."""
        return self._result.std

    @property
    def success(self) -> bool:
        """Whether the underlying optimisation converged."""
        return self._result.success

    @property
    def method(self) -> str:
        """Name of the estimation method used."""
        return self._method

    # ----- list-like interface (per-parameter priors) ------------------------

    def __iter__(self) -> iter:
        """Iterate over per-parameter priors."""
        return iter(self._priors)

    def __len__(self) -> int:
        """Return number of per-parameter priors."""
        return len(self._priors)

    def __getitem__(self, idx: int) -> Prior | None:
        """Get the prior for a specific parameter index."""
        return self._priors[idx]

    # ----- Prior interface ---------------------------------------------------

    def __call__(self, phi: jnp.ndarray) -> float:
        """Compute total log-probability as the sum of per-parameter priors.

        Parameters
        ----------
        phi : jnp.ndarray
            Parameter vector (in unconstrained space).

        Returns
        -------
        float
            Sum of log-probabilities from non-None per-parameter priors.
        """
        total = 0.0
        for i, prior_i in enumerate(self._priors):
            if prior_i is not None:
                total = total + prior_i(phi[i:i + 1])
        return total

    def sample(self, key: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Sample from per-parameter priors and concatenate.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        shape : tuple
            ``(n_particles, theta_dim)``

        Returns
        -------
        jnp.ndarray
            Samples with shape ``(n_particles, theta_dim)``.
        """
        n_particles = shape[0]
        columns = []
        for i, prior_i in enumerate(self._priors):
            subkey = jax.random.fold_in(key, i)
            if prior_i is not None:
                col = prior_i.sample(subkey, (n_particles, 1))
            else:
                # Fixed parameter — fill with the point estimate
                col = jnp.full((n_particles, 1), self._result.theta[i])
            columns.append(col)
        return jnp.concatenate(columns, axis=1)

    def plot(self, axes: np.ndarray | None = None, figsize: tuple[float, float] | None = None, return_axes: bool = False) -> np.ndarray | None:
        """Plot per-parameter prior distributions.

        Parameters
        ----------
        axes : array of Axes, optional
            Pre-existing axes (one per non-fixed parameter).
        figsize : tuple, optional
            Figure size when creating new axes.
        return_axes : bool, default=False
            If ``True``, return the axes array instead of calling ``plt.show()``.

        Returns
        -------
        array of Axes or None
        """
        import matplotlib.pyplot as plt

        free_indices = [i for i, p in enumerate(self._priors) if p is not None]
        n_free = len(free_indices)
        if n_free == 0:
            return None

        if axes is None:
            if figsize is None:
                figsize = (5, 3 * n_free)
            fig, axes = plt.subplots(n_free, 1, figsize=figsize, squeeze=False)
            axes = axes.ravel()

        for ax, idx in zip(axes, free_indices):
            self._priors[idx].plot(ax=ax, return_ax=True)
            ax.set_title(f'theta[{idx}]  {self._method}')

        plt.tight_layout()
        if return_axes:
            return axes
        else:
            plt.show()

    # ----- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        status = "converged" if self._result.success else "NOT converged"
        theta_str = np.array2string(
            np.asarray(self._result.theta), precision=4, separator=', ',
        )
        return (
            f"DataPrior(method={self._method}, {status}, "
            f"theta={theta_str})"
        )

class StepSizeSchedule:
    """
    Base class for step size schedules.

    Subclasses should implement __call__(iteration, particles) returning a scalar step size.
    """
    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        """
        Compute step size for given iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed)
        particles : jnp.ndarray, optional
            Current particle positions, shape (n_particles, theta_dim)

        Returns
        -------
        float
            Step size for this iteration
        """
        raise NotImplementedError

    def plot(self, nr_iter: int, figsize: tuple[float, float] | None = None, title: str | None = None, ax: matplotlib.axes.Axes | None = None, return_ax: bool = False) -> matplotlib.axes.Axes | None:
        """
        Plot the step size schedule over iterations.

        Parameters
        ----------
        nr_iter : int
            Number of iterations to plot
        figsize : tuple, default=(4, 3)
            Figure size (width, height) in inches
        title : str, optional
            Plot title. If None, uses class name
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        return_ax : bool, default=False
            If True, return the axes object. If False, call plt.show() instead.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            Axes object if return_ax=False, otherwise None

        Examples
        --------
        >>> schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=500.0)
        >>> ax = schedule.plot(nr_iter=2000)
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=None)
        # else:
        #     fig = ax.get_figure()

        # Compute schedule values
        iterations = np.arange(nr_iter)
        values = np.array([self(i) for i in iterations])

        # Plot
        # with phasic_theme():
        ax.plot(iterations, values, 'C1')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Step Size')
        ax.set_title(title or f'{self.__class__.__name__}')
        # ax.grid(True, alpha=0.3)

        # Add horizontal lines for first and last values if they exist
        if hasattr(self, 'first_step') and hasattr(self, 'last_step'):
            ax.axhline(self.first_step,
                       color=black_or_white,
                       linestyle='--', alpha=0.5,
                    label=f'first_step={self.first_step:.4g}')
            ax.axhline(self.last_step,
                       color=black_or_white,
                       linestyle='--', alpha=0.5,
                    label=f'last_step={self.last_step:.4g}')

        if ax or return_ax:
            return ax
        else:
            plt.show()


class ConstantStepSize(StepSizeSchedule):
    """
    Constant step size (default behavior).

    Parameters
    ----------
    step_size : float
        Fixed step size for all iterations

    Examples
    --------
    >>> schedule = ConstantStepSize(0.01)
    >>> schedule(0)  # iteration 0
    0.01
    >>> schedule(100)  # iteration 100
    0.01
    """
    def __init__(self, step_size: float = 0.01) -> None:
        self.step_size = step_size

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        return self.step_size


class ExpStepSize(StepSizeSchedule):
    """
    Exponential decay schedule: step_size = first_step * exp(-iteration/tau) + last_step * (1 - exp(-iteration/tau)).

    This schedule helps prevent divergence with large datasets by gradually reducing
    the step size as optimization progresses.

    Parameters
    ----------
    first_step : float, default=0.01
        Initial (first) step size at iteration 0
    last_step : float, default=1e-6
        Final (last) step size as iteration → ∞
    tau : float, default=1000.0
        Decay time constant (larger = slower decay)

    Examples
    --------
    >>> schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=500.0)
    >>> schedule(0)      # iteration 0
    0.1
    >>> schedule(500)    # iteration 500 (≈63% decay)
    0.037
    >>> schedule(5000)   # iteration 5000 (full decay)
    0.01
    """
    def __init__(self, first_step: float = 0.01, last_step: float = 1e-6, tau: float = 1000.0) -> None:
        self.first_step = first_step
        self.last_step = last_step
        self.tau = tau

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        decay = jnp.exp(-iteration / self.tau)
        return self.first_step * decay + self.last_step * (1 - decay)


class AdaptiveStepSize(StepSizeSchedule):
    """
    Adaptive step size based on particle spread (KL divergence proxy).

    Increases step size when particles are too concentrated (low KL),
    decreases when particles are too dispersed (high KL).

    Parameters
    ----------
    base_step : float, default=0.01
        Base step size
    kl_target : float, default=0.1
        Target KL divergence (in log-space particle spread)
    adjust_rate : float, default=0.1
        Rate of adjustment (0 = no adjustment, 1 = immediate)

    Examples
    --------
    >>> schedule = AdaptiveStepSize(base_step=0.01, kl_target=0.1)
    >>> particles = jnp.array([[1.0], [1.1], [0.9]])  # concentrated
    >>> schedule(10, particles)  # will increase step size
    0.011
    """
    def __init__(self, base_step: float = 0.01, kl_target: float = 0.1, adjust_rate: float = 0.1) -> None:
        self.base_step = base_step
        self.kl_target = kl_target
        self.adjust_rate = adjust_rate
        self.current_step = base_step

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        if particles is None:
            return self.current_step

        # Estimate KL divergence using particle spread
        particle_std = jnp.std(particles, axis=0)
        kl_estimate = jnp.mean(jnp.log(particle_std + 1e-8))

        # Adaptive adjustment
        if kl_estimate > self.kl_target:
            # Particles too spread out, reduce step size
            adjustment = 1.0 - self.adjust_rate
        else:
            # Particles too concentrated, increase step size
            adjustment = 1.0 + self.adjust_rate

        self.current_step = self.current_step * adjustment
        return self.current_step


class WarmupExpStepSize(StepSizeSchedule):
    """
    Linear warmup followed by exponential decay.

    Useful for Adam optimizer where initial learning rate should ramp up
    before decaying. Prevents large updates early when moment estimates
    are poorly calibrated.

    Parameters
    ----------
    peak_lr : float, default=0.001
        Maximum learning rate reached at end of warmup
    warmup_steps : int, default=100
        Number of iterations for linear warmup
    last_lr : float, default=1e-6
        Final learning rate as iteration → ∞
    tau : float, default=1000.0
        Decay time constant after warmup (larger = slower decay)

    Examples
    --------
    >>> schedule = WarmupExpStepSize(peak_lr=0.01, warmup_steps=70, last_lr=0.001, tau=500)
    >>> schedule(0)      # iteration 0: start of warmup
    0.0001
    >>> schedule(50)     # iteration 50: halfway through warmup
    0.0051
    >>> schedule(100)    # iteration 100: end of warmup, peak lr
    0.01
    >>> schedule(600)    # iteration 600: decaying after warmup
    0.0046
    """
    def __init__(self, peak_lr: float = 0.001, warmup_steps: int = 70, last_lr: float = 1e-6, tau: float = 1000.0) -> None:
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.last_lr = last_lr
        self.tau = tau
        # For plot() compatibility
        self.first_step = 0.0
        self.last_step = last_lr

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        warmup_steps = self.warmup_steps
        if iteration < warmup_steps:
            # Linear warmup from 0 to peak_lr
            return self.peak_lr * (iteration + 1) / warmup_steps
        else:
            # Exponential decay after warmup
            t = iteration - warmup_steps
            decay = jnp.exp(-t / self.tau)
            return self.peak_lr * decay + self.last_lr * (1 - decay)


# ============================================================================
# Optimizers
# ============================================================================

class Adam:
    """
    Adam optimizer for SVGD with per-parameter adaptive learning rates.

    Adam maintains running estimates of the first moment (mean) and second moment
    (uncentered variance) of gradients, using these to adaptively scale updates
    per-parameter. This is especially useful when:
    - Gradients have vastly different scales across parameters
    - Dataset size causes large gradient magnitudes
    - Optimization landscape has varying curvature

    Parameters
    ----------
    learning_rate : float or StepSizeSchedule, default=0.001
        Base learning rate (α in Adam paper). Can be a schedule (e.g., ExpStepSize,
        WarmupExpStepSize) for learning rate decay during optimization.
    beta1 : float or StepSizeSchedule, default=0.9
        Exponential decay rate for first moment estimates (momentum).
        Higher = more smoothing, slower adaptation. Can be a schedule for
        advanced warmup strategies.
    beta2 : float or StepSizeSchedule, default=0.999
        Exponential decay rate for second moment estimates (gradient variance).
        Higher = longer memory of gradient magnitudes. Can be a schedule.
    epsilon : float, default=1e-8
        Small constant for numerical stability in division.

    Attributes
    ----------
    m : array or None
        First moment estimate (shape: n_particles, theta_dim)
    v : array or None
        Second moment estimate (shape: n_particles, theta_dim)
    t : int
        Current timestep (for bias correction)

    Examples
    --------
    >>> from phasic import SVGD, Adam
    >>>
    >>> # Create optimizer with default settings
    >>> optimizer = Adam(learning_rate=0.01)
    >>>
    >>> # Use with SVGD
    >>> svgd = SVGD(
    ...     model=model,
    ...     observed_data=observations,
    ...     theta_dim=2,
    ...     optimizer=optimizer,
    ...     n_particles=50,
    ...     n_iterations=200
    ... )
    >>> svgd.fit()
    >>>
    >>> # Exponential decay learning rate
    >>> optimizer = Adam(learning_rate=ExpStepSize(first_step=0.01, last_step=0.001, tau=500))
    >>>
    >>> # Warmup + decay (recommended for large models)
    >>> optimizer = Adam(learning_rate=WarmupExpStepSize(peak_lr=0.01, warmup_steps=70))

    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    arXiv:1412.6980. https://arxiv.org/abs/1412.6980

    Notes
    -----
    When using Adam, the `learning_rate` parameter passed to SVGD is ignored
    in favor of the optimizer's learning rate.
    """

    def __init__(self, learning_rate: float | StepSizeSchedule = 0.001, beta1: float | StepSizeSchedule = 0.9, beta2: float | StepSizeSchedule = 0.999, epsilon: float = 1e-8) -> None:
        # Accept either scalar or schedule for each hyperparameter
        self.lr_schedule = self._to_schedule(learning_rate)
        self.beta1_schedule = self._to_schedule(beta1)
        self.beta2_schedule = self._to_schedule(beta2)
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Timestep

    @staticmethod
    def _to_schedule(value: float | int | StepSizeSchedule) -> StepSizeSchedule:
        """Convert scalar to ConstantStepSize, or return schedule unchanged."""
        if isinstance(value, (int, float)):
            return ConstantStepSize(float(value))
        return value

    @property
    def lr(self) -> float:
        """Current learning rate (for display/logging)."""
        return self.lr_schedule(self.t) if self.t > 0 else self.lr_schedule(0)

    @property
    def beta1(self) -> float:
        """Current beta1 value."""
        return self.beta1_schedule(self.t) if self.t > 0 else self.beta1_schedule(0)

    @property
    def beta2(self) -> float:
        """Current beta2 value."""
        return self.beta2_schedule(self.t) if self.t > 0 else self.beta2_schedule(0)

    def reset(self, shape: tuple[int, ...]) -> None:
        """
        Reset optimizer state for given particle shape.

        Called at the start of optimization to initialize moment estimates.

        Parameters
        ----------
        shape : tuple
            Shape of particles array (n_particles, theta_dim) or
            (n_particles, learnable_dim) if fixed parameters are used.
        """
        self.m = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.t = 0

    def step(self, phi: jnp.ndarray, particles: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Compute Adam update given SVGD gradient direction.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction: (K @ grad_log_p + sum(grad_K)) / n_particles
            This is the direction of steepest ascent in the RKHS.
        particles : array (n_particles, theta_dim), optional
            Current particle positions. Not used by base Adam, but available
            for subclasses (e.g., Adamelia jitter).

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles. Each element is adaptively
            scaled based on the history of gradients for that parameter.
        """
        self.t += 1

        # Get current hyperparameter values from schedules
        lr = self.lr_schedule(self.t)
        beta1 = self.beta1_schedule(self.t)
        beta2 = self.beta2_schedule(self.t)

        # Update biased first moment estimate (momentum)
        self.m = beta1 * self.m + (1 - beta1) * phi

        # Update biased second raw moment estimate (RMSprop-style)
        self.v = beta2 * self.v + (1 - beta2) * (phi ** 2)

        # Compute bias-corrected estimates
        # Early iterations have m,v biased toward zero; this corrects for that
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        # Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        # - Large gradients → large v_hat → smaller effective step
        # - Small gradients → small v_hat → larger effective step
        return lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)


class Adamelia(Adam):
    """
    Adam with Dynamic Adaptation via Monitoring Excessive Learning-rate Induced Anomalies.

    Monitors gradient direction sign flips to detect when learning rate is too high.
    When oscillation is detected, automatically reduces learning rate to stabilize
    optimization. Optionally injects jitter noise to restore variance lost during
    overshooting.

    Parameters
    ----------
    learning_rate : float or StepSizeSchedule, default=0.001
        Initial learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for first moment.
    beta2 : float, default=0.999
        Exponential decay rate for second moment.
    epsilon : float, default=1e-8
        Small constant for numerical stability.
    oscillation_threshold : float, default=0.3
        Fraction of components with sign flips to trigger oscillation detection.
        0.3 = oscillation detected when 30%+ components flip.
    patience : int, default=3
        Number of consecutive oscillation detections before reducing LR.
    lr_reduction_factor : float, default=0.5
        Factor to multiply learning rate by when reducing.
    min_lr : float, default=1e-7
        Minimum learning rate floor.
    verbose : bool, default=False
        Print messages when oscillation detected and LR reduced.
    jitter_scale : float, default=0.1
        Base multiplier for jitter magnitude when restoring variance.
        Final jitter = jitter_scale * overshoot_amplitude * particle_std * lr_multiplier.
    jitter_on_oscillation : bool, default=True
        Whether to inject jitter when LR is reduced due to oscillation.
        Helps restore variance lost during overshooting.
    seed : int, optional
        Random seed for jitter generation. Defaults to 42 for reproducibility.

    Examples
    --------
    >>> optimizer = Adamelia(learning_rate=0.01, verbose=True)
    >>> # If oscillation detected 3 times in a row, LR drops to 0.005
    >>> # and jitter is added to restore variance
    >>>
    >>> # Disable jitter if you prefer deterministic behavior
    >>> optimizer = Adamelia(learning_rate=0.01, jitter_on_oscillation=False)
    """

    def __init__(self, learning_rate: float | StepSizeSchedule = 0.3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 oscillation_threshold: float = 0.3, patience: int = 3, lr_reduction_factor: float = 0.5,
                 min_lr: float = 1e-6, verbose: bool = False, jitter_scale: float = 0.1,
                 jitter_on_oscillation: bool = True, seed: int | None = None) -> None:
        super().__init__(learning_rate, beta1, beta2, epsilon)

        self.oscillation_threshold = oscillation_threshold
        self.patience = patience
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.verbose = verbose

        # Jitter parameters
        self.jitter_scale = jitter_scale
        self.jitter_on_oscillation = jitter_on_oscillation
        self._seed = seed if seed is not None else 42
        self.rng_key = jax.random.PRNGKey(self._seed)

        # Oscillation detection state
        self.phi_prev = None
        self.oscillation_count = 0
        self.lr_reductions = 0
        self.lr_multiplier = 1.0  # Applied on top of schedule

    def reset(self, shape: tuple[int, ...]) -> None:
        """Reset optimizer state including oscillation detection and jitter RNG."""
        super().reset(shape)
        self.phi_prev = None
        self.oscillation_count = 0
        self.lr_reductions = 0
        self.lr_multiplier = 1.0
        self.rng_key = jax.random.PRNGKey(self._seed)

    def _detect_oscillation(self, phi: jnp.ndarray) -> bool:
        """Detect oscillation by checking gradient sign flips."""
        if self.phi_prev is None:
            return False

        # Count sign flips: where sign(phi) != sign(phi_prev)
        sign_flips = jnp.sum(jnp.sign(phi) * jnp.sign(self.phi_prev) < 0)
        total_components = phi.size
        flip_ratio = sign_flips / total_components

        return float(flip_ratio) > self.oscillation_threshold

    def _reduce_learning_rate(self) -> None:
        """Reduce effective learning rate by multiplier."""
        new_multiplier = self.lr_multiplier * self.lr_reduction_factor

        # Check if we'd go below min_lr
        base_lr = float(self.lr_schedule(self.t))
        if base_lr * new_multiplier < self.min_lr:
            new_multiplier = self.min_lr / base_lr

        self.lr_multiplier = new_multiplier
        self.lr_reductions += 1

        if self.verbose:
            effective_lr = base_lr * self.lr_multiplier
            logger.info(f"Adamelia Oscillation detected. "
                        f"LR reduced to {effective_lr:.2e} "
                        f"(reduction #{self.lr_reductions})")

    def _estimate_overshoot_amplitude(self, phi: jnp.ndarray) -> float:
        """
        Estimate how large the oscillation swing was.

        Measures the magnitude of gradient reversal on components that flipped sign.

        Returns
        -------
        float
            Mean magnitude of gradient reversal across flipped components.
            Returns 0.0 if no previous gradient is available.
        """
        if self.phi_prev is None:
            return 0.0

        # Identify components where sign flipped
        sign_flipped = jnp.sign(phi) * jnp.sign(self.phi_prev) < 0

        # Measure magnitude of reversal on flipped components
        reversal = jnp.abs(phi - self.phi_prev) * sign_flipped

        return float(jnp.mean(reversal))

    def _compute_jitter(self, particles: jnp.ndarray, overshoot_amplitude: float) -> jnp.ndarray:
        """
        Compute jitter scaled to particle spread, overshoot magnitude, and LR decay.

        Parameters
        ----------
        particles : array (n_particles, theta_dim)
            Current particle positions.
        overshoot_amplitude : float
            Estimated magnitude of the oscillation swing.

        Returns
        -------
        array (n_particles, theta_dim)
            Jitter to add to particles to restore variance.
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Per-dimension scaling based on current particle spread
        particle_std = jnp.std(particles, axis=0)

        # Combined scaling: decays with lr_multiplier so successive jitters are smaller
        jitter_magnitude = (
            self.jitter_scale
            * overshoot_amplitude
            * particle_std
            * self.lr_multiplier  # Decay factor
        )

        return jax.random.normal(subkey, particles.shape) * jitter_magnitude

    def step(self, phi: jnp.ndarray, particles: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Compute Adam update with oscillation detection and optional jitter.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction.
        particles : array (n_particles, theta_dim), optional
            Current particle positions. Required for jitter computation.

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles, possibly including jitter.
        """
        self.t += 1

        # Check for oscillation
        is_oscillating = self._detect_oscillation(phi)

        jitter = None
        if is_oscillating:
            self.oscillation_count += 1
            if self.oscillation_count >= self.patience:
                # Compute overshoot amplitude before reducing LR
                overshoot_amplitude = self._estimate_overshoot_amplitude(phi)

                self._reduce_learning_rate()
                self.oscillation_count = 0  # Reset counter after reduction

                # Add jitter to restore variance (scaled by lr_multiplier so it decays)
                if self.jitter_on_oscillation and particles is not None:
                    jitter = self._compute_jitter(particles, overshoot_amplitude)
                    if self.verbose:
                        jitter_mag = float(jnp.mean(jnp.abs(jitter)))
                        logger.info(f"Adamelia Jitter applied: mean magnitude {jitter_mag:.2e}")
        else:
            # Decay oscillation count (don't reset immediately)
            self.oscillation_count = max(0, self.oscillation_count - 1)

        # Store current phi for next iteration
        self.phi_prev = phi

        # Get current hyperparameter values from schedules
        lr = self.lr_schedule(self.t) * self.lr_multiplier
        beta1 = self.beta1_schedule(self.t)
        beta2 = self.beta2_schedule(self.t)

        # Standard Adam update
        self.m = beta1 * self.m + (1 - beta1) * phi
        self.v = beta2 * self.v + (1 - beta2) * (phi ** 2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        update = lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

        if jitter is not None:
            return update + jitter
        return update

    @property
    def lr(self) -> float:
        """Current effective learning rate (including reductions)."""
        base_lr = self.lr_schedule(self.t) if self.t > 0 else self.lr_schedule(0)
        return float(base_lr) * self.lr_multiplier


class SGDMomentum:
    """
    SGD with momentum optimizer for SVGD.

    Momentum helps accelerate gradients in the right direction and dampens
    oscillations. It accumulates a velocity vector in directions of persistent
    gradient descent.

    Parameters
    ----------
    learning_rate : float or StepSizeSchedule, default=0.01
        Step size for parameter updates. Can be a schedule for learning rate decay.
    momentum : float or StepSizeSchedule, default=0.9
        Momentum coefficient. Higher values give more weight to past gradients.
        Typical values: 0.9 (standard), 0.99 (high momentum). Can be a schedule.
    max_velocity : float or None, default=1.0
        Maximum absolute velocity to prevent unbounded accumulation.
        Set to None to disable velocity clipping (not recommended with
        positive_params=True as it can cause numerical issues).

    Attributes
    ----------
    v : array or None
        Velocity (accumulated gradient), shape (n_particles, theta_dim)

    Examples
    --------
    >>> from phasic import SVGD, SGDMomentum
    >>>
    >>> optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
    >>> svgd = SVGD(
    ...     model=model,
    ...     observed_data=observations,
    ...     theta_dim=2,
    ...     optimizer=optimizer
    ... )

    Notes
    -----
    Update rule: v = momentum * v + lr * gradient; params += v

    Velocity is clipped to [-max_velocity, max_velocity] to prevent unbounded
    growth, which can cause numerical issues when using positive_params=True.
    """

    def __init__(self, learning_rate: float | StepSizeSchedule = 0.01, momentum: float | StepSizeSchedule = 0.9, max_velocity: float | None = 1.0) -> None:
        # Accept either scalar or schedule for each hyperparameter
        self.lr_schedule = self._to_schedule(learning_rate)
        self.momentum_schedule = self._to_schedule(momentum)
        self.max_velocity = max_velocity
        self.v = None
        self.t = 0  # Timestep for schedules

    @staticmethod
    def _to_schedule(value: float | int | StepSizeSchedule) -> StepSizeSchedule:
        """Convert scalar to ConstantStepSize, or return schedule unchanged."""
        if isinstance(value, (int, float)):
            return ConstantStepSize(float(value))
        return value

    @property
    def lr(self) -> float:
        """Current learning rate (for display/logging)."""
        return self.lr_schedule(self.t) if self.t > 0 else self.lr_schedule(0)

    @property
    def momentum(self) -> float:
        """Current momentum value."""
        return self.momentum_schedule(self.t) if self.t > 0 else self.momentum_schedule(0)

    def reset(self, shape: tuple[int, ...]) -> None:
        """Reset optimizer state for given particle shape."""
        self.v = jnp.zeros(shape)
        self.t = 0

    def step(self, phi: jnp.ndarray, particles: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Compute SGD with momentum update.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction.
        particles : array (n_particles, theta_dim), optional
            Current particle positions. Not used by SGDMomentum.

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles.
        """
        del particles  # Unused
        self.t += 1

        # Get current hyperparameter values from schedules
        lr = self.lr_schedule(self.t)
        momentum = self.momentum_schedule(self.t)

        self.v = momentum * self.v + lr * phi
        # Clip velocity to prevent unbounded accumulation
        if self.max_velocity is not None:
            self.v = jnp.clip(self.v, -self.max_velocity, self.max_velocity)
        return self.v


class RMSprop:
    """
    RMSprop optimizer for SVGD.

    RMSprop adapts the learning rate for each parameter by dividing by an
    exponentially decaying average of squared gradients. This helps with
    non-stationary objectives and noisy gradients.

    Parameters
    ----------
    learning_rate : float or StepSizeSchedule, default=0.001
        Base learning rate. Can be a schedule for learning rate decay.
    decay : float or StepSizeSchedule, default=0.99
        Decay rate for the moving average of squared gradients.
        Higher values give longer memory. Can be a schedule.
    epsilon : float, default=1e-8
        Small constant for numerical stability.

    Attributes
    ----------
    v : array or None
        Moving average of squared gradients, shape (n_particles, theta_dim)

    Examples
    --------
    >>> from phasic import SVGD, RMSprop
    >>>
    >>> optimizer = RMSprop(learning_rate=0.001, decay=0.99)
    >>> svgd = SVGD(
    ...     model=model,
    ...     observed_data=observations,
    ...     theta_dim=2,
    ...     optimizer=optimizer
    ... )

    References
    ----------
    Hinton, G. (2012). Lecture 6.5 - RMSprop. Coursera: Neural Networks for
    Machine Learning.

    Notes
    -----
    Update rule: v = decay * v + (1 - decay) * gradient²; params += lr * gradient / (√v + ε)
    """

    def __init__(self, learning_rate: float | StepSizeSchedule = 0.001, decay: float | StepSizeSchedule = 0.99, epsilon: float = 1e-8) -> None:
        # Accept either scalar or schedule for each hyperparameter
        self.lr_schedule = self._to_schedule(learning_rate)
        self.decay_schedule = self._to_schedule(decay)
        self.epsilon = epsilon
        self.v = None
        self.t = 0  # Timestep for schedules

    @staticmethod
    def _to_schedule(value: float | int | StepSizeSchedule) -> StepSizeSchedule:
        """Convert scalar to ConstantStepSize, or return schedule unchanged."""
        if isinstance(value, (int, float)):
            return ConstantStepSize(float(value))
        return value

    @property
    def lr(self) -> float:
        """Current learning rate (for display/logging)."""
        return self.lr_schedule(self.t) if self.t > 0 else self.lr_schedule(0)

    @property
    def decay(self) -> float:
        """Current decay value."""
        return self.decay_schedule(self.t) if self.t > 0 else self.decay_schedule(0)

    def reset(self, shape: tuple[int, ...]) -> None:
        """Reset optimizer state for given particle shape."""
        self.v = jnp.zeros(shape)
        self.t = 0

    def step(self, phi: jnp.ndarray, particles: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Compute RMSprop update.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction.
        particles : array (n_particles, theta_dim), optional
            Current particle positions. Not used by RMSprop.

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles.
        """
        del particles  # Unused
        self.t += 1

        # Get current hyperparameter values from schedules
        lr = self.lr_schedule(self.t)
        decay = self.decay_schedule(self.t)

        self.v = decay * self.v + (1 - decay) * (phi ** 2)
        return lr * phi / (jnp.sqrt(self.v) + self.epsilon)


class Adagrad:
    """
    Adagrad optimizer for SVGD.

    Adagrad adapts the learning rate for each parameter based on the
    accumulated sum of squared gradients. Parameters with large gradients
    get smaller learning rates, and parameters with small gradients get
    larger learning rates.

    Parameters
    ----------
    learning_rate : float or StepSizeSchedule, default=0.01
        Base learning rate. Can be a schedule for learning rate decay.
    epsilon : float, default=1e-8
        Small constant for numerical stability.

    Attributes
    ----------
    G : array or None
        Accumulated sum of squared gradients, shape (n_particles, theta_dim)

    Examples
    --------
    >>> from phasic import SVGD, Adagrad
    >>>
    >>> optimizer = Adagrad(learning_rate=0.01)
    >>> svgd = SVGD(
    ...     model=model,
    ...     observed_data=observations,
    ...     theta_dim=2,
    ...     optimizer=optimizer
    ... )

    References
    ----------
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods
    for Online Learning and Stochastic Optimization. JMLR 12:2121-2159.

    Notes
    -----
    Update rule: G += gradient²; params += lr * gradient / (√G + ε)

    Warning: The learning rate decays over time as G accumulates. For long
    runs, consider using RMSprop or Adam which have bounded effective learning rates.
    """

    def __init__(self, learning_rate: float | StepSizeSchedule = 0.01, epsilon: float = 1e-8) -> None:
        # Accept either scalar or schedule for learning rate
        self.lr_schedule = self._to_schedule(learning_rate)
        self.epsilon = epsilon
        self.G = None
        self.t = 0  # Timestep for schedules

    @staticmethod
    def _to_schedule(value: float | int | StepSizeSchedule) -> StepSizeSchedule:
        """Convert scalar to ConstantStepSize, or return schedule unchanged."""
        if isinstance(value, (int, float)):
            return ConstantStepSize(float(value))
        return value

    @property
    def lr(self) -> float:
        """Current learning rate (for display/logging)."""
        return self.lr_schedule(self.t) if self.t > 0 else self.lr_schedule(0)

    def reset(self, shape: tuple[int, ...]) -> None:
        """Reset optimizer state for given particle shape."""
        self.G = jnp.zeros(shape)
        self.t = 0

    def step(self, phi: jnp.ndarray, particles: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Compute Adagrad update.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction.
        particles : array (n_particles, theta_dim), optional
            Current particle positions. Not used by Adagrad.

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles.
        """
        del particles  # Unused
        self.t += 1

        # Get current learning rate from schedule
        lr = self.lr_schedule(self.t)

        self.G = self.G + phi ** 2
        return lr * phi / (jnp.sqrt(self.G) + self.epsilon)


# ============================================================================
# Regularization Schedule Classes
# ============================================================================

class RegularizationSchedule:
    """
    Base class for regularization schedules.

    Subclasses should implement __call__(iteration, particles) returning a scalar regularization value.
    """
    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        """
        Compute regularization strength for given iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed)
        particles : jnp.ndarray, optional
            Current particle positions, shape (n_particles, theta_dim)

        Returns
        -------
        float
            Regularization strength for this iteration
        """
        raise NotImplementedError

    def plot(self, nr_iter: int, figsize: tuple[float, float] | None = None, title: str | None = None, ax: matplotlib.axes.Axes | None = None, return_ax: bool = False) -> matplotlib.axes.Axes | None:
        """
        Plot the regularization schedule over iterations.

        Parameters
        ----------
        nr_iter : int
            Number of iterations to plot
        figsize : tuple, default=(4, 3)
            Figure size (width, height) in inches
        title : str, optional
            Plot title. If None, uses class name
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        return_ax : bool, default=False
            If True, return the axes object. If False, call plt.show() instead.

        Returns
        -------
        ax : matplotlib.axes.Axes or None
            Axes object if return_ax=False, otherwise None

        Examples
        --------
        >>> schedule = ExpRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
        >>> ax = schedule.plot(nr_iter=2000)
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=None)
        else:
            fig = ax.get_figure()

        # Compute schedule values
        iterations = np.arange(nr_iter)
        values = np.array([self(i) for i in iterations])

        # Plot
        # with phasic_theme():
        ax.plot(iterations, values, 'C2')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Regularization Strength')
        ax.set_title(title or f'{self.__class__.__name__}')
        # ax.grid(True, alpha=0.3)

        # Add horizontal lines for first and last values if they exist
        if hasattr(self, 'first_reg') and hasattr(self, 'last_reg'):
            ax.axhline(self.first_reg,
                       color=black_or_white,
                       linestyle='--', alpha=0.5,
                    label=f'first_reg={self.first_reg:.4g}')
            ax.axhline(self.last_reg,
                       color=black_or_white,
                       linestyle='--', alpha=0.5,
                    label=f'last_reg={self.last_reg:.4g}')

        if return_ax:
            return ax
        else:
            plt.show()


class ConstantRegularization(RegularizationSchedule):
    """
    Constant regularization (default behavior).

    Parameters
    ----------
    regularization : float, default=0.0
        Fixed regularization strength for all iterations

    Examples
    --------
    >>> schedule = ConstantRegularization(1.0)
    >>> schedule(0)  # iteration 0
    1.0
    >>> schedule(100)  # iteration 100
    1.0
    """
    def __init__(self, regularization: float = 0.0) -> None:
        self.regularization = regularization

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        return self.regularization


class ExpRegularization(RegularizationSchedule):
    """
    Exponential decay schedule: reg = first_reg * exp(-iteration/tau) + last_reg * (1 - exp(-iteration/tau)).

    This schedule helps by starting with strong moment regularization to guide
    initial exploration, then gradually reducing regularization as optimization
    converges to allow fine-tuning.

    Parameters
    ----------
    first_reg : float, default=1.0
        Initial (first) regularization strength at iteration 0
    last_reg : float, default=0.0
        Final (last) regularization strength as iteration → ∞
    tau : float, default=1000.0
        Decay time constant (larger = slower decay)

    Examples
    --------
    >>> schedule = ExpRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> schedule(0)      # iteration 0
    5.0
    >>> schedule(500)    # iteration 500 (≈63% decay)
    0.1925
    >>> schedule(5000)   # iteration 5000 (full decay)
    0.1
    """
    def __init__(self, first_reg: float = 1.0, last_reg: float = 0.0, tau: float = 1000.0) -> None:
        self.first_reg = first_reg
        self.last_reg = last_reg
        self.tau = tau

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        decay = jnp.exp(-iteration / self.tau)
        return self.first_reg * decay + self.last_reg * (1 - decay)


class ExponentialCDFRegularization(RegularizationSchedule):
    """
    Exponential CDF schedule: reg = first_reg + (last_reg - first_reg) * (1 - exp(-iteration/tau)).

    This schedule uses the exponential cumulative distribution function (CDF) to create
    a smooth S-curve transition between first_reg and last_reg. Unlike exponential decay,
    this is bidirectional and works naturally for both increasing and decreasing schedules.

    The CDF approach provides:
    - Smooth, continuous transitions
    - Fast initial change that gradually slows
    - Natural interpretation: tau is the "characteristic time" (63% transition at tau)
    - Works equally well for increasing or decreasing regularization

    Parameters
    ----------
    first_reg : float, default=0.0
        Initial (first) regularization strength at iteration 0
    last_reg : float, default=1.0
        Final (last) regularization strength as iteration → ∞
    tau : float, default=1000.0
        Transition time constant (larger = slower transition)

    Examples
    --------
    >>> # Increasing regularization (useful for progressive regularization)
    >>> schedule = ExponentialCDFRegularization(first_reg=0.0, last_reg=1.0, tau=500.0)
    >>> schedule(0)      # iteration 0
    0.0
    >>> schedule(500)    # iteration 500 (≈63% transition)
    0.632
    >>> schedule(5000)   # iteration 5000 (nearly complete)
    0.993

    >>> # Decreasing regularization (similar to exponential decay)
    >>> schedule = ExponentialCDFRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> schedule(0)      # iteration 0
    5.0
    >>> schedule(500)    # iteration 500 (≈63% transition)
    1.9
    >>> schedule(5000)   # iteration 5000 (nearly complete)
    0.1
    """
    def __init__(self, first_reg: float = 0.0, last_reg: float = 1.0, tau: float = 1000.0) -> None:
        self.first_reg = first_reg
        self.last_reg = last_reg
        self.tau = tau

    def __call__(self, iteration: int, particles: jnp.ndarray | None = None) -> float:
        cdf = 1.0 - jnp.exp(-iteration / self.tau)
        return self.first_reg + (self.last_reg - self.first_reg) * cdf


# class BandwidthSchedule:
#     """
#     Base class for bandwidth schedules.

#     Subclasses should implement __call__(particles) returning bandwidth(s).
#     """
#     def __call__(self, particles):
#         """
#         Compute bandwidth for current particle configuration.

#         Parameters
#         ----------
#         particles : jnp.ndarray
#             Current particle positions, shape (n_particles, theta_dim)

#         Returns
#         -------
#         float or jnp.ndarray
#             Bandwidth (scalar for global, array for local)
#         """
#         raise NotImplementedError


# class MedianBandwidth(BandwidthSchedule):
#     """
#     Median heuristic bandwidth (default behavior).

#     Sets bandwidth to median of pairwise distances between particles.

#     Examples
#     --------
#     >>> schedule = MedianBandwidth()
#     >>> particles = jnp.array([[0.0], [1.0], [2.0]])
#     >>> schedule(particles)
#     1.0
#     """
#     def __call__(self, particles):
#         n_particles = particles.shape[0]
#         pairwise_dists = jnp.array([
#             jnp.linalg.norm(particles[i] - particles[j])
#             for i in range(n_particles)
#             for j in range(i + 1, n_particles)
#         ])
#         return jnp.median(pairwise_dists)


# class FixedBandwidth(BandwidthSchedule):
#     """
#     Fixed bandwidth for all iterations.

#     Parameters
#     ----------
#     bandwidth : float
#         Fixed bandwidth value

#     Examples
#     --------
#     >>> schedule = FixedBandwidth(1.0)
#     >>> particles = jnp.array([[0.0], [1.0]])
#     >>> schedule(particles)
#     1.0
#     """
#     def __init__(self, bandwidth=1.0):
#         self.bandwidth = bandwidth

#     def __call__(self, particles):
#         return self.bandwidth


# class LocalAdaptiveBandwidth(BandwidthSchedule):
#     """
#     Local adaptive bandwidth using k-nearest neighbors.

#     Computes per-particle bandwidth based on distance to k-nearest neighbors.

#     Parameters
#     ----------
#     alpha : float, default=0.9
#         Scaling factor for local bandwidth
#     k_frac : float, default=0.1
#         Fraction of particles to use as k-nearest neighbors

#     Examples
#     --------
#     >>> schedule = LocalAdaptiveBandwidth(alpha=0.9, k_frac=0.1)
#     >>> particles = jnp.array([[0.0], [1.0], [10.0]])
#     >>> bandwidths = schedule(particles)
#     >>> bandwidths.shape
#     (3,)
#     """
#     def __init__(self, alpha=0.9, k_frac=0.1):
#         self.alpha = alpha
#         self.k_frac = k_frac

#     def __call__(self, particles):
#         n_particles = particles.shape[0]
#         k_nn = max(1, int(n_particles * self.k_frac))

#         bandwidths = []
#         for i in range(n_particles):
#             # Compute distances to all other particles
#             distances = jnp.array([
#                 jnp.linalg.norm(particles[i] - particles[j])
#                 for j in range(n_particles) if j != i
#             ])
#             # Take k-nearest neighbors
#             knn_distances = jnp.sort(distances)[:k_nn]
#             local_bw = jnp.mean(knn_distances) * self.alpha
#             bandwidths.append(local_bw)

#         return jnp.array(bandwidths)


# ============================================================================
# End of Schedule Classes
# ============================================================================


# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2g},{b:.2g})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap
# # "_iridis" color map (viridis without the deep purple):
# _iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)


# @jit
# def calculate_param_dim(k, m):
#     """Calculate parameter dimension for discrete phase-type distribution
    
#     Parameters:
#     - k: number of dimensions (absorption states)  
#     - m: number of transient states
    
#     Returns:
#     - Total parameter dimension
#     """
#     # Initial distribution: m parameters (no constraint)
#     alpha_dim = m
    
#     # Sub-intensity matrix: m×m parameters with row-sum constraints
#     # Each row sums to <= 0, so m-1 free parameters per row
#     sub_Q_dim = m * (m - 1) 
    
#     # Exit rates: k×m parameters (all free)
#     exit_rates_dim = k * m
    
#     return alpha_dim + sub_Q_dim + exit_rates_dim

# def example_ptd_spec(key, k=1, m=2):
#     """Generate example discrete phase-type distribution parameters
    
#     Returns flattened parameter vector for the distribution with:
#     - k absorption states (dimensions)
#     - m transient states
#     """
#     # Generate initial distribution (normalized)
#     key, subkey = jax.random.split(key)
#     alpha_raw = jax.random.exponential(subkey, shape=(m,))
#     alpha = alpha_raw / jnp.sum(alpha_raw)
    
#     # Generate sub-intensity matrix Q (m×m)
#     key, subkey = jax.random.split(key)
#     # Off-diagonal elements (positive, will be made negative)
#     off_diag = jax.random.exponential(subkey, shape=(m, m))
#     off_diag = off_diag.at[jnp.diag_indices(m)].set(0)  # Zero diagonal
    
#     # Make off-diagonal negative and set diagonal to ensure row sums < 0
#     Q = -off_diag
#     row_sums = jnp.sum(Q, axis=1)
#     Q = Q.at[jnp.diag_indices(m)].set(-jnp.abs(row_sums) - 0.1)  # Ensure diagonal < row sum
    
#     # Generate exit rates (k×m, all positive)
#     key, subkey = jax.random.split(key)
#     exit_rates = jax.random.exponential(subkey, shape=(k, m))
    
#     # Flatten into parameter vector
#     # Structure: [alpha (m), Q off-diagonal (m*(m-1)), exit_rates (k*m)]
#     q_off_diag = jnp.concatenate([Q[i, :i].flatten() for i in range(m)] + 
#                                  [Q[i, i+1:].flatten() for i in range(m)])
    
#     params = jnp.concatenate([alpha, q_off_diag, exit_rates.flatten()])
#     return params

def unpack_theta(params: jnp.ndarray, k: int, m: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Unpack flattened parameter vector into components using JAX operations.

    Parameters
    ----------
    params : jnp.ndarray
        Flattened parameter vector.
    k : int
        Number of absorption states (dimensions).
    m : int
        Number of transient states.

    Returns
    -------
    alpha : jnp.ndarray
        Initial distribution vector of length ``m``.
    Q : jnp.ndarray
        Sub-intensity matrix of shape ``(m, m)``.
    exit_rates : jnp.ndarray
        Exit rate matrix of shape ``(k, m)``.
    """
    # Calculate dimensions
    alpha_dim = m
    sub_Q_dim = m * (m - 1)
    
    # Extract components using standard slicing (will be handled by JAX)
    alpha = params[:alpha_dim]
    q_off_diag = params[alpha_dim:alpha_dim + sub_Q_dim]
    exit_rates_flat = params[alpha_dim + sub_Q_dim:alpha_dim + sub_Q_dim + k * m]
    
    # Reconstruct Q matrix - simplified approach for any m
    Q = jnp.zeros((m, m))
    
    # For general case, use a more systematic approach
    # Fill off-diagonal elements in order
    idx = 0
    for i in range(m):
        for j in range(m):
            if i != j:  # Skip diagonal
                Q = Q.at[i, j].set(q_off_diag[idx])
                idx += 1
    
    # Set diagonal elements to ensure valid sub-intensity matrix
    row_sums = jnp.sum(Q, axis=1)
    Q = Q.at[jnp.diag_indices(m)].set(-jnp.abs(row_sums) - 0.1)
    
    # Reshape exit rates
    exit_rates = exit_rates_flat.reshape(k, m)
    
    return alpha, Q, exit_rates

# def simulate_example_data(key, params, k, m, n_samples):
#     """Simulate data from discrete phase-type distribution"""
#     alpha, Q, exit_rates = unpack_theta(params, k, m)
    
#     # Simple simulation - generate random absorption times
#     # This is a placeholder - real DPH simulation would be more complex
#     key, subkey = jax.random.split(key)
    
#     # Generate samples using approximation
#     # Sample from geometric distributions and combine
#     samples = []
#     for _ in range(n_samples):
#         key, subkey = jax.random.split(key)
#         # Simple approximation: sample absorption times
#         absorption_times = jax.random.geometric(subkey, 0.3, shape=(k,))
#         samples.append(absorption_times)
    
#     return jnp.array(samples)

def log_pmf_dph(x: jnp.ndarray, params: jnp.ndarray, k: int, m: int) -> float:
    """Log probability mass function for discrete phase-type distribution."""
    alpha, Q, exit_rates = unpack_theta(params, k, m)
    
    # Simple approximation for discrete phase-type log-pmf
    # Real implementation would involve matrix exponentials
    
    # Ensure x is properly shaped
    x = jnp.atleast_1d(x)
    if x.shape[0] != k:
        # Pad or truncate to match k dimensions
        if x.shape[0] < k:
            x = jnp.concatenate([x, jnp.ones(k - x.shape[0])])
        else:
            x = x[:k]
    
    # Approximate log-pmf using geometric distribution mixture
    log_prob = 0.0
    for i in range(k):
        for j in range(m):
            rate = jnp.abs(exit_rates[i, j])
            # Geometric log-pmf approximation
            p = rate / (1.0 + rate)
            log_prob += jnp.log(p) + (x[i] - 1) * jnp.log(1 - p)
    
    # Add initial distribution contribution
    log_prob += jnp.sum(jnp.log(alpha + 1e-8))
    
    return log_prob

# Simpler approach: direct parameter mapping
@jit
def z_to_theta(z: jnp.ndarray) -> jnp.ndarray:
    """Convert latent variable to parameter space."""
    return z  # Direct mapping for simplicity

# SVGD functions
@jit
def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, bandwidth: float) -> float:
    """RBF kernel function."""
    diff = x - y
    return jnp.exp(-jnp.sum(diff**2) / (2 * bandwidth**2))

# @jit
# def median_heuristic(particles):
#     """Median heuristic for bandwidth selection"""
#     n_particles = particles.shape[0]
#     distances = []
#     for i in range(n_particles):
#         for j in range(i+1, n_particles):
#             dist = jnp.linalg.norm(particles[i] - particles[j])
#             distances.append(dist)
#     distances = jnp.array(distances)
#     median_dist = jnp.median(distances)
#     return median_dist / jnp.log(n_particles + 1)

@jit 
def batch_median_heuristic(particles: jnp.ndarray) -> float:
    """Vectorized median heuristic for bandwidth selection."""
    n_particles = particles.shape[0]
    # Compute pairwise distances
    diff = particles[:, None, :] - particles[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2)
    # Get upper triangular part (excluding diagonal)
    triu_indices = jnp.triu_indices(n_particles, k=1)
    pairwise_dists = distances[triu_indices]
    median_dist = jnp.median(pairwise_dists)
    return median_dist / jnp.log(n_particles + 1)

@jit
def batch_median_heuristic_per_dim(particles: jnp.ndarray) -> jnp.ndarray:
    """Per-dimension median heuristic for anisotropic RBF kernel.

    Computes a separate bandwidth for each parameter dimension using
    h_d^2 = median((xi_d - xj_d)^2) / log(n+1), which gives a bandwidth
    scaled consistently with the isotropic median heuristic.

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions

    Returns
    -------
    bandwidth : array (theta_dim,)
        Per-dimension bandwidth values
    """
    n_particles = particles.shape[0]
    # Pairwise differences per dimension: (n_particles, n_particles, theta_dim)
    diff = particles[:, None, :] - particles[None, :, :]
    sq_diff = diff**2
    # Upper triangular indices (excluding diagonal)
    triu_indices = jnp.triu_indices(n_particles, k=1)
    # Extract upper triangle: (n_pairs, theta_dim)
    pairwise_sq = sq_diff[triu_indices]
    # Median of squared differences: (theta_dim,)
    median_sq_per_dim = jnp.median(pairwise_sq, axis=0)
    # h_d^2 = median_sq_d / log(n+1), so h_d = sqrt(median_sq_d / log(n+1))
    h_sq = median_sq_per_dim / jnp.log(n_particles + 1)
    # Clamp to avoid degenerate dimensions
    return jnp.maximum(jnp.sqrt(h_sq), 1e-8)

@jit
def rbf_kernel_median(particles: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RBF kernel with median heuristic bandwidth."""
    bandwidth = batch_median_heuristic(particles)
    n_particles = particles.shape[0]
    
    # Compute kernel matrix
    K = jnp.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            K = K.at[i, j].set(rbf_kernel(particles[i], particles[j], bandwidth))
    
    # Compute gradients
    grad_K = jnp.zeros((n_particles, n_particles, particles.shape[1]))
    for i in range(n_particles):
        for j in range(n_particles):
            diff = particles[i] - particles[j]
            grad_K = grad_K.at[i, j].set(-K[i, j] * diff / bandwidth**2)
    
    return K, grad_K

# Define log probability functions
@jit
def logp(theta: jnp.ndarray, data: jnp.ndarray, k: int, m: int) -> float:
    """Log probability of data given parameters."""
    return jnp.sum(vmap(lambda x: log_pmf_dph(x, theta, k, m))(data))

# @jit  
# def logp_z(z, k, m):
#     """Log probability function for latent variables"""
#     theta = z_to_theta(z)
#     # Add prior (standard normal on z)
#     log_prior = -0.5 * jnp.sum(z**2)
#     return log_prior

# # Adaptive step size functions
# @jit
# def decayed_kl_target(iteration, base=0.1, decay=0.01):
#     """Exponentially decaying KL target"""
#     return base * jnp.exp(-decay * iteration)

# @jit  
# def step_size_schedule(iteration, max_step=0.001, min_step=1e-6):
#     """Step size schedule"""
#     decay = jnp.exp(-iteration / 1000.0)
#     return max_step * decay + min_step * (1 - decay)

@jit
def local_adaptive_bandwidth(particles: jnp.ndarray, alpha: float = 0.9) -> jnp.ndarray:
    """Local adaptive bandwidth selection."""
    n_particles = particles.shape[0]
    # Use k-nearest neighbors approach
    k_nn = max(1, n_particles // 10)
    
    bandwidths = []
    for i in range(n_particles):
        # Compute distances to all other particles
        distances = jnp.array([jnp.linalg.norm(particles[i] - particles[j]) 
                              for j in range(n_particles) if j != i])
        # Take k-nearest neighbors
        knn_distances = jnp.sort(distances)[:k_nn]
        local_bw = jnp.mean(knn_distances) * alpha
        bandwidths.append(local_bw)
    
    return jnp.array(bandwidths)

@jit
def kl_adaptive_step(particles: jnp.ndarray, kl_target: float = 0.1) -> float:
    """Adaptive step size based on KL divergence estimate."""
    # Estimate KL divergence using particle approximation
    n_particles = particles.shape[0]
    
    # Simple KL estimate based on particle spread
    particle_std = jnp.std(particles, axis=0)
    kl_estimate = jnp.mean(jnp.log(particle_std + 1e-8))
    
    # Adaptive step using JAX conditional
    step_factor = jnp.where(kl_estimate > kl_target, 0.9, 1.1)
    
    return step_factor

# # SVGD update functions
# def svgd_update_z(particles_z, data, k, m, step_size=0.001, kl_target=0.1):
#     """SVGD update for latent variables"""
#     n_particles = particles_z.shape[0]
    
#     # Convert to parameter space for likelihood evaluation
#     particles_theta = jnp.array([z_to_theta(z) for z in particles_z])
    
#     # Compute log probability gradients
#     def logp_single(theta):
#         return logp(theta, data, k, m)
    
#     grad_logp = vmap(grad(logp_single))(particles_theta)
    
#     # Compute kernels
#     K, grad_K = rbf_kernel_median(particles_z)
    
#     # SVGD update
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         # Positive term: weighted gradient
#         positive_term = jnp.sum(K[i, :, None] * grad_logp, axis=0) / n_particles
        
#         # Negative term: kernel gradient
#         negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
        
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step size
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     adaptive_step = step_size * step_factor
    
#     return particles_z + adaptive_step * phi

# # More sophisticated SVGD updates
# @jit
# def update_median_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """SVGD update with median bandwidth and KL-adaptive step"""
#     n_particles = particles_z.shape[0]
    
#     # Gradients in latent space (prior only for now)
#     grad_logp_z = -particles_z  # Gradient of standard normal prior
    
#     # Compute kernel and its gradients
#     K, grad_K = rbf_kernel_median(particles_z)
    
#     # SVGD update
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         positive_term = jnp.sum(K[i, :, None] * grad_logp_z, axis=0) / n_particles
#         negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
#     return particles_z + step_size * phi

# @jit
# def update_local_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """SVGD update with local bandwidth and KL-adaptive step"""
#     n_particles = particles_z.shape[0]
    
#     # Get local bandwidths
#     local_bws = local_adaptive_bandwidth(particles_z)
    
#     # Gradients  
#     grad_logp_z = -particles_z
    
#     # Compute updates with local bandwidths
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         # Local kernel computations
#         local_K = jnp.array([rbf_kernel(particles_z[i], particles_z[j], local_bws[i]) 
#                             for j in range(n_particles)])
        
#         # Local kernel gradients
#         local_grad_K = jnp.array([
#             -local_K[j] * (particles_z[i] - particles_z[j]) / (local_bws[i]**2)
#             for j in range(n_particles)
#         ])
        
#         positive_term = jnp.sum(local_K[:, None] * grad_logp_z, axis=0) / n_particles
#         negative_term = jnp.sum(local_grad_K, axis=0) / n_particles
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
#     return particles_z + step_size * phi

# # Distributed SVGD
# def distributed_svgd_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """Distributed SVGD step using pjit"""
#     return update_median_bw_kl_step(particles_z, k, m, kl_target, max_step)

# # Main SVGD function
# def run_variable_dim_svgd(key, data, k, m, n_particles=40, n_steps=70, lr=0.001):
#     """Run SVGD for variable-dimension discrete phase-type distributions"""
    
#     # Calculate parameter dimension
#     param_dim = calculate_param_dim(k, m)
#     print(f"Running SVGD for k={k}, m={m} (param_dim={param_dim})")
    
#     # Generate true parameters
#     key, subkey = jax.random.split(key)
#     true_params = example_ptd_spec(subkey, k, m)
    
#     # SVGD parameters
#     n_devices = min(8, n_particles)  # Don't exceed available devices
#     kl_target_base = 0.1
#     kl_target_decay = 0.01
#     max_step = lr
#     min_step = 1e-7
#     max_step_scaler = 0.1
    
#     if n_particles % n_devices != 0:
#         n_particles = (n_particles // n_devices) * n_devices
#         print(f"Adjusted n_particles to {n_particles} for even sharding")
    
#     # Initial particles
#     key, subkey = jax.random.split(key)
#     particles_z = jax.random.normal(subkey, shape=(n_particles, param_dim))
    
#     # Shard particles over devices
#     devices = mesh_utils.create_device_mesh((n_devices,))
#     mesh = Mesh(devices, axis_names=("i",))
#     sharding = NamedSharding(mesh, P("i", None))
#     particles_z = jax.device_put(particles_z, sharding)
    
#     # SVGD iterations
#     particle_z_history = [particles_z]
#     every = max(1, n_steps // 10)  # Save every 10% of iterations
#     prev = None
    
#     with mesh:
#         # for i in range(n_steps):
#         for i in trange(n_steps):
#             kl_target = decayed_kl_target(i, base=kl_target_base, decay=kl_target_decay)
#             particles_z = distributed_svgd_step(particles_z, k, m, kl_target=kl_target, max_step=max_step)
            
#             if not i % every:
#                 particle_z_history.append(particles_z)
    
#     # Extract final results
#     particles = jnp.array([z_to_theta(z) for z in particles_z])
    
#     print(f"\nResults for k={k}, m={m}:")
#     print(f"True parameters shape: {true_params.shape}")
#     print(f"Estimated parameters shape: {particles.shape}")
#     print(f"Parameter means: {jnp.mean(particles, axis=0)}")
#     print(f"True parameters: {true_params}")
    
#     return particles, particle_z_history, true_params

# ==============================================================================
# Main SVGD API for external use
# ==============================================================================

@jit
def _compute_kernel_grad_impl(particles: jnp.ndarray, bandwidth: float | jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled RBF kernel computation (core implementation)

    Supports both isotropic (scalar bandwidth) and anisotropic
    (per-dimension bandwidth vector) kernels.

    For anisotropic kernel:
        K(x,y) = exp(-sum_d ((x_d - y_d)^2 / (2 * h_d^2)))
        dK/dx_d = -K * (x_d - y_d) / h_d^2

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    bandwidth : float or array (theta_dim,)
        Kernel bandwidth. Scalar for isotropic, vector for anisotropic.

    Returns
    -------
    K : array (n_particles, n_particles)
        Kernel matrix
    grad_K : array (n_particles, n_particles, theta_dim)
        Gradient of kernel matrix
    """
    # Vectorized computation - no Python loops!
    # Shape: (n_particles, n_particles, theta_dim)
    diff = particles[:, None, :] - particles[None, :, :]

    # Scaled squared distances: (n_particles, n_particles)
    # bandwidth can be scalar or (theta_dim,) — broadcasting handles both
    sq_dist = jnp.sum(diff**2 / (2 * bandwidth**2), axis=2)

    # Kernel matrix: K[i,j] = exp(-sum_d (x_i_d - x_j_d)^2 / (2*h_d^2))
    K = jnp.exp(-sq_dist)

    # Kernel gradient: ∇_d K[i,j] = -K[i,j] * (x_i_d - x_j_d) / h_d^2
    # Shape: (n_particles, n_particles, theta_dim)
    grad_K = -K[:, :, None] * diff / bandwidth**2

    return K, grad_K


class _PreconditionerBase:
    """Shared base for preconditioners (reference search + interface).

    Parameters
    ----------
    model : callable
        Model function: model(theta, data, rewards=None) -> (pmf, moments)
    observed_data : array
        Observation data points
    theta_dim : int
        Number of parameters (learnable dimensions only if fixed params exist)
    param_transform : callable or None
        Transformation from unconstrained to constrained space (e.g., softplus)
    rewards : array or None
        Optional rewards for multivariate models
    epsilon : float, default=1e-8
        Floor for scaling values to avoid division by zero
    """

    def __init__(self, model: Callable, observed_data: jnp.ndarray | SparseObservations,
                 theta_dim: int, param_transform: Callable | None = None,
                 rewards: jnp.ndarray | None = None, epsilon: float = 1e-8) -> None:
        self.model = model
        self.observed_data = observed_data
        self.theta_dim = theta_dim
        self.param_transform = param_transform
        self.rewards = rewards
        self.epsilon = epsilon
        self.scaling = None

    def _find_moment_matching_reference(self, theta_init: jnp.ndarray) -> jnp.ndarray:
        """Find reference point where model moments match data moments.

        Uses a data-driven search range: in unconstrained (phi) space,
        searches over a log-spaced grid from -2 to phi(10 * data_mean),
        adapting to the scale of the observed data.

        Parameters
        ----------
        theta_init : array (theta_dim,)
            Starting point in unconstrained space.

        Returns
        -------
        theta_ref : array (theta_dim,)
            Improved reference point in unconstrained space.
        """
        # Handle sparse vs dense observation format
        if is_sparse_observations(self.observed_data):
            times = self.observed_data  # Keep as SparseObservations for model call
            data_mean = float(jnp.mean(self.observed_data.values))
        else:
            times = jnp.atleast_1d(jnp.array(self.observed_data))
            data_mean = float(jnp.mean(times))

        # Data-driven search range in unconstrained space
        # inverse softplus: log(exp(x) - 1) ≈ x for large x
        x = max(10.0 * data_mean, 5.0)
        upper = float(np.log(np.expm1(x) + 1e-10)) if x < 30 else float(x)
        upper = max(upper, 5.0)
        candidates = np.concatenate([
            np.array([-2.0, -1.0, 0.0]),
            np.linspace(0.5, upper, 12)
        ])

        theta_ref = jnp.array(theta_init)

        for j in range(self.theta_dim):
            best_val = float(theta_ref[j])
            best_err = np.inf

            for c in candidates:
                theta_try = theta_ref.at[j].set(c)
                if self.param_transform is not None:
                    theta_try_c = self.param_transform(theta_try)
                else:
                    theta_try_c = theta_try

                try:
                    _, moments = self.model(theta_try_c, times, rewards=self.rewards)
                    if moments.ndim == 2:
                        model_mean = float(jnp.mean(moments[:, 0]))
                    else:
                        model_mean = float(moments[0])
                    err = abs(model_mean - data_mean)
                    if err < best_err:
                        best_err = err
                        best_val = c
                except Exception:
                    continue

            theta_ref = theta_ref.at[j].set(best_val)

        logger.debug("%s: moment-matching reference (unconstrained) = %s",
                      type(self).__name__, theta_ref)
        if self.param_transform is not None:
            logger.debug("%s: moment-matching reference (constrained) = %s",
                          type(self).__name__, self.param_transform(theta_ref))
        logger.debug("%s: data mean = %.4f", type(self).__name__, data_mean)

        return theta_ref

    def compute_scaling(self, theta_ref: jnp.ndarray) -> None:
        """Compute scaling factors. Must be overridden by subclasses."""
        raise NotImplementedError


class MomentJacobianPreconditioner(_PreconditionerBase):
    """Moment Jacobian preconditioner for multi-scale SVGD.

    Computes J[k,j] = d(moment_k)/d(theta_j) via finite differences at a
    reference point, then uses column norms as scaling factors.
    Scaling D_j = ||J[:,j]|| (column norm), normalized to mean 1.

    This is simpler and more robust than Fisher preconditioning because it
    avoids dividing by PMF values (which can blow up when PMF is small).

    Parameters
    ----------
    model : callable
        Model function: model(theta, data, rewards=None) -> (pmf, moments)
    observed_data : array
        Observation data points
    theta_dim : int
        Number of parameters (learnable dimensions only if fixed params exist)
    param_transform : callable or None
        Transformation from unconstrained to constrained space (e.g., softplus)
    rewards : array or None
        Optional rewards for multivariate models
    epsilon : float, default=1e-8
        Floor for scaling values to avoid division by zero
    """

    def compute_scaling(self, theta_ref: jnp.ndarray) -> None:
        """Compute Jacobian column norms at reference point and derive scaling.

        Uses moment-matching to find a better reference point before computing
        the Jacobian. The provided theta_ref is used as a starting point for
        the moment-matching search.

        Parameters
        ----------
        theta_ref : array (theta_dim,)
            Initial reference point in unconstrained space (same space as particles).
            Used as starting point for moment-matching refinement.
        """
        logger.debug("MomentJacobianPreconditioner: computing scaling for %d dimensions",
                      self.theta_dim)
        logger.debug("MomentJacobianPreconditioner: initial theta_ref (unconstrained) = %s",
                      theta_ref)

        theta_ref = self._find_moment_matching_reference(theta_ref)
        logger.debug("MomentJacobianPreconditioner: refined theta_ref (unconstrained) = %s",
                      theta_ref)

        if self.param_transform is not None:
            theta_c = self.param_transform(theta_ref)
        else:
            theta_c = theta_ref

        # Handle sparse vs dense observation format
        if is_sparse_observations(self.observed_data):
            times = self.observed_data  # Keep as SparseObservations
        else:
            times = jnp.atleast_1d(jnp.array(self.observed_data))
        _, moments_ref = self.model(theta_c, times, rewards=self.rewards)
        moments_ref = moments_ref.flatten()
        n_moments = len(moments_ref)

        eps = 1e-5
        J = np.zeros((n_moments, self.theta_dim))

        for j in range(self.theta_dim):
            theta_plus = theta_ref.at[j].add(eps)
            theta_minus = theta_ref.at[j].add(-eps)
            if self.param_transform is not None:
                tp_c = self.param_transform(theta_plus)
                tm_c = self.param_transform(theta_minus)
            else:
                tp_c, tm_c = theta_plus, theta_minus

            _, moments_plus = self.model(tp_c, times, rewards=self.rewards)
            _, moments_minus = self.model(tm_c, times, rewards=self.rewards)
            J[:, j] = np.asarray(
                (moments_plus.flatten() - moments_minus.flatten()) / (2 * eps)
            )

        col_norms = np.linalg.norm(J, axis=0)
        D = np.maximum(col_norms, self.epsilon)
        D = D / np.mean(D)
        self.scaling = jnp.array(D)

        logger.debug("MomentJacobianPreconditioner: Jacobian matrix:\n%s", J)
        logger.debug("MomentJacobianPreconditioner: column norms = %s", col_norms)
        logger.debug("MomentJacobianPreconditioner: final scaling = %s", self.scaling)


class FisherPreconditioner(_PreconditionerBase):
    """Diagonal Fisher information preconditioner for multi-scale SVGD.

    Computes the diagonal of the empirical Fisher information matrix at a
    reference parameter point, then uses it to normalize the kernel's
    particle space so that all dimensions have comparable information content.

    Parameters
    ----------
    model : callable
        Model function: model(theta, data, rewards=None) -> (pmf, moments)
    observed_data : array
        Observation data points
    theta_dim : int
        Number of parameters (learnable dimensions only if fixed params exist)
    param_transform : callable or None
        Transformation from unconstrained to constrained space (e.g., softplus)
    rewards : array or None
        Optional rewards for multivariate models
    epsilon : float, default=1e-8
        Floor for Fisher values to avoid division by zero
    """

    def compute_scaling(self, theta_ref: jnp.ndarray) -> None:
        """Compute Fisher diagonal at reference point and derive scaling.

        Uses moment-matching to find a better reference point before computing
        the Fisher information. The provided theta_ref is used as a starting
        point for the moment-matching search.

        Parameters
        ----------
        theta_ref : array (theta_dim,)
            Initial reference point in unconstrained space (same space as particles).
            Used as starting point for moment-matching refinement.
        """
        logger.debug("FisherPreconditioner: computing scaling for %d dimensions", self.theta_dim)
        logger.debug("FisherPreconditioner: initial theta_ref (unconstrained) = %s", theta_ref)

        # Find a better reference point via moment matching
        theta_ref = self._find_moment_matching_reference(theta_ref)
        logger.debug("FisherPreconditioner: refined theta_ref (unconstrained) = %s", theta_ref)

        # Transform to constrained space for model evaluation
        if self.param_transform is not None:
            theta_c = self.param_transform(theta_ref)
            logger.debug("FisherPreconditioner: theta_ref (constrained) = %s", theta_c)
        else:
            theta_c = theta_ref
            logger.debug("FisherPreconditioner: no param_transform, using raw theta_ref")

        # Handle sparse vs dense observation format
        if is_sparse_observations(self.observed_data):
            times = self.observed_data  # Keep as SparseObservations
            n_times = len(self.observed_data.values)
        else:
            times = jnp.atleast_1d(jnp.array(self.observed_data))
            n_times = len(times)
        logger.debug("FisherPreconditioner: %d observation points", n_times)

        # Reference PMF values: p(x_n | theta_ref) for all observations
        pmf_ref, _ = self.model(theta_c, times, rewards=self.rewards)
        # Handle 2D (multivariate) pmf by flattening
        pmf_flat = pmf_ref.flatten()
        logger.debug("FisherPreconditioner: reference PMF range [%.4e, %.4e], "
                      "mean=%.4e, %d zero/negative values",
                      float(jnp.min(pmf_flat)), float(jnp.max(pmf_flat)),
                      float(jnp.mean(pmf_flat)),
                      int(jnp.sum(pmf_flat <= 0)))

        # Finite-difference score: d log p(x_n|theta) / d theta_j = (d pmf_n / d theta_j) / pmf_n
        eps = 1e-5
        n_obs = len(pmf_flat)
        scores = np.zeros((n_obs, self.theta_dim))

        for j in range(self.theta_dim):
            theta_plus = theta_ref.at[j].add(eps)
            theta_minus = theta_ref.at[j].add(-eps)
            if self.param_transform is not None:
                theta_plus_c = self.param_transform(theta_plus)
                theta_minus_c = self.param_transform(theta_minus)
            else:
                theta_plus_c = theta_plus
                theta_minus_c = theta_minus

            pmf_plus, _ = self.model(theta_plus_c, times, rewards=self.rewards)
            pmf_minus, _ = self.model(theta_minus_c, times, rewards=self.rewards)

            # Central difference for d pmf / d theta_j
            dpmf_dtheta_j = (pmf_plus.flatten() - pmf_minus.flatten()) / (2 * eps)
            # Score = d log pmf / d theta_j = (d pmf / d theta_j) / pmf
            scores[:, j] = np.asarray(dpmf_dtheta_j) / (np.asarray(pmf_flat) + 1e-30)
            logger.debug("FisherPreconditioner: dim %d: score range [%.4e, %.4e], "
                          "mean_abs=%.4e",
                          j, float(np.min(scores[:, j])),
                          float(np.max(scores[:, j])),
                          float(np.mean(np.abs(scores[:, j]))))

        # Empirical Fisher diagonal: F_j = (1/N) sum_n score[n,j]^2
        fisher_diag = np.mean(scores**2, axis=0)  # Shape: (theta_dim,)
        logger.debug("FisherPreconditioner: Fisher diagonal (raw) = %s", fisher_diag)

        D = np.sqrt(np.maximum(fisher_diag, self.epsilon))
        logger.debug("FisherPreconditioner: sqrt(Fisher) before normalization = %s", D)

        # Normalize to mean 1 (only relative scaling matters)
        D = D / np.mean(D)
        self.scaling = jnp.array(D)
        logger.debug("FisherPreconditioner: final scaling (normalized) = %s", self.scaling)


class SVGDKernel:
    """RBF kernel for SVGD with automatic bandwidth selection.

    Parameters
    ----------
    bandwidth : str or float or jnp.ndarray, default='median_per_dim'
        Bandwidth selection method or fixed value.
    preconditioner : _PreconditionerBase or None, default=None
        Optional preconditioner for normalizing particle space.
    """

    def __init__(self, bandwidth: str | float | jnp.ndarray = 'median_per_dim', preconditioner: _PreconditionerBase | None = None) -> None:
        """
        Parameters
        ----------
        bandwidth : str, float, or np.ndarray, default='median_per_dim'
            Bandwidth selection method. Options:
            - 'median_per_dim': Per-dimension median heuristic (default).
              Computes a separate bandwidth for each parameter dimension,
              giving an anisotropic kernel that adapts to different parameter scales.
            - 'median': Scalar median heuristic (isotropic kernel)
            - float: Fixed scalar bandwidth value
            - np.ndarray: Fixed per-dimension bandwidth vector
        preconditioner : MomentJacobianPreconditioner, FisherPreconditioner, or None, default=None
            If provided, particles are normalized by the preconditioner's scaling
            before kernel computation, and kernel gradients are transformed back.
            This makes the kernel isotropic in the preconditioned space.
        """
        self.bandwidth_method = bandwidth
        self.preconditioner = preconditioner

    def compute_kernel_grad(self, particles: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute RBF kernel matrix and its gradient (JIT-compiled)

        Parameters
        ----------
        particles : jnp.ndarray
            Current particle positions, shape ``(n_particles, theta_dim)``.

        Returns
        -------
        K : jnp.ndarray
            Kernel matrix, shape ``(n_particles, n_particles)``.
        grad_K : jnp.ndarray
            Gradient of kernel matrix w.r.t. original (unnormalized) particles,
            shape ``(n_particles, n_particles, theta_dim)``.
        """
        # Normalize particles if preconditioner is set
        if self.preconditioner is not None and self.preconditioner.scaling is not None:
            D = self.preconditioner.scaling  # (theta_dim,)
            particles_norm = particles * D[None, :]  # compress high-info dims
            logger.debug("SVGDKernel: preconditioner active, scaling=%s", D)
            logger.debug("SVGDKernel: particle range before normalization: "
                          "min=%s, max=%s",
                          jnp.min(particles, axis=0),
                          jnp.max(particles, axis=0))
            logger.debug("SVGDKernel: particle range after normalization: "
                          "min=%s, max=%s",
                          jnp.min(particles_norm, axis=0),
                          jnp.max(particles_norm, axis=0))
        else:
            particles_norm = particles

        # Compute bandwidth in normalized space
        if isinstance(self.bandwidth_method, str):
            if self.bandwidth_method == 'median_per_dim':
                bandwidth = batch_median_heuristic_per_dim(particles_norm)
            elif self.bandwidth_method == 'median':
                bandwidth = batch_median_heuristic(particles_norm)
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth_method!r}. "
                                 f"Options: 'median_per_dim', 'median'")
        else:
            bandwidth = jnp.asarray(self.bandwidth_method)

        # Compute kernel in normalized space
        K, grad_K_norm = _compute_kernel_grad_impl(particles_norm, bandwidth)

        # Transform gradient back: dK/dtheta_j = dK/dz_j * D_j
        if self.preconditioner is not None and self.preconditioner.scaling is not None:
            grad_K = grad_K_norm * D[None, None, :]
            logger.debug("SVGDKernel: gradient transformed back, "
                          "grad_K_norm range=[%.4e, %.4e], "
                          "grad_K range=[%.4e, %.4e]",
                          float(jnp.min(grad_K_norm)),
                          float(jnp.max(grad_K_norm)),
                          float(jnp.min(grad_K)),
                          float(jnp.max(grad_K)))
        else:
            grad_K = grad_K_norm

        return K, grad_K


@jit
def _svgd_update_jitted(particles: jnp.ndarray, K: jnp.ndarray, grad_K: jnp.ndarray, grad_log_p: jnp.ndarray, step_size: float) -> jnp.ndarray:
    """
    JIT-compiled SVGD update (core computation)

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    K : array (n_particles, n_particles)
        Kernel matrix
    grad_K : array (n_particles, n_particles, theta_dim)
        Kernel gradient
    grad_log_p : array (n_particles, theta_dim)
        Log probability gradients
    step_size : float
        Step size for update

    Returns
    -------
    array (n_particles, theta_dim)
        Updated particles
    """
    n_particles = particles.shape[0]

    # SVGD update: phi = (K @ grad_log_p + sum(grad_K)) / n
    # Vectorized computation - no Python loop!
    # K: (n_particles, n_particles)
    # grad_log_p: (n_particles, theta_dim)
    # K @ grad_log_p -> (n_particles, theta_dim)
    positive_term = jnp.einsum('ij,jk->ik', K, grad_log_p) / n_particles

    # grad_K: (n_particles, n_particles, theta_dim)
    # Sum over all particle interactions -> (n_particles, theta_dim)
    negative_term = jnp.sum(grad_K, axis=1) / n_particles

    phi = positive_term + negative_term

    return particles + step_size * phi


def svgd_step(particles: jnp.ndarray, log_prob_fn: callable, kernel: SVGDKernel, step_size: float,
              compiled_grad: callable | None = None,
              parallel_mode: str = 'vmap', n_devices: int | None = None,
              fixed_mask: jnp.ndarray | None = None, fixed_values: jnp.ndarray | None = None,
              optimizer: Adam | SGDMomentum | RMSprop | Adagrad | OptaxOptimizer | None = None) -> jnp.ndarray:
    """
    Perform single SVGD update step

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    log_prob_fn : callable
        Log probability function: theta -> scalar
    kernel : SVGDKernel
        Kernel object for computing K and grad_K
    step_size : float
        Step size for update
    compiled_grad : callable, optional
        Precompiled gradient function for faster execution
    parallel_mode : str, default='vmap'
        Parallelization strategy: 'vmap', 'pmap', or 'none'
    n_devices : int, optional
        Number of devices to use for pmap (only used if parallel_mode='pmap')
    fixed_mask : array (theta_dim,), optional
        Binary mask indicating which parameters to fix.
        - 0: Optimize this parameter
        - 1: Fix at value specified in fixed_values
        If provided, SVGD operates in reduced parameter space for efficiency.
    fixed_values : array (theta_dim,), optional
        Values to fix parameters at (for dimensions where fixed_mask=1).
        Defaults to all 1.0 if not provided.
    optimizer : Adam, SGDMomentum, RMSprop, Adagrad, OptaxOptimizer
        If provided, uses optimizer for adaptive per-parameter learning rates.
        When used, the `step_size` parameter is ignored.

    Returns
    -------
    array (n_particles, theta_dim)
        Updated particles
    """
    n_particles = particles.shape[0]

    # Handle fixed parameters via parameter projection
    if fixed_mask is not None:
        # Identify learnable dimensions
        learnable_mask = (fixed_mask == 0)
        learnable_indices = jnp.where(learnable_mask)[0]
        fixed_indices = jnp.where(fixed_mask == 1)[0]

        # Project to learnable subspace (computational efficiency!)
        particles_learnable = particles[:, learnable_indices]

        # Create wrapped log_prob that handles projection
        def log_prob_fn_reduced(theta_learnable):
            # Expand to full space: learnable → full
            # NOTE: Fixed dims initialized to values from fixed_values (untransformed space)
            # If param_transform is active (e.g., softplus), these will be transformed
            # IMPORTANT: Must create fresh array copy for JAX gradient tracing
            if fixed_values is not None:
                theta_full = jnp.array(fixed_values)  # Create fresh copy
            else:
                theta_full = jnp.ones(len(fixed_mask))
            theta_full = theta_full.at[learnable_indices].set(theta_learnable)
            return log_prob_fn(theta_full)

        # Wrap compiled_grad if provided
        if compiled_grad is not None:
            def compiled_grad_reduced(theta_learnable):
                # IMPORTANT: Must create fresh array copy for JAX gradient tracing
                if fixed_values is not None:
                    theta_full = jnp.array(fixed_values)  # Create fresh copy
                else:
                    theta_full = jnp.ones(len(fixed_mask))
                theta_full = theta_full.at[learnable_indices].set(theta_learnable)
                grad_full = compiled_grad(theta_full)
                return grad_full[learnable_indices]  # Extract learnable gradients
            compiled_grad_to_use = compiled_grad_reduced
        else:
            compiled_grad_to_use = None

        # Use reduced space for gradient computation
        particles_for_grad = particles_learnable
        log_prob_for_grad = log_prob_fn_reduced
    else:
        # No fixed parameters - standard SVGD
        particles_for_grad = particles
        log_prob_for_grad = log_prob_fn
        compiled_grad_to_use = compiled_grad

    # Use provided parallelization strategy
    actual_parallel_mode = parallel_mode
    actual_n_devices = n_devices

    # Compute log probability gradients based on parallelization strategy
    if actual_parallel_mode == 'pmap' and actual_n_devices is not None:
        # Parallel gradient computation across devices (pmap)
        particles_per_device = n_particles // actual_n_devices
        particles_sharded = particles_for_grad.reshape(actual_n_devices, particles_per_device, -1)

        # NOTE: JAX 0.8+ requires explicit device mesh to avoid conflicts
        # Create mesh for current pmap operation
        from jax.experimental import mesh_utils
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        # In multi-host environments, use only local devices (not global)
        local_devices = jax.local_devices()[:actual_n_devices]
        devices = mesh_utils.create_device_mesh((actual_n_devices,), devices=local_devices)
        mesh = Mesh(devices, axis_names=("batch",))

        # Use explicit mesh context for pmap
        # pmap over devices, vmap over particles within each device
        with mesh:
            if compiled_grad_to_use is not None:
                grad_log_p_sharded = pmap(vmap(compiled_grad_to_use), axis_name="batch")(particles_sharded)
            else:
                grad_log_p_sharded = pmap(vmap(grad(log_prob_for_grad)), axis_name="batch")(particles_sharded)

        grad_log_p = grad_log_p_sharded.reshape(n_particles, -1)
    elif actual_parallel_mode == 'vmap':
        # Single device vectorization - use vmap only
        if compiled_grad_to_use is not None:
            grad_log_p = vmap(compiled_grad_to_use)(particles_for_grad)
        else:
            grad_log_p = vmap(grad(log_prob_for_grad))(particles_for_grad)
    elif actual_parallel_mode == 'none':
        # No parallelization - sequential computation (useful for debugging)
        if compiled_grad_to_use is not None:
            grad_log_p = jnp.array([compiled_grad_to_use(p) for p in particles_for_grad])
        else:
            grad_fn = grad(log_prob_for_grad)
            grad_log_p = jnp.array([grad_fn(p) for p in particles_for_grad])
    else:
        raise ValueError(f"Invalid parallel_mode: {actual_parallel_mode}")

    # Compute kernel and kernel gradient (in reduced space if fixed_mask provided)
    K, grad_K = kernel.compute_kernel_grad(particles_for_grad)

    # ##############    

    # # phi = jnp.zeros_like(particles)
    # # for i in range(n_particles):
    # #     positive_term = jnp.sum(K[i, :, None] * grad_log_p, axis=0) / n_particles
    # #     negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
    # #     phi = phi.at[i].set(positive_term + negative_term)
    
    # positive_term = jnp.sum(K[:, :, None] * grad_log_p, axis=1) / n_particles
    # negative_term = jnp.sum(grad_K, axis=1) / n_particles
    # phi = positive_term + negative_term

    # # Adaptive step
    # # step_factor = kl_adaptive_step(particles, kl_target)
    # # _step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)  * phi 
    # _step_size = step_size * phi   

    #  # Call JIT-compiled update
    # return _svgd_update_jitted(particles, K, grad_K, grad_log_p, _step_size)

    # ##############

    # SVGD update: phi = (K @ grad_log_p + sum(grad_K)) / n
    # Compute in reduced space (learnable dims only) if fixed_mask provided
    positive_term = jnp.einsum('ij,jk->ik', K, grad_log_p) / n_particles
    negative_term = jnp.sum(grad_K, axis=1) / n_particles
    phi = positive_term + negative_term

    # Compute update: either Adam (adaptive) or fixed step size
    if optimizer is not None:
        update = optimizer.step(phi, particles=particles_for_grad)
    else:
        update = step_size * phi

    # Apply update to particles
    if fixed_mask is not None:
        # Update in reduced space (learnable dimensions only)
        particles_learnable_new = particles_for_grad + update

        # Expand back to full space with custom fixed values
        if fixed_values is not None:
            particles_new = jnp.tile(fixed_values, (particles.shape[0], 1))
        else:
            particles_new = jnp.ones_like(particles)  # Default: fixed dims = 1.0
        particles_new = particles_new.at[:, learnable_indices].set(particles_learnable_new)

        return particles_new
    else:
        # Standard SVGD update (no fixed parameters)
        return particles + update


def run_svgd(log_prob_fn: Callable | None, theta_init: jnp.ndarray, n_steps: int,
             learning_rate: float | StepSizeSchedule | None = None,
             kernel: SVGDKernel | None = None, return_history: bool = True,
             verbose: bool = True, progress: bool = False,
             compiled_grad: Callable | None = None,
             parallel_mode: str = 'vmap', n_devices: int | None = None,
             log_prob_fn_factory: Callable | None = None,
             regularization_schedule: RegularizationSchedule | None = None,
             lr_scale: float = 1.0,
             fixed_mask: jnp.ndarray | None = None,
             fixed_values: jnp.ndarray | None = None,
             optimizer: Adam | SGDMomentum | RMSprop | Adagrad | OptaxOptimizer | None = None
             ) -> dict:
    """
    Run Stein Variational Gradient Descent

    Parameters
    ----------
    log_prob_fn : callable
        Log probability function: theta -> scalar
        Should return log p(data|theta) + log p(theta)
    theta_init : array (n_particles, theta_dim)
        Initial particle positions
    n_steps : int
        Number of SVGD iterations
    learning_rate : float or StepSizeSchedule
        Step size. Can be:
        - float: constant step size (backward compatible)
        - StepSizeSchedule object: dynamic schedule
    kernel : SVGDKernel
        Kernel specification.
    return_history : bool
        If True, return particle positions at each iteration
    verbose : bool
        Print progress information
    progress : bool
        Display progress bar during optimization
    compiled_grad : callable, optional
        Precompiled gradient function for faster execution
    parallel_mode : str, default='vmap'
        Parallelization strategy: 'vmap', 'pmap', or 'none'
    n_devices : int, optional
        Number of devices for pmap (only used if parallel_mode='pmap')
    optimizer : Adam, SGDMomentum, RMSprop, Adagrad, OptaxOptimizer
        If provided, uses optimizer for adaptive per-parameter learning rates.
        When used, the `learning_rate` parameter is ignored.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'particles': Final particles (n_particles, theta_dim)
        - 'history': Particle history if return_history=True
        - 'theta_mean': Posterior mean
        - 'theta_std': Posterior standard deviation
    """

    # Initialize
    particles = theta_init

    history = [particles] if return_history else None
    history_iterations = [0] if return_history else []  # Track iteration numbers for history snapshots

    # # Handle step size schedule (backward compatible)
    # if isinstance(learning_rate, StepSizeSchedule):
    #     step_schedule = learning_rate
    #     use_schedule = True
    # elif isinstance(learning_rate, (int, float)):
    #     step_schedule = ConstantStepSize(float(learning_rate))
    #     use_schedule = False  # Can still use constant value
    # elif learning_rate is None:
    #     # When using an optimizer (e.g., Adam, Adamelia), learning_rate may be None
    #     # The optimizer handles step sizing internally
    #     step_schedule = None
    #     use_schedule = False
    # else:
    #     raise TypeError(
    #         f"learning_rate must be float, StepSizeSchedule, or None, got: {type(learning_rate)}"
    #     )

    # # Initialize optimizer if provided
    # if optimizer is not None:
    #     # Get shape for optimizer state (reduced space if fixed parameters)
    #     if fixed_mask is not None:
    #         learnable_dim = int(jnp.sum(fixed_mask == 0))
    #         optimizer_shape = (len(theta_init), learnable_dim)
    #     else:
    #         optimizer_shape = theta_init.shape
    #     optimizer.reset(optimizer_shape)
    #     if verbose:
    #         # Print optimizer-specific information
    #         opt_name = type(optimizer).__name__
    #         if hasattr(optimizer, 'beta1') and hasattr(optimizer, 'beta2'):
    #             print(f"Using {opt_name} (lr={optimizer.lr}, β1={optimizer.beta1}, β2={optimizer.beta2})")
    #         elif hasattr(optimizer, 'momentum'):
    #             print(f"Using {opt_name} (lr={optimizer.lr}, momentum={optimizer.momentum})")
    #         elif hasattr(optimizer, 'decay'):
    #             print(f"Using {opt_name} (lr={optimizer.lr}, decay={optimizer.decay})")
    #         else:
    #             print(f"Using {opt_name} (lr={optimizer.lr})")

    if learning_rate is None and optimizer is None:
        raise ValueError("Either learning_rate or optimizer must be provided.")

    if optimizer is None:
        if isinstance(learning_rate, StepSizeSchedule):
            step_schedule = learning_rate
            use_schedule = True
        elif isinstance(learning_rate, (int, float)):
            step_schedule = ConstantStepSize(float(learning_rate))
            use_schedule = False  # Can still use constant value
        else:        
            raise TypeError(
                f"learning_rate must be float, StepSizeSchedule, or None, got: {type(learning_rate)}"
            )
    else:
        step_schedule = None
        use_schedule = False

        if regularization_schedule is not None:
            raise ValueError("When optimizer is provided, regularization_schedule must be None.")

        # Get shape for optimizer state (reduced space if fixed parameters)
        if fixed_mask is not None:
            learnable_dim = int(jnp.sum(fixed_mask == 0))
            optimizer_shape = (len(theta_init), learnable_dim)
        else:
            optimizer_shape = theta_init.shape
        optimizer.reset(optimizer_shape)
        if verbose:
            # Print optimizer-specific information
            opt_name = type(optimizer).__name__
            if hasattr(optimizer, 'beta1') and hasattr(optimizer, 'beta2'):
                print(f"Using {opt_name} (lr={optimizer.lr}, β1={optimizer.beta1}, β2={optimizer.beta2})")
            elif hasattr(optimizer, 'momentum'):
                print(f"Using {opt_name} (lr={optimizer.lr}, momentum={optimizer.momentum})")
            elif hasattr(optimizer, 'decay'):
                print(f"Using {opt_name} (lr={optimizer.lr}, decay={optimizer.decay})")
            else:
                print(f"Using {opt_name} (lr={optimizer.lr})")

    # SVGD iterations
    if verbose:
        print(f"Running SVGD: {n_steps} steps, {len(particles)} particles")

    # for step in range(n_steps) if verbose else range(n_steps):
    for step in trange(n_steps) if progress else range(n_steps):
        # Compute current step size from schedule
        # Note: When optimizer is used, step_size is ignored by svgd_step
        if use_schedule:
            current_step_size = step_schedule(step, particles) * lr_scale
        elif learning_rate is not None:
            current_step_size = learning_rate * lr_scale
        else:
            # learning_rate=None means optimizer handles step sizing
            current_step_size = 0.0  # Ignored by svgd_step when optimizer is used

        # Compute current regularization and create log_prob_fn if using schedule
        if regularization_schedule is not None:
            current_reg = regularization_schedule(step, particles)
            # Create log_prob_fn with current regularization
            log_prob_fn = log_prob_fn_factory(current_reg)
            # Gradient is computed on-the-fly (no precompilation benefit)
            compiled_grad_to_use = None
        else:
            compiled_grad_to_use = compiled_grad

        # Perform SVGD update
        particles = svgd_step(particles, log_prob_fn, kernel, current_step_size,
                             compiled_grad=compiled_grad_to_use,
                             parallel_mode=parallel_mode,
                             n_devices=n_devices,
                             fixed_mask=fixed_mask,
                             fixed_values=fixed_values,
                             optimizer=optimizer)

        # Store history
        if return_history: # and (step % max(1, n_steps // 20) == 0):
            history.append(particles)
            history_iterations.append(step)

    # Final history
    if return_history:
        history.append(particles)
        history_iterations.append(n_steps)

    # Compute summary statistics
    theta_mean = jnp.mean(particles, axis=0)
    theta_std = jnp.std(particles, axis=0)

    results = {
        'particles': particles,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
    }

    if return_history:
        results['history'] = history
        results['history_iterations'] = history_iterations

    # Note: Final summary is printed by SVGD.fit() with transformed values

    return results


# ============================================================================
# Helper Functions for Moment-Based Regularization
# ============================================================================

def compute_sample_moments(data: jnp.ndarray | SparseObservations, nr_moments: int) -> jnp.ndarray:
    """
    Compute sample moments from observed data.

    For multivariate data (2D or SparseObservations), computes moments
    independently per feature, ignoring NaN values within each feature.

    Parameters
    ----------
    data : np.ndarray or SparseObservations
        Observed data points. Can be:
        - 1D array: (n_times,) for univariate observations
        - 2D array: (n_times, n_features) for multivariate observations
        - SparseObservations: sparse format with values and feature indices
    nr_moments : int
        Number of moments to compute

    Returns
    -------
    jnp.array
        Sample moments:
        - 1D data: Shape (nr_moments,) with [mean, mean(X^2), ..., mean(X^k)]
        - 2D/sparse data: Shape (n_features, nr_moments) with per-feature moments

    Examples
    --------
    >>> # 1D case
    >>> data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> moments = compute_sample_moments(data, nr_moments=2)
    >>> print(moments)  # [3.0, 11.0] = [mean, mean of squares]
    >>>
    >>> # 2D case (multivariate)
    >>> data = jnp.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    >>> moments = compute_sample_moments(data, nr_moments=2)
    >>> print(moments.shape)  # (2, 2) = (n_features, nr_moments)
    >>> # Feature 0: mean=2.0, Feature 1: mean=20.0 (ignoring NaN)
    >>>
    >>> # Sparse format
    >>> sparse = SparseObservations(
    ...     values=jnp.array([1.0, 2.0, 3.0, 10.0, 30.0]),
    ...     features=jnp.array([0, 0, 0, 1, 1]),
    ...     n_features=2
    ... )
    >>> moments = compute_sample_moments(sparse, nr_moments=2)
    >>> print(moments.shape)  # (2, 2) = (n_features, nr_moments)
    """
    # Handle sparse observations
    if is_sparse_observations(data):
        n_features = data.n_features
        moments = []
        for j in range(n_features):
            # Extract values for this feature
            mask = data.features == j
            values_j = data.values[mask]

            feature_moments = []
            for k in range(1, nr_moments + 1):
                # Compute k-th moment - no NaN since sparse format has only valid values
                if len(values_j) > 0:
                    feature_moments.append(jnp.mean(values_j**k))
                else:
                    # No observations for this feature - use NaN
                    feature_moments.append(jnp.nan)
            moments.append(feature_moments)
        return jnp.array(moments)  # Shape: (n_features, nr_moments)

    # Handle dense arrays
    data = jnp.array(data)

    if data.ndim == 1:
        # 1D data: backward compatible
        moments = []
        for k in range(1, nr_moments + 1):
            moments.append(jnp.nanmean(data**k))
        return jnp.array(moments)  # Shape: (nr_moments,)

    elif data.ndim == 2:
        # 2D data: compute moments per feature
        n_features = data.shape[1]
        moments = []
        for j in range(n_features):
            feature_moments = []
            for k in range(1, nr_moments + 1):
                # Compute k-th moment for feature j, ignoring NaN
                feature_moments.append(jnp.nanmean(data[:, j]**k))
            moments.append(feature_moments)
        return jnp.array(moments)  # Shape: (n_features, nr_moments)

    else:
        raise ValueError(
            f"Data must be 1D, 2D, or SparseObservations. "
            f"Got type: {type(data)}, shape: {data.shape}"
        )


# ============================================================================
# SVGD Class for Object-Oriented Interface
# ============================================================================

class SVGD:
    """
    Stein Variational Gradient Descent (SVGD) for Bayesian parameter inference.

    This class provides an object-oriented interface for SVGD inference with
    automatic result storage and diagnostic plotting capabilities.

    Parameters
    ----------
    model : callable
        JAX-compatible parameterized model with signature: model(theta, data) -> values
    observed_data : np.ndarray
        Observed data points
    prior : callable or list of Prior objects, optional
        Log prior function for parameters. Can be:
        - Single callable: prior(theta) -> scalar, applied to entire theta vector
        - List of Prior objects: One prior per parameter dimension.
          Use None for fixed parameters: prior=[GaussPrior(...), None, GaussPrior(...)]

        If None, uses standard normal prior: log p(theta) = -0.5 * sum(theta^2)

        **With fixed parameters**:
        When using a list of priors with the `fixed` parameter, you must provide None
        at indices corresponding to fixed parameters. This is validated at initialization.

        Example:
            prior=[GaussPrior(ci=[0,1]), None, GaussPrior(ci=[0,1])],
            fixed=[(1, 0.5)]  # theta[1] fixed, prior[1] must be None
    n_particles : int, default is 20 times length of theta
        Number of SVGD particles
    n_iterations : int, default=1000
        Number of SVGD optimization steps
    learning_rate : float, StepSizeSchedule, or None, default=None
        SVGD step size. Can be:
        - None: Uses Adam optimizer with adaptive learning rates (default)
        - float: constant step size (uses fixed learning rate approach)
        - StepSizeSchedule object: dynamic step size schedule

        When None and no optimizer is provided, Adam is used as the default
        optimizer (unless regularization > 0, which falls back to fixed lr=0.001).
        Examples: ConstantStepSize(0.01), ExpStepSize(0.1, 0.01, 500.0)
    bandwidth : str, float, or np.ndarray, default='median_per_dim'
        Kernel bandwidth selection. Can be:
        - 'median_per_dim': Per-dimension median heuristic (default). Uses a
          separate bandwidth for each parameter dimension, giving an anisotropic
          kernel that adapts to different parameter scales.
        - 'median': Scalar median heuristic (isotropic kernel)
        - float: Fixed scalar bandwidth value
        - np.ndarray: Fixed per-dimension bandwidth vector
    theta_init : np.ndarray, optional
        Initial particle positions (n_particles, theta_dim)
    theta_dim : int, optional
        Dimension of theta parameter vector (required if theta_init is None)
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
    progress : bool, default=False
        Display progress bar during optimization
    jit : bool or None, default=None
        Enable JIT compilation. If None, uses value from phasic.get_config().jit.
        If True, requires JAX to be available (raises PTDConfigError otherwise).
        JIT compilation provides significant speedup but adds initial compilation overhead.
    parallel : str or None, default=None
        Parallelization strategy:
        - 'vmap': Vectorize across particles (single device)
        - 'pmap': Parallelize across devices (uses multiple CPUs/GPUs)
        - 'none': No parallelization (sequential, useful for debugging)
        - None: Auto-select (pmap if multiple devices, vmap otherwise)

        **Single-machine multi-CPU**: Auto-selection uses pmap for multi-core parallelization.
        **Multi-node SLURM**: Call initialize_distributed() then set parallel='pmap' explicitly.
    n_devices : int or None, default=None
        Number of devices to use for pmap. Only used when parallel='pmap'.
        If None, uses all available devices. Must be <= number of available JAX devices.
        See: jax.devices() to check available devices, or configure via PTDALG_CPUS
        environment variable before import.
    precompile : bool, default=True
        (Deprecated: use jit parameter instead)
        Precompile model and gradient functions for faster execution.
        Implies jit=True for backward compatibility.
        First run will take longer (compilation time) but subsequent
        iterations will be much faster. Compiled functions are cached
        in memory and on disk (~/.phasic_cache/).
    compilation_config : CompilationConfig, dict, str, or Path, optional
        JAX compilation optimization configuration. Can be:
        - CompilationConfig object from phasic.CompilationConfig
        - dict with CompilationConfig parameters
        - str/Path to JSON config file
        - None (uses default balanced configuration)

        The configuration controls JAX/XLA compilation behavior including:
        - Persistent cache directory for cross-session caching
        - Optimization level (0-3)
        - Parallel compilation settings

        Examples:
        - Use preset: CompilationConfig.fast_compile()
        - Load from file: 'my_config.json'
        - Custom dict: {'optimization_level': 2, 'cache_dir': '/tmp/cache'}
    positive_params : bool, default=True
        If True, constrains parameters to positive domain using softplus transformation.
        SVGD operates in unconstrained space (can be negative) but results are
        transformed to positive values.

        DEFAULT=True because phase-type distribution parameters (rates) must be positive.
        Set to False only if you have a specific reason to allow negative parameters
        (e.g., regression coefficients, log-space parameterization).
    param_transform : callable, optional
        Custom parameter transformation function. Overrides positive_params if provided.
        Should map unconstrained space to constrained space (e.g., lambda x: jax.nn.sigmoid(x)
        for parameters in [0,1]). Cannot be used together with positive_params=True.
    regularization : float or RegularizationSchedule, default=0.0
        Moment-based regularization strength. Can be:
        - float: constant regularization (0.0 = no regularization, >0.0 = regularized SVGD)
        - RegularizationSchedule object: dynamic regularization schedule
        Examples: ConstantRegularization(1.0), ExpRegularization(5.0, 0.1, 500.0)

        If > 0.0, adds penalty term to match model moments to sample moments.
        Sample moments are computed from observed_data at initialization.

        **Note**: Using RegularizationSchedule disables gradient precompilation for flexibility,
        which may be slower than constant regularization but allows dynamic strategies.
    nr_moments : int, default=2
        Number of moments to use for regularization. Only used if regularization > 0.
        Typical values: 2 (mean and variance) or 3 (mean, variance, skewness).
    fixed : np.ndarray or list of tuples, optional
        Specifies which parameters to fix during optimization. Two formats supported:

        **Format 1 (Binary mask)**: Array indicating which parameters to fix at 1.0
        - 0: Optimize this parameter
        - 1: Fix at 1.0 (do not optimize)
        Must have length theta_dim.
        Example: fixed=[0, 1] fixes theta[1]=1.0 while optimizing theta[0].

        **Format 2 (Index-value tuples)**: List of (index, value) pairs
        - Each tuple specifies (parameter_index, fixed_value)
        - Only listed parameters are fixed; others are optimized
        - Values can be any positive number (not just 1.0)
        Example: fixed=[(1, 0.01)] fixes theta[1]=0.01 while optimizing theta[0].
        Example: fixed=[(0, 2.5), (2, 0.1)] fixes theta[0]=2.5 and theta[2]=0.1.

        **Performance**: SVGD operates in reduced parameter space (learnable dims only)
        for computational efficiency. Kernel bandwidth is computed only over varying
        dimensions, improving convergence.
    optimizer : Adam, SGDMomentum, RMSprop, Adagrad, OptaxOptimizer
        Optimizer for adaptive per-parameter learning rates. Default is None.

        Options include:
        - Adam: (default)Standard Adam optimizer
        - SGDMomentum: SGD with momentum
        - RMSprop: RMSprop optimizer
        - Adagrad: Adagrad optimizer

        When an optimizer is provided, the `learning_rate` parameter is ignored
        (the optimizer has its own learning rate).

        Example:
            >>> from phasic import SVGD, Adam
            >>> # Default: uses Adam
            >>> svgd = SVGD(model, data, theta_dim=2)
            >>> # Explicit optimizer
            >>> svgd = SVGD(model, data, theta_dim=2, optimizer=Adam(learning_rate=0.01))

    Attributes
    ----------
    particles : array
        Final posterior samples (n_particles, theta_dim)
    theta_mean : array
        Posterior mean estimate
    theta_std : array
        Posterior standard deviation
    history : list of arrays, optional
        Particle evolution over iterations (if fit was called with return_history=True)
    is_fitted : bool
        Whether fit() has been called

    Examples
    --------
    >>> # Basic usage with auto-configuration
    >>> svgd = SVGD(model, observed_data, theta_dim=1)
    >>> svgd.fit()
    >>>
    >>> # Explicit single-device configuration
    >>> svgd = SVGD(model, observed_data, theta_dim=1, jit=True, parallel='vmap')
    >>> svgd.fit()
    >>>
    >>> # Multi-device parallelization
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             jit=True, parallel='pmap', n_devices=8)
    >>> svgd.fit()
    >>>
    >>> # No JIT (for debugging)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, jit=False, parallel='none')
    >>> svgd.fit()
    >>>
    >>> # Multi-node SLURM (explicit distributed initialization)
    >>> from phasic import initialize_distributed
    >>> dist = initialize_distributed()  # Auto-detects SLURM environment
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             jit=True, parallel='pmap', n_devices=dist.num_processes)
    >>> svgd.fit()
    >>>
    >>> # Using step size schedules to prevent divergence
    >>> from phasic import ExpStepSize
    >>> schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=500.0)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, learning_rate=schedule)
    >>> svgd.fit()
    >>>
    >>> # Using adaptive step size based on particle spread
    >>> from phasic import AdaptiveStepSize
    >>> schedule = AdaptiveStepSize(base_step=0.01, kl_target=0.1, adjust_rate=0.1)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, learning_rate=schedule)
    >>> svgd.fit()
    >>>
    >>> # Using regularization schedules for moment matching
    >>> from phasic import ExpRegularization
    >>> reg_schedule = ExpRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             regularization=reg_schedule, nr_moments=2)
    >>> svgd.fit()  # Starts with strong regularization, gradually reduces
    >>>
    >>> # Using CDF-based regularization schedule (bidirectional)
    >>>
    >>> # Constant regularization (no schedule)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=1.0, nr_moments=2)
    >>> svgd.fit()
    >>>
    >>> # Using custom bandwidth schedule
    >>> from phasic import LocalAdaptiveBandwidth
    >>> bandwidth = LocalAdaptiveBandwidth(alpha=0.9, k_frac=0.1)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, kernel=bandwidth)
    >>> svgd.fit()
    >>>
    >>> # Access results
    >>> print(svgd.theta_mean)
    >>> print(svgd.theta_std)
    >>>
    >>> # Generate diagnostic plots
    >>> svgd.plot_posterior()
    >>> svgd.plot_trace()
    """

    # Class-level cache for compiled models (shared across instances)
    _compiled_cache = {}

    def __init__(self, model: Callable, observed_data: jnp.ndarray | SparseObservations,
                 prior: Prior | list[Prior | None] | DataPrior | None = None,
                 n_particles: int | None = None,
                 n_iterations: int = 700,
                 learning_rate: float | StepSizeSchedule | None = None,
                 bandwidth: str | float = 'median_per_dim',
                 theta_init: jnp.ndarray | None = None,
                 theta_dim: int | None = None,
                 seed: int | None = None,
                 verbose: bool = True,
                 progress: bool = False,
                 jit: bool | None = None,
                 parallel: str | None = None,
                 n_devices: int | None = None,
                 precompile: bool = True,
                 compilation_config: dict | str | None = None,
                 positive_params: bool = True,
                 param_transform: Callable | None = None,
                 regularization: float | RegularizationSchedule = 0.0,
                 nr_moments: int = 2,
                 rewards: jnp.ndarray | None = None,
                 fixed: dict | None = None,
                 optimizer: Adam | SGDMomentum | RMSprop | Adagrad | OptaxOptimizer | None = None,
                 preconditioner: str | _PreconditionerBase = 'auto') -> None:

        if n_particles is None:
            n_particles = 20 * theta_dim

        if seed is None:
            seed = np.random.randint(1, 10000)

        # Get configuration
        config = get_config()

        # Validate JIT parameter against config
        if jit is None:
            jit = config._use_jit  # Use config default
        elif jit and not config._use_jax:
            raise PTDConfigError(
                "jit=True requires JAX.\n"
                "  Current config: JAX disabled (compute='cpu')\n"
                "  Fix: phasic.configure(compute='jax-cpu')"
            )

        # Validate parallel parameter
        if parallel is None:
            # Default: use pmap if multiple devices, vmap otherwise
            # This enables multi-core parallelization on single machines
            # For multi-node SLURM: call initialize_distributed() + set parallel='pmap' explicitly
            parallel = 'pmap' if len(jax.devices()) > 1 else 'vmap'
            if verbose:
                print(f"Auto-selected parallel='{parallel}' ({len(jax.devices())} devices available)")
        elif parallel not in ['vmap', 'pmap', 'none']:
            raise ValueError(
                f"parallel must be 'vmap', 'pmap', or 'none', got: {parallel}"
            )

        # Validate n_devices parameter and check for misconfigurations
        # In multi-host environments, pmap requires local device count, not global
        if jax.process_count() > 1:
            # Multi-host: use local devices only
            available_devices = jax.local_device_count()
            if verbose and n_devices is None:
                print(f"Multi-host environment detected: {jax.process_count()} processes")
                print(f"Using {available_devices} local devices per process")
        else:
            # Single-host: use all devices
            available_devices = len(jax.devices())

        if parallel == 'pmap':
            if available_devices == 1:
                import warnings
                warnings.warn(
                    "parallel='pmap' requested but only 1 JAX device available. "
                    "Using 'vmap' instead. To use pmap, configure more devices via "
                    "PTDALG_CPUS environment variable or initialize_distributed().",
                    UserWarning,
                    stacklevel=2
                )
                parallel = 'vmap'
                n_devices = None
            else:
                if n_devices is None:
                    n_devices = available_devices
                    if verbose:
                        print(f"Using all {n_devices} devices for pmap")
                elif n_devices > available_devices:
                    raise PTDConfigError(
                        f"n_devices={n_devices} but only {available_devices} devices available.\n"
                        f"  JAX devices: {jax.devices()}\n"
                        f"  Fix: Set n_devices<={available_devices} or configure more devices\n"
                        f"  See: PTDALG_CPUS environment variable or phasic.configure()"
                    )
                elif n_devices < 1:
                    raise ValueError(f"n_devices must be >= 1, got: {n_devices}")
        elif n_devices is not None:
            if verbose:
                print(f"Warning: n_devices={n_devices} ignored (only used with parallel='pmap')")
            n_devices = None

        # if verbose:
        #     print("---------------------------------------------")
        #     print(f"SVGD Configuration:")
        #     print(f"  JIT compilation:        {jit}")
        #     print(f"  Parallelization mode:   {parallel}")
        #     if parallel == 'pmap':
        #         print(f"  Number of devices:      {n_devices} (available: {available_devices})")    
        #     print("---------------------------------------------")

        # Store configuration (parallel may have been modified by validation)
        self.jit_enabled = jit
        self.parallel_mode = parallel
        self.n_devices = n_devices

        # Backward compatibility: precompile implies jit (deprecated)
        if precompile is not None and not precompile:
            import warnings
            warnings.warn(
                "precompile parameter is deprecated and will be removed in v1.0. "
                "Use jit=True/False instead.",
                DeprecationWarning,
                stacklevel=2
            )
        if precompile and not jit:
            if verbose:
                print("Warning: precompile=True but jit=False. Setting jit=True for backward compatibility.")
            self.jit_enabled = True

        # Handle compilation configuration
        if compilation_config is not None:
            from pathlib import Path
            try:
                from .jax_config import CompilationConfig
            except ImportError:
                # If running from svgd.py directly without package import
                try:
                    from jax_config import CompilationConfig
                except ImportError:
                    CompilationConfig = None

            # Parse compilation_config
            if isinstance(compilation_config, str) or isinstance(compilation_config, Path):
                # Load from file
                if CompilationConfig:
                    config = CompilationConfig.load_from_file(compilation_config)
                    config.apply(force=False)
                    if verbose:
                        print(f"Loaded compilation config from: {compilation_config}")
            elif isinstance(compilation_config, dict):
                # Create from dictionary
                if CompilationConfig:
                    config = CompilationConfig(**compilation_config)
                    config.apply(force=False)
                    if verbose:
                        print(f"Applied compilation config from dict")
            elif CompilationConfig and isinstance(compilation_config, CompilationConfig):
                # Already a CompilationConfig object
                compilation_config.apply(force=False)
                if verbose:
                    print(f"Applied compilation config")
            else:
                if verbose:
                    print(f"Warning: Could not parse compilation_config, using defaults")

        self.model = model

        # Handle sparse vs dense observation format
        if is_sparse_observations(observed_data):
            self.observed_data = observed_data  # Keep as SparseObservations
            self._sparse_format = True
            n_observations = float(len(observed_data.values))
            if verbose:
                print(f"Using sparse observation format: {len(observed_data.values)} observations across {observed_data.n_features} features")
        else:
            self.observed_data = jnp.array(observed_data)
            self._sparse_format = False
            n_observations = float(self.observed_data.shape[0])

        self.prior = prior
        # Detect per-parameter priors (list/tuple of Prior objects)
        self.prior_list = list(prior) if isinstance(prior, (list, tuple, DataPrior)) else None
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.fixed = fixed

        # Handle step size schedule and optimizer selection
        # Auto-scale learning rate by number of observations to prevent gradient explosion
        # Gradients scale with n_observations (not total elements), so we normalize by that
        lr_scale = 1.0 / max(1.0, n_observations / 1000.0)  # Scale down for > 1000 observations

        if optimizer is not None:
            if learning_rate is not None:
                raise ValueError(
                    "Cannot provide both optimizer and learning_rate. "
                    "The optimizer has its own learning rate."
                )
            if regularization:
                raise ValueError(
                    "When using an optimizer, regularization must be 0.0. "
                    "Moment regularization requires fixed learning rates."
                )
            self.step_schedule = None  # Not used when optimizer is set
            self.learning_rate = None
            self.lr_scale = lr_scale
        elif isinstance(learning_rate, StepSizeSchedule):
            self.step_schedule = learning_rate
            self.learning_rate = None  # Will be computed dynamically
            self.lr_scale = lr_scale
        elif isinstance(learning_rate, (int, float)):
            scaled_lr = float(learning_rate) * lr_scale
            self.step_schedule = ConstantStepSize(scaled_lr)
            self.learning_rate = scaled_lr
            self.lr_scale = lr_scale
            if lr_scale < 1.0:
                logger.debug(
                    f"Auto-scaled learning rate: {learning_rate} → {scaled_lr:.6g} "
                    f"({int(n_observations)} observations)"
                )
        elif learning_rate is None:
            # User provided optimizer explicitly, or regularization > 0 with no learning_rate
            # Use a sensible default learning rate
            default_lr = 0.001
            scaled_lr = default_lr * lr_scale
            self.step_schedule = ConstantStepSize(scaled_lr)
            self.learning_rate = scaled_lr
            self.lr_scale = lr_scale
            if regularization > 0.0:
                logger.debug(f"Using default learning rate {default_lr} (regularization={regularization})")
        else:
            raise TypeError(
                f"learning_rate must be float, StepSizeSchedule, or None, got: {type(learning_rate)}"
            )

        if optimizer is not None:
            # Optimizer handles step sizing; override any step_schedule that was set
            self.step_schedule = None
            logging.info(f"Using {optimizer.__class__.__name__} as optimizer for SVGD")

        self.bandwidth = bandwidth
        self.theta_dim = theta_dim
        self.seed = seed
        self.verbose = verbose
        self.progress = progress
        self.precompile = precompile
        self.compilation_config = compilation_config

        # Handle parameter transformation
        if positive_params and param_transform is not None:
            raise ValueError(
                "Cannot specify both positive_params=True and param_transform. "
                "Use positive_params=True for automatic softplus transformation, "
                "or provide a custom param_transform function."
            )

        if positive_params:
            # Clamp minimum to avoid edge weights becoming non-positive
            # Use 1e-9 as minimum to avoid numerical precision issues in C code
            # (causing floating-point errors that produce -0.0 weights)
            self.param_transform = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
            if verbose:
                print("Using softplus transformation to constrain parameters to positive domain")
        elif param_transform is not None:
            if not callable(param_transform):
                raise ValueError("param_transform must be a callable function")
            self.param_transform = param_transform
            if verbose:
                print("Using custom parameter transformation")
        else:
            self.param_transform = None

        # Wrap priors with transformation info for Jacobian correction
        # This allows priors to be defined in THETA space while SVGD works in PHI space
        if self.param_transform is not None:
            if self.prior_list is not None:
                for prior_i in self.prior_list:
                    if hasattr(prior_i, '_transform'):
                        prior_i._transform = self.param_transform
            elif self.prior is not None and hasattr(self.prior, '_transform'):
                self.prior._transform = self.param_transform

        # Validate and initialize particles
        if theta_init is None and theta_dim is None:
            raise ValueError(
                "Either theta_init or theta_dim must be provided. "
                "If you don't have initial particles, specify theta_dim (the number of parameters)."
            )

        # Validate per-parameter prior list length
        if self.prior_list is not None and theta_dim is not None:
            if len(self.prior_list) != theta_dim:
                raise ValueError(
                    f"prior list length ({len(self.prior_list)}) must match theta_dim ({theta_dim})"
                )

            # Check that None entries in prior_list align with fixed parameters
            if fixed is not None:
                # Determine fixed indices from fixed parameter
                if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
                    fixed_indices = set(idx for idx, _ in fixed)
                else:
                    fixed_indices = set(i for i, val in enumerate(fixed) if val == 1)

                # Check None alignment
                for i, prior_i in enumerate(self.prior_list):
                    if prior_i is None and i not in fixed_indices:
                        raise ValueError(
                            f"prior[{i}] is None but theta[{i}] is not fixed.\n"
                            f"Use None in prior list only for fixed parameters.\n"
                            f"Fixed indices: {sorted(fixed_indices)}\n"
                            f"Example: prior=[GaussPrior(...), None, ...] with fixed=[(1, value)]"
                        )
                    elif prior_i is not None and i in fixed_indices:
                        raise ValueError(
                            f"prior[{i}] is provided but theta[{i}] is fixed.\n"
                            f"Use None in prior list for fixed parameters.\n"
                            f"Example: prior=[GaussPrior(...), None, GaussPrior(...)] with fixed=[(1, value)]"
                        )

        # Adjust n_particles for pmap if needed
        if self.parallel_mode == 'pmap' and self.n_devices is not None:
            if n_particles % self.n_devices != 0:
                adjusted_n_particles = ((n_particles + self.n_devices - 1) // self.n_devices) * self.n_devices
                if verbose:
                    print(f"Adjusted n_particles from {n_particles} to {adjusted_n_particles} "
                          f"for even distribution across {self.n_devices} devices")
                n_particles = adjusted_n_particles
                self.n_particles = n_particles

        # Initialize particles
        key = jax.random.PRNGKey(seed)
        if theta_init is None:
            # Check for per-parameter priors (list of Prior objects)
            if self.prior_list is not None:
                # Sample each dimension from its respective prior (skip None for fixed params)
                samples = []
                for i, prior_i in enumerate(self.prior_list):
                    if prior_i is None:
                        # Fixed parameter - will be set later based on fixed_values
                        # Use placeholder for now (normal distribution)
                        key, subkey = jax.random.split(key)
                        s_i = jax.random.normal(subkey, (n_particles, 1))
                        samples.append(s_i)
                    else:
                        if not hasattr(prior_i, 'sample'):
                            raise ValueError(
                                f"Prior at index {i} does not have a sample() method. "
                                f"Per-parameter priors must be Prior objects with sample() method."
                            )
                        key, subkey = jax.random.split(key)
                        s_i = prior_i.sample(subkey, (n_particles, 1))
                        samples.append(s_i)
                self.theta_init = jnp.concatenate(samples, axis=1)
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from per-parameter priors:")
                    for i, prior_i in enumerate(self.prior_list):
                        if prior_i is None:
                            print(f"    θ[{i}]: (fixed parameter)")
                        elif hasattr(prior_i, 'mu') and hasattr(prior_i, 'sigma'):
                            print(f"    θ[{i}]: N({prior_i.mu:.2g}, {prior_i.sigma:.2g}²)")
                        elif hasattr(prior_i, 'scale'):
                            print(f"    θ[{i}]: HalfCauchy(scale={prior_i.scale:.2g})")
                        else:
                            print(f"    θ[{i}]: {type(prior_i).__name__}")
            # Check if prior is a Prior object with sample method
            elif self.prior is not None and hasattr(self.prior, 'sample'):
                # Sample from prior distribution
                self.theta_init = self.prior.sample(key, (n_particles, theta_dim))
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from prior")
                    if hasattr(self.prior, 'mu') and hasattr(self.prior, 'sigma'):
                        print(f"  (Prior: N({self.prior.mu:.2g}, {self.prior.sigma:.2g}²))")
                    elif hasattr(self.prior, 'scale'):
                        print(f"  (Prior: HalfCauchy(scale={self.prior.scale:.2g}))")
            elif self.param_transform is not None:
                # Fallback: For transformed parameters, initialize in a range that maps to reasonable positive values
                # softplus(x) ≈ x for x >> 0, and softplus(0) ≈ 0.69
                # Initialize around N(1, 1) so softplus gives values around 1-2
                self.theta_init = jax.random.normal(key, (n_particles, theta_dim)) + 1.0
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from N(1,1)")
                    print(f"  (Transformed range: softplus(N(1,1)) ≈ [0.7, 3.5])")
            else:
                self.theta_init = jax.random.normal(key, (n_particles, theta_dim))
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from N(0,1)")
        else:
            self.theta_init = jnp.array(theta_init)
            if self.theta_init.ndim != 2:
                raise ValueError(
                    f"theta_init must be 2D array (n_particles, theta_dim), "
                    f"got shape {self.theta_init.shape}"
                )
            self.n_particles = self.theta_init.shape[0]
            self.theta_dim = self.theta_init.shape[1]
            if verbose:
                print(f"Using provided initial particles: {self.theta_init.shape}")

        # Validate and store fixed parameter mask
        if fixed is not None:
            # Detect format: list of tuples [(index, value), ...] vs binary mask [0, 1, ...]
            if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
                # Format: [(index, value), ...] - fix specific parameters at specific values
                self.fixed_mask = jnp.zeros(self.theta_dim)
                self.fixed_values = jnp.ones(self.theta_dim)  # Default to 1.0 for non-fixed

                for idx, value in fixed:
                    if not isinstance(idx, (int, jnp.integer)):
                        raise TypeError(f"Parameter index must be integer, got {type(idx)}")
                    if idx < 0 or idx >= self.theta_dim:
                        raise ValueError(
                            f"Invalid parameter index {idx}, must be in [0, {self.theta_dim})"
                        )
                    self.fixed_mask = self.fixed_mask.at[idx].set(1)
                    # CRITICAL: Fixed values are specified in THETA space (constrained).
                    # But SVGD operates in PHI space (unconstrained), and log_prob_fn applies
                    # param_transform (e.g., softplus) to convert phi -> theta.
                    # So we must store fixed values in PHI space, i.e., apply inverse transform.
                    # inv_softplus(theta) = log(exp(theta) - 1) for theta > 0
                    if positive_params and value > 0:
                        # Apply inverse softplus to convert theta -> phi
                        phi_value = float(jnp.log(jnp.exp(value) - 1))
                        self.fixed_values = self.fixed_values.at[idx].set(phi_value)
                    else:
                        # No transform or value <= 0 (edge case)
                        self.fixed_values = self.fixed_values.at[idx].set(value)

                n_fixed = len(fixed)
                n_learnable = self.theta_dim - n_fixed
                if verbose:
                    print(f"Fixed parameters: {n_fixed}/{self.theta_dim} parameters fixed at custom values")
                    for idx, value in fixed:
                        print(f"  θ_{idx} fixed at {value}")
                    print(f"  Learnable: {n_learnable}, Fixed: {n_fixed}")
                    if n_learnable == 0:
                        raise ValueError("All parameters are fixed! At least one parameter must be learnable.")

                # Initialize fixed dimensions to their specified values (in PHI space)
                if theta_init is None:
                    for idx, value in fixed:
                        if positive_params and value > 0:
                            # Convert theta -> phi
                            phi_value = float(jnp.log(jnp.exp(value) - 1))
                            self.theta_init = self.theta_init.at[:, idx].set(phi_value)
                        else:
                            self.theta_init = self.theta_init.at[:, idx].set(value)
            else:
                # Format: [0, 1, 0, ...] - binary mask (backward compatible)
                fixed_array = jnp.array(fixed)
                if len(fixed_array) != self.theta_dim:
                    raise ValueError(
                        f"fixed must have length {self.theta_dim} (theta_dim), got {len(fixed_array)}"
                    )
                if not jnp.all((fixed_array == 0) | (fixed_array == 1)):
                    raise ValueError(
                        "fixed must contain only 0 (optimize) or 1 (fix at 1.0). "
                        f"Got values: {jnp.unique(fixed_array)}"
                    )
                self.fixed_mask = fixed_array
                # Fixed values default to theta=1.0, but we need to store in phi space
                if positive_params:
                    # inv_softplus(1.0) = log(exp(1) - 1) ≈ 0.5413
                    phi_one = float(jnp.log(jnp.exp(1.0) - 1))
                    self.fixed_values = jnp.full(self.theta_dim, phi_one)
                else:
                    self.fixed_values = jnp.ones(self.theta_dim)  # All fixed values = 1.0
                n_fixed = int(jnp.sum(fixed_array))
                n_learnable = self.theta_dim - n_fixed
                if verbose:
                    print(f"Fixed parameters: {n_fixed}/{self.theta_dim} parameters fixed at 1.0")
                    print(f"  Learnable: {n_learnable}, Fixed: {n_fixed}")
                    if n_learnable == 0:
                        raise ValueError("All parameters are fixed! At least one parameter must be learnable.")

                # Initialize fixed dimensions to phi value corresponding to theta=1.0
                if theta_init is None:
                    fixed_indices = jnp.where(fixed_array == 1)[0]
                    if positive_params:
                        # inv_softplus(1.0) ≈ 0.5413
                        phi_one = float(jnp.log(jnp.exp(1.0) - 1))
                        self.theta_init = self.theta_init.at[:, fixed_indices].set(phi_one)
                    else:
                        self.theta_init = self.theta_init.at[:, fixed_indices].set(1.0)
        else:
            self.fixed_mask = None
            self.fixed_values = None

        # Store regularization settings and handle regularization schedule (backward compatible)
        if isinstance(regularization, RegularizationSchedule):
            self.regularization_schedule = regularization
            self.use_regularization_schedule = True
            # Evaluate at iteration 0 to get initial value
            self.regularization = regularization(0)
            if verbose:
                print(f"Using regularization schedule (initial value: {self.regularization})")
        elif isinstance(regularization, (int, float)):
            self.regularization_schedule = ConstantRegularization(float(regularization))
            self.use_regularization_schedule = False
            self.regularization = float(regularization)
        else:
            raise TypeError(
                f"regularization must be float or RegularizationSchedule, got: {type(regularization)}"
            )

        self.nr_moments = nr_moments
        self.rewards = rewards  # Can be None, 1D (n_vertices,), or 2D (n_vertices, n_features)
        self.optimizer = optimizer

        # Validate and store preconditioner setting
        if isinstance(preconditioner, str):
            if preconditioner in ('auto', 'jacobian'):
                self.preconditioner_method = 'jacobian'
            elif preconditioner == 'fisher':
                self.preconditioner_method = 'fisher'
            elif preconditioner == 'none':
                self.preconditioner_method = None
            else:
                raise ValueError(
                    f"preconditioner must be 'auto', 'jacobian', 'fisher', 'none', None, "
                    f"or a preconditioner instance, got: {preconditioner!r}"
                )
        elif preconditioner is None:
            self.preconditioner_method = None
        elif isinstance(preconditioner, (FisherPreconditioner, MomentJacobianPreconditioner)):
            self.preconditioner_method = preconditioner
        else:
            raise TypeError(
                f"preconditioner must be str, None, FisherPreconditioner, "
                f"or MomentJacobianPreconditioner, got: {type(preconditioner)}"
            )

        if self.regularization > 0.0 or self.use_regularization_schedule:
            if self.nr_moments == 0:
                raise ValueError(
                    "nr_moments must be > 0 when using regularization."
                )

        # Compute sample moments if initial regularization > 0 or using schedule
        # (schedule might start at 0 but increase later, so we need moments ready)
        if self.regularization > 0.0 or self.use_regularization_schedule:
            self.sample_moments = compute_sample_moments(self.observed_data, nr_moments)
            if verbose and self.regularization > 0.0:
                print(f"Computed {nr_moments} sample moments for regularization={self.regularization}")
        else:
            self.sample_moments = None

        # Validate that model returns (pmf, moments) tuple
        # All models must use Graph.pmf_and_moments_from_graph()
        try:
            test_theta = self.theta_init[0]
            # Use abs() to ensure positive test values when param_transform is set
            # (Actual transformation applied in _log_prob methods during optimization)
            # This avoids negative edge weights and FFI initialization crashes
            if self.param_transform is not None:
                test_theta = jnp.abs(test_theta)

            # Extract test times (handle sparse vs dense format)
            if self._sparse_format:
                # For sparse, take first 2 observations from the values array
                n_test = min(2, len(self.observed_data.values))
                test_times = SparseObservations(
                    values=self.observed_data.values[:n_test],
                    features=self.observed_data.features[:n_test],
                    n_features=self.observed_data.n_features
                )
            else:
                test_times = self.observed_data[:min(2, len(self.observed_data))]

            # Test with rewards if provided
            if self.rewards is not None:
                # For 2D rewards, extract first 2 columns to match test_times
                # (only for dense format - sparse format doesn't need this)
                if not self._sparse_format and jnp.asarray(self.rewards).ndim == 2 and test_times.ndim == 2:
                    test_rewards = jnp.asarray(self.rewards)[:, :test_times.shape[1]]
                else:
                    test_rewards = self.rewards
                result = self.model(test_theta, test_times, rewards=test_rewards)
            else:
                result = self.model(test_theta, test_times, rewards=None)

            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    "Model must return (pmf, moments) tuple. "
                    f"Got: {type(result)}. "
                    "Use Graph.pmf_and_moments_from_graph() to create model, "
                    "not Graph.pmf_from_graph()."
                )

            # Validate number of moments matches nr_moments parameter (if using regularization)
            if self.nr_moments > 0 and (self.regularization > 0.0 or self.use_regularization_schedule):
                pmf_vals, model_moments = result

                # Handle 2D moments (multivariate case)
                if model_moments.ndim == 2:
                    # Check shape: (n_features, nr_moments)
                    actual_nr_moments = model_moments.shape[1]
                else:
                    # 1D moments
                    actual_nr_moments = len(model_moments)

                if actual_nr_moments < self.nr_moments:
                    raise ValueError(
                        f"Model returns {actual_nr_moments} moments but SVGD is configured to use {self.nr_moments} moments. "
                        f"Create model with: Graph.pmf_and_moments_from_graph(graph, nr_moments={self.nr_moments})"
                    )

            if verbose:
                print("Model validated: returns (pmf, moments) tuple")
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Other errors during model evaluation
            raise ValueError(
                f"Model validation failed. Error: {e}\n"
                "Ensure model has signature: model(theta, times, rewards=None) -> (pmf, moments)"
            )

        # Results (initialized after fit())
        self.particles = None
        self.theta_mean = None
        self.theta_std = None
        self.history = None
        self.history_iterations = None
        self.is_fitted = False

        # Compiled model and gradient (set by _precompile_model if jit_enabled=True)
        self.compiled_model = None
        self.compiled_grad = None

        # Precompilation now happens in optimize() based on regularization settings
        # This allows caching for both regularized and non-regularized cases
        # Old behavior: if self.jit_enabled: self._precompile_model()

    def _log_prob(self, theta: jnp.ndarray) -> float:
        """
        Log probability function: log p(data|theta) + log p(theta)

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)

        Returns
        -------
        scalar
            Log probability
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Log-likelihood
        try:
            result = self.model(theta_transformed, self.observed_data)

            # Handle both (pmf, moments) and pmf-only models
            if isinstance(result, tuple):
                model_values = result[0]  # Extract PMF values
            else:
                model_values = result
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model has signature model(theta, times). "
                f"Error: {e}"
            )

        # Prevent log(0) by adding small epsilon
        log_lik = jnp.sum(jnp.log(model_values + 1e-10))

        # Log-prior (evaluated in unconstrained space)
        if self.prior_list is not None:
            # Per-parameter priors: sum log-probabilities, skip None (fixed params)
            log_pri = sum(
                self.prior_list[i](theta[i:i+1])
                for i in range(len(self.prior_list))
                if self.prior_list[i] is not None
            )
        elif self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior on unconstrained parameters
            log_pri = -0.5 * jnp.sum(theta**2)

        return log_lik + log_pri

    def _log_prob_regularized(self, theta: jnp.ndarray, sample_moments: jnp.ndarray, nr_moments: int, regularization: float) -> float:
        """
        Regularized log probability with moment matching term.

        log p(theta | data, moments) = log p(data|theta) + log p(theta) - λ * ||E[T^k|theta] - sample_moments||^2

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)
        sample_moments : array
            Sample moments computed from observed data
        nr_moments : int
            Number of moments to use for regularization
        regularization : float
            Strength of moment regularization (λ in objective)

        Returns
        -------
        scalar
            Regularized log probability
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Evaluate model to get PMF and moments
        try:
            result = self.model(theta_transformed, self.observed_data)
            if isinstance(result, tuple) and len(result) == 2:
                pmf_vals, model_moments = result
            else:
                raise ValueError("Model must return (pmf, moments) tuple for regularized SVGD")
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model signature is model(theta, times) -> (pmf, moments). "
                f"Error: {e}"
            )

        # Standard log-likelihood term
        log_lik = jnp.sum(jnp.log(pmf_vals + 1e-10))

        # Log-prior term (evaluated in unconstrained space)
        if self.prior_list is not None:
            # Per-parameter priors: sum log-probabilities, skip None (fixed params)
            log_pri = sum(
                self.prior_list[i](theta[i:i+1])
                for i in range(len(self.prior_list))
                if self.prior_list[i] is not None
            )
        elif self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior
            log_pri = -0.5 * jnp.sum(theta**2)

        # Moment regularization penalty
        # We want to minimize (model_moments - sample_moments)^2
        # So we subtract this from log probability
        moment_diff = model_moments[:nr_moments] - sample_moments
        moment_penalty = regularization * jnp.sum(moment_diff**2)

        return log_lik + log_pri - moment_penalty

    def _log_prob_unified(self, theta: jnp.ndarray, nr_moments: int = 0, sample_moments: jnp.ndarray | None = None,
                         regularization: float = 0.0, rewards: jnp.ndarray | None = None) -> float:
        """
        Unified log probability with optional moment regularization.

        This replaces both _log_prob() and _log_prob_regularized() with a single
        implementation that handles both cases based on the regularization parameter.

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)
        nr_moments : int, default=0
            Number of moments to use for regularization (only used if regularization > 0)
        sample_moments : array or None
            Sample moments from observed data (required if regularization > 0)
        regularization : float, default=0.0
            Strength of moment regularization (λ)
            - 0.0: No regularization
            - > 0.0: Moment-based regularization penalty
        rewards : array or None
            Reward vector/matrix for multivariate phase-type distributions.
            - 1D array (n_vertices,): Single reward vector for univariate distribution
            - 2D array (n_features, n_vertices): Reward matrix for multivariate distribution
              where each row rewards[j, :] defines the reward vector for feature j

        Returns
        -------
        scalar
            Log probability (with or without moment regularization penalty)

        Raises
        ------
        ValueError
            If regularization > 0 but model doesn't return moments
        ValueError
            If regularization > 0 but sample_moments is None
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Evaluate model
        try:
            result = self.model(theta_transformed, self.observed_data, rewards=rewards)
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model has signature model(theta, times, rewards=None). "
                f"Error: {e}"
            )

        # Always expect (pmf, moments) tuple
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Model must return (pmf, moments) tuple. "
                f"Got: {type(result)}. "
                "Use Graph.pmf_and_moments_from_graph() to create model."
            )

        pmf_vals, model_moments = result

        # Log-likelihood term - handle sparse vs dense format differently
        if self._sparse_format:
            # Sparse format: all values are valid (no NaN), simpler computation
            # pmf_vals should be 1D array matching the number of observations
            pmf_mask = ~jnp.isnan(pmf_vals)

            def check_nan_pmf_sparse(pmf_mask):
                """Callback to check for NaN PMF values (executed during runtime, not tracing)"""
                if not np.all(pmf_mask):
                    nan_count = np.sum(~pmf_mask)
                    raise ValueError(
                        f"Model returned NaN PMF values for valid observations. "
                        f"Check model implementation and parameter values. "
                        f"NaN count: {nan_count}"
                    )

            # Register debug callback (only executes at runtime, not during tracing)
            jax.debug.callback(check_nan_pmf_sparse, pmf_mask)

            # All observations are valid in sparse format - simple sum
            log_lik = jnp.sum(jnp.log(pmf_vals + 1e-10))

        else:
            # Dense format: handle missing observations via NaN
            # Distinguish between NaN observations (expected) and NaN PMF values (error)
            obs_mask = ~jnp.isnan(self.observed_data)  # Valid observations
            pmf_mask = ~jnp.isnan(pmf_vals)             # Valid PMF computations

            # Check for invalid PMF using debug callback (JAX-compatible error checking)
            # This won't block JIT compilation but will warn if NaN PMF occurs
            invalid_pmf = obs_mask & ~pmf_mask

            def check_nan_pmf(invalid_mask):
                """Callback to check for NaN PMF values (executed during runtime, not tracing)"""
                if np.any(invalid_mask):
                    raise ValueError(
                        f"Model returned NaN PMF values for valid observations. "
                        f"Check model implementation and parameter values. "
                        f"NaN count: {np.sum(invalid_mask)}"
                    )

            # Register debug callback (only executes at runtime, not during tracing)
            jax.debug.callback(check_nan_pmf, invalid_pmf)

            # Compute log-likelihood only on valid observations (skip NaN observations)
            log_lik = jnp.sum(jnp.where(obs_mask, jnp.log(pmf_vals + 1e-10), 0.0))

        # Log-prior term (evaluated in unconstrained space)
        if self.prior_list is not None:
            # Per-parameter priors: sum log-probabilities, skip None (fixed params)
            log_pri = sum(
                self.prior_list[i](theta[i:i+1])
                for i in range(len(self.prior_list))
                if self.prior_list[i] is not None
            )
        elif self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior on unconstrained parameters
            log_pri = -0.5 * jnp.sum(theta**2)

        # Moment regularization penalty
        # Always compute penalty if moments available (but it's zero if regularization=0)
        # This avoids Python control flow on potentially-traced values
        if sample_moments is not None and nr_moments > 0:
            # Handle different moment dimensionalities
            if model_moments.ndim == 2 and sample_moments.ndim == 2:
                # Both 2D: compute penalty per feature, then sum
                # Shape: (n_features, nr_moments)
                moment_diff = model_moments[:, :nr_moments] - sample_moments[:, :nr_moments]
                moment_penalty = regularization * jnp.sum(moment_diff**2)
            elif model_moments.ndim == 2 and sample_moments.ndim == 1:
                # Model is 2D but sample is 1D: aggregate model moments
                # This can happen if user manually provides 1D sample_moments
                model_moments_agg = jnp.mean(model_moments, axis=0)
                moment_diff = model_moments_agg[:nr_moments] - sample_moments
                moment_penalty = regularization * jnp.sum(moment_diff**2)
            else:
                # 1D moments (standard case)
                # Shape: (nr_moments,)
                moment_diff = model_moments[:nr_moments] - sample_moments
                moment_penalty = regularization * jnp.sum(moment_diff**2)

            return log_lik + log_pri - moment_penalty
        else:
            # No regularization: moments computed but not used
            return log_lik + log_pri

    def _get_cache_path(self) -> pathlib.Path:
        """Generate cache path for this model configuration."""
        # Create cache key from model id and shapes
        theta_shape = (self.theta_dim,)
        if self._sparse_format:
            times_shape = (len(self.observed_data.values), self.observed_data.n_features)
        else:
            times_shape = self.observed_data.shape
        cache_key = f"{id(self.model)}_{theta_shape}_{times_shape}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

        # Cache directory
        cache_dir = pathlib.Path.home() / '.phasic_cache'
        cache_dir.mkdir(exist_ok=True)

        return cache_dir / f"compiled_svgd_{cache_hash}.pkl"

    def _get_cache_key_unified(self, nr_moments: int, regularization: float, rewards: tuple | None = None) -> str:
        """
        Generate cache key including regularization parameters.

        Different regularization settings require different compiled gradients,
        so we include nr_moments, regularization, and rewards in the cache key.

        Parameters
        ----------
        nr_moments : int
            Number of moments for regularization
        regularization : float
            Regularization strength
        rewards : tuple or None, optional
            Reward vector as tuple for hashing

        Returns
        -------
        str
            Cache hash for this configuration
        """
        theta_shape = (self.theta_dim,)
        if self._sparse_format:
            times_shape = (len(self.observed_data.values), self.observed_data.n_features)
        else:
            times_shape = self.observed_data.shape
        # Include nr_moments, regularization, and rewards in cache key
        rewards_str = str(rewards) if rewards is not None else "None"
        cache_key = f"{id(self.model)}_{theta_shape}_{times_shape}_{nr_moments}_{regularization}_{rewards_str}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        return cache_hash

    def _save_compiled(self, cache_path: pathlib.Path) -> None:
        """Save compiled model and gradient to disk."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model': self.compiled_model,
                    'grad': self.compiled_grad
                }, f)
            if self.verbose:
                print(f"  Saved compiled functions to cache: {cache_path.name}")
        except Exception as e:
            # Disk caching is best-effort; memory cache still works
            # Pickling JIT functions with closures often fails - this is expected
            pass

    def _load_compiled(self, cache_path: pathlib.Path) -> bool:
        """Load compiled model and gradient from disk."""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.compiled_model = cached['model']
                self.compiled_grad = cached['grad']
                if self.verbose:
                    print(f"  Loaded compiled functions from cache: {cache_path.name}")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load cache: {e}")
                return False
        return False

    def _precompile_model(self) -> None:
        """Precompile model and gradient for known shapes."""
        # Generate cache key
        theta_shape = (self.theta_dim,)
        if self._sparse_format:
            times_shape = (len(self.observed_data.values), self.observed_data.n_features)
        else:
            times_shape = self.observed_data.shape
        memory_cache_key = (id(self.model), theta_shape, times_shape)

        # Check memory cache first
        if memory_cache_key in SVGD._compiled_cache:
            cached = SVGD._compiled_cache[memory_cache_key]
            self.compiled_model = cached['model']
            self.compiled_grad = cached['grad']
            if self.verbose:
                print(f"  Using cached compiled functions from memory")
            return

        # Check disk cache
        cache_path = self._get_cache_path()
        if self._load_compiled(cache_path):
            # Store in memory cache for future instances
            SVGD._compiled_cache[memory_cache_key] = {
                'model': self.compiled_model,
                'grad': self.compiled_grad
            }
            return

        # Need to compile
        if self.verbose:
            print(f"\nPrecompiling gradient function...")
            print(f"  Theta shape: {theta_shape}, Times shape: {times_shape}")
            print(f"  This may take several minutes for large models...")

        # Create dummy inputs with correct shapes
        dummy_theta = jnp.zeros(theta_shape)

        # JIT compile gradient (use jit without lower/compile so it can be vmapped/pmapped)
        if self.verbose:
            print(f"  JIT compiling gradient...")
        start = time()
        grad_fn = jax.grad(self._log_prob)
        self.compiled_grad = jax.jit(grad_fn)
        # Trigger compilation with dummy call
        _ = self.compiled_grad(dummy_theta)
        if self.verbose:
            print(f"  Gradient JIT compiled in {time() - start:.1g}s")
            print(f"  Precompilation complete!")

        # Save to both caches
        SVGD._compiled_cache[memory_cache_key] = {
            'model': self.compiled_model,
            'grad': self.compiled_grad
        }
        self._save_compiled(cache_path)

    def _create_log_prob_fn_with_regularization(self, regularization_value: float) -> Callable:
        """
        Create log_prob function with specific regularization value.

        This factory method is used for regularization schedules, where the
        regularization value changes per iteration.

        Parameters
        ----------
        regularization_value : float
            Current regularization strength for this iteration

        Returns
        -------
        callable
            Log probability function with signature: theta -> scalar
        """
        return partial(
            self._log_prob_unified,
            nr_moments=self.nr_moments,
            sample_moments=self.sample_moments,
            regularization=regularization_value,
            rewards=self.rewards 
        )

    def _precompile_unified(self, nr_moments: int, sample_moments: jnp.ndarray | None, regularization: float, rewards: jnp.ndarray | None = None) -> Callable:
        """
        Precompile gradient for unified log_prob with given regularization settings.

        Handles caching (both memory and disk) for compiled gradients with different
        regularization parameters.

        Parameters
        ----------
        nr_moments : int
            Number of moments for regularization
        sample_moments : array or None
            Sample moments from data
        regularization : float
            Regularization strength
        rewards : array or None, optional
            Optional reward vector/matrix for multivariate phase-type distributions.
            - 1D array (n_vertices,): Single reward vector
            - 2D array (n_features, n_vertices): Reward matrix where rewards[j, :] is feature j's vector

        Returns
        -------
        compiled_grad : callable
            JIT-compiled gradient function
        """
        # Generate cache key including regularization params and rewards
        # Convert rewards to hashable tuple (JAX arrays aren't hashable)
        if rewards is not None:
            import numpy as np
            rewards_tuple = tuple(np.asarray(rewards).flatten())
        else:
            rewards_tuple = None
        cache_hash = self._get_cache_key_unified(nr_moments, regularization, rewards_tuple)
        if self._sparse_format:
            obs_shape = (len(self.observed_data.values), self.observed_data.n_features)
        else:
            obs_shape = self.observed_data.shape
        memory_cache_key = (id(self.model), self.theta_dim, obs_shape,
                           nr_moments, regularization, rewards_tuple)

        # Check memory cache first
        if memory_cache_key in SVGD._compiled_cache:
            cached = SVGD._compiled_cache[memory_cache_key]
            compiled_grad = cached['grad']
            if self.verbose:
                print(f"  Using cached compiled gradient from memory")
            return compiled_grad

        # Check disk cache
        cache_dir = pathlib.Path.home() / '.phasic_cache'
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"compiled_svgd_{cache_hash}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                compiled_grad = cached['grad']
                if self.verbose:
                    print(f"  Loaded compiled gradient from disk cache: {cache_path.name}")
                # Store in memory cache
                SVGD._compiled_cache[memory_cache_key] = {'grad': compiled_grad}
                return compiled_grad
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load cache: {e}")

        # Need to compile
        if self.verbose:
            print(f"\nPrecompiling gradient function...")
            print(f"  Theta shape: {(self.theta_dim,)}, Times shape: {obs_shape}")
            if regularization > 0:
                print(f"  Moment regularization: λ={regularization}, nr_moments={nr_moments}")
            print(f"  This may take several minutes for large models...")

        # Create log_prob function using partial
        log_prob_fn = partial(
            self._log_prob_unified,
            nr_moments=nr_moments,
            sample_moments=sample_moments,
            regularization=regularization,
            rewards=rewards
        )

        # JIT compile gradient
        start = time()
        grad_fn = jax.grad(log_prob_fn)
        compiled_grad = jax.jit(grad_fn)
        # Trigger compilation with dummy call
        dummy_theta = jnp.zeros((self.theta_dim,))
        _ = compiled_grad(dummy_theta)
        if self.verbose:
            print(f"  Gradient JIT compiled in {time() - start:.1g}s")
            print(f"  Precompilation complete!")

        # Save to both caches
        SVGD._compiled_cache[memory_cache_key] = {'grad': compiled_grad}
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'grad': compiled_grad}, f)
            if self.verbose:
                print(f"  Saved compiled gradient to cache: {cache_path.name}")
        except Exception:
            # Disk caching is best-effort
            pass

        return compiled_grad

    def optimize(self, rewards: jnp.ndarray | None = None, return_history: bool = True) -> SVGD:
        """
        Run SVGD inference with optional moment-based regularization.

        Regularization settings are configured at SVGD initialization via
        regularization and nr_moments parameters.

        Parameters
        ----------
        rewards : np.ndarray, optional
            Reward vector/matrix for multivariate phase-type distributions.
            - 1D array (n_vertices,): Single reward vector for univariate distribution
            - 2D array (n_features, n_vertices): Reward matrix for multivariate distribution
              where each row rewards[j, :] defines the reward vector for feature j
            Must be provided if model requires rewards (multivariate distributions).
            Each reward serves as multiplier of vertex value in trace.
        return_history : bool, default=True
            If True, store particle positions throughout optimization

        Returns
        -------
        SVGD
            Returns self for method chaining.

        Raises
        ------
        NotImplementedError
            If rewards parameter is provided (not yet implemented)

        Examples
        --------
        >>> # Standard SVGD (no regularization)
        >>> model = Graph.pmf_and_moments_from_graph(graph)
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=0.0)
        >>> svgd.optimize()

        >>> # SVGD with moment regularization
        >>> model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=1.0, nr_moments=2)
        >>> svgd.optimize()

        >>> # With custom moments and strong regularization
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=5.0, nr_moments=3)
        >>> svgd.optimize()

        Notes
        -----
        - Supports all JAX transformations: jit, grad, vmap, pmap
        - Supports multi-core parallelization via parallel='vmap'/'pmap'
        - Supports multi-machine distribution via initialize_distributed()
        - Gradient compilation is cached (both memory and disk) for performance
        - All functionality from fit() and fit_regularized() is preserved
        """
        # FIX: Use self.rewards if rewards parameter not provided
        if rewards is None:
            rewards = self.rewards

        # Compute preconditioner (before kernel creation)
        preconditioner_obj = None
        if self.preconditioner_method is not None:
            if isinstance(self.preconditioner_method, (FisherPreconditioner, MomentJacobianPreconditioner)):
                # User provided a pre-built preconditioner
                preconditioner_obj = self.preconditioner_method
                logger.debug("SVGD.optimize: using user-provided %s "
                              "(scaling=%s)", type(preconditioner_obj).__name__,
                              preconditioner_obj.scaling)
            elif self.preconditioner_method in ('jacobian', 'fisher'):
                # Determine dimensionality (learnable dims only if fixed params)
                if self.fixed_mask is not None:
                    n_learnable = int(jnp.sum(self.fixed_mask == 0))
                    learnable_indices = jnp.where(self.fixed_mask == 0)[0]
                    logger.debug("SVGD.optimize: fixed params detected, "
                                  "%d learnable dims (indices=%s)",
                                  n_learnable, learnable_indices)
                else:
                    n_learnable = self.theta_dim
                    learnable_indices = None
                    logger.debug("SVGD.optimize: no fixed params, "
                                  "all %d dims learnable", n_learnable)

                preconditioner_kwargs = dict(
                    model=self.model,
                    observed_data=self.observed_data,
                    theta_dim=n_learnable,
                    param_transform=self.param_transform,
                    rewards=rewards
                )
                if self.preconditioner_method == 'jacobian':
                    preconditioner_obj = MomentJacobianPreconditioner(**preconditioner_kwargs)
                else:
                    preconditioner_obj = FisherPreconditioner(**preconditioner_kwargs)

                # Compute reference point from particle mean
                theta_ref_full = jnp.mean(self.theta_init, axis=0)
                if learnable_indices is not None:
                    theta_ref = theta_ref_full[learnable_indices]
                else:
                    theta_ref = theta_ref_full
                logger.debug("SVGD.optimize: %s reference point (unconstrained) = %s",
                              self.preconditioner_method, theta_ref)

                # For fixed params, wrap model to expand learnable -> full space
                if self.fixed_mask is not None:
                    original_model = preconditioner_obj.model
                    def _model_with_fixed(theta_c, times, rewards=None):
                        if self.param_transform is not None:
                            # theta_c is already in constrained space from preconditioner
                            # We need to reconstruct the full constrained vector
                            full_constrained = self.param_transform(
                                jnp.array(self.fixed_values)
                            )
                        else:
                            full_constrained = jnp.array(self.fixed_values)
                        full_constrained = full_constrained.at[learnable_indices].set(theta_c)
                        return original_model(full_constrained, times, rewards=rewards)
                    preconditioner_obj.model = _model_with_fixed
                    logger.debug("SVGD.optimize: wrapped model for fixed params "
                                  "(fixed_values=%s)", self.fixed_values)

                preconditioner_obj.compute_scaling(theta_ref)
                if self.verbose:
                    print(f"  Preconditioner scaling ({self.preconditioner_method}): "
                          f"{preconditioner_obj.scaling}")
        else:
            logger.debug("SVGD.optimize: no preconditioner configured")

        # Create kernel
        kernel = SVGDKernel(bandwidth=self.bandwidth, preconditioner=preconditioner_obj)

        # Use regularization settings from __init__
        use_regularization = (self.regularization > 0.0 or self.use_regularization_schedule)

        # Run SVGD - split into two paths based on regularization type
        if self.use_regularization_schedule:
            # Dynamic regularization - cannot precompile gradient
            # Gradient is computed on-the-fly each iteration with current regularization
            if self.verbose:
                print(f"\nStarting SVGD inference with regularization schedule...")
                print(f"  Model: parameterized phase-type distribution")
                n_data = len(self.observed_data.values) if self._sparse_format else len(self.observed_data)
                print(f"  Data points: {n_data}")
                print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")
                print(f"  Regularization: dynamic schedule (initial λ = {self.regularization})")
                print(f"  Nr moments: {self.nr_moments}")
                print(f"  Note: Gradient precompilation disabled for schedule flexibility")

            # Create factory that captures rewards parameter
            def log_prob_factory(reg_value):
                return partial(
                    self._log_prob_unified,
                    nr_moments=self.nr_moments,
                    sample_moments=self.sample_moments,
                    regularization=reg_value,
                    rewards=rewards
                )

            results = run_svgd(
                log_prob_fn=None,  # Created dynamically per iteration
                theta_init=self.theta_init,
                n_steps=self.n_iterations,
                learning_rate=self.step_schedule,
                kernel=kernel,
                return_history=return_history,
                verbose=self.verbose,
                progress=self.progress,
                compiled_grad=None,  # Cannot precompile with dynamic regularization
                parallel_mode=self.parallel_mode,
                n_devices=self.n_devices,
                log_prob_fn_factory=log_prob_factory,
                regularization_schedule=self.regularization_schedule,
                lr_scale=self.lr_scale,
                fixed_mask=self.fixed_mask,
                fixed_values=self.fixed_values,
                optimizer=self.optimizer
            )
        else:
            # Static regularization - use current precompiled approach
            # Precompile gradient with caching (if JIT enabled)
            if self.jit_enabled:
                compiled_grad = self._precompile_unified(self.nr_moments, self.sample_moments, self.regularization, rewards)
            else:
                # Create log_prob function using partial (no JIT)
                log_prob_fn = partial(
                    self._log_prob_unified,
                    nr_moments=self.nr_moments,
                    sample_moments=self.sample_moments,
                    regularization=self.regularization,
                    rewards=rewards
                )
                compiled_grad = jax.grad(log_prob_fn)  # Not JIT compiled

            # Create log_prob function for run_svgd
            log_prob_fn = partial(
                self._log_prob_unified,
                nr_moments=self.nr_moments,
                sample_moments=self.sample_moments,
                regularization=self.regularization,
                rewards=rewards
            )

            # Print info
            if self.verbose:
                print(f"\nStarting SVGD inference...")
                print(f"  Model: parameterized phase-type distribution")
                n_data = len(self.observed_data.values) if self._sparse_format else len(self.observed_data)
                print(f"  Data points: {n_data}")
                print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")
                if use_regularization:
                    print(f"  Moment regularization: λ = {self.regularization}")
                    print(f"  Nr moments: {self.nr_moments}")
                else:
                    print(f"  Moment regularization: disabled")

            results = run_svgd(
                log_prob_fn=log_prob_fn,
                theta_init=self.theta_init,
                n_steps=self.n_iterations,
                learning_rate=self.step_schedule,
                kernel=kernel,
                return_history=return_history,
                verbose=self.verbose,
                progress=self.progress,
                compiled_grad=compiled_grad,
                parallel_mode=self.parallel_mode,
                n_devices=self.n_devices,
                log_prob_fn_factory=None,
                regularization_schedule=None,
                lr_scale=self.lr_scale,
                fixed_mask=self.fixed_mask,
                fixed_values=self.fixed_values,
                optimizer=self.optimizer
            )

        # Store results as attributes
        self.particles = results['particles']
        self.theta_mean = results['theta_mean']
        self.theta_std = results['theta_std']

        if return_history:
            self.history = np.array(results['history'])
            self.history_iterations = results['history_iterations']

        self.is_fitted = True

        # Print summary with transformed values if verbose
        if self.verbose:
            print(f"\nSVGD complete!")
            transformed_results = self.get_results()
            print(f"Posterior mean: {transformed_results['theta_mean']}")
            print(f"Posterior std:  {transformed_results['theta_std']}")

        return self


    def get_results(self) -> dict:
        """
        Get inference results as a dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - 'particles': Final posterior samples (in constrained space if using transformation)
            - 'theta_mean': Posterior mean (in constrained space if using transformation)
            - 'theta_std': Posterior standard deviation (in constrained space if using transformation)
            - 'history': Particle evolution (if available, in constrained space if using transformation)
            - 'particles_unconstrained': Particles in unconstrained space (only if using transformation)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before accessing results")

        # Transform particles to constrained space if transformation is active
        if self.param_transform is not None:
            # Transform ALL dimensions (both learnable and fixed) from PHI space to THETA space
            # Fixed dimensions are stored in PHI space (e.g., inv_softplus(1.0) = 0.541)
            # and need to be transformed back to THETA space for reporting
            particles_constrained = jnp.array([self.param_transform(p) for p in self.particles])

            theta_mean = particles_constrained.mean(axis=0)
            theta_std = particles_constrained.std(axis=0)

            # HPD intervals per dimension
            hpd_lower = np.empty(self.theta_dim)
            hpd_upper = np.empty(self.theta_dim)
            for i in range(self.theta_dim):
                hpd_lower[i], hpd_upper[i] = _compute_hpd(
                    np.asarray(particles_constrained[:, i])
                )

            results = {
                'particles': particles_constrained,
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'hpd_lower': hpd_lower,
                'hpd_upper': hpd_upper,
                'particles_unconstrained': self.particles,  # Also return unconstrained
            }

            if self.history is not None:
                # Transform history as well - ALL dimensions need transformation
                history_constrained = jnp.array([[self.param_transform(p) for p in step] for step in self.history])
                results['history'] = history_constrained
                results['history_unconstrained'] = self.history
        else:
            # HPD intervals per dimension
            hpd_lower = np.empty(self.theta_dim)
            hpd_upper = np.empty(self.theta_dim)
            for i in range(self.theta_dim):
                hpd_lower[i], hpd_upper[i] = _compute_hpd(
                    np.asarray(self.particles[:, i])
                )

            results = {
                'particles': self.particles,
                'theta_mean': self.theta_mean,
                'theta_std': self.theta_std,
                'hpd_lower': hpd_lower,
                'hpd_upper': hpd_upper,
            }

            if self.history is not None:
                results['history'] = self.history

        return results

    def map_estimate_from_particles(self, unconstrained: bool = False) -> tuple[list, float]:
        """
        Find the MAP estimate from a set of particles by finding the particle
        with the highest log probability.

        Parameters
        ----------
        unconstrained : bool, default=False
            If False, return constrained (model-space) parameter values.
            If True, return unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        tuple[list, float]
            Tuple of (parameter values as list, log probability).
        """
        n_particles = self.particles.shape[0]

        log_prob_fn = partial(
            self._log_prob_unified,
            # nr_moments=self.nr_moments,
            # sample_moments=self.sample_moments,
            # regularization=self.regularization,
            rewards=self.rewards
        )

        # Compute log probability for each particle
        log_probs = jnp.array([log_prob_fn(self.particles[i]) for i in range(n_particles)])

        # Find the particle with the highest log probability
        map_idx = jnp.argmax(log_probs)

        map_particle = self.particles[map_idx]

        # Transform to constrained space unless unconstrained is requested
        if not unconstrained and self.param_transform is not None:
            map_particle = self.param_transform(map_particle)

        return map_particle.tolist(), log_probs[map_idx].item()


    def map_estimate_with_optimization(self, n_steps: int = 70, step_size: float = 0.01, unconstrained: bool = False) -> tuple[list, float]:
        """
        Refine MAP estimate by starting from the best particle and performing
        gradient ascent on the log probability.

        Parameters
        ----------
        n_steps : int, default=70
            Number of optimization steps.
        step_size : float, default=0.01
            Step size for gradient ascent.
        unconstrained : bool, default=False
            If False, return constrained (model-space) parameter values.
            If True, return unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        tuple[list, float]
            Tuple of (refined parameter values as list, log probability).
        """

        print("Rewards not yet implemented")
        log_prob_fn = partial(
            self._log_prob_unified,
            nr_moments=self.nr_moments,
            sample_moments=self.sample_moments,
            regularization=self.regularization,
            rewards=self.rewards
        )

        # Start with the best particle (in unconstrained space for optimization)
        map_particle, _ = self.map_estimate_from_particles(unconstrained=True)

        # Define gradient of log probability
        grad_log_prob = jax.grad(log_prob_fn)

        # Perform gradient ascent to refine the MAP estimate
        x = map_particle
        for _ in range(n_steps):
            grad = grad_log_prob(x)
            x = x + step_size * grad

        # Transform to constrained space unless unconstrained is requested
        if not unconstrained and self.param_transform is not None:
            x = self.param_transform(x)

        return x.tolist(), log_prob_fn(x).item()


    def plot_posterior(self, true_theta: jnp.ndarray | list | None = None,
                      param_names: list[str] | None = None, bins: int = 20,
                      figsize: tuple[float, float] | None = None,
                      save_path: str | None = None,
                      unconstrained: bool = False, return_fig: bool = False,
                      ci_method: str = 'hpd', ci_level: float = 0.95):
        """
        Plot posterior distributions for each parameter.

        Parameters
        ----------
        true_theta : np.ndarray, optional
            True parameter values (if known) to overlay on plot
        param_names : list of str, optional
            Names for each parameter dimension
        bins : int, default=20
            Number of histogram bins
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.
        return_fig : bool, default=True
            If True, return (fig, axes). If False, call plt.show() instead.
        ci_method : str, default='hpd'
            Method for credible intervals shown as a shaded region on each
            histogram.

            - ``'hpd'``: Highest Posterior Density interval — the shortest
              interval containing ``ci_level`` fraction of posterior samples.
              Computed by sorting the particles and sliding a window of
              ``ceil(n * ci_level)`` samples to find the narrowest span.
              Better centred on the mode for skewed posteriors.
            - ``'percentile'``: Equal-tailed percentile interval using
              symmetric quantiles.

        ci_level : float, default=0.95
            Credible level for the shaded interval (e.g. 0.95 for 95%).

        Returns
        -------
        tuple[matplotlib.figure.Figure, numpy.ndarray]
            Matplotlib figure and axes array (only if return_fig=True).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if ci_method not in ('hpd', 'percentile'):
            raise ValueError(f"ci_method must be 'hpd' or 'percentile', got '{ci_method}'")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate particle representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            particles = results['particles']
            theta_mean = results['theta_mean']
            space_label = ""
        else:
            particles = results.get('particles_unconstrained', results['particles'])
            theta_mean = jnp.mean(particles, axis=0)
            space_label = " (unconstrained)"

        n_params = self.theta_dim

        # Determine subplot layout
        if n_params == 1:
            nrows, ncols = 1, 1
            figsize = figsize or None
        elif n_params == 2:
            nrows, ncols = 1, 2
            figsize = figsize or (8, 3)
        else:
            ncols = min(3, n_params)
            nrows = (n_params + ncols - 1) // ncols
            figsize = figsize or (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        pct = int(ci_level * 100)
        ci_label = f"HPD {pct}%" if ci_method == 'hpd' else f"CI {pct}%"

        for i in range(n_params):
            ax = axes[i]

            # Histogram of posterior samples
            ax.hist(particles[:, i], bins=bins, alpha=0.7, density=True,
                   edgecolor='black', label='Posterior')

            # Credible interval shading
            samples_i = np.asarray(particles[:, i])
            if ci_method == 'hpd':
                ci_lo, ci_hi = _compute_hpd(samples_i, alpha=ci_level)
            else:
                lo_pct = (1 - ci_level) / 2 * 100
                hi_pct = (1 + ci_level) / 2 * 100
                ci_lo = float(np.percentile(samples_i, lo_pct))
                ci_hi = float(np.percentile(samples_i, hi_pct))
            ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='steelblue',
                       label=f'{ci_label} [{ci_lo:.3g}, {ci_hi:.3g}]')

            # Posterior mean
            ax.axvline(theta_mean[i],
                       color=black_or_white,
                       linestyle='--',
                       label=f'Mean = {theta_mean[i]:.3g}')

            # True value (if provided)
            if true_theta is not None:
                true_val = jnp.array(true_theta)[i]
                ax.axvline(true_val, color='magenta', linestyle='--',
                           label=f'True = {true_val:.3g}')

            # Labels
            param_name = param_names[i] if param_names else rf"$\theta_{i}$"
            ax.set_xlabel(param_name + space_label)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {param_name}')
            ax.legend()
            # ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if return_fig:
            return fig, axes
        else:
            plt.show()


    def plot_trace(self, param_names: list[str] | None = None,
                   figsize: tuple[float, float] | None = None,
                   skip: int = 0, max_particles: int | None = None,
                   save_path: str | None = None, unconstrained: bool = False,
                   hide_fixed: bool = True,
                   return_fig: bool = False):
        """
        Plot trace plots showing particle evolution over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each parameter dimension
        figsize : tuple, optional
            Figure size (width, height)
        skip : int, optional
            Number of initial iterations to skip. Defaults to 0.
        max_particles : int, optional
            Max number of particles to plot. Defaults to all particles.
        save_path : str, optional
            Path to save the plot
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.
        return_fig : bool, default=True
            If True, return (fig, axes). If False, call plt.show() instead.

        Returns
        -------
        tuple[matplotlib.figure.Figure, numpy.ndarray]
            Matplotlib figure and axes array (only if return_fig=True).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.history is None:
            raise RuntimeError("History not available. Call fit(return_history=True) first")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        n_params = self.theta_dim
        param_indices = list(range(n_params))

        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_params) if not self.fixed_mask[i]]   
            n_params = len(param_indices)

        # Get appropriate history representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            history = results.get('history', self.history)
            theta_mean = results['theta_mean']
            space_label = ""
        else:
            history = results.get('history_unconstrained', self.history)
            theta_mean = jnp.mean(history[-1], axis=0)
            space_label = " (unconstrained)"

        cols = int(n_params > 1) + 1
        rows = n_params // 2 + n_params % 2

        # Determine subplot layout
        if n_params > 1:
            figsize = figsize or (min(14, 3.5 * cols), min(12, 2.7 * rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Convert history to array: (n_snapshots, n_particles, theta_dim)
        history_array = jnp.stack(history)
        n_snapshots = len(history)

        # for i in range(n_params):
            # ax = axes[i]
        for i, ax in zip_longest(param_indices, axes, fillvalue=None):

            if i is None and ax is not None:
                ax.axis('off')
                continue

            # Plot each particle's trajectory
            max_plotted = self.n_particles if max_particles is None else max_particles
            for p in range(max_plotted):  # Plot first 10 particles
                y = history_array[:, p, i]
                x = np.arange(y.size)
                ax.plot(x[skip:], y[skip:], alpha=1, linewidth=0.5)

            # Plot mean trajectory
            mean_trajectory = jnp.mean(history_array[:, :, i], axis=1)
            y = mean_trajectory
            x = np.arange(y.size)

            ax.plot(x[skip:], y[skip:], 
                    color=black_or_white, 
                    linestyle='dashed', label=f'Mean = {theta_mean[i]:.3g}')

            # Labels
            param_name = param_names[i] if param_names else rf"$\theta_{i}$"
            ax.set_xlabel('SVGD Iteration')
            ax.set_ylabel(param_name + space_label)
            ax.set_title(f'Trace: {param_name}')
            ax.legend()
            # ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if return_fig:
            return fig, axes
        else:
            plt.show()

    def plot_convergence(self, figsize: tuple[float, float] = (7, 3),
                         save_path: str | None = None, skip: int = 0,
                         unconstrained: bool = False, 
                         hide_fixed: bool = True,
                         return_fig: bool = False):
        """
        Plot convergence diagnostics showing mean and std over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        figsize : tuple, default=(7, 4)
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        skip : int, optional
            Number of initial iterations to skip. Defaults to 0.
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.
        return_fig : bool, default=True
            If True, return (fig, axes). If False, call plt.show() instead.

        Returns
        -------
        tuple[matplotlib.figure.Figure, numpy.ndarray]
            Matplotlib figure and axes array (only if return_fig=True).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.history is None:
            raise RuntimeError("History not available. Call fit(return_history=True) first")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate history representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            history = results.get('history', self.history)
            space_label = ""
        else:
            history = results.get('history_unconstrained', self.history)
            space_label = " (unconstrained)"

        n_params = self.theta_dim
        param_indices = list(range(n_params))

        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_params) if not self.fixed_mask[i]]   
            n_params = len(param_indices)

        # Convert history to array
        history_array = jnp.stack(history)

        # Compute mean and std at each snapshot
        mean_over_time = jnp.mean(history_array, axis=1)  # (n_snapshots, theta_dim)
        std_over_time = jnp.std(history_array, axis=1)    # (n_snapshots, theta_dim)

        # Get iteration numbers for x-axis
        iterations = self.history_iterations if self.history_iterations is not None else range(len(history))

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, axes[1] = axes

        # Plot 1: Mean convergence
        for i in param_indices:
            param_name = rf"$\theta_{i}$"
            x, y = iterations, mean_over_time[:, i]
            ax1.plot(x[skip:], y[skip:], label=param_name, )

        ax1.set_xlabel('SVGD Iteration')
        ax1.set_ylabel('Posterior Mean' + space_label)
        ax1.set_title('Mean')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Std convergence
        for i in param_indices:
            param_name = rf"$\theta_{i}$"
            x, y = iterations, std_over_time[:, i]
            axes[1].plot(x[skip:], y[skip:], label=param_name, )

        axes[1].set_xlabel('SVGD Iteration')
        axes[1].set_ylabel('Posterior Std' + space_label)
        axes[1].set_title('Std')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if return_fig:
            return fig, axes
        else:
            plt.show()

    def plot_ci(self, figsize: tuple[float, float] = (7, 3),
                save_path: str | None = None, skip: int = 0,
                unconstrained: bool = False,
                true_theta: jnp.ndarray | list | None = None,
                ci: float = 0.95, alpha: float = 0.2,
                target: jnp.ndarray | list | None = None,
                median: bool = False, return_fig: bool = False,
                ci_method: str = 'hpd',
                hide_fixed: bool = True):
        """
        Plot mean parameter trajectory with credible interval ribbon.

        Shows the posterior mean over iterations with a shaded region representing
        the specified credible interval (default 95%).

        Parameters
        ----------
        figsize : tuple, default=(7, 3)
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        skip : int, default=0
            Number of initial iterations to skip
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
        true_theta : np.ndarray, optional
            True parameter values to overlay as horizontal lines
        ci : float, default=0.95
            Credible interval level (e.g., 0.95 for 95% CI)
        alpha : float, default=0.2
            Transparency of the CI ribbon
        target : np.ndarray, optional
            Target parameter value to overlay as horizontal line
        median : bool, default=False
            If True, plot the median trajectory as a dashed line
        return_fig : bool, default=True
            If True, return (fig, ax). If False, call plt.show() instead.
        ci_method : str, default='hpd'
            Method for credible intervals.

            - ``'hpd'``: Highest Posterior Density interval — the shortest
              interval containing ``ci`` fraction of posterior samples at each
              iteration. Computed by sorting the particles and sliding a
              window of ``ceil(n * ci)`` samples to find the narrowest span.
            - ``'percentile'``: Equal-tailed percentile interval using
              symmetric quantiles.
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.              

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Matplotlib figure and axes (only if return_fig=True).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.history is None:
            raise RuntimeError("History not available. Call fit(return_history=True) first")

        if ci_method not in ('hpd', 'percentile'):
            raise ValueError(f"ci_method must be 'hpd' or 'percentile', got '{ci_method}'")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate history representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            history = results.get('history', self.history)
            space_label = ""
        else:
            history = results.get('history_unconstrained', self.history)
            space_label = " (unconstrained)"

        # Convert history to array: (n_iterations, n_particles, theta_dim)
        history_array = jnp.stack(history)

        n_params = self.theta_dim
        param_indices = list(range(n_params))
        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_params) if not self.fixed_mask[i]]   
            n_params = len(param_indices)

        mean_over_time = jnp.mean(history_array, axis=1)  # (n_iterations, theta_dim)

        n_iters = history_array.shape[0]
        n_params = history_array.shape[2]

        if ci_method == 'hpd':
            # Compute HPD interval per iteration and dimension
            lower_ci = np.empty((n_iters, n_params))
            upper_ci = np.empty((n_iters, n_params))
            for t in range(n_iters):
                for d in range(n_params):
                    lower_ci[t, d], upper_ci[t, d] = _compute_hpd(
                        np.asarray(history_array[t, :, d]), alpha=ci
                    )
            ci_label = "HPD"
        else:
            lower_pct = (1 - ci) / 2 * 100
            upper_pct = (1 + ci) / 2 * 100
            lower_ci = jnp.percentile(history_array, lower_pct, axis=1)
            upper_ci = jnp.percentile(history_array, upper_pct, axis=1)
            ci_label = "CI"

        param_indices = list(range(n_params))
        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_params) if not self.fixed_mask[i]]   

        if median:
            median_over_time = jnp.median(history_array, axis=1)  # (n_iterations, theta_dim)

        # Get iteration numbers for x-axis
        iterations = self.history_iterations if self.history_iterations is not None else range(len(history))
        iterations = list(iterations)  # Convert to list for slicing

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        colors = plt.cm.tab10.colors

        if target is not None:
            ax.axhline(target, color='C1', linestyle='--', alpha=0.7)

        for i in param_indices:
            param_name = rf"$\theta_{i}$"
            color = colors[i % len(colors)]
            x = iterations[skip:]
            y_mean = mean_over_time[skip:, i]
            y_lower = lower_ci[skip:, i]
            y_upper = upper_ci[skip:, i]

            # Plot mean line
            ax.plot(x, y_mean, label=param_name, color=color)

            # Plot CI ribbon
            ax.fill_between(x, y_lower, y_upper, alpha=alpha, color=color)

            # Plot median line if requested
            if median:
                y_median = median_over_time[skip:, i]
                ax.plot(x, y_median, linestyle='--', color=color, alpha=0.7)

        # Add true theta lines if provided
        if true_theta is not None:
            true_theta = jnp.atleast_1d(jnp.asarray(true_theta))
            for i, val in enumerate(true_theta):
                color = colors[i % len(colors)]
                ax.axhline(val, color=color, linestyle='--', alpha=0.7)

        ax.set_xlabel('SVGD Iteration')
        ax.set_ylabel(f'Posterior Mean ± {int(ci*100)}% {ci_label}' + space_label)
        ax.set_title(f'Parameter Convergence with {int(ci*100)}% {ci_label}')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if return_fig:
            return ax
        else:
            plt.show()


    # def plot_svgd_posterior_1d(self, particles=None, true_params=None, obs_stats=None,
    #                         map_est=None,
    #                         ax=None, title="SVGD Posterior Approximation",
    #                         unconstrained=False):
    #     """
    #     Plot 1D posterior approximation (SVGD particle distribution)

    #     Args:
    #         particles: shape (n_particles, 1) array of SVGD particles
    #         true_params: optional true parameter value for comparison
    #         title: plot title
    #         unconstrained : bool, default=False
    #             If False, show constrained (model-space) parameter values.
    #             If True, show unconstrained (optimization-space) values.
    #             Only relevant when using parameter transformations.
    #     """
    #     if ax is None:
    #         plt.figure(figsize=(8, 6))
    #         ax = plt.gca()

    #     if particles is None:
    #         # Get appropriate particle representation
    #         results = self.get_results()
    #         if not unconstrained or self.param_transform is None:
    #             if unconstrained and self.param_transform is None:
    #                 raise ValueError(
    #                     "unconstrained=True has no effect when no parameter transformation is used. "
    #                     "Either set unconstrained=False, or use positive_params=True / param_transform "
    #                     "to enable parameter transformation."
    #                 )
    #             particles = results['particles']
    #         else:
    #             particles = results.get('particles_unconstrained', results['particles'])

    #     # Extract 1D values
    #     x = particles.flatten()
        
    #     # Plot histogram of particles
    #     ax.hist(x, bins=30, density=True, alpha=0.4, label='Particle histogram')
        
    #     # # Plot KDE of posterior
    #     # kde = gaussian_kde(x)
    #     # xx = np.linspace(min(x), max(x), 1000)
    #     # ax.plot(xx, kde(xx), color='orange', lw=2, label='KDE posterior')

    #     # Fit curve
    #     def gengamma_curve_fit(data):
    #         a, c, loc, scale = gengamma.fit(data, floc=0)
    #         x = np.linspace(data.max(), data.max(), 1000)
    #         y = gengamma.pdf(x, a, c, loc=0, scale=scale)
    #         return x, y

    #     ax.plot(*gengamma_curve_fit(x), color='orange', lw=2, label='Generalized gamma fit')

    #     # Add true parameter if provided
    #     if true_params is not None:
    #         ax.axvline(true_params, color='hotpink', linestyle='--', 
    #                 label=f'True value: {true_params:.2g}')
            
    #     # Add data statistics
    #     if obs_stats is not None:
    #         ax.axvline(obs_stats, color='magenta',
    #                 label=f'Observed value: {obs_stats:.2g}')    
    #     if map_est is not None:
    #         ax.axvline(map_est, color='orange', linestyle='dashed',
    #                 label=f'MAP value: {map_est:.2g}')       
        
    #     ax.set_title(title)
    #     ax.set_xlabel('Parameter value')
    #     ax.set_ylabel('Density')
    #     ax.legend()
    #     sns.despine(ax=ax)


    def check_convergence(self, every: int = 1, text: str | list[str] | None = None, param_indices: list[int] | None = None) -> None:
        """Monitor convergence of SVGD by tracking statistics for n-dimensional parameters."""
        mean_params = []
        std_params = []
        log_probs = []
        
        log_p_fn, data = self._log_prob_fn, self.data
        particle_history = self.history  # Shape: (n_iterations, n_particles, n_dims)

        n_dims = particle_history.shape[2]
        
        # If no specific parameters selected, use first few
        if param_indices is None:
            param_indices = list(range(min(3, n_dims)))  # Show up to 3 parameters
        
        # Validate indices
        param_indices = [idx for idx in param_indices if idx < n_dims]
        
        def scale_labels(ax, every):
            """Scale x-ticks to match parameter values"""
            vals = ax.get_xticks()[1:-1]
            labels = (vals * every).astype(int)
            ax.set_xticks(vals, labels=labels)

        for i in range(particle_history.shape[0]):
            particles = particle_history[i, :, :]  # Shape: (n_particles, n_dims)
            # track parameter statistics
            mean_params.append(np.mean(particles, axis=0))
            std_params.append(np.std(particles, axis=0))
            # track average log probability
            avg_log_p = np.mean([log_p_fn(data, p) for p in particles])
            log_probs.append(avg_log_p)
        
        if text is not None:
            fig = plt.figure(figsize=(10, 4))
            gs = GridSpec(2, 3, figure=fig, height_ratios=(4, 1))
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            if type(text) is str:
                text = [text]
                text_ax = [fig.add_subplot(gs[1, :])]
            else:
                text_ax = [
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1]),
                    fig.add_subplot(gs[1, 2])
                ]
            [ax.set_axis_off() for ax in text_ax]
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

        # Plot mean parameters (selected indices only)
        for i, param_idx in enumerate(param_indices):
            ax1.plot([p[param_idx] for p in mean_params], label=f'Param {param_idx}')
        ax1.set_title('Parameter Means')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value')
        ax1.legend()
        scale_labels(ax1, every)

        # Plot parameter standard deviations (selected indices only)
        for i, param_idx in enumerate(param_indices):
            ax2.plot([p[param_idx] for p in std_params], label=f'Param {param_idx}')
        ax2.set_title('Parameter Standard Deviations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Value')
        ax2.legend()
        scale_labels(ax2, every)

        # Plot log probabilities
        ax3.plot(log_probs)
        ax3.set_title('Average Log Probability')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Log Prob')
        scale_labels(ax3, every)
        
        if text is not None:
            for i, ax in enumerate(text_ax):
                ax.text(0, 0.9, text[i], fontsize=10,
                            #  horizontalalignment='left',
                            verticalalignment='top',
                            fontname='monospace', 
                            #  traansform=ax.transAxes,
                            # bbox=dict(facecolor='magenta', alpha=0.5)
                            )
        plt.tight_layout()


    def estimate_hdr(self, alpha: float = 0.95) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Estimate the Highest Density Region (HDR) from particles.

        Parameters
        ----------
        alpha : float, default=0.95
            Coverage probability (e.g., 0.95 for 95% HDR).

        Returns
        -------
        hdr_particles : array
            Particles that are within the HDR.
        threshold : float
            The log probability threshold that defines the HDR.
        """

        log_prob_fn = partial(
            self._log_prob_unified,
            nr_moments=self.nr_moments,
            sample_moments=self.sample_moments,
            regularization=self.regularization,
            rewards=self.rewards 
        )  

        n_particles = self.particles.shape[0]
        
        # Compute log probability for each particle
        # log_probs = jnp.array([log_prob_fn(particles[i]) for i in range(n_particles)])
        log_probs = vmap(log_prob_fn)(self.particles)

        # Sort particles by log probability (descending)
        sorted_indices = jnp.argsort(-log_probs)
        sorted_log_probs = log_probs[sorted_indices]
        
        # Find the log probability threshold for the HDR
        n_hdr = int(n_particles * alpha)
        threshold = sorted_log_probs[n_hdr-1]
        
        # Get particles in the HDR
        hdr_mask = log_probs >= threshold
        hdr_particles = self.particles[hdr_mask]
        
        return hdr_particles, threshold


    def plot_hdr(self, alphas: list[float] = [0.95, 0.5], idx: list[int] | None = None,
                    figsize: tuple[float, float] = (5, 4), hexgrid: bool = True,
                    trim: bool = True, n: int = 15, margin: float = 0.1,
                    xlim: tuple[float, float] | None = None,
                    ylim: tuple[float, float] | None = None,
                    palette: str = 'viridis', pad: int = 2,
                    unconstrained: bool = False, return_fig: bool = False,
                    hide_fixed: bool = True,                    
                    show_hpd: bool = False, hpd_alpha: float = 0.95):
        """Plot 2D highest-density region with optional marginal HPD bands.

        Displays a hex-grid log-likelihood heatmap and KDE-based HDR contours
        for two selected parameter dimensions.

        Parameters
        ----------
        alphas : list of float, default=[0.95, 0.5]
            HDR contour levels (fraction of mass enclosed).
        idx : list of int, default=None
            Indices of the two parameter dimensions to plot.
            Defaults to first two non-fixed parameters.
        figsize : tuple, default=(5, 4)
            Figure size (width, height).
        hexgrid : bool, default=True
            Whether to show the log-likelihood hex-grid heatmap.
        trim : bool, default=True
            Clip axes to the hex-grid extent.
        n : int, default=15
            Approximate number of hexagons along the shorter axis.
        margin : float, default=0.1
            Fractional margin around particle range.
        xlim, ylim : tuple, optional
            Manual axis limits.
        palette : str, default='viridis'
            Colormap for the hex-grid heatmap.
        pad : int, default=2
            Extra hex rows/columns beyond the data range.
        unconstrained : bool, default=False
            Show unconstrained (optimization-space) values.
        return_fig : bool, default=False
            If True, return the axes object instead of calling plt.show().
        show_hpd : bool, default=False
            If True, overlay marginal HPD intervals as translucent
            vertical and horizontal bands.  The HPD interval is the
            shortest contiguous interval containing ``hpd_alpha`` fraction
            of the marginal particle samples (computed by sorting the
            samples and sliding a window of ``ceil(n * hpd_alpha)``
            elements to find the narrowest span).
        hpd_alpha : float, default=0.95
            Credible level for the marginal HPD bands (only used when
            ``show_hpd=True``).
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.            
        """

        def hex_grid(x_min, x_max, y_min, y_max, n, aspect=1.0, flat_topped=False, pad=1):
            """Generate hex grid midpoints that fill the bounding box with equal-edged hexagons.
            
            Args:
                x_min, x_max, y_min, y_max: Bounding box
                n: Approximate number of hexagons along the shorter axis
                aspect: Aspect ratio (data_x / data_y) / (canvas_x / canvas_y)
                flat_topped: Hexagon orientation
                pad: Extra rows/columns to add beyond bounding box
            
            Returns:
                points: Nx2 array of midpoints
                size: Computed hex size (center to vertex)
                padded_bounds: (x_min_padded, x_max_padded, y_min_padded, y_max_padded)
            """
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Determine which axis is "shorter" in display units
            effective_x = x_range / aspect
            effective_y = y_range
            
            if flat_topped:
                if effective_x <= effective_y:
                    size = x_range / (1.5 * (n - 1) + 2) / aspect
                else:
                    size = y_range / (np.sqrt(3) * (n - 0.5))
                dx = 1.5 * size * aspect
                dy = np.sqrt(3) * size
            else:
                if effective_x <= effective_y:
                    size = x_range / (np.sqrt(3) * (n - 0.5)) / aspect
                else:
                    size = y_range / (1.5 * (n - 1) + 2)
                dx = np.sqrt(3) * size * aspect
                dy = 1.5 * size
            
            # Extend grid beyond bounds by pad rows/columns
            x_pad = dx * pad
            y_pad = dy * pad
            
            x_min_padded = x_min - x_pad
            x_max_padded = x_max + x_pad
            y_min_padded = y_min - y_pad
            y_max_padded = y_max + y_pad
            
            x_min_padded = max(x_min_padded, 0)
            y_min_padded = max(y_min_padded, 0)

            x = np.arange(x_min_padded, x_max_padded + dx, dx)
            y = np.arange(y_min_padded, y_max_padded + dy, dy)
            
            xx, yy = np.meshgrid(x, y)
            
            if flat_topped:
                yy[:, 1::2] += dy / 2
            else:
                xx[1::2] += dx / 2
            
            points = np.column_stack((xx.ravel(), yy.ravel()))
            padded_bounds = (x_min_padded, x_max_padded, y_min_padded, y_max_padded)
            return points, size, padded_bounds

        def hex_vertices(cx, cy, size, aspect=1.0, flat_topped=False):
            """Get vertices for a single hexagon with equal edges."""
            angles = np.linspace(0, 2 * np.pi, 7)
            if not flat_topped:
                angles += np.pi / 6
            x = cx + size * np.cos(angles) * aspect
            y = cy + size * np.sin(angles)
            return np.column_stack((x, y))

        def heatmap_hexagons(ax, points, z, size, aspect=1.0, flat_topped=False, **kwargs):
            """Draw hexagonal heatmap with touching hexagons."""
            hexagons = [hex_vertices(p[0], p[1], size, aspect, flat_topped) for p in points]
            collection = PolyCollection(hexagons, array=np.array(z), **kwargs)
            return collection

        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            full_history = results.get('history', self.history)
            space_label = ""
        else:
            full_history = results.get('history_unconstrained', self.history)
            space_label = " (unconstrained)"

        particles = full_history[-1]

        n_dims = particles.shape[1]
        param_indices = list(range(n_dims))

        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_dims) if not self.fixed_mask[i]]   
            n_dims = len(param_indices)

        if idx is None:
            idx = param_indices[:2]

        log_prob_fn = partial(
            self._log_prob_unified,
            nr_moments=self.nr_moments,
            sample_moments=self.sample_moments,
            regularization=self.regularization,
            rewards=self.rewards
        )

        if max(idx) >= n_dims:
            raise ValueError(f"Index {max(idx)} exceeds parameter dimension {n_dims}")

        # Get particles in display space for grid construction
        # When unconstrained=False and we have a transform, use constrained particles
        if not unconstrained and self.param_transform is not None:
            particles_display = results['particles']  # Already transformed
            theta_mean_display = jnp.mean(particles_display, axis=0)
            # Need inverse transform for log_prob evaluation
            inv_transform = _inverse_softplus
        else:
            particles_display = self.particles
            theta_mean_display = jnp.mean(self.particles, axis=0)
            inv_transform = lambda x: x  # Identity

        # Determine data limits from display-space particles
        if xlim is None:
            x_min, x_max = particles_display[:, idx[0]].min(), particles_display[:, idx[0]].max()
            _margin = (x_max - x_min) * margin
            xlim = (x_min - _margin, x_max + _margin)
        else:
            x_min, x_max = xlim

        if ylim is None:
            y_min, y_max = particles_display[:, idx[1]].min(), particles_display[:, idx[1]].max()
            _margin = (y_max - y_min) * margin
            ylim = (y_min - _margin, y_max + _margin)
        else:
            y_min, y_max = ylim

        # Compute aspect ratio to make hexagons appear with equal edges
        x_range = x_max - x_min
        y_range = y_max - y_min
        data_aspect = x_range / y_range
        canvas_aspect = figsize[0] / figsize[1]
        aspect = data_aspect / canvas_aspect

        # generate sparse hex grid with padding and evaluate log probability
        # at hex midpoints to find approximate log likelihood mode
        points, size, padded_bounds = hex_grid(x_min, x_max, y_min, y_max, n=5, aspect=aspect, pad=pad)
        # Build grid params in display space
        grid_params = jnp.tile(theta_mean_display, (len(points), 1))
        grid_params = grid_params.at[:, idx[0]].set(points[:, 0])
        grid_params = grid_params.at[:, idx[1]].set(points[:, 1])
        # Inverse transform for log_prob evaluation (log_prob_fn expects unconstrained)
        grid_params_unconstrained = inv_transform(grid_params)
        z_flat = vmap(log_prob_fn)(grid_params_unconstrained)

        # adjust limits to center likelihood mode
        max_idx = jnp.argmax(z_flat)
        mode_x, mode_y = points[max_idx]
        xlim = (min(xlim[0], mode_x - (xlim[1] - mode_x)/2),
                 max(xlim[1], mode_x + (mode_x - xlim[0])/2))
        ylim = (min(ylim[0], mode_y - (ylim[1] - mode_y)/2),
                 max(ylim[1], mode_y + (mode_y - ylim[0])/2))
        x_min, x_max = xlim
        y_min, y_max = ylim

        # repeat with full grid
        points, size, padded_bounds = hex_grid(x_min, x_max, y_min, y_max, n=n, aspect=aspect, pad=pad)
        # Build grid params in display space
        grid_params = jnp.tile(theta_mean_display, (len(points), 1))
        grid_params = grid_params.at[:, idx[0]].set(points[:, 0])
        grid_params = grid_params.at[:, idx[1]].set(points[:, 1])
        # Inverse transform for log_prob evaluation
        grid_params_unconstrained = inv_transform(grid_params)
        z_flat = vmap(log_prob_fn)(grid_params_unconstrained)


        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize)

        collection = heatmap_hexagons(
            ax, points, z_flat, size, aspect=aspect,
            cmap=palette,
            edgecolors='none',
            linewidths=0
        )

        if hexgrid:
            ax.add_collection(collection)

        # Plot particles
        ax.scatter(particles[:, idx[0]], particles[:, idx[1]], 
                color='magenta' if hexgrid else None,
                edgecolor='none', alpha=0.5, s=20)

        # Mark max grid point
        max_idx = jnp.argmax(z_flat)
        ax.scatter(points[max_idx, 0], points[max_idx, 1], 
                color='black' if hexgrid else black_or_white, 
                s=70, marker='x', alpha=1, label='Max grid logL')

        # Mark MAP estimate
        map_particle, _ = self.map_estimate_from_particles(unconstrained=unconstrained)
        if len(map_particle) > max(idx):
            ax.scatter(map_particle[idx[0]], map_particle[idx[1]], 
                    color='black' if hexgrid else black_or_white, 
                    s=70, marker='+', alpha=1, label='MAP estimate')

        # logL contours        
        from scipy.interpolate import griddata
        x_min_p, x_max_p, y_min_p, y_max_p = padded_bounds
        xi = np.linspace(x_min_p, x_max_p, 150)
        yi = np.linspace(y_min_p, y_max_p, 150)
        Xi, Yi = np.meshgrid(xi, yi)
        # Zi = griddata(points, np.array(z_flat), (Xi, Yi), method='cubic')
        # levels = []
        # label_map = {}
        # for alpha in sorted(alphas, reverse=True):
        #     _, threshold = self.estimate_hdr(alpha)
        #     level = threshold.item()
        #     levels.append(level)
        #     label_map[level] = f'{alpha:.0%}'
        # cont = ax.contour(Xi, Yi, Zi, levels=levels,
        #                     colors='black' if hexgrid else black_or_white,
        #                     alpha=0.7)
        # ax.clabel(cont, inline=True , fmt=label_map, fontsize=9)


        # HDR contours based on particle density (KDE)
        particles_2d = np.column_stack([particles[:, idx[0]], particles[:, idx[1]]])
        kde = gaussian_kde(particles_2d.T)

        # Evaluate KDE on grid
        positions = np.vstack([Xi.ravel(), Yi.ravel()])
        Zi_kde = kde(positions).reshape(Xi.shape)

        # Compute HDR thresholds from particle densities
        kde_at_particles = kde(particles_2d.T)
        kde_levels = []
        kde_label_map = {}
        for alpha in sorted(alphas, reverse=True):
            # Find density threshold that contains alpha fraction of particles
            sorted_densities = np.sort(kde_at_particles)[::-1]
            n_hdr = int(len(kde_at_particles) * alpha)
            threshold = sorted_densities[min(n_hdr - 1, len(sorted_densities) - 1)]
            kde_levels.append(threshold)
            kde_label_map[threshold] = f'{alpha:.0%}'

        linestyles = ['solid', 'dashed'] + ['dotted']*10
        kde_cont = ax.contour(Xi, Yi, Zi_kde, levels=kde_levels,
                              colors='black' if hexgrid else black_or_white,
                              linestyles=linestyles[len(kde_levels):],
                              alpha=0.7)
        ax.clabel(kde_cont, inline=True, fmt=kde_label_map, fontsize=9)

        # Marginal HPD intervals as shaded bands
        if show_hpd:
            pct = int(hpd_alpha * 100)
            x_lo, x_hi = _compute_hpd(
                np.asarray(particles_display[:, idx[0]]), alpha=hpd_alpha
            )
            y_lo, y_hi = _compute_hpd(
                np.asarray(particles_display[:, idx[1]]), alpha=hpd_alpha
            )
            band_color = 'white' if hexgrid else 'steelblue'
            ax.axvspan(x_lo, x_hi, alpha=0.15, color=band_color,
                       label=f'HPD {pct}%')
            ax.axhspan(y_lo, y_hi, alpha=0.15, color=band_color)

        # Clip to xlim/ylim to remove ragged edges
        if trim:
            x, y = zip(*points)
            ax.set_xlim((min(x), max(x)))
            ax.set_ylim((min(y), max(y)))

        ax.set_xlabel(rf"$\theta_{idx[0]}$" + space_label)
        ax.set_ylabel(rf"$\theta_{idx[1]}$" + space_label)


        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                fancybox=True, ncol=2, labelcolor=black_or_white)
        leg = ax.get_legend()
        for i in range(len(leg.legend_handles)):
            leg.legend_handles[i].set_facecolor(black_or_white)
            leg.legend_handles[i].set_edgecolor(black_or_white)

        if hexgrid:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(collection, cax=cax, label='Log likelihood')
            cb.outline.set_visible(False)

            ax.grid(False)

        if return_fig:
            return ax
        else:
            plt.show()


    # def plot_parameter_matrix(self, true_params=None, max_params=6, figsize=(12, 10)):
    #     """
    #     Plot ...

    #     """
    #     params = locals().copy()
    #     del params['self']
    #     svgd_plots.plot_parameter_matrix(self.particles, **params)


    # def plot_parameter_matrix(self, true_params=None, max_params=6, figsize=(12, 10)):
    #     """
    #     Create a matrix plot showing pairwise relationships between parameters
        
    #     Args:
    #         particles: Array of shape (n_particles, n_dims)
    #         true_params: optional true parameter values
    #         max_params: maximum number of parameters to show
    #         figsize: figure size
    #     """
    #     n_dims = self.particles.shape[1]
    #     n_show = min(max_params, n_dims)
        
    #     fig, axes = plt.subplots(n_show, n_show, figsize=figsize)
    #     if n_show == 1:
    #         axes = np.array([[axes]])
    #     elif axes.ndim == 1:
    #         axes = axes.reshape(1, -1)
        
    #     for i in range(n_show):
    #         for j in range(n_show):
    #             ax = axes[i, j]
                
    #             if i == j:
    #                 # Diagonal: show 1D marginal distribution
    #                 self.plot_svgd_posterior_1d(
    #                     self.particles[:, i],
    #                     true_params=true_params[i] if true_params is not None and len(true_params) > i else None,
    #                     ax=ax,
    #                     title=f"Parameter {i}"
    #                 )
    #             else:
    #                 # Off-diagonal: show 2D scatter plot
    #                 ax.scatter(self.particles[:, j], self.particles[:, i], alpha=0.5, s=5, edgecolor='none')
                    
    #                 if true_params is not None and len(true_params) > max(i, j):
    #                     ax.scatter(true_params[j], true_params[i], color='magenta', s=50, marker='*', 
    #                             label='True value')
                    
    #                 ax.set_xlabel(f'Parameter {j}')
    #                 ax.set_ylabel(f'Parameter {i}')
                    
    #             # Remove ticks for cleaner look
    #             if i < n_show - 1:
    #                 ax.set_xlabel('')
    #             if j > 0:
    #                 ax.set_ylabel('')
        
    #     plt.suptitle(f'Parameter Matrix Plot (showing {n_show}/{n_dims} parameters)', fontsize=14)
    #     plt.tight_layout()
    #     return fig


    # def animate_parameter_pairs(self, param_pairs=None, true_params=None, figsize=(15, 5), save_as_gif=None):    
    #     """
    #     Plot ...

    #     """
    #     params = locals().copy()
    #     del params['self']
    #     svgd_plots.animate_parameter_pairs(self.history, **params)


    def animate_parameter_pairs(self, param_pairs: list[tuple[int, int]] | None = None,
                            true_params: jnp.ndarray | list | None = None,
                            figsize: tuple[float, float] = (15, 5),
                            hide_fixed: bool = True,                            
                            save_as_gif: str | None = None):
        """
        Animate multiple parameter pairs simultaneously.

        Parameters
        ----------
        param_pairs : list of tuple[int, int], optional
            Parameter pairs to show, e.g. [(0, 1), (2, 3)].
            Defaults to consecutive pairs.
        true_params : np.ndarray, optional
            True parameter values for comparison.
        figsize : tuple, default=(15, 5)
            Figure size (width, height).
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.            
        save_as_gif : str, optional
            Path to save animation as GIF.
        """
        n_dims = self.history.shape[2]
        param_indices = list(range(n_dims))
        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_dims) if not self.fixed_mask[i]]   
            n_dims = len(param_indices)

        # Default to first few parameter pairs if not specified
        if param_pairs is None:
            param_pairs = [(param_indices[i], param_indices[i+1]) for i in range(0, min(6, n_dims-1), 2)]
        
        # Validate param_pairs
        param_pairs = [(i, j) for i, j in param_pairs if max(i, j) < n_dims]
        
        n_plots = len(param_pairs)
        if n_plots == 0:
            raise ValueError("No valid parameter pairs found")
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        # Initialize plots
        scatters = []
        texts = []
        
        for plot_idx, (i, j) in enumerate(param_pairs):
            ax = axes[plot_idx]
            
            # Get data ranges for this parameter pair
            x_data = self.history[:, :, j].flatten()
            y_data = self.history[:, :, i].flatten()
            
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            x_pad = 0.1 * (x_max - x_min)
            y_pad = 0.1 * (y_max - y_min)
            
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlabel(f'Parameter {j}')
            ax.set_ylabel(f'Parameter {i}')
            ax.set_title(f'Params {i} vs {j}')
            
            # Plot true values if available
            if true_params is not None and len(true_params) > max(i, j):
                ax.scatter(true_params[j], true_params[i], color='magenta', s=70, marker='+', 
                        label='True value', zorder=10)
                ax.legend()
            
            # Initialize scatter plot
            scatter = ax.scatter([], [], alpha=0.6, s=5, edgecolor='none')
            scatters.append(scatter)
            
            # Add iteration text
            text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
            texts.append(text)
        
        def init():
            for scatter in scatters:
                scatter.set_offsets(np.empty((0, 2)))
            for text in texts:
                text.set_text('')
            return scatters + texts
        
        def update(frame):
            for plot_idx, (i, j) in enumerate(param_pairs):
                particles_2d = self.history[frame, :, [j, i]]  # Note: [j, i] for x, y
                scatters[plot_idx].set_offsets(particles_2d)
                texts[plot_idx].set_text(f'Iter: {frame}')
            return scatters + texts
        
        anim = FuncAnimation(fig, update, frames=self.history.shape[0],
                            init_func=init, blit=True, interval=200)
        
        plt.tight_layout()
        
        if save_as_gif:
            anim.save(save_as_gif, writer='pillow', fps=10)
        
        from IPython.display import HTML
        return HTML(anim.to_jshtml())




    # ========================================================================
    # Convergence Analysis and Diagnostics
    # ========================================================================

    def _compute_particle_diversity(self, particles: jnp.ndarray) -> dict:
        """
        Compute particle diversity metrics.

        Parameters
        ----------
        particles : array (n_particles, theta_dim)
            Particle positions

        Returns
        -------
        dict
            'mean_distance': Mean pairwise distance
            'min_distance': Minimum pairwise distance
            'ess': Effective sample size estimate
        """
        n_particles = particles.shape[0]

        # Compute pairwise distances
        distances = jnp.array([
            jnp.linalg.norm(particles[i] - particles[j])
            for i in range(n_particles)
            for j in range(i + 1, n_particles)
        ])

        mean_dist = jnp.mean(distances)
        min_dist = jnp.min(distances)

        # Estimate ESS from particle weights (uniform weights for SVGD)
        # Use inverse participation ratio: ESS ≈ 1 / sum(w_i^2)
        # For SVGD, approximate based on particle spread
        particle_var = jnp.var(particles, axis=0)
        overall_var = jnp.mean(particle_var)

        # Rough ESS estimate: higher variance → better ESS
        # Normalize by expected variance for uniform particles
        ess_estimate = n_particles * (overall_var / (overall_var + 1e-10))

        return {
            'mean_distance': float(mean_dist),
            'min_distance': float(min_dist),
            'ess': float(ess_estimate),
            'ess_ratio': float(ess_estimate / n_particles)
        }

    def _detect_convergence_point(self, trajectory: jnp.ndarray, window: int = 50, threshold: float = 0.01) -> int | None:
        """
        Detect iteration where trajectory converged.

        Parameters
        ----------
        trajectory : array (n_iterations,)
            Trajectory of mean or std over iterations
        window : int
            Window size for stability check
        threshold : float
            Relative change threshold for convergence

        Returns
        -------
        int or None
            Iteration where converged, or None if not converged
        """
        if len(trajectory) < window * 2:
            return None

        for i in range(window, len(trajectory) - window):
            # Check if trajectory is stable in window around this point
            window_vals = trajectory[i:i + window]
            mean_val = jnp.mean(window_vals)

            if abs(mean_val) < 1e-10:
                continue  # Skip if near zero

            # Compute relative variation
            rel_var = jnp.std(window_vals) / abs(mean_val)

            if rel_var < threshold:
                return i

        return None

    def _detect_variance_collapse(self, history_array: jnp.ndarray) -> dict:
        """
        Detect if particles collapsed to same value (variance collapse).

        Parameters
        ----------
        history_array : array (n_iterations, n_particles, theta_dim)
            Full particle history

        Returns
        -------
        dict
            'collapsed': bool
            'collapse_iteration': int or None
            'final_diversity': float
        """
        n_iterations = history_array.shape[0]

        # Check variance over time
        std_over_time = jnp.std(history_array, axis=1)  # (n_iterations, theta_dim)
        mean_std_over_time = jnp.mean(std_over_time, axis=1)  # (n_iterations,)

        # Check if std drops to near-zero
        final_std = mean_std_over_time[-1]
        max_std = jnp.max(mean_std_over_time)

        collapsed = final_std < 0.01 * max_std

        # Find when collapse happened
        collapse_iter = None
        if collapsed:
            threshold = 0.1 * max_std
            for i in range(len(mean_std_over_time)):
                if mean_std_over_time[i] < threshold:
                    collapse_iter = i
                    break

        return {
            'collapsed': bool(collapsed),
            'collapse_iteration': int(collapse_iter) if collapse_iter is not None else None,
            'final_diversity': float(final_std),
            'max_diversity': float(max_std)
        }

    def _suggest_learning_rate(self, diagnostics: dict) -> dict:
        """
        Suggest learning rate improvements based on diagnostics.

        Parameters
        ----------
        diagnostics : dict
            Diagnostics from analyze_trace

        Returns
        -------
        dict
            'recommended': schedule object or float
            'reason': str explaining suggestion
        """
        # Extract key metrics
        converged = diagnostics['converged']
        conv_point = diagnostics.get('convergence_point')
        n_iterations = diagnostics['n_iterations']
        variance_collapsed = diagnostics['variance_collapse']['collapsed']

        # Get current learning rate info
        current_schedule = self.step_schedule

        # Decision logic
        if variance_collapsed:
            return {
                'recommended': ExpStepSize(
                    first_step=0.005, last_step=0.0005, tau=500.0
                ),
                'reason': 'Variance collapsed - reduce learning rate significantly'
            }
        elif not converged:
            # Not converged - might need more iterations or different schedule
            if isinstance(current_schedule, ConstantStepSize):
                return {
                    'recommended': ExpStepSize(
                        first_step=current_schedule.step_size * 1.5,
                        last_step=current_schedule.step_size * 0.1,
                        tau=n_iterations * 0.5
                    ),
                    'reason': 'Not converged - use decay schedule for better convergence'
                }
            else:
                return {
                    'recommended': 'increase n_iterations',
                    'reason': 'Not converged within current iterations'
                }
        elif conv_point and conv_point < n_iterations * 0.5:
            # Converged very early - could use higher learning rate
            if isinstance(current_schedule, ConstantStepSize):
                return {
                    'recommended': current_schedule.step_size * 1.5,
                    'reason': f'Converged early (iteration {conv_point}) - could converge faster'
                }
            else:
                return {
                    'recommended': 'current schedule is good',
                    'reason': 'Converged efficiently'
                }
        else:
            return {
                'recommended': 'current learning rate is appropriate',
                'reason': 'Good convergence behavior'
            }

    def _suggest_particles(self, diagnostics: dict) -> dict:
        """
        Suggest particle count based on diagnostics.

        Parameters
        ----------
        diagnostics : dict
            Diagnostics from analyze_trace

        Returns
        -------
        dict
            'recommended': int
            'reason': str
        """
        current_n = self.n_particles
        ess_ratio = diagnostics['diversity']['ess_ratio']
        variance_collapsed = diagnostics['variance_collapse']['collapsed']

        if variance_collapsed:
            return {
                'recommended': current_n * 2,
                'reason': 'Variance collapse detected - increase particles for diversity'
            }
        elif ess_ratio < 0.5:
            return {
                'recommended': int(current_n * 1.5),
                'reason': f'Low ESS ratio ({ess_ratio:.2g}) - increase particles'
            }
        elif ess_ratio > 0.9:
            return {
                'recommended': max(20, int(current_n * 0.8)),
                'reason': f'High ESS ratio ({ess_ratio:.2g}) - could reduce particles'
            }
        else:
            return {
                'recommended': current_n,
                'reason': 'Particle count is appropriate'
            }

    def analyze_trace(self, burnin: int | None = None, verbose: bool = True, return_dict: bool = False) -> dict | None:
        """
        Analyze SVGD convergence and suggest parameter improvements.

        Computes convergence diagnostics, detects issues, and recommends
        parameter updates for better performance.

        Parameters
        ----------
        burnin : int, optional
            Number of initial iterations to discard as burn-in.
            If None, auto-detects using convergence detection.
        verbose : bool, default=True
            Print detailed diagnostic report
        return_dict : bool, default=False
            Return full diagnostics dictionary

        Returns
        -------
        dict or None
            If return_dict=True, returns diagnostics dictionary with:
            - 'converged': bool - Whether SVGD converged
            - 'convergence_point': int or None - Iteration where converged
            - 'diversity': dict - Particle diversity metrics
            - 'variance_collapse': dict - Variance collapse diagnostics
            - 'suggestions': dict - Recommended parameter updates
            - 'issues': list - Detected problems

        Raises
        ------
        RuntimeError
            If fit() not called or history not available

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=1, n_iterations=700)
        >>> svgd.fit(return_history=True)
        >>> svgd.analyze_trace()  # Prints diagnostic report

        >>> # Get full diagnostics
        >>> diag = svgd.analyze_trace(return_dict=True, verbose=False)
        >>> if not diag['converged']:
        >>>     print("Need more iterations!")
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before analyzing trace")

        if self.history is None:
            raise RuntimeError(
                "History not available. Call fit(return_history=True) first"
            )

        # Get history in appropriate space
        results = self.get_results()
        if self.param_transform is not None:
            history = results.get('history', self.history)
            space_label = ""
        else:
            history = self.history
            space_label = " (unconstrained)"

        # Convert to array
        history_array = jnp.stack(history)  # (n_iterations, n_particles, theta_dim)
        n_iterations, n_particles, theta_dim = history_array.shape

        # Compute trajectories
        mean_over_time = jnp.mean(history_array, axis=1)  # (n_iterations, theta_dim)
        std_over_time = jnp.std(history_array, axis=1)    # (n_iterations, theta_dim)

        # Average across dimensions for overall convergence
        mean_trajectory = jnp.mean(mean_over_time, axis=1)
        std_trajectory = jnp.mean(std_over_time, axis=1)

        # Detect convergence
        mean_conv_point = self._detect_convergence_point(mean_trajectory, window=50, threshold=0.01)
        std_conv_point = self._detect_convergence_point(std_trajectory, window=50, threshold=0.05)

        converged = mean_conv_point is not None

        # Auto-detect burnin if not provided
        if burnin is None:
            burnin = mean_conv_point if mean_conv_point is not None else int(n_iterations * 0.2)

        # Compute particle diversity
        final_particles = history_array[-1]
        diversity = self._compute_particle_diversity(final_particles)

        # Detect variance collapse
        variance_collapse = self._detect_variance_collapse(history_array)

        # Build diagnostics dict
        diagnostics = {
            'converged': converged,
            'convergence_point': mean_conv_point,
            'std_convergence_point': std_conv_point,
            'n_iterations': n_iterations,
            'n_particles': n_particles,
            'theta_dim': theta_dim,
            'diversity': diversity,
            'variance_collapse': variance_collapse,
            'burnin': burnin,
        }

        # Get suggestions
        lr_suggestion = self._suggest_learning_rate(diagnostics)
        particle_suggestion = self._suggest_particles(diagnostics)

        # Detect issues
        issues = []
        if variance_collapse['collapsed']:
            issues.append(f"⚠ Variance collapse at iteration {variance_collapse['collapse_iteration']}")
        if not converged:
            issues.append("⚠ Did not converge within n_iterations")
        if diversity['ess_ratio'] < 0.5:
            issues.append(f"⚠ Low effective sample size ({diversity['ess_ratio']:.1%})")
        if converged and mean_conv_point < n_iterations * 0.7:
            pct = mean_conv_point / n_iterations * 100
            issues.append(f"ℹ Converged at {pct:.1g}% of iterations - could reduce n_iterations")

        diagnostics['issues'] = issues
        diagnostics['suggestions'] = {
            'learning_rate': lr_suggestion,
            'n_particles': particle_suggestion
        }

        # Print report if verbose
        if verbose:
            self._print_analysis_report(diagnostics, space_label)

        if return_dict:
            return diagnostics

    def _print_analysis_report(self, diag: dict, space_label: str = "") -> None:
        """Print formatted analysis report."""

        # Convergence status
        if diag['converged']:
            print(f"CONVERGED (iteration {diag['convergence_point']}/{diag['n_iterations']})")
            print(f"  Mean stabilized at iteration {diag['convergence_point']}")
            if diag['std_convergence_point']:
                print(f"  Std stabilized at iteration {diag['std_convergence_point']}")
        else:
            print(f"NOT CONVERGED after {diag['n_iterations']} iterations")

        # print()
        # print("Particle Diversity:")
        # div = diag['diversity']
        # print(f"  Mean inter-particle distance: {div['mean_distance']:.3g}")
        # print(f"  Effective sample size (ESS): {div['ess']:.1g} / {diag['n_particles']} particles ({div['ess_ratio']:.1%})")

        # if div['ess_ratio'] > 0.7:
        #     print("  Good particle diversity")
        # elif div['ess_ratio'] > 0.5:
        #     print("  Moderate particle diversity")
        # else:
        #     print("  Low particle diversity")

        if diag['variance_collapse']['collapsed']:
            print()
            print("Variance Collapse:")
            vc = diag['variance_collapse']
            print(f"  Particles collapsed at iteration {vc['collapse_iteration']}")
            print(f"  Final diversity: {vc['final_diversity']:.4g} (max was {vc['max_diversity']:.4g})")

        # Issues
        if diag['issues']:
            print()
            print("Detected Issues:")
            for issue in diag['issues']:
                print(f"  {issue}")

        # # Suggestions
        # print()
        # print("=" * 80)
        # print("Suggested Parameter Updates")
        # print("=" * 80)
        # print()

        # print("Current Configuration:")
        # print(f"  learning_rate={self.step_schedule}")
        # print(f"  n_particles={self.n_particles}")
        # print(f"  n_iterations={self.n_iterations}")
        # if self.optimizer is not None:
        #     opt = self.optimizer
        #     opt_name = type(opt).__name__
        #     if hasattr(opt, 'beta1'):  # Adam
        #         print(f"  optimizer={opt_name}(lr={opt.lr}, β1={opt.beta1}, β2={opt.beta2})")
        #     elif hasattr(opt, 'momentum'):  # SGDMomentum
        #         print(f"  optimizer={opt_name}(lr={opt.lr}, momentum={opt.momentum})")
        #     elif hasattr(opt, 'decay'):  # RMSprop
        #         print(f"  optimizer={opt_name}(lr={opt.lr}, decay={opt.decay})")
        #     else:  # Adagrad or custom
        #         print(f"  optimizer={opt_name}(lr={opt.lr})")
        # else:
        #     print(f"  optimizer=None (fixed step size)")
        # print()

        # Learning rate suggestion
        lr_sug = diag['suggestions']['learning_rate']
        print(f"Learning Rate: {lr_sug['reason']}")
        if isinstance(lr_sug['recommended'], str):
            print(f"  {lr_sug['recommended']}")
        elif isinstance(lr_sug['recommended'], ExpStepSize):
            sched = lr_sug['recommended']
            print(f"  ExpStepSize(")
            print(f"      first_step={sched.first_step},")
            print(f"      last_step={sched.last_step},")
            print(f"      tau={sched.tau}")
            print(f"  )")
        else:
            print(f"  {lr_sug['recommended']}")

        # print()

        # Particle suggestion
        part_sug = diag['suggestions']['n_particles']
        print(f"Particles: {part_sug['reason']}")
        if part_sug['recommended'] != self.n_particles:
            print(f"  n_particles={part_sug['recommended']}")
        else:
            print(f"  Keep n_particles={self.n_particles}")

        # print()

        # Iteration suggestion
        if diag['converged'] and diag['convergence_point'] < diag['n_iterations'] * 0.8:
            suggested_iters = int(diag['convergence_point'] * 1.2)  # Add 20% buffer
            print(f"Iterations: Converged early")
            print(f"  Could reduce to n_iterations={suggested_iters}")
            # print()
        elif not diag['converged']:
            suggested_iters = int(diag['n_iterations'] * 1.5)
            print(f"Iterations: Did not converge")
            print(f"  Increase to n_iterations={suggested_iters}")
            # print()

        # Optimizer suggestion based on diagnostics
        self._print_optimizer_suggestion(diag)

        # print("=" * 80)

    def _print_optimizer_suggestion(self, diag: dict) -> None:
        """Print optimizer recommendation based on diagnostics."""
        # Detect oscillation by checking sign changes in gradient direction
        has_oscillation = False
        has_variance_collapse = diag['variance_collapse']['collapsed']
        low_ess = diag['diversity']['ess_ratio'] < 0.5
        not_converged = not diag['converged']

        # Simple heuristic: if not converged or variance collapsed, suggest optimizer change
        if self.optimizer is None:
            # Currently using fixed step size
            if has_variance_collapse or not_converged:
                print("Optimizer: Consider using an adaptive optimizer")
                print("  Options (from phasic import Adam, SGDMomentum, RMSprop, Adagrad):")
                print("    Adam(lr=0.01)     - Adaptive LR + momentum (recommended)")
                print("    SGDMomentum(lr=0.01)      - Momentum only")
                print("    RMSprop(lr=0.001)         - Adaptive LR, no momentum")
                print("    Adagrad(lr=0.01)          - Cumulative gradient scaling")
                # print()
        else:
            opt_name = type(self.optimizer).__name__
            if has_variance_collapse:
                print(f"Optimizer: {opt_name} - variance collapsed, try lower learning rate")
                print(f"  Reduce optimizer lr (current: {self.optimizer.lr})")
                # print()
            elif not_converged and low_ess:
                print(f"Optimizer: {opt_name} - not converged with low ESS")
                print(f"  Try higher lr or more iterations")
                # print()
            elif diag['converged']:
                print(f"Optimizer: {opt_name} working well")
                # print()

    def plot_pairwise(self, true_theta: jnp.ndarray | list | None = None,
                     param_names: list[str] | None = None,
                     figsize: tuple[float, float] | None = None,
                     save_path: str | None = None,
                     hide_fixed: bool = True,
                     unconstrained: bool = False, return_fig: bool = False):
        """
        Plot pairwise scatter plots for all parameter pairs.

        Parameters
        ----------
        true_theta : np.ndarray, optional
            True parameter values (if known) to overlay on plot
        param_names : list of str, optional
            Names for each parameter dimension
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.
        hide_fixed : bool, default=True
            If True, hide fixed parameters in the trace plot.            
        return_fig : bool, default=True
            If True, return (fig, axes). If False, call plt.show() instead.

        Returns
        -------
        tuple[matplotlib.figure.Figure, numpy.ndarray]
            Matplotlib figure and axes array (only if return_fig=True).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.theta_dim < 2:
            raise ValueError("Pairwise plots require at least 2 parameters")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate particle representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            particles = results['particles']
            space_label = ""
        else:
            particles = results.get('particles_unconstrained', results['particles'])
            space_label = " (unconstrained)"

        n_params = self.theta_dim
        # n_fixed = len(self.fixed_mask)
        ax_dim = n_params #- n_fixed
        if ax_dim < 2:
            raise ValueError("Not enough free parameters to plot pairwise relationships.")
        figsize = figsize or (min(8, 3 * ax_dim), min(7, 2.3 * ax_dim))

        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        if self.fixed_mask is not None and hide_fixed:
            param_indices = [i for i in range(n_params) if not self.fixed_mask[i]]   
        else:
            param_indices = list(range(n_params))

        for i in param_indices:
            for j in param_indices:

                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(particles[:, i], bins=20, alpha=0.7,
                           edgecolor='black')
                    param_name = param_names[i] if param_names else rf"$\theta_{i}$"
                    ax.set_ylabel('Count')

                    if true_theta is not None:
                        true_val = jnp.array(true_theta)[i]
                        ax.axvline(true_val, color='magenta', linestyle='--', alpha=0.5)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(particles[:, j], particles[:, i],
                             alpha=0.5, s=20)

                    if true_theta is not None:
                        true_val_i = jnp.array(true_theta)[i]
                        true_val_j = jnp.array(true_theta)[j]
                        ax.scatter([true_val_j], [true_val_i], color='magenta',
                                 alpha=0.5, s=20)

                # Labels
                if i == n_params - 1:
                    param_name_j = param_names[j] if param_names else rf"$\theta_{j}$"
                    ax.set_xlabel(param_name_j + space_label)
                if j == 0:
                    param_name_i = param_names[i] if param_names else rf"$\theta_{i}$"
                    ax.set_ylabel(param_name_i + space_label)

                # ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if return_fig:
            return fig, axes
        else:
            plt.show()

    def _validate_animation_params(self, skip: int) -> tuple:
        """Validate common animation parameters and import dependencies."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before animating")

        if self.history is None:
            raise RuntimeError(
                "No history available. Call fit(return_history=True) to record particle evolution."
            )

        # Validate skip parameter
        n_iterations = len(self.history)
        if skip >= n_iterations:
            raise ValueError(
                f"skip ({skip}) must be less than number of iterations ({n_iterations})"
            )

        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            return plt, FuncAnimation
        except ImportError:
            raise ImportError(
                "matplotlib is required for animation. Install with: pip install matplotlib"
            )

    def _save_animation(self, anim: object, save_as_gif: str | None, save_as_mp4: str | None, interval: int) -> None:
        """Save animation to file if requested."""
        if save_as_gif:
            try:
                anim.save(save_as_gif, writer='pillow', fps=int(1000/interval))
                if self.verbose:
                    print(f"Animation saved as GIF: {save_as_gig}")
            except Exception as e:
                print(f"Warning: Could not save GIF: {e}")

        if save_as_mp4:
            try:
                anim.save(save_as_mp4, writer='ffmpeg', fps=int(1000/interval))
                if self.verbose:
                    print(f"Animation saved as MP4: {save_as_mp4}")
            except Exception as e:
                print(f"Warning: Could not save MP4: {e}")

    def _return_animation_html(self, anim: object) -> object:
        """Return animation as HTML for Jupyter display."""
        try:
            from IPython.display import HTML
            return HTML(anim.to_jshtml())
        except ImportError:
            print("Warning: IPython not available. Returning animation object.")
            return anim

    def animate(self, param_idx: int = 0, true_theta: jnp.ndarray | list | None = None,
                param_name: str | None = None,
                figsize: tuple[float, float] = (8, 3), skip: int = 0, thin: int = 1,
                interval: int = 100, duration: int | None = None, bins: int = 30,
                show_particles: bool = True, max_particles: int = 20,
                save_as_gif: str | None = None, save_as_mp4: str | None = None,
                unconstrained: bool = False):
        """
        Create an animation showing the evolution of a single parameter's distribution.

        This method creates a side-by-side animation with:
        - Left panel: Histogram of current parameter distribution
        - Right panel: Particle trajectories over time

        Parameters
        ----------
        param_idx : int, default=0
            Index of the parameter to animate (0-indexed)
        true_theta : np.ndarray, optional
            True parameter values. If provided, will overlay the true value for param_idx.
        param_name : str, optional
            Name for the parameter (e.g., 'jump rate'). If None, uses 'θ_{param_idx}'.
        figsize : tuple, default=(8, 3)
            Figure size (width, height)
        skip : int, default=0
            Number of initial iterations to skip in the animation
        thin : int, thin=1
            Interval of interations to plot/annimate
        interval : int, default=100
            Delay between frames in milliseconds
        duration : int, default=None
            Duration of the animation in seconds, overrides interval and thin if set
        bins : int, default=30
            Number of histogram bins
        show_particles : bool, default=True
            If True, show individual particle trajectories in right panel
        max_particles : int, default=20
            Maximum number of particle trajectories to show (for clarity)
        save_as_gif : str, optional
            Path to save animation as GIF (requires pillow)
        save_as_mp4 : str, optional
            Path to save animation as MP4 (requires ffmpeg)
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        IPython.display.HTML
            Animation as HTML for Jupyter notebook display

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=3, n_iterations=70)
        >>> svgd.fit(return_history=True)
        >>> anim = svgd.animate(param_idx=0, true_theta=[2.0, 3.0, 2.0],
        ...                     param_name='jump rate')
        """
        plt, FuncAnimation = self._validate_animation_params(skip)

        if param_idx < 0 or param_idx >= self.theta_dim:
            raise ValueError(f"param_idx ({param_idx}) out of range [0, {self.theta_dim-1}]")

        # Get appropriate history representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            full_history = results.get('history', self.history)
            space_label = ""
        else:
            full_history = results.get('history_unconstrained', self.history)
            space_label = " (unconstrained)"

        if duration is not None:
            interval = 40
            iterations = len(full_history) - skip
            thin = interval / (duration / iterations * 1000)
            if thin < 1:
                interval *= interval / thin
            interval *= round(thin) / thin    
            thin = round(thin)
            thin, interval = int(thin), int(interval)

        # Get history subset
        history = jnp.stack(full_history[skip::thin])
        param_history = history[:, :, param_idx]

        # Compute axis limits
        param_min = jnp.min(param_history)
        param_max = jnp.max(param_history)
        param_range = param_max - param_min
        param_lim = (param_min - 0.1 * param_range, param_max + 0.1 * param_range)

        param_name = param_name or f'θ_{param_idx}'
        true_val = jnp.array(true_theta)[param_idx] if true_theta is not None else None

        # Create figure
        fig, (ax_hist, ax_traj) = plt.subplots(1, 2, figsize=figsize)

        # Setup histogram panel
        ax_hist.set_xlim(param_lim)
        ax_hist.set_ylim(0, self.n_particles * 0.4)
        ax_hist.set_xlabel(param_name + space_label)
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Current Distribution')
        ax_hist.grid(alpha=0.3)
        if true_val is not None:
            ax_hist.axvline(true_val, color='magenta', linestyle='--',
                           label='True value', zorder=10)
            ax_hist.legend()

        # Setup trajectory panel
        ax_traj.set_xlim(0, len(history))
        ax_traj.set_ylim(param_lim)
        ax_traj.set_xlabel('Iteration')
        ax_traj.set_ylabel(param_name + space_label)
        ax_traj.set_title('Particle Trajectories')
        ax_traj.grid(alpha=0.3)
        if true_val is not None:
            ax_traj.axhline(true_val, color='magenta', linestyle='--',
                           label='True value', zorder=10)

        # Initialize trajectory lines
        particle_lines = []
        if show_particles:
            n_show = min(max_particles, self.n_particles)
            for _ in range(n_show):
                line, = ax_traj.plot([], [], alpha=0.3, )
                particle_lines.append(line)

        mean_line, = ax_traj.plot([], [], 
                                  color=black_or_white,  
                                  label='Mean', zorder=5)
        current_marker = ax_traj.axvline(0, color='blue', linestyle=':',  alpha=0.7)
        ax_traj.legend()

        iteration_text = fig.text(0.5, 0.98, '', ha='center', va='top')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        def init():
            for line in particle_lines:
                line.set_data([], [])
            mean_line.set_data([], [])
            iteration_text.set_text('')
            return particle_lines + [mean_line, iteration_text]

        def update(frame):
            particles_current = param_history[frame]

            # Update histogram
            ax_hist.clear()
            ax_hist.hist(particles_current, bins=bins, alpha=0.7,
                        edgecolor='black', range=param_lim)
            ax_hist.set_xlim(param_lim)
            ax_hist.set_xlabel(param_name)
            ax_hist.set_ylabel('Count')
            ax_hist.set_title('Current Distribution')
            ax_hist.grid(alpha=0.3)
            if true_val is not None:
                ax_hist.axvline(true_val, color='magenta', linestyle='--',  zorder=10)

            # Update trajectories
            iterations = jnp.arange(frame + 1)
            if show_particles:
                n_show = min(max_particles, self.n_particles)
                for p in range(n_show):
                    particle_lines[p].set_data(iterations, param_history[:frame+1, p])

            mean_trajectory = jnp.mean(param_history[:frame+1], axis=1)
            mean_line.set_data(iterations, mean_trajectory)
            current_marker.set_xdata([frame, frame])
            iteration_text.set_text(f'Iteration: {skip + frame}/{skip + len(history) - 1}')

            return particle_lines + [mean_line, iteration_text]

        anim = FuncAnimation(fig, update, frames=len(history),
                           init_func=init, blit=False, interval=interval)
        plt.close(fig)

        self._save_animation(anim, save_as_gif, save_as_mp4, interval)
        return self._return_animation_html(anim)

    def animate_pairwise(self, true_theta: jnp.ndarray | list | None = None,
                        param_names: list[str] | None = None,
                        figsize: tuple[float, float] | None = None,
                        skip: int = 0, thin: int = 1, interval: int = 100,
                        duration: int | None = None,
                        save_as_gif: str | None = None,
                        save_as_mp4: str | None = None,
                        unconstrained: bool = False):
        """
        Create an animated pairwise scatter plot showing SVGD particle evolution.

        Parameters
        ----------
        true_theta : np.ndarray, optional
            True parameter values (if known) to overlay as red 'x' markers
        param_names : list of str, optional
            Names for each parameter dimension (e.g., ['jump', 'flood_left', 'flood_right'])
        figsize : tuple, optional
            Figure size (width, height). Auto-sized based on parameter dimension if None.
        skip : int, default=0
            Number of initial iterations to skip in the animation
        thin : int, thin=1
            Interval of interations to plot/annimate
        interval : int, default=100
            Delay between frames in milliseconds
        duration : int, default=None
            Duration of the animation in seconds, overrides interval and thin if set
        save_as_gif : str, optional
            Path to save animation as GIF (requires pillow)
        save_as_mp4 : str, optional
            Path to save animation as MP4 (requires ffmpeg)
        unconstrained : bool, default=False
            If False, show constrained (model-space) parameter values.
            If True, show unconstrained (optimization-space) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        IPython.display.HTML
            Animation as HTML for Jupyter notebook display

        Raises
        ------
        RuntimeError
            If fit() was not called with return_history=True
        ImportError
            If matplotlib or required animation backend is not installed

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=3, n_iterations=70)
        >>> svgd.fit(return_history=True)
        >>> anim = svgd.animate_pairwise(
        ...     true_theta=[2.0, 3.0, 2.0],
        ...     param_names=['jump', 'flood_left', 'flood_right'],
        ...     save_as_gif='svgd_evolution.gif'
        ... )
        """
        plt, FuncAnimation = self._validate_animation_params(skip)

        if self.theta_dim < 2:
            raise ValueError("Pairwise plots require at least 2 parameters")

        # Get appropriate history representation
        results = self.get_results()
        if not unconstrained or self.param_transform is None:
            if unconstrained and self.param_transform is None:
                raise ValueError(
                    "unconstrained=True has no effect when no parameter transformation is used. "
                    "Either set unconstrained=False, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            full_history = results.get('history', self.history)
            space_label = ""
        else:
            full_history = results.get('history_unconstrained', self.history)
            space_label = " (unconstrained)"

        n_params = self.theta_dim
        figsize = figsize or (min(14, 3 * n_params), min(12, 2.3 * n_params))

        if duration is not None:
            interval = 40
            iterations = len(full_history) - skip
            thin = interval / (duration / iterations * 1000)
            if thin < 1:
                interval *= interval / thin
            interval *= round(thin) / thin    
            thin = round(thin)
            thin, interval = int(thin), int(interval)

        # Get history subset
        history = full_history[skip::thin]

        # Compute global axis limits based on all history
        all_particles = jnp.concatenate(history, axis=0)
        param_mins = jnp.min(all_particles, axis=0)
        param_maxs = jnp.max(all_particles, axis=0)
        param_ranges = param_maxs - param_mins
        param_lims = [(param_mins[i] - 0.1 * param_ranges[i],
                       param_maxs[i] + 0.1 * param_ranges[i])
                      for i in range(n_params)]

        # Create figure and axes
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        # Initialize scatter plots and histograms
        scatter_plots = {}
        hist_data = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for i in range(n_params):
                for j in range(n_params):
                    ax = axes[i, j]
                    ax.set_xlim(param_lims[j])

                    if i == j:
                        # Diagonal: histogram (will be updated each frame)
                        ax.set_ylim(0, self.n_particles * 0.3)  # Will adjust dynamically
                        param_name = param_names[i] if param_names else f'θ_{i}'
                        ax.set_ylabel('Count')

                        if true_theta is not None:
                            true_val = jnp.array(true_theta)[i]
                            ax.axvline(true_val, color='magenta', linestyle='--',  zorder=10)

                        hist_data[(i, j)] = None  # Placeholder for histogram artists
                    else:
                        # Off-diagonal: scatter plot
                        ax.set_ylim(param_lims[i])
                        scatter = ax.scatter([], [], alpha=0.5, s=20)
                        scatter_plots[(i, j)] = scatter

                        if true_theta is not None:
                            true_val_i = jnp.array(true_theta)[i]
                            true_val_j = jnp.array(true_theta)[j]
                            ax.scatter([true_val_j], [true_val_i], color='magenta',
                                    s=70, marker='+', linewidths=3, zorder=10)

                    # Labels
                    if i == n_params - 1:
                        param_name_j = param_names[j] if param_names else rf"$\theta_{j}$"
                        ax.set_xlabel(param_name_j + space_label)
                    if j == 0:
                        param_name_i = param_names[i] if param_names else rf"$\theta_{i}$"
                        ax.set_ylabel(param_name_i + space_label)

#                ax.grid(alpha=0.3)

        # Add iteration counter
        iteration_text = fig.text(0.5, 0.98, '', ha='center', va='top')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        def init():
            """Initialize animation."""
            for scatter in scatter_plots.values():
                scatter.set_offsets(jnp.empty((0, 2)))
            iteration_text.set_text('')
            return list(scatter_plots.values()) + [iteration_text]

        def update(frame):
            """Update function for each animation frame."""
            particles = history[frame]  # Shape: (n_particles, n_params)

            # Update scatter plots
            for (i, j), scatter in scatter_plots.items():
                scatter.set_offsets(jnp.column_stack([particles[:, j], particles[:, i]]))

            # Update histograms
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)            
                for i in range(n_params):
                    ax = axes[i, i]
                    ax.clear()
                    ax.hist(particles[:, i], bins=20, alpha=0.7, edgecolor='black', range=param_lims[i])
                    ax.set_xlim(param_lims[i])
                    ax.set_ylabel('Count')

                    param_name = param_names[i] if param_names else rf"$\theta_{i}$"
                    if i == n_params - 1:
                        ax.set_xlabel(param_name)

                    if true_theta is not None:
                        true_val = jnp.array(true_theta)[i]
                        ax.axvline(true_val, color='magenta', linestyle='--',  zorder=10)

                    # ax.grid(alpha=0.3)

            # Update iteration counter
            iteration_text.set_text(f'Iteration: {skip + frame}/{skip + len(history) - 1}')

            return list(scatter_plots.values()) + [iteration_text]

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(history),
                           init_func=init, blit=False, interval=interval)

        plt.close(fig)  # Prevent duplicate display in notebooks

        self._save_animation(anim, save_as_gif, save_as_mp4, interval)
        return self._return_animation_html(anim)


    def summary(self, ci_method: str = 'hpd', ci_level: float = 0.95) -> None:
        """Print a summary of the inference results.

        Parameters
        ----------
        ci_method : str, default='hpd'
            Method for credible intervals.

            - ``'hpd'``: Highest Posterior Density interval — the shortest
              interval containing ``ci_level`` fraction of the posterior
              samples. Computed by sorting the particles and sliding a window
              of ``ceil(n * ci_level)`` samples to find the narrowest span.
              Better centred on the mode for skewed posteriors.
            - ``'percentile'``: Equal-tailed percentile interval using the
              ``(1 - ci_level)/2`` and ``(1 + ci_level)/2`` quantiles.

        ci_level : float, default=0.95
            Credible level (e.g. 0.95 for 95% intervals).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting summary")

        if ci_method not in ('hpd', 'percentile'):
            raise ValueError(f"ci_method must be 'hpd' or 'percentile', got '{ci_method}'")

        # Get transformed results if using parameter transformation
        results = self.get_results()
        particles = results['particles']
        theta_mean = results['theta_mean']
        theta_std = results['theta_std']

        theta_map, _ = self.map_estimate_from_particles(unconstrained=False)

        pct = int(ci_level * 100)
        if ci_method == 'hpd':
            lo_label = f"HPD {pct}% lo"
            hi_label = f"HPD {pct}% hi"
        else:
            lo_pct = (1 - ci_level) / 2 * 100
            hi_pct = (1 + ci_level) / 2 * 100
            lo_label = f"CI {lo_pct:.1g}%"
            hi_label = f"CI {hi_pct:.1g}%"

        fields = ["Parameter", "Fixed", "MAP", "Mean", "SD", lo_label, hi_label]
        fmt_str = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}"
        print(fmt_str.format(*fields))

        for i in range(self.theta_dim):
            val_fmt = f'{{:.3e}}' if np.log10(abs(theta_mean[i]) + 1e-300) > 2 else f'{{:.4g}}'

            if self.fixed_mask is not None and self.fixed_mask[i]:
                fields = [i,
                    'Yes',
                    val_fmt.format(theta_mean[i]),
                    'NA',
                    'NA',
                    'NA',
                    'NA',
                    ]
            else:
                if ci_method == 'hpd':
                    ci_lower, ci_upper = _compute_hpd(
                        np.asarray(particles[:, i]), alpha=ci_level
                    )
                else:
                    lo_pct = (1 - ci_level) / 2 * 100
                    hi_pct = (1 + ci_level) / 2 * 100
                    ci_lower = jnp.percentile(particles[:, i], lo_pct).item()
                    ci_upper = jnp.percentile(particles[:, i], hi_pct).item()

                fields = [i,
                        'Yes' if self.fixed_mask is not None and self.fixed_mask[i] else 'No',
                        val_fmt.format(theta_map[i]),
                        val_fmt.format(theta_mean[i]),
                        val_fmt.format(theta_std[i]),
                        val_fmt.format(ci_lower),
                        val_fmt.format(ci_upper),
                        ]
            print(fmt_str.format(*fields))

        print()
        print(f"Particles: {self.n_particles}, Iterations: {self.n_iterations}")
    




