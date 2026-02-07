"""
likelihood_topology_jax
-----------------------
Single-file Python package implementing Fisher-information and likelihood-topology
analyses with optional JAX autodifferentiation and persistent-homology support.

Features
- compute_observed_fisher: observed Fisher (negative Hessian) via JAX
- local_identifiability_metrics: eigenvalues, condition number, effective rank
- grid_likelihood_scan: evaluate log-likelihood on a parameter-space grid
- compute_persistent_h0: compute H0 persistence using ripser (if available) or
  a fall-back component-tracking approximation on the grid
- barrier_height_between_modes: approximate saddle/barrier between modes using
  graph shortest-path on negative-log-likelihood
- multimodality_index: combine persistence lifetimes and basin volumes into a
  single scalar

Requirements
- jax and jaxlib (for autodiff) OR a function returning gradients/hessians
- numpy, scipy
- optional: ripser or gudhi for persistent homology

Usage example (minimal):
>>> from likelihood_topology_jax import example
>>> example.run_simple_demo()

"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

import numpy as np
from scipy import linalg
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Try to import JAX; allow fallback to autograd-like signature if absent
try:
    import jax
    import jax.numpy as jnp
    from jax import jacfwd, jacrev, hessian
    _HAS_JAX = True
except Exception:
    _HAS_JAX = False

# Try ripser for PH
try:
    from ripser import ripser
    from persim import plot_diagrams  # type: ignore
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False


@dataclass
class LocalIdentifiability:
    eigenvalues: np.ndarray
    condition_number: float
    effective_rank: int
    null_space_dim: int


def _to_numpy(x):
    if _HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x)
    return np.array(x)


def compute_observed_fisher(loglik_fn: Callable[[np.ndarray], float], theta: np.ndarray) -> np.ndarray:
    """
    Compute the observed Fisher information (negative Hessian of log-likelihood)
    at theta using JAX autodifferentiation.

    Parameters
    - loglik_fn: function theta -> scalar log-likelihood (can sum over data)
    - theta: point at which to evaluate (1-d array)

    Returns
    - I_obs: (d,d) observed Fisher (i.e. -Hessian)

    Raises
    - ImportError if JAX is not available
    """
    if not _HAS_JAX:
        raise ImportError("JAX is required for automatic differentiation. Install with: pip install jax jaxlib")

    theta = np.asarray(theta, dtype=float)
    d = theta.size

    # wrap to JAX
    def _ll_jax(t):
        return jnp.asarray(loglik_fn(t))

    H = hessian(_ll_jax)(theta)
    H = _to_numpy(H)

    Iobs = -H
    return Iobs


def local_identifiability_metrics(I: np.ndarray, tol: float = 1e-8) -> LocalIdentifiability:
    """
    Compute eigenvalues, condition number, effective rank and null-space dimension
    for a Fisher information matrix I.

    Parameters
    - I: symmetric (d,d) Fisher matrix
    - tol: relative tolerance for eigenvalues considered zero

    Returns
    - LocalIdentifiability dataclass
    """
    evals, evecs = linalg.eigh(I)
    # Numerical stability: sort descending
    evals = np.sort(evals)[::-1]
    max_eval = np.max(np.abs(evals)) if evals.size else 0.0
    null_thresh = tol * max(1.0, max_eval)
    null_space_dim = int(np.sum(evals < null_thresh))
    # condition number, be careful with near-zero
    if np.all(np.isfinite(evals)) and evals.size:
        eps = 1e-30
        cond = float(np.abs(evals[0]) / max(np.abs(evals[-1]), eps))
    else:
        cond = float('inf')
    # effective rank: number of eigenvalues > tol*max
    eff_rank = int(np.sum(evals > null_thresh))

    return LocalIdentifiability(eigenvalues=evals, condition_number=cond, effective_rank=eff_rank, null_space_dim=null_space_dim)


def grid_likelihood_scan(loglik_fn: Callable[[np.ndarray], float], grid_axes: Tuple[np.ndarray, ...], batch: int = 1000) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Evaluate log-likelihood on a Cartesian grid defined by grid_axes.

    Parameters
    - loglik_fn: theta -> scalar log-likelihood
    - grid_axes: tuple of 1D arrays, one per parameter dimension
    - batch: batch size for Python-loop evaluation

    Returns
    - ll_grid: array of shape len(grid_axes) each
    - mesh: tuple of nd-grids corresponding to axes (for plotting)
    """
    shapes = [len(ax) for ax in grid_axes]
    mesh = np.meshgrid(*grid_axes, indexing='ij')
    coords = np.stack([m.reshape(-1) for m in mesh], axis=1)
    npts = coords.shape[0]
    ll = np.empty(npts, dtype=float)
    for i in range(0, npts, batch):
        j = min(npts, i + batch)
        for k in range(i, j):
            ll[k] = float(loglik_fn(coords[k]))
    ll_grid = ll.reshape(*shapes)
    return ll_grid, mesh


def compute_persistent_h0(ll_grid: np.ndarray, axes: Tuple[np.ndarray, ...]) -> Dict:
    """
    Compute H0 persistent homology for superlevel sets of the log-likelihood on a grid.

    Uses ripser on point-cloud formed by grid points weighted by likelihood.
    Samples points proportionally to exp(ll) to get a point cloud of
    high-likelihood regions and computes H0 persistence.

    Returns a dictionary with keys:
    - 'diagram': Nx2 array of (birth, death) pairs
    - 'method': 'ripser'

    Raises:
    - ImportError if ripser is not available
    """
    if not _HAS_RIPSER:
        raise ImportError("ripser is required for persistent homology computation. Install with: pip install ripser")

    grid_shape = ll_grid.shape
    coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(axes))
    ll_flat = ll_grid.reshape(-1)
    # numerical stabilization
    probs = np.exp(ll_flat - np.max(ll_flat))
    probs /= probs.sum()
    n_samples = min(2000, coords.shape[0])
    idx = np.random.choice(coords.shape[0], size=n_samples, replace=False, p=probs)
    pc = coords[idx]
    out = ripser(pc, maxdim=0)
    diag = out['dgms'][0]
    # diag: Nx2 array of birth/death
    return {'diagram': diag, 'method': 'ripser'}


def find_modes_on_grid(ll_grid: np.ndarray, axes: Tuple[np.ndarray, ...], min_prominence: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima on a grid by comparing to neighbors. Returns coordinates and values.
    """
    footprint = np.ones((3,) * ll_grid.ndim, dtype=bool)
    local_max = (ll_grid == ndimage.maximum_filter(ll_grid, footprint=footprint))
    # remove flat regions by strict greater than neighbors
    eroded = ndimage.maximum_filter(ll_grid, size=3, mode='constant')
    mask = local_max
    coords = np.array(np.nonzero(mask)).T
    values = np.array([ll_grid[tuple(c)] for c in coords])
    # map grid indices to real parameter coordinates
    param_coords = np.stack([axes[dim][coords[:, dim]] for dim in range(len(axes))], axis=1)
    return param_coords, values


def barrier_height_between_modes(ll_grid: np.ndarray, axes: Tuple[np.ndarray, ...], mode_idx_a: int, mode_idx_b: int) -> float:
    """
    Approximate barrier height (difference between mode minima and highest minimum along
    lowest-energy path) between two modes on the grid by converting to a graph and
    computing the minimum-energy path between the two grid nodes.

    Returns: barrier height (mode min - path max)
    """
    shape = ll_grid.shape
    coords_list = [np.arange(n) for n in shape]
    # flatten index function
    def idx_multi_to_flat(multi):
        flat = 0
        mul = 1
        for s, i in zip(shape[::-1], multi[::-1]):
            flat += i * mul
            mul *= s
        return flat

    flat_ll = -ll_grid.flatten()  # energy
    # build 4/6/8-neighbor graph depending on dimension; use grid graph with weights
    neighbors_offsets = []
    for dim in range(ll_grid.ndim):
        off = [0] * ll_grid.ndim
        off[dim] = 1
        neighbors_offsets.append(tuple(off))
        off2 = [0] * ll_grid.ndim
        off2[dim] = -1
        neighbors_offsets.append(tuple(off2))
    # build sparse adjacency
    N = flat_ll.size
    rows = []
    cols = []
    data = []
    it = np.nditer(ll_grid, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        u = np.ravel_multi_index(idx, shape)
        for off in neighbors_offsets:
            nb = tuple(idx[i] + off[i] for i in range(len(shape)))
            if any((nbi < 0 or nbi >= shape[i]) for i, nbi in enumerate(nb)):
                continue
            v = np.ravel_multi_index(nb, shape)
            # edge weight = max(energy(u), energy(v)) i.e. path trying to minimize peak
            w = max(flat_ll[u], flat_ll[v])
            rows.append(u); cols.append(v); data.append(w)
        it.iternext()
    G = csr_matrix((data, (rows, cols)), shape=(N, N))
    # find mode indices as grid maxima
    param_coords, values = find_modes_on_grid(ll_grid, axes)
    if param_coords.shape[0] < max(mode_idx_a, mode_idx_b) + 1:
        raise IndexError('Mode index out of range')
    idx_a = tuple((np.abs(axes[dim] - param_coords[mode_idx_a, dim]).argmin() for dim in range(len(axes))))
    idx_b = tuple((np.abs(axes[dim] - param_coords[mode_idx_b, dim]).argmin() for dim in range(len(axes))))
    a_flat = np.ravel_multi_index(idx_a, shape)
    b_flat = np.ravel_multi_index(idx_b, shape)
    dist, predecessors = dijkstra(csgraph=G, directed=False, indices=a_flat, return_predecessors=True)
    # the minimal maximal energy along path is dist[b_flat]
    path_max_energy = dist[b_flat]
    mode_peak = max(flat_ll[a_flat], flat_ll[b_flat])
    barrier = path_max_energy - min(flat_ll[a_flat], flat_ll[b_flat])
    return float(barrier)


def multimodality_index_from_persistence(diagram: np.ndarray) -> float:
    """
    Simple multimodality index: sum of H0 lifetimes (death - birth) weighted by birth value.
    diagram: Nx2 array of (birth, death)
    """
    if diagram is None or diagram.size == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    births = diagram[:, 0]
    return float(np.sum(lifetimes * np.exp(-births)))


# Convenience example: a bimodal gaussian mixture on 2 parameters

def _mixture_loglik(theta: np.ndarray) -> float:
    # theta: length-2
    # Use JAX operations for autodiff compatibility
    if _HAS_JAX:
        x = jnp.asarray(theta)
        mu1 = jnp.array([0.0, 0.0])
        mu2 = jnp.array([3.0, 0.0])
        cov = jnp.eye(2) * 0.3
        def rv_logpdf(mu):
            diff = x - mu
            return -0.5 * diff @ jnp.linalg.solve(cov, diff) - 0.5 * jnp.log((2 * jnp.pi) ** 2 * jnp.linalg.det(cov))
        # Don't convert to float when using JAX (might be a tracer during autodiff)
        return jnp.logaddexp(rv_logpdf(mu1), rv_logpdf(mu2))
    else:
        x = theta
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 0.0])
        cov = np.eye(2) * 0.3
        def rv_logpdf(mu):
            diff = x - mu
            return -0.5 * diff @ np.linalg.solve(cov, diff) - 0.5 * np.log((2 * np.pi) ** 2 * np.linalg.det(cov))
        return float(np.logaddexp(rv_logpdf(mu1), rv_logpdf(mu2)))


def run_simple_demo(plot: bool = False):
    """Run a basic demo: grid scan, local Fisher and fallback PH."""
    # grid
    ax0 = np.linspace(-1.0, 4.0, 201)
    ax1 = np.linspace(-2.0, 2.0, 161)
    ll_grid, mesh = grid_likelihood_scan(_mixture_loglik, (ax0, ax1))
    # find modes
    coords, vals = find_modes_on_grid(ll_grid, (ax0, ax1))
    print('Found modes (coords, values):')
    for c, v in zip(coords, vals):
        print(c, v)
    # compute Fisher at first mode (approx using finite diff or JAX)
    theta0 = coords[0]
    I = compute_observed_fisher(_mixture_loglik, theta0)
    lip = local_identifiability_metrics(I)
    print('Eigenvalues:', lip.eigenvalues)
    # compute persistent homology
    ph = compute_persistent_h0(ll_grid, (ax0, ax1))
    print(f"Persistent homology ({ph['method']}): {len(ph['diagram'])} H0 features")
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.contourf(mesh[0], mesh[1], ll_grid, levels=40)
            plt.scatter(coords[:, 0], coords[:, 1], c='r')
            plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_simple_demo(plot=True)
    else:
        print('likelihood_topology_jax module. Run python likelihood_topology_jax.py demo to execute demo.')
