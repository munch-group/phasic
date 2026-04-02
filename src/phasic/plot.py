"""
Shared plotting utilities for phasic inference results.

Standalone functions that accept a ``results`` dict from either
``SVGD.get_results()`` or ``MCMC.get_results()`` and produce
publication-quality plots.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
    _text_color = matplotlib.rcParams['text.color']
except ImportError:
    HAS_MPL = False
    _text_color = 'black'


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def _compute_hpd(samples, alpha=0.95):
    """Compute Highest Posterior Density interval."""
    samples = np.sort(samples)
    n = len(samples)
    interval_size = int(np.ceil(n * alpha))
    if interval_size >= n:
        return float(samples[0]), float(samples[-1])
    widths = samples[interval_size:] - samples[:n - interval_size]
    best = int(np.argmin(widths))
    return float(samples[best]), float(samples[best + interval_size])


def plot_posterior(results: dict,
                   true_theta: list | np.ndarray | None = None,
                   param_names: list[str] | None = None,
                   bins: int = 20,
                   figsize: tuple[float, float] | None = None,
                   save_path: str | None = None,
                   ci_method: str = 'hpd',
                   ci_level: float = 0.95,
                   return_fig: bool = False):
    """Plot posterior distributions for each parameter.

    Works with results from both SVGD and MCMC.

    Parameters
    ----------
    results : dict
        From ``SVGD.get_results()`` or ``MCMC.get_results()``.
        Must contain ``'particles'`` and ``'theta_mean'``.
    true_theta : array-like, optional
        True parameter values to overlay.
    param_names : list of str, optional
        Names for each parameter.
    bins : int, default=20
        Histogram bins.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.
    ci_method : str, default='hpd'
        ``'hpd'`` or ``'percentile'``.
    ci_level : float, default=0.95
        Credible interval level.
    return_fig : bool, default=False
        If True, return (fig, axes) instead of showing.

    Returns
    -------
    tuple or None
        (fig, axes) if return_fig=True.
    """
    _require_matplotlib()

    particles = np.asarray(results['particles'])
    theta_mean = np.asarray(results['theta_mean'])
    n_params = particles.shape[1]

    if n_params <= 2:
        nrows, ncols = 1, n_params
        figsize = figsize or (4 * n_params, 3)
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
        samples_i = particles[:, i]

        ax.hist(samples_i, bins=bins, alpha=0.7, density=True,
                edgecolor='black', label='Posterior')

        # Credible interval
        if ci_method == 'hpd':
            ci_lo, ci_hi = _compute_hpd(samples_i, alpha=ci_level)
        else:
            lo_pct = (1 - ci_level) / 2 * 100
            hi_pct = (1 + ci_level) / 2 * 100
            ci_lo = float(np.percentile(samples_i, lo_pct))
            ci_hi = float(np.percentile(samples_i, hi_pct))
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='steelblue',
                   label=f'{ci_label} [{ci_lo:.3f}, {ci_hi:.3f}]')

        # Mean
        ax.axvline(theta_mean[i], color=_text_color, linestyle='--',
                   label=f'Mean = {theta_mean[i]:.3f}')

        # True value
        if true_theta is not None:
            true_val = np.asarray(true_theta)[i]
            ax.axvline(true_val, color='magenta', linestyle='--',
                       label=f'True = {true_val:.3f}')

        name = param_names[i] if param_names else rf"$\theta_{i}$"
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {name}')
        ax.legend(fontsize=8)

    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if return_fig:
        return fig, axes
    else:
        plt.show()


def plot_chains(results: dict,
                true_theta: list | np.ndarray | None = None,
                param_names: list[str] | None = None,
                figsize: tuple[float, float] | None = None,
                save_path: str | None = None,
                return_fig: bool = False,
                sharey: bool = False):
    """Plot MCMC chain traces (all chains overlaid per parameter).

    Parameters
    ----------
    results : dict
        Must contain ``'chains'`` with shape ``(n_chains, n_samples, theta_dim)``.
    true_theta : array-like, optional
        True parameter values.
    param_names : list of str, optional
        Names for each parameter.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.
    return_fig : bool, default=False
        If True, return (fig, axes).

    Returns
    -------
    tuple or None
        (fig, axes) if return_fig=True.
    """
    _require_matplotlib()

    chains = np.asarray(results['chains'])
    n_chains, n_samples, n_params = chains.shape

    if n_params <= 2:
        nrows, ncols = 1, n_params
        figsize = figsize or (5 * n_params, 3)
    else:
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharey=sharey)
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        for c in range(n_chains):
            ax.plot(chains[c, :, i], alpha=0.6, lw=0.5, label=f'Chain {c+1}')

        if true_theta is not None:
            ax.axhline(np.asarray(true_theta)[i], color='red', ls='--',
                       lw=1.5, label=f'True')

        name = param_names[i] if param_names else rf"$\theta_{i}$"
        ax.set_xlabel('Sample')
        ax.set_ylabel(name)
        ax.set_title(f'Trace: {name}')
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if return_fig:
        return fig, axes
    else:
        plt.show()


def plot_autocorrelation(results: dict,
                         max_lag: int = 50,
                         param_names: list[str] | None = None,
                         figsize: tuple[float, float] | None = None,
                         save_path: str | None = None,
                         return_fig: bool = False):
    """Plot autocorrelation function for each parameter.

    Parameters
    ----------
    results : dict
        Must contain ``'particles'`` with shape ``(n_samples, theta_dim)``.
    max_lag : int, default=50
        Maximum lag to compute.
    param_names : list of str, optional
        Names for each parameter.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.
    return_fig : bool, default=False
        If True, return (fig, axes).

    Returns
    -------
    tuple or None
        (fig, axes) if return_fig=True.
    """
    _require_matplotlib()

    particles = np.asarray(results['particles'])
    n_samples, n_params = particles.shape

    if n_params <= 2:
        nrows, ncols = 1, n_params
        figsize = figsize or (5 * n_params, 3)
    else:
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        x = particles[:, i]
        mean = np.mean(x)
        var = np.var(x)
        if var == 0:
            ax.bar(range(max_lag), [1.0] + [0.0] * (max_lag - 1), width=1, alpha=0.7)
        else:
            acf = []
            for k in range(max_lag):
                if k == 0:
                    acf.append(1.0)
                else:
                    rho = np.mean((x[:n_samples-k] - mean) * (x[k:] - mean)) / var
                    acf.append(rho)
            ax.bar(range(max_lag), acf, width=1, alpha=0.7, color='steelblue')
            ax.axhline(0, color='black', lw=0.5)
            ax.axhline(1.96 / np.sqrt(n_samples), color='red', ls='--', lw=0.5)
            ax.axhline(-1.96 / np.sqrt(n_samples), color='red', ls='--', lw=0.5)

        name = param_names[i] if param_names else rf"$\theta_{i}$"
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title(f'Autocorrelation: {name}')

    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if return_fig:
        return fig, axes
    else:
        plt.show()


def plot_pairwise(results: dict,
                  true_theta: list | np.ndarray | None = None,
                  param_names: list[str] | None = None,
                  figsize: tuple[float, float] | None = None,
                  save_path: str | None = None,
                  return_fig: bool = False):
    """Plot pairwise scatter of posterior samples.

    Parameters
    ----------
    results : dict
        Must contain ``'particles'`` and ``'theta_mean'``.
    true_theta : array-like, optional
        True parameter values.
    param_names : list of str, optional
        Names for each parameter.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.
    return_fig : bool, default=False
        If True, return (fig, axes).

    Returns
    -------
    tuple or None
        (fig, axes) if return_fig=True.
    """
    _require_matplotlib()

    particles = np.asarray(results['particles'])
    theta_mean = np.asarray(results['theta_mean'])
    n_params = particles.shape[1]

    if n_params < 2:
        raise ValueError("Need at least 2 parameters for pairwise plot")

    n_pairs = n_params * (n_params - 1) // 2
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    figsize = figsize or (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    pair_idx = 0
    for i in range(n_params):
        for j in range(i + 1, n_params):
            ax = axes[pair_idx]
            ax.scatter(particles[:, i], particles[:, j], alpha=0.15, s=5, c='C0')

            ax.plot(theta_mean[i], theta_mean[j], 'C1*', markersize=12,
                    label=f'Mean ({theta_mean[i]:.2f}, {theta_mean[j]:.2f})')

            if true_theta is not None:
                tv = np.asarray(true_theta)
                ax.plot(tv[i], tv[j], 'r*', markersize=12,
                        label=f'True ({tv[i]:.2f}, {tv[j]:.2f})')

            ni = param_names[i] if param_names else rf"$\theta_{i}$"
            nj = param_names[j] if param_names else rf"$\theta_{j}$"
            ax.set_xlabel(ni)
            ax.set_ylabel(nj)
            ax.set_title(f'{ni} vs {nj}')
            ax.legend(fontsize=7)
            pair_idx += 1

    for i in range(pair_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if return_fig:
        return fig, axes
    else:
        plt.show()
