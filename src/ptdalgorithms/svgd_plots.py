from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from jax import vmap
from jax import numpy as jnp
import jax


# "iridis" color map (viridis without the deep purple)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)



def plot_svgd_posterior_1d(particles, true_params=None, obs_stats=None, 
                           map_est=None,
                           ax=None, title="SVGD Posterior Approximation"):
    """
    Plot 1D posterior approximation from SVGD particles
    
    Args:
        particles: shape (n_particles, 1) array of SVGD particles
        true_params: optional true parameter value for comparison
        title: plot title
    """
    if ax is None:        
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    # Extract 1D values
    x = particles.flatten()
    
    # Plot histogram of particles
    ax.hist(x, bins=30, density=True, alpha=0.4, label='Particle histogram')
    
    # Plot KDE of posterior
    kde = gaussian_kde(x)
    xx = np.linspace(min(x), max(x), 1000)
    ax.plot(xx, kde(xx), color='orange', lw=2, label='KDE posterior')
    
    # Add true parameter if provided
    if true_params is not None:
        ax.axvline(true_params, color='hotpink', linestyle='--', 
                   label=f'True value: {true_params:.2f}')
        
    # Add data statistics
    if obs_stats is not None:
        ax.axvline(obs_stats, color='magenta',
                   label=f'Observed value: {obs_stats:.2f}')    
    if map_est is not None:
        ax.axvline(map_est, color='orange', linestyle='dashed',
                   label=f'MAP value: {map_est:.2f}')       
    
    ax.set_title(title)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Density')
    ax.legend()
    sns.despine(ax=ax)

def plot_svgd_posterior_2d(particles, true_params=None, obs_stats=None,
                          map_est=None, idx=(0, 1),
                          figsize=(10, 8),
                          labels=None,
                          title=None):
    """
    Plot 2D posterior approximation from SVGD particles
    
    Args:
        particles: shape (n_particles, n_dims) array of SVGD particles
        true_params: optional array of true parameter values
        idx: tuple of parameter indices to plot (default: (0, 1))
        labels: parameter names for axes (auto-generated if None)
        title: plot title
    """
    n_dims = particles.shape[1]
    
    # Validate indices
    if max(idx) >= n_dims:
        raise ValueError(f"Index {max(idx)} exceeds parameter dimension {n_dims}")
    
    # Auto-generate labels if not provided
    if labels is None:
        labels = [f"Parameter {idx[0]}", f"Parameter {idx[1]}"]
    
    print(f"Plotting parameters {idx[0]} vs {idx[1]} from {n_dims}-dimensional space")
    if true_params is not None:
        print(f"True parameter values: {true_params}")
    
    plt.rcParams['animation.embed_limit'] = 100 # Mb

    plt.figure(figsize=figsize)
    
    # Extract parameters
    x = particles[:, idx[0]]
    y = particles[:, idx[1]]
    
    # Create 2D histogram
    plt.subplot(2, 2, 1)
    plt.hist2d(x, y, bins=30, cmap='viridis')
    plt.colorbar(label='Particle count')
    if true_params is not None and len(true_params) > max(idx):
        plt.plot(true_params[idx[0]], true_params[idx[1]], ls='', color='hotpink', marker='*', markersize=10, 
                label='True value')        
    if obs_stats is not None and len(obs_stats) > max(idx):
        plt.plot(obs_stats[idx[0]], obs_stats[idx[1]], ls='', color='magenta', marker='*', markersize=10, 
            label='Obs value')        
    if map_est is not None and len(map_est) > max(idx):
        plt.plot(map_est[idx[0]], map_est[idx[1]], ls='', color='orange', marker='*', markersize=10, 
                 label=f'MAP value')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('2D Histogram')
    
    # Create scatter plot
    plt.subplot(2, 2, 2)
    if true_params is not None and len(true_params) > max(idx):
        plt.gca().axvline(true_params[idx[0]], color='hotpink', linewidth=0.5, linestyle='--', zorder=-1)   
        plt.gca().axhline(true_params[idx[1]], color='hotpink', linewidth=0.5, linestyle='--', zorder=-1, label='True value')   
        plt.legend()
    plt.scatter(x, y, alpha=0.5, s=5, edgecolor='none')
    if obs_stats is not None and len(obs_stats) > max(idx):
        plt.plot(obs_stats[idx[0]], obs_stats[idx[1]], ls='', color='magenta', marker='*', markersize=10, 
            label='Obs value')        
    if map_est is not None and len(map_est) > max(idx):
        plt.plot(map_est[idx[0]], map_est[idx[1]], ls='', color='orange', marker='*', markersize=10, 
                 label='MAP value')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Particle Distribution')
    
    plt.subplot(2, 2, 3)
    plot_svgd_posterior_1d(
        x,
        true_params=true_params[idx[0]] if true_params is not None and len(true_params) > idx[0] else None,
        map_est=map_est[idx[0]] if map_est is not None and len(map_est) > idx[0] else None,
        ax=plt.gca(),
        title=f"Posterior Distribution of {labels[0]}"
    )

    plt.subplot(2, 2, 4)
    plot_svgd_posterior_1d(
        y,
        true_params=true_params[idx[1]] if true_params is not None and len(true_params) > idx[1] else None,
        map_est=map_est[idx[1]] if map_est is not None and len(map_est) > idx[1] else None,
        ax=plt.gca(),
        title=f"Posterior Distribution of {labels[1]}"
    )
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def animate_svgd_2d(particle_history, true_params=None, obs_stats=None,
                    map_est=None, idx=[0, 1],
                    skip=0,
                    figsize=(5, 5),
                    labels=None,
                    save_as_gif=None, save_as_mp4=None,
                    title=None):
    """
    Create an animation of SVGD particle evolution for any 2D projection
    
    Args:
        particle_history: array of shape (n_iterations, n_particles, n_dims)
        true_params: optional true parameter values
        idx: list of parameter indices to animate [param1_idx, param2_idx]
        labels: parameter names for axes (auto-generated if None)
        title: animation title
    """
    n_dims = particle_history.shape[2]
    
    # Validate indices
    if max(idx) >= n_dims:
        raise ValueError(f"Index {max(idx)} exceeds parameter dimension {n_dims}")
    
    # Auto-generate labels if not provided
    if labels is None:
        labels = [f"Parameter {idx[0]}", f"Parameter {idx[1]}"]
    
    fig, ax = plt.subplots(figsize=figsize)

    assert skip < particle_history.shape[0], "Skip value must be less than number of iterations"
    
    # Select only the parameters of interest: shape (n_iterations, n_particles, 2)
    selected_history = particle_history[skip:, :, idx]

    # min/max for axis limits
    x_min = np.min(selected_history[:, :, 0])
    x_max = np.max(selected_history[:, :, 0])
    y_min = np.min(selected_history[:, :, 1])
    y_max = np.max(selected_history[:, :, 1])

    # padding
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title if title else f"SVGD Evolution: {labels[0]} vs {labels[1]}")
    
    # Plot true parameters if provided
    if true_params is not None and len(true_params) > max(idx):
        ax.plot(true_params[idx[0]], true_params[idx[1]], ls='', color='hotpink', marker='*', markersize=5, label='True value')
    if obs_stats is not None and len(obs_stats) > max(idx):
        ax.plot(obs_stats[idx[0]], obs_stats[idx[1]], ls='', color='magenta', marker='*', markersize=5, label='Observed value')
    if map_est is not None and len(map_est) > max(idx):
        ax.plot(map_est[idx[0]], map_est[idx[1]], ls='', color='orange', marker='*', markersize=5, label='MAP value')

    # Initialize scatter plot
    scatter = ax.scatter([], [], alpha=1, s=5, edgecolor='none')
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        iteration_text.set_text('')
        return scatter, iteration_text
    
    def update(frame):
        # Get particles for this frame: shape (n_particles, 2)
        particles_2d = selected_history[frame]
        scatter.set_offsets(particles_2d)
        iteration_text.set_text(f'Iteration: {frame}')
        return scatter, iteration_text
    
    anim = FuncAnimation(fig, update, frames=len(selected_history),
                         init_func=init, blit=True, interval=100)
    
    plt.close()  # Prevent duplicate display in notebooks

    # Save animation as a gif or mp4
    if save_as_gif:
        anim.save(save_as_gif, writer='pillow', fps=10)
    if save_as_mp4:
        anim.save(save_as_mp4, writer='ffmpeg', fps=10)


    from IPython.display import HTML
    return HTML(anim.to_jshtml())


def check_convergence(particle_history, log_p_fn, data, every=1, text=None, param_indices=None):
    """Monitor convergence of SVGD by tracking statistics for n-dimensional parameters"""
    mean_params = []
    std_params = []
    log_probs = []
    
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
                        # bbox=dict(facecolor='red', alpha=0.5)
                        )
    
    # axes[0].annotate('axes fraction',
    #         xy=(2, 1), xycoords='data',
    #         xytext=(0.36, 0.68), textcoords='axes fraction',
    #         arrowprops=dict(facecolor='black', shrink=0.05),
    #         horizontalalignment='right', verticalalignment='top')

    plt.tight_layout()


def estimate_hdr(particles, log_prob_fn, alpha=0.95):
    """
    Estimate the Highest Density Region (HDR) from particles.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
        alpha: Coverage probability (e.g., 0.95 for 95% HDR)
    
    Returns:
        List of particles that are within the HDR
        The log probability threshold that defines the HDR
    """
    n_particles = particles.shape[0]
    
    # Compute log probability for each particle
    # log_probs = jnp.array([log_prob_fn(particles[i]) for i in range(n_particles)])
    log_probs = vmap(log_prob_fn)(particles)

    # Sort particles by log probability (descending)
    sorted_indices = jnp.argsort(-log_probs)
    sorted_log_probs = log_probs[sorted_indices]
    
    # Find the log probability threshold for the HDR
    n_hdr = int(n_particles * alpha)
    threshold = sorted_log_probs[n_hdr-1]
    
    # Get particles in the HDR
    hdr_mask = log_probs >= threshold
    hdr_particles = particles[hdr_mask]
    
    return hdr_particles, threshold


def visualize_hdr_2d(particles, log_prob_fn, idx=[0, 1], alphas=[0.95], 
                     grid_size=50, margin=0.1, xlim=None, ylim=None):
    """
    Visualize the Highest Density Region (HDR) for any 2D projection of n-dimensional distribution.
    
    Args:
        particles: Array of shape (n_particles, n_dims)
        log_prob_fn: Function that computes log probability
        idx: indices of parameters to visualize [param1_idx, param2_idx]
        alphas: list of coverage probabilities (e.g., [0.95] for 95% HDR)
        grid_size: Size of the grid for visualization
        xlim, ylim: Limits for the grid
    
    Returns:
        Figure with HDR visualization
    """
    n_dims = particles.shape[1]
    
    # Validate indices
    if max(idx) >= n_dims:
        raise ValueError(f"Index {max(idx)} exceeds parameter dimension {n_dims}")
    
    # Determine limits if not provided
    if xlim is None:
        x_min, x_max = particles[:, idx[0]].min(), particles[:, idx[0]].max()
        _margin = (x_max - x_min) * margin
        xlim = (x_min - _margin, x_max + _margin)
    
    if ylim is None:
        y_min, y_max = particles[:, idx[1]].min(), particles[:, idx[1]].max()
        _margin = (y_max - y_min) * margin
        ylim = (y_min - _margin, y_max + _margin)
    
    # Create grid
    x = jnp.linspace(xlim[0], xlim[1], grid_size)
    y = jnp.linspace(ylim[0], ylim[1], grid_size)
    X, Y = jnp.meshgrid(x, y)

    # Evaluate log probability on grid using mean values for other parameters
    theta_mean = jnp.mean(particles, axis=0)
    Z = jnp.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            p = theta_mean.copy()
            p = p.at[idx[0]].set(X[i, j])
            p = p.at[idx[1]].set(Y[i, j])
            Z = Z.at[i, j].set(log_prob_fn(p))

    # Get HDR threshold
    levels = []
    for alpha in alphas:
        _, threshold = estimate_hdr(particles, log_prob_fn, alpha)
        levels.append((threshold.item(), alpha))

    fig, ax = plt.subplots(figsize=(7, 5))

    # plot grid log likelihoods
    x_flat, y_flat, z_flat = X.ravel(), Y.ravel(), Z.ravel()
    scatter = sns.scatterplot(x=x_flat, y=y_flat, 
                              hue=z_flat, palette=iridis,
                              edgecolor='none', alpha=0.5, s=5, legend=False)
    # Find and mark logL grid point
    max_idx = jnp.argmax(z_flat)
    ax.scatter(x_flat[max_idx], y_flat[max_idx], color='red', s=70, 
            marker='x', alpha=1,
            label='Max grid LogL')

    # Find and mark MAP estimate
    map_particle, _ = map_estimate_from_particles(particles, log_prob_fn)
    if len(map_particle) > max(idx):
        ax.scatter(map_particle[idx[0]], map_particle[idx[1]], color='orange', s=70, 
                   marker='x', alpha=1,
                   label='MAP estimate')
    
    # plot particles (only the selected dimensions)
    logLikelihoods = vmap(lambda p: log_prob_fn(p))(particles)    
    scatter = sns.scatterplot(x=particles[:, idx[0]], y=particles[:, idx[1]], 
                              hue=logLikelihoods, palette=iridis,
                              edgecolor='none', alpha=0.5, s=10, legend=False)
    # plot contour lines for HDR
    levels, alphas = zip(*sorted(levels))
    contour = ax.contour(X, Y, Z, levels=levels, cmap=iridis, linestyles='dashed', alpha=0.7)
    
    ax.set_xlabel(f'Parameter {idx[0]}')
    ax.set_ylabel(f'Parameter {idx[1]}')
    ax.set_title(f'HDR Visualization: Param {idx[0]} vs Param {idx[1]}')
    ax.legend()
    
    return fig


def map_estimate_from_particles(particles, log_prob_fn):
    """
    Find the MAP estimate from a set of particles by finding the particle
    with the highest log probability.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
    
    Returns:
        The particle with highest log probability
    """
    n_particles = particles.shape[0]
    
    # Compute log probability for each particle
    log_probs = jnp.array([log_prob_fn(particles[i]) for i in range(n_particles)])
    
    # Find the particle with the highest log probability
    map_idx = jnp.argmax(log_probs)
    
    return particles[map_idx], log_probs[map_idx]


def map_estimate_with_optimization(particles, log_prob_fn, n_steps=100, step_size=0.01):
    """
    Refine MAP estimate by starting from the best particle and performing
    gradient ascent on the log probability.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
        n_steps: Number of optimization steps
        step_size: Step size for gradient ascent
    
    Returns:
        The refined MAP estimate after optimization
    """
    # Start with the best particle
    map_particle, _ = map_estimate_from_particles(particles, log_prob_fn)
    
    # Define gradient of log probability
    grad_log_prob = jax.grad(log_prob_fn)
    
    # Perform gradient ascent to refine the MAP estimate
    x = map_particle
    for _ in range(n_steps):
        grad = grad_log_prob(x)
        x = x + step_size * grad
    
    return x, log_prob_fn(x)


def plot_parameter_matrix(particles, true_params=None, max_params=6, figsize=(12, 10)):
    """
    Create a matrix plot showing pairwise relationships between parameters
    
    Args:
        particles: Array of shape (n_particles, n_dims)
        true_params: optional true parameter values
        max_params: maximum number of parameters to show
        figsize: figure size
    """
    n_dims = particles.shape[1]
    n_show = min(max_params, n_dims)
    
    fig, axes = plt.subplots(n_show, n_show, figsize=figsize)
    if n_show == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        for j in range(n_show):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show 1D marginal distribution
                plot_svgd_posterior_1d(
                    particles[:, i],
                    true_params=true_params[i] if true_params is not None and len(true_params) > i else None,
                    ax=ax,
                    title=f"Parameter {i}"
                )
            else:
                # Off-diagonal: show 2D scatter plot
                ax.scatter(particles[:, j], particles[:, i], alpha=0.5, s=2, edgecolor='none')
                
                if true_params is not None and len(true_params) > max(i, j):
                    ax.scatter(true_params[j], true_params[i], color='red', s=50, marker='*', 
                             label='True value')
                
                ax.set_xlabel(f'Parameter {j}')
                ax.set_ylabel(f'Parameter {i}')
                
            # Remove ticks for cleaner look
            if i < n_show - 1:
                ax.set_xlabel('')
            if j > 0:
                ax.set_ylabel('')
    
    plt.suptitle(f'Parameter Matrix Plot (showing {n_show}/{n_dims} parameters)', fontsize=14)
    plt.tight_layout()
    return fig


def animate_parameter_pairs(particle_history, param_pairs=None, true_params=None, 
                           figsize=(15, 5), save_as_gif=None):
    """
    Animate multiple parameter pairs simultaneously
    
    Args:
        particle_history: array of shape (n_iterations, n_particles, n_dims)
        param_pairs: list of tuples [(i1,j1), (i2,j2), ...] for parameter pairs to show
        true_params: optional true parameter values
        figsize: figure size
    """
    n_dims = particle_history.shape[2]
    
    # Default to first few parameter pairs if not specified
    if param_pairs is None:
        param_pairs = [(i, i+1) for i in range(0, min(6, n_dims-1), 2)]
    
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
        x_data = particle_history[:, :, j].flatten()
        y_data = particle_history[:, :, i].flatten()
        
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
            ax.scatter(true_params[j], true_params[i], color='red', s=50, marker='*', 
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
            particles_2d = particle_history[frame, :, [j, i]]  # Note: [j, i] for x, y
            scatters[plot_idx].set_offsets(particles_2d)
            texts[plot_idx].set_text(f'Iter: {frame}')
        return scatters + texts
    
    anim = FuncAnimation(fig, update, frames=particle_history.shape[0],
                         init_func=init, blit=True, interval=100)
    
    plt.tight_layout()
    
    if save_as_gif:
        anim.save(save_as_gif, writer='pillow', fps=10)
    
    from IPython.display import HTML
    return HTML(anim.to_jshtml())

