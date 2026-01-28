"""Optional Optax integration for SVGD optimization.

This module provides wrapper classes to use Optax optimizers with phasic's SVGD
interface. Optax is a gradient processing and optimization library for JAX that
provides a wide variety of optimizers and gradient transformations.

To use this module, install phasic with the optax extra:
    pip install phasic[optax]

Example
-------
>>> from phasic import optax_adam, SVGD
>>>
>>> # Use Optax Adam optimizer
>>> svgd = SVGD(
...     model=model,
...     observed_data=observations,
...     theta_dim=2,
...     optimizer=optax_adam(0.001)
... )
>>> svgd.fit()
>>>
>>> # Use Optax with gradient clipping
>>> import optax
>>> from phasic import optax_chain
>>> optimizer = optax_chain(
...     optax.clip_by_global_norm(1.0),
...     optax.adam(0.001)
... )
>>> svgd = SVGD(model=model, observed_data=data, theta_dim=2, optimizer=optimizer)
"""

try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False
    optax = None


class OptaxOptimizer:
    """Wrapper to use Optax optimizers with phasic's SVGD interface.

    This class adapts Optax's functional optimizer interface to phasic's
    object-oriented optimizer interface expected by SVGD.

    Parameters
    ----------
    optax_optimizer : optax.GradientTransformation
        An Optax optimizer or chain of gradient transformations.

    Attributes
    ----------
    optimizer : optax.GradientTransformation
        The underlying Optax optimizer.
    state : optax.OptState or None
        The optimizer state, initialized by reset().

    Examples
    --------
    >>> import optax
    >>> from phasic.optax_wrapper import OptaxOptimizer
    >>>
    >>> # Wrap any Optax optimizer
    >>> optimizer = OptaxOptimizer(optax.adam(learning_rate=0.001))
    >>>
    >>> # Or use with gradient clipping
    >>> optimizer = OptaxOptimizer(optax.chain(
    ...     optax.clip_by_global_norm(1.0),
    ...     optax.adam(0.001)
    ... ))
    """

    def __init__(self, optax_optimizer):
        """Initialize wrapper with an Optax optimizer.

        Parameters
        ----------
        optax_optimizer : optax.GradientTransformation
            An Optax optimizer created via optax.adam(), optax.sgd(), etc.

        Raises
        ------
        ImportError
            If optax is not installed.
        """
        if not HAS_OPTAX:
            raise ImportError(
                "Optax is required for this optimizer. "
                "Install with: pip install phasic[optax]"
            )
        self.optimizer = optax_optimizer
        self.state = None
        self._t = 0
        self._params = None  # Track params for optimizers that need them (e.g., adamw)

    @property
    def lr(self):
        """Current learning rate (for display/logging).

        Note: This is approximate for Optax optimizers with schedules,
        as Optax handles learning rate internally.
        """
        # Optax doesn't expose learning rate directly; return placeholder
        return None

    @property
    def t(self):
        """Current timestep."""
        return self._t

    def reset(self, shape, params=None):
        """Initialize optimizer state for given particle shape.

        Called at the start of optimization to initialize the Optax state.

        Parameters
        ----------
        shape : tuple
            Shape of particles array (n_particles, theta_dim).
        params : array, optional
            Initial parameter values. If provided, used for optimizers
            that need current params (e.g., adamw for weight decay).
        """
        import jax.numpy as jnp
        if params is None:
            params = jnp.zeros(shape)
        self._params = params
        self.state = self.optimizer.init(params)
        self._t = 0

    def step(self, phi, params=None, particles=None):
        """Compute update using Optax optimizer.

        Parameters
        ----------
        phi : array (n_particles, theta_dim)
            SVGD gradient direction: (K @ grad_log_p + sum(grad_K)) / n_particles
        params : array, optional
            Current parameter values. Required for optimizers with weight decay
            (e.g., adamw). If not provided, uses internally tracked params.
        particles : array, optional
            Alias for params, for compatibility with phasic optimizer interface.

        Returns
        -------
        update : array (n_particles, theta_dim)
            Scaled update to add to particles.
        """
        self._t += 1
        if params is None:
            params = particles
        if params is None:
            params = self._params
        # Optax convention: updates = -lr * adapted_grad (gradient descent).
        # SVGD convention: phi is the ascent direction, update = +lr * adapted_phi.
        # Negate the returned updates to convert descent -> ascent.
        updates, self.state = self.optimizer.update(phi, self.state, params)
        updates = -updates
        # Update internal params tracking
        if params is not None:
            self._params = params + updates
        return updates


# Convenience factory functions

def optax_adam(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Create Optax Adam optimizer wrapped for phasic.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=0.001
        Learning rate. Can be a scalar or an Optax schedule.
    b1 : float, default=0.9
        Exponential decay rate for first moment.
    b2 : float, default=0.999
        Exponential decay rate for second moment.
    eps : float, default=1e-8
        Small constant for numerical stability.

    Returns
    -------
    OptaxOptimizer
        Wrapped Adam optimizer compatible with phasic SVGD.

    Examples
    --------
    >>> from phasic import optax_adam, SVGD
    >>> svgd = SVGD(model=model, data=data, theta_dim=2, optimizer=optax_adam(0.001))
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.adam(learning_rate, b1, b2, eps))


def optax_adamw(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    """Create Optax AdamW optimizer wrapped for phasic.

    AdamW implements Adam with decoupled weight decay regularization,
    which can help prevent overfitting.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=0.001
        Learning rate.
    b1 : float, default=0.9
        Exponential decay rate for first moment.
    b2 : float, default=0.999
        Exponential decay rate for second moment.
    eps : float, default=1e-8
        Small constant for numerical stability.
    weight_decay : float, default=0.01
        Weight decay coefficient.

    Returns
    -------
    OptaxOptimizer
        Wrapped AdamW optimizer compatible with phasic SVGD.
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.adamw(learning_rate, b1, b2, eps, weight_decay=weight_decay))


def optax_sgd(learning_rate=0.01, momentum=0.0, nesterov=False):
    """Create Optax SGD optimizer wrapped for phasic.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=0.01
        Learning rate.
    momentum : float, default=0.0
        Momentum coefficient. Set to 0.9 for standard momentum.
    nesterov : bool, default=False
        Whether to use Nesterov momentum.

    Returns
    -------
    OptaxOptimizer
        Wrapped SGD optimizer compatible with phasic SVGD.
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.sgd(learning_rate, momentum, nesterov))


def optax_rmsprop(learning_rate=0.001, decay=0.9, eps=1e-8, momentum=0.0):
    """Create Optax RMSprop optimizer wrapped for phasic.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=0.001
        Learning rate.
    decay : float, default=0.9
        Decay rate for exponential moving average.
    eps : float, default=1e-8
        Small constant for numerical stability.
    momentum : float, default=0.0
        Momentum coefficient.

    Returns
    -------
    OptaxOptimizer
        Wrapped RMSprop optimizer compatible with phasic SVGD.
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.rmsprop(learning_rate, decay, eps, momentum=momentum))


def optax_adagrad(learning_rate=0.01, initial_accumulator_value=0.1, eps=1e-7):
    """Create Optax Adagrad optimizer wrapped for phasic.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=0.01
        Learning rate.
    initial_accumulator_value : float, default=0.1
        Initial value for accumulator.
    eps : float, default=1e-7
        Small constant for numerical stability.

    Returns
    -------
    OptaxOptimizer
        Wrapped Adagrad optimizer compatible with phasic SVGD.
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.adagrad(learning_rate, initial_accumulator_value, eps))


def optax_chain(*transforms):
    """Create chained Optax transforms wrapped for phasic.

    This allows combining multiple gradient transformations, such as
    gradient clipping followed by an optimizer.

    Parameters
    ----------
    *transforms : optax.GradientTransformation
        Optax gradient transformations to chain together.

    Returns
    -------
    OptaxOptimizer
        Wrapped chained optimizer compatible with phasic SVGD.

    Examples
    --------
    >>> import optax
    >>> from phasic import optax_chain
    >>>
    >>> # Adam with gradient clipping
    >>> optimizer = optax_chain(
    ...     optax.clip_by_global_norm(1.0),
    ...     optax.adam(0.001)
    ... )
    >>>
    >>> # Learning rate warmup with Adam
    >>> schedule = optax.warmup_cosine_decay_schedule(
    ...     init_value=0.0,
    ...     peak_value=0.01,
    ...     warmup_steps=100,
    ...     decay_steps=1000
    ... )
    >>> optimizer = optax_chain(
    ...     optax.adam(schedule)
    ... )
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.chain(*transforms))


def optax_lion(learning_rate=1e-4, b1=0.9, b2=0.99, weight_decay=0.0):
    """Create Optax Lion optimizer wrapped for phasic.

    Lion (Evolved Sign Momentum) is a memory-efficient optimizer discovered
    through program search. It uses sign-based updates and typically requires
    a lower learning rate than Adam.

    Parameters
    ----------
    learning_rate : float or optax.Schedule, default=1e-4
        Learning rate. Typically 3-10x smaller than Adam.
    b1 : float, default=0.9
        Exponential decay rate for first moment.
    b2 : float, default=0.99
        Exponential decay rate for the gradient moving average.
    weight_decay : float, default=0.0
        Weight decay coefficient.

    Returns
    -------
    OptaxOptimizer
        Wrapped Lion optimizer compatible with phasic SVGD.

    References
    ----------
    Chen et al. (2023). Symbolic Discovery of Optimization Algorithms.
    arXiv:2302.06675. https://arxiv.org/abs/2302.06675
    """
    if not HAS_OPTAX:
        raise ImportError(
            "Optax is required. Install with: pip install phasic[optax]"
        )
    return OptaxOptimizer(optax.lion(learning_rate, b1, b2, weight_decay=weight_decay))
