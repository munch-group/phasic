from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

class VariableDimPTDDecoder(eqx.Module):
    """Neural network decoder mapping latent variables to PTD parameters.

    Uses a simple MLP with tanh activations to transform a latent vector
    into phase-type distribution parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for weight initialization.
    latent_dim : int
        Dimensionality of the latent input space.
    k : int
        Number of phases in the phase-type distribution.
    m : int
        Number of absorbing states.
    param_dim : int
        Total number of output parameters.

    Attributes
    ----------
    layers : list[eqx.nn.Linear]
        MLP layers (latent_dim -> 64 -> 32 -> param_dim).
    k : int
        Number of phases.
    m : int
        Number of absorbing states.
    param_dim : int
        Output parameter dimensionality.
    """
    layers: list
    k: int
    m: int
    param_dim: int

    def __init__(self, key: jax.Array, latent_dim: int, k: int, m: int, param_dim: int) -> None:
        self.param_dim = param_dim
        self.k = k
        self.m = m

        # Simple MLP
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(latent_dim, 64, key=keys[0]),
            eqx.nn.Linear(64, 32, key=keys[1]),
            eqx.nn.Linear(32, self.param_dim, key=keys[2])
        ]

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the decoder network.

        Parameters
        ----------
        z : jnp.ndarray
            Latent input vector of shape ``(latent_dim,)``.

        Returns
        -------
        jnp.ndarray
            PTD parameters of shape ``(param_dim,)``.
        """
        x = z
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x

class LessThanOneDecoder(eqx.Module):
    """Decoder that outputs two probabilities summing to less than one.

    Uses a softmax over three logits and returns the first two components,
    guaranteeing ``a + b < 1`` with ``a, b > 0``.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent input space.
    key : jax.random.PRNGKey
        Random key for weight initialization.

    Attributes
    ----------
    linear : eqx.nn.Linear
        Linear layer mapping latent space to 3 logits.
    """
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(latent_dim, 3, key=key)

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning two probabilities that sum to less than one.

        Parameters
        ----------
        z : jnp.ndarray
            Latent input vector of shape ``(latent_dim,)``.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Pair ``(a, b)`` where ``a, b > 0`` and ``a + b < 1``.
        """
        probs = jax.nn.softmax(self.linear(z))
        return probs[0], probs[1]  # a, b ∈ (0,1), a + b < 1

class SumToOneDecoder(eqx.Module):
    """Decoder that outputs two probabilities summing to exactly one.

    Uses a sigmoid to produce ``s``, then returns ``(s, 1 - s)``.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent input space.
    key : jax.random.PRNGKey
        Random key for weight initialization.

    Attributes
    ----------
    linear : eqx.nn.Linear
        Linear layer mapping latent space to 1 logit.
    """
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(latent_dim, 1, key=key)

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning two probabilities summing to one.

        Parameters
        ----------
        z : jnp.ndarray
            Latent input vector of shape ``(latent_dim,)``.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Pair ``(s, 1 - s)`` where ``s`` is in ``(0, 1)``.
        """
        s = jax.nn.sigmoid(self.linear(z)[0])
        return s, 1.0 - s

class IndependentProbDecoder(eqx.Module):
    """Decoder that outputs two independent probabilities.

    Each probability is produced by applying sigmoid to an independent logit,
    so ``a`` and ``b`` are unconstrained relative to each other.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent input space.
    key : jax.random.PRNGKey
        Random key for weight initialization.

    Attributes
    ----------
    linear : eqx.nn.Linear
        Linear layer mapping latent space to 2 logits.
    """
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(latent_dim, 2, key=key)

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning two independent probabilities.

        Parameters
        ----------
        z : jnp.ndarray
            Latent input vector of shape ``(latent_dim,)``.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Pair ``(a, b)`` where each is independently in ``(0, 1)``.
        """
        logits = self.linear(z)  # shape (2,)
        a, b = jax.nn.sigmoid(logits[0]), jax.nn.sigmoid(logits[1])
        return a, b
