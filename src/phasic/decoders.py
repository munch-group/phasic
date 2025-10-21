import equinox as eqx
import jax
import jax.numpy as jnp

class VariableDimPTDDecoder(eqx.Module):
    """Neural network to decode latent variables to PTD parameters"""
    layers: list
    k: int
    m: int
    param_dim: int
    
    # def __init__(self, key, latent_dim, k, m):
    #     self.param_dim = calculate_param_dim(k, m)
    def __init__(self, key, latent_dim, k, m, param_dim):
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
    
    def __call__(self, z):
        x = z
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x
    
class LessThanOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 3, key=key)

    def __call__(self, z: jnp.ndarray):
        probs = jax.nn.softmax(self.linear(z))
        return probs[0], probs[1]  # a, b ∈ (0,1), a + b < 1
    
class SumToOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 1, key=key)

    def __call__(self, z: jnp.ndarray):
        s = jax.nn.sigmoid(self.linear(z)[0])
        return s, 1.0 - s

class IndependentProbDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 2, key=key)

    def __call__(self, z: jnp.ndarray):
        logits = self.linear(z)  # shape (2,)
        a, b = jax.nn.sigmoid(logits[0]), jax.nn.sigmoid(logits[1])
        return a, b