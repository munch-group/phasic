"""Check JAX device configuration"""
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")
print(f"Default backend: {jax.default_backend()}")

# Test if code runs on CPU
x = jnp.array([1.0, 2.0, 3.0])
print(f"Array device: {x.device()}")
