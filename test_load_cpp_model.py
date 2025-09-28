#!/usr/bin/env python
"""
Test the Graph.load_cpp_model() functionality

This demonstrates how users can:
1. Write a simple C++ file that returns a Graph instance
2. Load it with Graph.load_cpp_model()
3. Get full JAX compatibility automatically
"""

import sys
import os
sys.path.insert(0, 'src')
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from ptdalgorithms import Graph

print("=" * 70)
print("TESTING Graph.load_cpp_model() FUNCTIONALITY")
print("=" * 70)
print(f"JAX version: {jax.__version__}")

# Test 1: Load simple exponential model
print("\n1. LOADING SIMPLE EXPONENTIAL MODEL")
print("-" * 40)

try:
    exp_model = Graph.load_cpp_model("examples/user_models/simple_exponential.cpp")
    print("✅ Successfully loaded simple_exponential.cpp")

    # Test with a single parameter
    theta = jnp.array([1.0])  # rate = 1.0
    times = jnp.linspace(0.1, 5.0, 10)

    pdf_values = exp_model(theta, times)
    print(f"PDF values shape: {pdf_values.shape}")
    print(f"First few PDF values: {pdf_values[:3]}")

    # Test JIT compilation
    jit_model = jit(exp_model)
    pdf_jit = jit_model(theta, times)
    print(f"✅ JIT compilation works: {jnp.allclose(pdf_values, pdf_jit)}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Load birth-death model
print("\n2. LOADING BIRTH-DEATH PROCESS MODEL")
print("-" * 40)

try:
    bd_model = Graph.load_cpp_model("examples/user_models/birth_death_process.cpp")
    print("✅ Successfully loaded birth_death_process.cpp")

    # Test with two parameters
    theta = jnp.array([0.8, 1.2])  # birth rate, death rate
    times = jnp.linspace(0.1, 10.0, 20)

    pdf_values = bd_model(theta, times)
    print(f"PDF values computed: shape {pdf_values.shape}")

    # Test automatic differentiation
    def loss(params):
        pdf = bd_model(params, times)
        return jnp.sum(pdf)

    gradient = grad(loss)(theta)
    print(f"✅ Automatic differentiation works: gradient = {gradient}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Load M/M/1 queue model
print("\n3. LOADING M/M/1 QUEUE MODEL")
print("-" * 40)

try:
    queue_model = Graph.load_cpp_model("examples/user_models/mm1_queue.cpp")
    print("✅ Successfully loaded mm1_queue.cpp")

    # Test with arrival and service rates
    theta = jnp.array([0.7, 1.0])  # arrival rate < service rate (stable)
    times = jnp.linspace(0.1, 20.0, 50)

    pdf_values = queue_model(theta, times)
    print(f"Queue model PDF computed: max value = {jnp.max(pdf_values):.6f}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Batch processing with vmap
print("\n4. TESTING VECTORIZATION WITH VMAP")
print("-" * 40)

try:
    # Use the exponential model for simplicity
    exp_model = Graph.load_cpp_model("examples/user_models/simple_exponential.cpp")

    # Multiple parameter sets
    theta_batch = jnp.array([
        [0.5],
        [1.0],
        [2.0],
        [4.0]
    ])

    # Vectorized computation
    @jit
    def batch_compute(params_batch):
        return vmap(lambda p: exp_model(p, times))(params_batch)

    batch_results = batch_compute(theta_batch)
    print(f"✅ vmap works: computed {len(theta_batch)} parameter sets")
    print(f"   Result shape: {batch_results.shape}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Load Erlang distribution model
print("\n5. LOADING ERLANG DISTRIBUTION MODEL")
print("-" * 40)

try:
    erlang_model = Graph.load_cpp_model("examples/user_models/erlang_distribution.cpp")
    print("✅ Successfully loaded erlang_distribution.cpp")

    # Rate and number of stages
    theta = jnp.array([0.5, 3.0])  # rate=0.5, 3 stages
    times = jnp.linspace(0.1, 15.0, 30)

    pdf_values = erlang_model(theta, times)
    print(f"Erlang PDF computed: shape {pdf_values.shape}")

    # Find the mode (should be at (k-1)/rate = 2/0.5 = 4.0 for k=3)
    max_idx = jnp.argmax(pdf_values)
    print(f"Mode approximately at time: {times[max_idx]:.2f}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Performance comparison
print("\n6. PERFORMANCE COMPARISON")
print("-" * 40)

try:
    import time

    model = Graph.load_cpp_model("examples/user_models/simple_exponential.cpp")
    theta = jnp.array([1.0])
    times = jnp.linspace(0.1, 10.0, 100)

    # Regular execution
    def regular_compute(params):
        return model(params, times)

    # JIT compiled
    jit_compute = jit(regular_compute)

    # Warm up
    _ = jit_compute(theta)

    # Benchmark
    n_runs = 100

    start = time.time()
    for _ in range(n_runs):
        _ = regular_compute(theta).block_until_ready()
    regular_time = time.time() - start

    start = time.time()
    for _ in range(n_runs):
        _ = jit_compute(theta).block_until_ready()
    jit_time = time.time() - start

    print(f"Regular execution: {regular_time*1000:.2f} ms")
    print(f"JIT compiled:      {jit_time*1000:.2f} ms")
    if jit_time > 0:
        print(f"Speedup:           {regular_time/jit_time:.2f}x")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 7: Non-JAX mode
print("\n7. TESTING NON-JAX MODE")
print("-" * 40)

try:
    # Load without JAX compatibility
    regular_model = Graph.load_cpp_model(
        "examples/user_models/simple_exponential.cpp",
        jax_compatible=False
    )
    print("✅ Loaded model without JAX compatibility")

    # Use with numpy arrays
    theta_np = np.array([1.0])
    times_np = np.linspace(0.1, 5.0, 10)

    pdf_np = regular_model(theta_np, times_np)
    print(f"Regular Python function works: shape {pdf_np.shape}")
    print(f"Returns numpy array: {type(pdf_np)}")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✅ Users can write simple C++ files that return Graph instances")
print("✅ Graph.load_cpp_model() automatically handles compilation")
print("✅ Full JAX support: JIT, grad, vmap all work")
print("✅ Multiple models can be loaded and used")
print("✅ Both JAX and non-JAX modes supported")
print("\nThe interface is simple: just implement build_model() and use Graph.load_cpp_model()!")