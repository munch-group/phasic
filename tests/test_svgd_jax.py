"""
SVGD Configuration Options Showcase
====================================

This script demonstrates the different configuration options for SVGD in PtDAlgorithms.
It shows how to control JAX compilation, parallelization, and device usage explicitly.

The explicit configuration system eliminates all silent fallback behavior, giving users
full control over how SVGD executes.
"""

from ptdalgorithms import Graph, SVGD
import ptdalgorithms as ptd
import numpy as np
import jax.numpy as jnp
import jax
import time


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_config_info():
    """Print current configuration and available options"""
    print("Current Configuration:")
    config = ptd.get_config()
    print(f"  jax={config.jax}, jit={config.jit}, backend='{config.backend}'")

    print("\nAvailable Options on This System:")
    opts = ptd.get_available_options()
    print(f"  JAX available: {opts['jax']}")
    print(f"  Backends: {opts['backends']}")
    print(f"  Platforms: {opts['platforms']}")
    print(f"  Number of JAX devices: {len(jax.devices())}")
    print(f"  Device list: {jax.devices()}")


def build_simple_exponential():
    """
    Build simple exponential distribution for quick testing.

    Graph structure: S → [2] → [1]
    - S: Starting vertex (implicit state [0])
    - [2]: Initial transient state
    - [1]: Absorbing state
    - Transition rate: θ (single parameter)
    """
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])

    start.add_edge(v2, 1.0)  # S → [2] with probability 1.0
    v2.add_edge_parameterized(v1, 0.0, [1.0])  # [2] → [1] with rate θ

    return g


def build_coalescent(nr_samples=None):

    def coalescent(state, nr_samples=None):
        if not state.size:
            ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]  # should the ipv be parameterized?
            # ipv = [[[nr_samples]+[0]*nr_samples, 1, [1]]]  # should the ipv be parameterized?
            return ipv
        else:
            transitions = []
            for i in range(nr_samples):
                for j in range(i, nr_samples):            
                    same = int(i == j)
                    if same and state[i] < 2:
                        continue
                    if not same and (state[i] < 1 or state[j] < 1):
                        continue 
                    new = state.copy()
                    new[i] -= 1
                    new[j] -= 1
                    new[i+j+1] += 1
                    transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
            return transitions

    # Build the graph
    nr_samples = 10
    graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

    return graph


def generate_test_data(true_theta=2.0, n_obs=50):
    """Generate observed data from true model"""
    graph = build_simple_exponential()
    graph.update_parameterized_weights([true_theta])
    observed_data = np.array(graph.sample(n_obs))
    return observed_data


def run_svgd_test(name, description, svgd_kwargs, observed_data):
    """
    Run SVGD with given configuration and print results.

    Parameters
    ----------
    name : str
        Test name
    description : str
        What this test demonstrates
    svgd_kwargs : dict
        SVGD constructor arguments
    observed_data : array
        Observed data for inference
    """
    print(f"\n{name}")
    print("-" * 80)
    print(f"Description: {description}")
    print(f"\nSVGD Parameters:")
    for key, val in svgd_kwargs.items():
        if key not in ['model', 'observed_data']:
            print(f"  {key}={val}")

    try:
        # Create SVGD instance
        svgd = SVGD(**svgd_kwargs)

        start = time.time()

        # Print what was selected
        print(f"\nSelected Configuration:")
        print(f"  jit_enabled: {svgd.jit_enabled}")
        print(f"  parallel_mode: {svgd.parallel_mode}")
        print(f"  n_devices: {svgd.n_devices}")

        # Run SVGD
        print(f"\nRunning SVGD...")
        svgd.fit()

        # Print results
        print(f"\n✓ Success!")
        print(f"  Posterior mean: {svgd.theta_mean}")
        print(f"  Posterior std:  {svgd.theta_std}")

        return time.time() - start
        # return True

    except Exception as e:
        print(f"\n✗ Failed with error:")
        print(f"  {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    """Run comprehensive SVGD configuration showcase"""

    print_section("SVGD Configuration Options Showcase")

    # Check current system configuration
    print_config_info()

    # Generate test data
    print_section("Generating Test Data")
    true_theta = 5.0
    nr_samples = 10000
    _graph = build_coalescent(nr_samples=nr_samples)
    _graph.update_parameterized_weights([true_theta])
    observed_data = _graph.sample(10)

    # observed_data = generate_test_data(true_theta, n_obs=50)
    print(f"Generated {len(observed_data)} observations from Exponential({true_theta})")
    print(f"Sample mean: {np.mean(observed_data):.3f} (theoretical: {1/true_theta:.3f})")


    # Build parameterized model
    graph = build_coalescent(nr_samples=nr_samples)
    # graph = build_simple_exponential()
    model = Graph.pmf_from_graph(graph, discrete=False, param_length=1)

    # Common parameters for all tests
    common_params = {
        'model': model,
        'observed_data': observed_data,
        'theta_dim': 1,
        'n_particles': 20,
        'n_iterations': 100,
    }

    # ==========================================================================
    # Test 1: Default Configuration (Auto-Select Everything)
    # ==========================================================================
    print_section("Test 1: Default Configuration (Auto-Select)")

    running_time = run_svgd_test(
        name="Auto-Select (Defaults)",
        description="""
        Uses default configuration - all parameters set to None.
        SVGD automatically selects:
        - jit: from config (default True)
        - parallel: 'pmap' if multiple devices, 'vmap' otherwise
        - n_devices: all available devices for pmap

        How it works under the hood:
        - jit_enabled comes from ptd.get_config().jit
        - parallel_mode auto-detects based on len(jax.devices())
        - For single device: uses vmap (vectorize across particles)
        - For multiple devices: uses pmap (parallelize across devices)
        """,
        svgd_kwargs={
            **common_params,
            'verbose': True,
            # jit=None (default)
            # parallel=None (default)
            # n_devices=None (default)
        },
        observed_data=observed_data
    )
    print(f"Running time: {running_time:.2f} seconds") if running_time else "N/A"


    # ==========================================================================
    # Test 2: Explicit Single-Device (vmap)
    # ==========================================================================
    print_section("Test 2: Explicit Single-Device Vectorization")

    running_time = run_svgd_test(
        name="Explicit vmap",
        description="""
        Explicitly requests single-device vectorization.

        How it works under the hood:
        - jit=True: JIT compiles gradient function for speed
        - parallel='vmap': Uses JAX's vmap to vectorize gradient computation
          across all particles on a single device
        - All particles computed in parallel on one device
        - Best for: Small-medium particle counts, single GPU/CPU

        Implementation:
        - svgd_step() calls: vmap(grad(log_prob_fn))(particles)
        - Each particle gradient computed in parallel via SIMD
        """,
        svgd_kwargs={
            **common_params,
            'jit': True,
            'parallel': 'vmap',
            'verbose': False,
        },
        observed_data=observed_data
    )
    print(f"Running time: {running_time:.2f} seconds") if running_time else "N/A"


    # ==========================================================================
    # Test 3: Multi-Device (pmap) if available
    # ==========================================================================
    print_section("Test 3: Multi-Device Parallelization")

    n_devices = len(jax.devices())
    if n_devices > 1:
        running_time = run_svgd_test(
            name="Explicit pmap (Multi-Device)",
            description=f"""
            Explicitly requests multi-device parallelization.
            System has {n_devices} devices available.

            How it works under the hood:
            - jit=True: JIT compiles gradient function
            - parallel='pmap': Uses JAX's pmap to distribute computation
              across multiple devices
            - Particles are sharded: {common_params['n_particles'] // n_devices} particles per device
            - Each device computes gradients for its shard in parallel

            Implementation:
            - Particles reshaped to (n_devices, particles_per_device, theta_dim)
            - svgd_step() calls: pmap(vmap(grad(log_prob_fn)))(particles_sharded)
            - Outer pmap: parallelize across devices
            - Inner vmap: vectorize within each device
            - Results gathered and reshaped back

            Best for: Large particle counts, multiple GPUs/TPUs
            """,
            svgd_kwargs={
                **common_params,
                'jit': True,
                'parallel': 'pmap',
                'n_devices': n_devices,
                'n_particles': n_devices * 10,  # Ensure divisible by n_devices
                'verbose': False,
            },
            observed_data=observed_data
        )
        print(f"Running time: {running_time:.2f} seconds") if running_time else "N/A"
    else:
        print(f"Skipping pmap test (only {n_devices} device available)")
        print("To test pmap, set PTDALG_CPUS environment variable before import:")
        print("  export PTDALG_CPUS=4")
        print("  python tests/test_svgd_jax.py")

    # ==========================================================================
    # Test 4: No Parallelization (Sequential, for debugging)
    # ==========================================================================
    print_section("Test 4: No Parallelization (Sequential)")

    running_time = run_svgd_test(
        name="Sequential (parallel='none')",
        description="""
        No parallelization - sequential gradient computation.

        How it works under the hood:
        - jit=True: Still compiles gradient function
        - parallel='none': Computes gradients one particle at a time
        - Python loop over particles

        Implementation:
        - svgd_step() calls: [grad_fn(p) for p in particles]
        - Each particle gradient computed sequentially
        - Results stacked into array

        Best for: Debugging, understanding behavior, tiny models
        Warning: VERY SLOW for large particle counts
        """,
        svgd_kwargs={
            **common_params,
            'jit': True,
            'parallel': 'none',
            'n_particles': 10,  # Smaller for speed
            'n_iterations': 5,   # Fewer iterations
            'verbose': False,
        },
        observed_data=observed_data
    )
    print(f"Running time: {running_time:.2f} seconds") if running_time else "N/A"
    print()

    # ==========================================================================
    # Test 5: No JIT (Interpreted Mode)
    # ==========================================================================
    print_section("Test 5: No JIT Compilation (Interpreted Mode)")

    running_time = run_svgd_test(
        name="No JIT (jit=False)",
        description="""
        Disable JIT compilation - run in interpreted mode.

        How it works under the hood:
        - jit=False: No compilation, pure Python/NumPy execution
        - parallel='vmap': Still uses vmap for vectorization (works without JIT)
        - Gradients computed via JAX autograd in eager mode

        Implementation:
        - No jit() wrapper around gradient function
        - Each SVGD step executes immediately without compilation
        - Still gets automatic differentiation from JAX

        Best for: Rapid prototyping, debugging, small models
        Warning: Much slower than JIT for production use
        """,
        svgd_kwargs={
            **common_params,
            'jit': False,
            'parallel': 'vmap',
            'n_particles': 20,
            'n_iterations': 5,  # Fewer iterations since it's slow
            'verbose': False,
        },
        observed_data=observed_data
    )
    print(f"Running time: {running_time:.2f} seconds") if running_time else "N/A"


    # ==========================================================================
    # Test 6: Backward Compatibility (precompile parameter)
    # ==========================================================================
    print_section("Test 6: Backward Compatibility")

    run_svgd_test(
        name="Legacy precompile=True",
        description="""
        Legacy parameter for backward compatibility.

        How it works under the hood:
        - precompile=True: Sets jit_enabled=True internally
        - Maintained for backward compatibility with old code
        - Equivalent to jit=True in new API

        Implementation:
        - if precompile and not jit: self.jit_enabled = True
        - Warning printed if both precompile and jit specified

        Recommendation: Use jit parameter for new code
        """,
        svgd_kwargs={
            **common_params,
            'precompile': True,  # Old parameter
            'verbose': False,
        },
        observed_data=observed_data
    )

    # ==========================================================================
    # Test 7: Error Handling - Invalid Configurations
    # ==========================================================================
    print_section("Test 7: Error Handling")

    print("\n7a. Invalid parallel mode")
    print("-" * 80)
    run_svgd_test(
        name="Invalid parallel='invalid'",
        description="Should raise ValueError with helpful message",
        svgd_kwargs={
            **common_params,
            'parallel': 'invalid',
            'verbose': False,
        },
        observed_data=observed_data
    )

    print("\n\n7b. Excessive n_devices")
    print("-" * 80)
    run_svgd_test(
        name="n_devices=999 (exceeds available)",
        description="Should raise PTDConfigError with actionable fix",
        svgd_kwargs={
            **common_params,
            'parallel': 'pmap',
            'n_devices': 999,
            'verbose': False,
        },
        observed_data=observed_data
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_section("Summary of Configuration Options")

    print("""
Configuration Parameters for SVGD:
-----------------------------------

1. jit : bool or None (default=None)
   - True: Enable JIT compilation for speed
   - False: Interpreted mode for debugging
   - None: Use value from ptd.get_config().jit (default True)

2. parallel : str or None (default=None)
   - 'vmap': Single-device vectorization (good for 1 GPU/CPU)
   - 'pmap': Multi-device parallelization (good for multiple GPUs/CPUs)
   - 'none': Sequential computation (debugging only)
   - None: Auto-select ('pmap' if multi-device, 'vmap' otherwise)

3. n_devices : int or None (default=None)
   - Number of devices to use for pmap
   - None: Use all available devices
   - Only used when parallel='pmap'
   - Must be <= len(jax.devices())

4. precompile : bool (default=True, deprecated)
   - Legacy parameter, use jit instead
   - Equivalent to jit=True


Under the Hood - How It Works:
-------------------------------

JAX Device Configuration:
- Set via PTDALG_CPUS environment variable before import
- Or via ptd.configure() before importing SVGD
- Controls len(jax.devices()) for pmap

Gradient Computation Strategies:

1. vmap (Single Device):
   grad_log_p = vmap(grad(log_prob_fn))(particles)
   - Vectorizes across all particles on one device
   - Uses SIMD parallelization

2. pmap (Multi Device):
   particles_sharded = particles.reshape(n_devices, particles_per_device, -1)
   grad_log_p = pmap(vmap(grad(log_prob_fn)))(particles_sharded)
   - Outer pmap: parallelize across devices
   - Inner vmap: vectorize within each device
   - Best utilization of multiple GPUs/CPUs

3. none (Sequential):
   grad_log_p = [grad(log_prob_fn)(p) for p in particles]
   - Loop over particles
   - Useful for debugging only


Performance Characteristics:
----------------------------

Setup                     | Relative Speed | Use Case
--------------------------|----------------|---------------------------
jit=True, parallel='vmap' | 1.0x (baseline)| Single GPU, production
jit=True, parallel='pmap' | 1.5-3.0x       | Multiple GPUs, large scale
jit=False, parallel='vmap'| 0.1-0.5x       | Debugging, prototyping
jit=True, parallel='none' | 0.01-0.05x     | Debugging only
jit=False, parallel='none'| 0.001-0.01x    | Understanding behavior only


Recommended Configurations:
---------------------------

Production (single device):
    SVGD(model, data, theta_dim=d, jit=True, parallel='vmap')

Production (multi-device):
    SVGD(model, data, theta_dim=d, jit=True, parallel='pmap', n_devices=8)

Debugging:
    SVGD(model, data, theta_dim=d, jit=False, parallel='none',
         n_particles=10, n_iterations=5)

Default (auto-select):
    SVGD(model, data, theta_dim=d)  # Sensible defaults chosen automatically
    """)

    print("\n" + "="*80)
    print("  All tests complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
