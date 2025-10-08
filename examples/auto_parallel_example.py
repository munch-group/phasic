#!/usr/bin/env python3
"""
Complete Example: Automatic Parallelization with PtDAlgorithms

This example demonstrates the complete workflow for using automatic parallelization
across different environments (local machine, SLURM single-node, SLURM multi-node).

Features demonstrated:
- Phase 1: Automatic environment detection and configuration
- Phase 2: Batch-aware methods with automatic parallelization
- Phase 3: Context managers for temporary configuration changes

Requirements:
- ptdalgorithms (with C++ extension built)
- Optional: JAX for advanced parallelization
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Part 1: Initialization and Configuration (Phase 1)
# ============================================================================

print("=" * 80)
print("Part 1: Automatic Environment Detection and Configuration")
print("=" * 80)
print()

import ptdalgorithms as pta

# Initialize parallel computing
# This automatically:
# - Detects your environment (Jupyter, SLURM, script)
# - Counts available CPUs/devices
# - Configures JAX if available
# - Selects parallelization strategy (pmap/vmap/none)

config = pta.init_parallel()

print(f"Detected configuration:")
print(f"  Environment: {config.env_info.env_type if config.env_info else 'unknown'}")
print(f"  Strategy: {config.strategy}")
print(f"  Devices: {config.device_count}")
print()

# ============================================================================
# Part 2: Build a Phase-Type Model
# ============================================================================

print("=" * 80)
print("Part 2: Build a Phase-Type Distribution Model")
print("=" * 80)
print()

# Example: Two-phase Erlang distribution
# This represents a process with two sequential exponential phases

g = pta.Graph(1)  # 1-dimensional state space
start = g.starting_vertex()

# Create vertices
v0 = g.find_or_create_vertex([0])  # First phase
v1 = g.find_or_create_vertex([1])  # Second phase (absorbing)

# Add transitions
start.add_edge(v0, 1.0)    # Enter first phase with rate 1.0
v0.add_edge(v1, 2.0)        # Transition to second phase with rate 2.0

# Normalize the graph
g.normalize()

print(f"Graph built with {g.vertices_length()} vertices")
print()

# ============================================================================
# Part 3: Batch Evaluation with Automatic Parallelization (Phase 2)
# ============================================================================

print("=" * 80)
print("Part 3: Batch Evaluation (Automatic Parallelization)")
print("=" * 80)
print()

# Evaluate PDF at many time points
# This automatically uses the parallel configuration from Part 1
times = np.linspace(0.1, 5.0, 1000)

print(f"Evaluating PDF at {len(times)} time points...")
print(f"  Using strategy: {config.strategy}")

pdf_values = g.pdf_batch(times)

print(f"  Completed!")
print(f"  Result shape: {pdf_values.shape}")
print(f"  Peak PDF value: {np.max(pdf_values):.4f}")
print()

# ============================================================================
# Part 4: Context Managers for Temporary Configuration (Phase 3)
# ============================================================================

print("=" * 80)
print("Part 4: Temporary Configuration Changes (Context Managers)")
print("=" * 80)
print()

# Example 1: Temporarily disable parallelization for debugging
print("Example 1: Debugging with serial execution")
print(f"  Current strategy: {pta.get_parallel_config().strategy}")

with pta.disable_parallel():
    print(f"  Inside context: {pta.get_parallel_config().strategy}")

    # This will run serially, making it easier to debug
    debug_times = np.array([1.0, 2.0, 3.0])
    debug_pdf = g.pdf_batch(debug_times)
    print(f"  Debug result: {debug_pdf}")

print(f"  After context: {pta.get_parallel_config().strategy}")
print()

# Example 2: Temporarily change parallelization strategy
print("Example 2: Temporary strategy override")

with pta.parallel_config(strategy='none'):
    # Force serial execution for this block
    small_batch = np.linspace(0.1, 1.0, 10)
    result_serial = g.pdf_batch(small_batch)
    print(f"  Serial execution: {len(result_serial)} results")

print(f"  Back to original strategy: {pta.get_parallel_config().strategy}")
print()

# ============================================================================
# Part 5: Visualization
# ============================================================================

print("=" * 80)
print("Part 5: Visualization")
print("=" * 80)
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: PDF
axes[0].plot(times, pdf_values, linewidth=2)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('PDF')
axes[0].set_title('Two-Phase Erlang Distribution')
axes[0].grid(True, alpha=0.3)

# Plot 2: CDF (cumulative)
cdf_values = np.cumsum(pdf_values) * (times[1] - times[0])  # Approximate CDF
axes[1].plot(times, cdf_values, linewidth=2, color='orange')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('CDF')
axes[1].set_title('Cumulative Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('auto_parallel_example.png', dpi=150)
print("Saved plot to 'auto_parallel_example.png'")
print()

# ============================================================================
# Part 6: Advanced - Discrete Phase-Type Distribution
# ============================================================================

print("=" * 80)
print("Part 6: Discrete Phase-Type Distribution (Batch Evaluation)")
print("=" * 80)
print()

# Discretize the continuous model
g_discrete, rewards = g.discretize(reward_rate=0.1)
g_discrete.normalize()

print(f"Discrete graph created with {g_discrete.vertices_length()} vertices")

# Evaluate PMF at many jump counts
jumps = np.arange(0, 50)
print(f"Evaluating PMF at {len(jumps)} jump counts...")

pmf_values = g_discrete.dph_pmf_batch(jumps)

print(f"  Completed!")
print(f"  Result shape: {pmf_values.shape}")
print(f"  Total probability mass: {np.sum(pmf_values):.4f}")
print()

# ============================================================================
# Part 7: Moments Calculation
# ============================================================================

print("=" * 80)
print("Part 7: Moments Calculation (Batch Evaluation)")
print("=" * 80)
print()

# Calculate multiple moments at once
moment_orders = np.arange(1, 6)  # Compute moments 1 through 5
print(f"Computing moments {list(moment_orders)}...")

moments = g.moments_batch(moment_orders)

print(f"  Completed!")
for i, order in enumerate(moment_orders):
    print(f"    E[T^{order}] = {moments[i]:.4f}")
print()

# Calculate mean and variance from moments
mean = moments[0]
variance = moments[1] - mean**2
print(f"Distribution statistics:")
print(f"  Mean: {mean:.4f}")
print(f"  Variance: {variance:.4f}")
print(f"  Std Dev: {np.sqrt(variance):.4f}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("Summary: Automatic Parallelization Features")
print("=" * 80)
print()
print("✓ Phase 1: Environment detection and configuration")
print("✓ Phase 2: Batch-aware methods (pdf_batch, dph_pmf_batch, moments_batch)")
print("✓ Phase 3: Context managers (parallel_config, disable_parallel)")
print()
print("Benefits:")
print("  - Write once, run anywhere (local, SLURM, cloud)")
print("  - No manual parallelization code needed")
print("  - Automatic resource detection and utilization")
print("  - Easy debugging with disable_parallel()")
print("  - Flexible configuration with context managers")
print()
print("For more examples, see:")
print("  - docs/examples/python/ (Jupyter notebooks)")
print("  - PHASE1_COMPLETE.md (Phase 1 documentation)")
print("  - PHASE2_COMPLETE.md (Phase 2 documentation)")
print("  - PHASE3_COMPLETE.md (Phase 3 documentation)")
print()
