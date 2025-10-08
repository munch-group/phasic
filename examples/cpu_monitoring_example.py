#!/usr/bin/env python3
"""
CPU Monitoring Example for PtDAlgorithms

This script demonstrates how to use the CPU monitoring features
in PtDAlgorithms for both local and SLURM environments.

Features demonstrated:
- Context manager usage
- Decorator usage
- Custom width and update interval
- Per-core CPU monitoring
- Summary statistics

Requirements:
- ptdalgorithms with psutil and rich
- Optional: JAX for parallel computations

Author: PtDAlgorithms Team
Date: 2025-10-08
"""

import time
import numpy as np
import ptdalgorithms as pta

print("=" * 80)
print("CPU Monitoring Examples")
print("=" * 80)
print()

# ============================================================================
# Example 1: Basic Context Manager
# ============================================================================

print("Example 1: Basic CPU Monitoring with Context Manager")
print("-" * 80)

def simulate_computation(duration=3, intensity='medium'):
    """
    Simulate CPU-intensive computation.

    Parameters
    ----------
    duration : float
        How long to run (seconds)
    intensity : str
        'light', 'medium', or 'heavy' - affects CPU load
    """
    end_time = time.time() + duration

    if intensity == 'light':
        # Light computation - mostly sleeping
        while time.time() < end_time:
            x = sum(range(1000))
            time.sleep(0.1)

    elif intensity == 'medium':
        # Medium computation - some work, some rest
        while time.time() < end_time:
            x = sum(range(100000))
            time.sleep(0.01)

    else:  # heavy
        # Heavy computation - continuous work
        while time.time() < end_time:
            x = sum(range(1000000))


print("Running light computation (3 seconds)...")
with pta.CPUMonitor():
    simulate_computation(duration=3, intensity='light')

print("\n")

# ============================================================================
# Example 2: Custom Configuration
# ============================================================================

print("Example 2: Custom Width and Update Interval")
print("-" * 80)

print("Running medium computation with custom settings...")
with pta.CPUMonitor(width=100, update_interval=0.25):
    simulate_computation(duration=3, intensity='medium')

print("\n")

# ============================================================================
# Example 3: Decorator Usage
# ============================================================================

print("Example 3: Using @monitor_cpu Decorator")
print("-" * 80)

@pta.monitor_cpu
def my_computation():
    """A decorated function that will be monitored."""
    print("Running decorated computation...")
    simulate_computation(duration=3, intensity='heavy')
    return "Computation complete!"

result = my_computation()
print(f"Result: {result}")

print("\n")

# ============================================================================
# Example 4: Decorator with Custom Settings
# ============================================================================

print("Example 4: Decorator with Custom Settings")
print("-" * 80)

@pta.monitor_cpu(update_interval=1.0, width=80)
def another_computation():
    """Another decorated function with custom monitor settings."""
    print("Running another computation...")
    simulate_computation(duration=4, intensity='medium')
    return 42

value = another_computation()
print(f"Returned value: {value}")

print("\n")

# ============================================================================
# Example 5: Real Computation with NumPy
# ============================================================================

print("Example 5: Real NumPy Computation")
print("-" * 80)

def matrix_operations():
    """Perform some real CPU-intensive matrix operations."""
    print("Performing matrix operations...")

    # Create large matrices
    n = 2000
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    # Matrix multiplication
    C = np.dot(A, B)

    # Eigenvalue computation
    eigenvalues = np.linalg.eigvals(C[:500, :500])

    # SVD
    U, s, Vh = np.linalg.svd(C[:500, :500], full_matrices=False)

    return s

print("Running matrix computations...")
with pta.CPUMonitor():
    singular_values = matrix_operations()
    print(f"Computed {len(singular_values)} singular values")

print("\n")

# ============================================================================
# Example 6: With PtDAlgorithms Graph Operations (if available)
# ============================================================================

print("Example 6: Monitoring PtDAlgorithms Operations")
print("-" * 80)

try:
    # Create a phase-type distribution
    g = pta.Graph(1)
    start = g.starting_vertex()

    # Create vertices
    v0 = g.find_or_create_vertex([0])
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    # Add transitions
    start.add_edge(v0, 1.0)
    v0.add_edge(v1, 2.0)
    v1.add_edge(v2, 1.5)

    g.normalize()

    print(f"Created graph with {g.vertices_length()} vertices")

    # Evaluate PDF with CPU monitoring
    print("Evaluating PDF at 10,000 time points...")
    times = np.linspace(0.1, 10.0, 10000)

    with pta.CPUMonitor():
        pdf_values = g.pdf_batch(times)

    print(f"Computed PDF, max value: {np.max(pdf_values):.4f}")

except Exception as e:
    print(f"Could not run Graph example: {e}")

print("\n")

# ============================================================================
# Example 7: Multiple Sequential Operations
# ============================================================================

print("Example 7: Monitoring Multiple Operations")
print("-" * 80)

with pta.CPUMonitor():
    print("Phase 1: Light computation (2s)")
    simulate_computation(duration=2, intensity='light')

    print("Phase 2: Heavy computation (2s)")
    simulate_computation(duration=2, intensity='heavy')

    print("Phase 3: Medium computation (2s)")
    simulate_computation(duration=2, intensity='medium')

print("\n")

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("CPU Monitoring Examples Complete!")
print("=" * 80)
print()
print("Key Features:")
print("  ✓ Context manager: with pta.CPUMonitor(): ...")
print("  ✓ Decorator: @pta.monitor_cpu")
print("  ✓ Custom settings: width, update_interval")
print("  ✓ Per-core monitoring with Unicode bars")
print("  ✓ Summary statistics after completion")
print("  ✓ SLURM-aware (detects allocated nodes/CPUs)")
print()
print("For Jupyter notebook usage, see:")
print("  examples/cpu_monitoring_notebook.ipynb")
print()
print("In Jupyter, you can use the %%usage cell magic:")
print("  %%usage")
print("  # your code here")
print()
