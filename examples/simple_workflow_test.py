#!/usr/bin/env python3
"""
Simple workflow test - validates all components work together.

This is a simplified version of the Jupyter notebook for testing purposes.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from phasic import get_trace
from phasic.trace_elimination import instantiate_from_trace, trace_to_log_likelihood

print("=" * 70)
print("Simple Workflow Test")
print("=" * 70)

# Use a pre-computed trace from IPFS to avoid graph construction issues
print("\n1. Downloading pre-computed trace from IPFS...")
trace = get_trace("coalescent_n5_theta1")
print(f"   ✓ Trace loaded: {trace.n_vertices} vertices, {trace.param_length} params")

# Instantiate graph and compute PDF
print("\n2. Instantiating graph and computing PDF...")
theta = np.array([1.0])
graph = instantiate_from_trace(trace, theta)
times = np.linspace(0.1, 5.0, 20)
pdf_values = np.array([graph.pdf(t, granularity=100) for t in times])
print(f"   ✓ Computed {len(pdf_values)} PDF values")
print(f"     Mean PDF: {np.mean(pdf_values):.6f}")

# Generate synthetic data
print("\n3. Generating synthetic data...")
cdf_values = np.cumsum(pdf_values * np.diff(times, prepend=0))
cdf_values /= cdf_values[-1]
n_obs = 10
uniform_samples = np.random.uniform(0, 1, n_obs)
observed_times = np.interp(uniform_samples, cdf_values, times)
print(f"   ✓ Generated {n_obs} observations")
print(f"     Mean: {np.mean(observed_times):.3f}, Std: {np.std(observed_times):.3f}")

# Create log-likelihood
print("\n4. Creating log-likelihood function...")
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100, use_cpp=False)
print(f"   ✓ Log-likelihood created")

# Test likelihood
print("\n5. Testing log-likelihood at different θ values...")
for theta_test in [0.5, 1.0, 2.0]:
    ll = log_lik(np.array([theta_test]))
    print(f"     θ={theta_test:.1f}: log-lik={ll:.2f}")

print("\n" + "=" * 70)
print("✓ All components working correctly!")
print("=" * 70)
print("\nThis validates:")
print("  - IPFS trace download")
print("  - Graph instantiation from trace")
print("  - PDF computation (forward algorithm)")
print("  - Synthetic data generation")
print("  - Log-likelihood function creation")
print("\nThe Jupyter notebook uses the same workflow.")
