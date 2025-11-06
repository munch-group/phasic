#!/usr/bin/env python3
"""
Test moments and PDF correctness with reward transformation.
Compare against known theoretical values from the notebook.
"""

import numpy as np
import phasic
from phasic import Graph

def coalescent(state, nr_samples=None):
    """Coalescent model callback"""
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1.0, []]]
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


true_theta = 10.0
nr_samples = 4

print("="*80)
print("Reward Transformation Correctness Test")
print("="*80)
print(f"True θ = {true_theta}")
print()

# Create graph
graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph.update_parameterized_weights(np.array([true_theta]))

print("Original graph (no rewards):")
print(f"  E[T] = {graph.expectation():.6f}")
print()

# Get rewards from states
rewards = graph.states()[:, :-2]
print(f"Rewards shape: {rewards.shape}")
print("Rewards matrix (vertices × features):")
print(rewards)
print()

# Expected values from the notebook (simple_example.ipynb)
# These are the theoretical values we should match
expected_values = {
    0: 0.200000,  # Feature 0: reward=[0,4,2,0,1,0]
    1: 0.100000,  # Feature 1: reward=[0,0,1,2,0,0]
    2: 0.066667,  # Feature 2: reward=[0,0,0,0,1,0]
}

print("="*80)
print("Test 1: Moment Computation (E[X] for each feature)")
print("="*80)
print()

all_moments_pass = True
for feat_idx in range(rewards.shape[1]):
    reward_vec = rewards[:, feat_idx]

    # Compute expectation with reward transformation
    E_X = graph.expectation(rewards=reward_vec)
    expected = expected_values[feat_idx]
    error = abs(E_X - expected)
    relative_error = error / expected * 100

    status = "✅ PASS" if relative_error < 0.01 else "❌ FAIL"
    if relative_error >= 0.01:
        all_moments_pass = False

    print(f"Feature {feat_idx}:")
    print(f"  Reward vector: {reward_vec}")
    print(f"  Expected E[X]: {expected:.6f}")
    print(f"  Computed E[X]: {E_X:.6f}")
    print(f"  Absolute error: {error:.9f}")
    print(f"  Relative error: {relative_error:.6f}%")
    print(f"  {status}")
    print()

print("="*80)
print("Test 2: PDF Computation (sample from each feature)")
print("="*80)
print()

# Generate samples and check if PDF is reasonable
np.random.seed(42)
n_samples_test = 1000

all_pdf_pass = True
for feat_idx in range(rewards.shape[1]):
    reward_vec = rewards[:, feat_idx]

    # Generate samples
    samples = graph.sample(n_samples_test, rewards=reward_vec)
    sample_mean = np.mean(samples)
    expected = expected_values[feat_idx]

    # Check if sample mean is close to expectation
    error = abs(sample_mean - expected)
    relative_error = error / expected * 100

    # With 1000 samples, we expect ~3% error or less
    status = "✅ PASS" if relative_error < 5.0 else "❌ FAIL"
    if relative_error >= 5.0:
        all_pdf_pass = False

    print(f"Feature {feat_idx}:")
    print(f"  Expected E[X]: {expected:.6f}")
    print(f"  Sample mean:   {sample_mean:.6f}")
    print(f"  Relative error: {relative_error:.2f}%")
    print(f"  {status}")
    print()

print("="*80)
print("Test 3: PDF Decreasing with θ (correct direction)")
print("="*80)
print()

# Test that PDF decreases with increasing θ (holding data fixed)
# This is a KEY property - higher θ means faster coalescence, so longer times are less likely

test_time = 0.15  # Around the mean
thetas = [8.0, 10.0, 12.0]
pdf_monotonic_pass = True

for feat_idx in range(rewards.shape[1]):
    reward_vec = rewards[:, feat_idx]

    print(f"Feature {feat_idx} (reward={reward_vec}):")

    pdf_values = []
    for theta in thetas:
        graph_test = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
        graph_test.update_parameterized_weights(np.array([theta]))
        graph_transformed = graph_test.reward_transform(reward_vec)
        pdf = graph_transformed.pdf(test_time)
        pdf_values.append(pdf)
        print(f"  θ={theta:5.1f}: PDF({test_time}) = {pdf:.6e}")

    # Check monotonicity (PDF should decrease with θ)
    is_decreasing = all(pdf_values[i] >= pdf_values[i+1] for i in range(len(pdf_values)-1))

    status = "✅ PASS" if is_decreasing else "❌ FAIL"
    if not is_decreasing:
        pdf_monotonic_pass = False

    print(f"  PDF decreasing with θ: {status}")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Test 1 (Moments):         {'✅ PASS' if all_moments_pass else '❌ FAIL'}")
print(f"Test 2 (PDF sampling):    {'✅ PASS' if all_pdf_pass else '❌ FAIL'}")
print(f"Test 3 (PDF monotonic):   {'✅ PASS' if pdf_monotonic_pass else '❌ FAIL'}")
print()

if all_moments_pass and all_pdf_pass and pdf_monotonic_pass:
    print("✅ ALL TESTS PASSED")
    exit(0)
else:
    print("❌ SOME TESTS FAILED")
    exit(1)
