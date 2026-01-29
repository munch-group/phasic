"""
Test sampling with rewards - following exact notebook pattern
"""

import numpy as np
from phasic import Graph


def test_simple_sample_no_rewards():
    """Test basic sampling without rewards"""

    print("="*60)
    print("Test 1: Simple sampling (no rewards)")
    print("="*60)

    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    # Create and update
    _graph = Graph(callback)
    _graph.update_parameterized_weights(np.array([10.0]))

    # Sample
    samples = _graph.sample(10)
    print(f"Samples: {samples}")
    print(f"Mean: {np.mean(samples):.6f}")
    print(f"Expected: {1/10:.6f}")


def test_sample_with_rewards():
    """Test sampling with rewards"""

    print("\n" + "="*60)
    print("Test 2: Sampling with rewards")
    print("="*60)

    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    # Create and update
    _graph = Graph(callback)
    print(f"Vertices: {_graph.vertices_length()}")

    _graph.update_parameterized_weights(np.array([10.0]))

    # Try different reward vectors
    rewards_neutral = [1.0, 1.0]
    rewards_scaled = [2.0, 2.0]

    print(f"\nWith neutral rewards {rewards_neutral}:")
    samples_neutral = _graph.sample(10, rewards=rewards_neutral)
    print(f"  Samples: {samples_neutral}")
    print(f"  Mean: {np.mean(samples_neutral):.6f}")
    print(f"  Expected: {1/10:.6f}")

    print(f"\nWith scaled rewards {rewards_scaled}:")
    samples_scaled = _graph.sample(10, rewards=rewards_scaled)
    print(f"  Samples: {samples_scaled}")
    print(f"  Mean: {np.mean(samples_scaled):.6f}")
    print(f"  Expected (θ*r): {1/(10*2):.6f}")


def test_notebook_pattern():
    """Test exact pattern from notebook"""

    print("\n" + "="*60)
    print("Test 3: Exact notebook pattern")
    print("="*60)

    def coalescent(state, nr_samples=None):
        if not state.size:
            ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]
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

    true_theta = np.array([10.0])
    nr_samples = 3

    _graph = Graph(coalescent, nr_samples=nr_samples)
    print(f"Graph vertices: {_graph.vertices_length()}")

    _graph.update_parameterized_weights(true_theta)

    # Sample without rewards
    print(f"\nSampling without rewards:")
    samples_no_rewards = _graph.sample(10)
    print(f"  Samples: {samples_no_rewards}")
    print(f"  Mean: {np.mean(samples_no_rewards):.6f}")

    # Get rewards from states
    rewards = _graph.states()[:, :-2]
    print(f"\nRewards from states:")
    print(f"  Shape: {rewards.shape}")
    print(f"  Values:\n{rewards.T}")

    # Sample with each reward vector
    for i in range(rewards.shape[1]):
        reward_i = rewards[:, i]
        print(f"\nFeature {i}, rewards={list(reward_i)}:")
        samples_i = _graph.sample(10, rewards=reward_i)
        print(f"  Samples: {samples_i}")
        print(f"  Mean: {np.mean(samples_i):.6f}")


if __name__ == "__main__":
    print("Testing sample() with rewards...\n")

    test_simple_sample_no_rewards()
    test_sample_with_rewards()
    test_notebook_pattern()

    print("\n" + "="*60)
    print("Complete")
    print("="*60)
