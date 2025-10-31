"""
Test reward scaling for all features (sparse and dense)
"""
import phasic
import numpy as np

def coalescent(state, nr_samples=None):
    """Coalescent model"""
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

# Setup
theta = 10.0
nr_samples = 6

_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(np.array([theta]))
rewards = _graph.states()[:, :-2]

print("="*70)
print("Testing reward scaling for all features")
print("="*70)

nr_observations = 5000

for feature_idx in range(rewards.shape[1]):
    reward_1x = rewards[:, feature_idx]
    reward_2x = reward_1x * 2.0

    # Count non-zero entries
    n_nonzero = np.sum(reward_1x != 0)
    sparsity = 1.0 - (n_nonzero / len(reward_1x))

    # Generate samples
    samples_1x = np.array(_graph.sample(nr_observations, rewards=reward_1x.tolist()))
    samples_2x = np.array(_graph.sample(nr_observations, rewards=reward_2x.tolist()))

    mean_1x = np.mean(samples_1x)
    mean_2x = np.mean(samples_2x)

    ratio = mean_2x / mean_1x
    error = abs(ratio - 2.0)

    print(f"\nFeature {feature_idx} (sparsity={sparsity:.2f}, non-zero={n_nonzero}/12):")
    print(f"  1x mean: {mean_1x:.6f}")
    print(f"  2x mean: {mean_2x:.6f}")
    print(f"  Ratio:   {ratio:.6f} (expected: 2.0, error: {error:.6f})")

    if error < 0.05:
        print(f"  ✓ Scales correctly")
    elif error < 0.2:
        print(f"  ⚠️  Small error")
    else:
        print(f"  ❌ LARGE error!")

print(f"\n{'='*70}")
print("Pattern: Do sparser rewards have larger scaling errors?")
print("="*70)
