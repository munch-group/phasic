"""Diagnostic: Check multivariate likelihood computation"""
import phasic
import numpy as np
import jax.numpy as jnp

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

print("=" * 70)
print("MULTIVARIATE DIAGNOSTIC")
print("=" * 70)

true_theta = np.array([10.0])
nr_samples = 5
nr_observations = 10

print(f"\n1. True theta: {true_theta}")
print(f"   Nr samples: {nr_samples}")
print(f"   Nr observations: {nr_observations}")

# Create graph
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print(f"\n2. Graph created: {graph.vertices_length()} vertices")

# Setup rewards
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
states = _graph.states()
print(f"\n3. States shape: {states.shape}")
print(f"   States:\n{states}")

rewards = states[:, :-1]  # (n_vertices, n_features)
print(f"\n4. Rewards shape: {rewards.shape}")
print(f"   Rewards:\n{rewards}")

# Sample observed data
print(f"\n5. Sampling {nr_observations} observations per feature...")
observed_data_list = []
for i in range(rewards.shape[1]):
    samples = np.array(_graph.sample(nr_observations, rewards=rewards[:, i]))
    print(f"   Feature {i}: mean={samples.mean():.3f}, std={samples.std():.3f}, samples={samples[:3]}")
    observed_data_list.append(samples)

observed_data = jnp.array(observed_data_list).T  # (n_observations, n_features)
print(f"\n6. Observed data shape: {observed_data.shape}")
print(f"   Data (first 3 rows):\n{observed_data[:3]}")

# Create model
print(f"\n7. Creating multivariate model...")
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=0, discrete=False, use_ffi=False, param_length=1
)
print("   ✓ Model created")

# Test likelihood at TRUE theta
print(f"\n8. Testing model at TRUE theta={true_theta[0]}...")
test_theta = jnp.array([true_theta[0]])
try:
    pmf, moments = model(test_theta, observed_data, rewards=rewards)
    print(f"   PMF shape: {pmf.shape}")
    print(f"   PMF (first 3 rows):\n{pmf[:3]}")
    log_lik_true = jnp.sum(jnp.log(pmf + 1e-10))
    print(f"   Log-likelihood at TRUE theta: {log_lik_true:.6f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test likelihood at different theta values
print(f"\n9. Testing log-likelihoods at different theta values...")
test_thetas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 700.0]
log_liks = []
for theta_val in test_thetas:
    test_theta = jnp.array([theta_val])
    pmf_test, _ = model(test_theta, observed_data, rewards=rewards)
    log_lik = jnp.sum(jnp.log(pmf_test + 1e-10))
    log_liks.append(float(log_lik))
    print(f"   theta={theta_val:6.1f}: log-lik={log_lik:10.4f}")

best_theta = test_thetas[np.argmax(log_liks)]
print(f"\n   Best theta from grid: {best_theta} (log-lik={max(log_liks):.4f})")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
