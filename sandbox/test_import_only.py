"""Test if just importing causes issues"""
import sys

print("Step 1: Basic imports")
sys.stdout.flush()

import numpy as np
print("  ✓ numpy imported")
sys.stdout.flush()

import jax.numpy as jnp
print("  ✓ jax imported")
sys.stdout.flush()

import phasic
print("  ✓ phasic imported")
sys.stdout.flush()

print("\nStep 2: Define coalescent")
sys.stdout.flush()

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

print("  ✓ coalescent defined")
sys.stdout.flush()

print("\nStep 3: Create first graph")
sys.stdout.flush()

try:
    graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
    print("  ✓ first graph created")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Create second graph")
sys.stdout.flush()

try:
    graph2 = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
    print("  ✓ second graph created")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Update weights")
sys.stdout.flush()

try:
    true_theta = np.array([10.0])
    graph2.update_parameterized_weights(true_theta)
    print("  ✓ weights updated")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll steps passed!")
