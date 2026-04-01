from phasic import Graph, with_ipv, StateIndexer, Property
import numpy as np

nr_samples = 2
indexer = StateIndexer(lineages=[Property('ton', min_value=1, max_value=nr_samples)])
ipv = [0] * indexer.state_length
ipv[indexer.props_to_index(ton=1)] = nr_samples

@with_ipv(ipv)
def coal(state, indexer=None):
    from itertools import combinations_with_replacement
    transitions = []
    for i, j in combinations_with_replacement(range(state.size), 2):
        same = int(i == j)
        if same and state[i] < 2: continue
        if not same and (state[i] < 1 or state[j] < 1): continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return transitions

g = Graph(coal, indexer=indexer)
jg = g.joint_prob_graph(indexer, tot_reward_limit=2, mutation_rate=1.0)
print(f"Joint graph: {jg.vertices_length()} vertices, param_length={jg.param_length()}")
print(f"is_discrete: {jg.is_discrete}")

# Show graph structure
for i in range(jg.vertices_length()):
    v = jg.vertices()[i]
    edges = v.edges()
    if edges:
        weights = [round(e.weight(), 3) for e in edges]
        targets = [e.to().index() for e in edges]
        print(f"  v{i} state={v.state()} -> {list(zip(targets, weights))}")
    else:
        print(f"  v{i} state={v.state()} (absorbing)")

# --- Homogeneous: constant N=1, mutation_rate=1 ---
print("\n--- Homogeneous (N=1) ---")
jg.update_weights([1.0, 1.0])
ctx = jg.distribution_context()
for step in range(30):
    ctx.step()
    if step % 5 == 0 or ctx.cdf() > 0.99:
        print(f"  step {step}: cdf={ctx.cdf():.4f}, jumps={ctx.jumps()}")
    if ctx.cdf() > 0.999:
        break
sp_hom = ctx.stop_probability()
print(f"Stop probs: {[(i, round(p, 4)) for i, p in enumerate(sp_hom) if p > 0.001]}")

# --- Inhomogeneous: N=1 for first 3 jumps, then N=4 ---
print("\n--- Inhomogeneous (N=1 then N=4 at jump 3) ---")
jg.update_weights([1.0, 1.0])
ctx2 = jg.distribution_context()
for step in range(3):
    ctx2.step()
print(f"  At epoch boundary: cdf={ctx2.cdf():.4f}, jumps={ctx2.jumps()}")

jg.update_weights([0.25, 1.0])  # N=4 means coal rate = 1/4
for step in range(50):
    ctx2.step()
    if step % 10 == 0 or ctx2.cdf() > 0.999:
        print(f"  step {step}: cdf={ctx2.cdf():.4f}, jumps={ctx2.jumps()}")
    if ctx2.cdf() > 0.999:
        break
sp_inhom = ctx2.stop_probability()
print(f"Stop probs: {[(i, round(p, 4)) for i, p in enumerate(sp_inhom) if p > 0.001]}")
