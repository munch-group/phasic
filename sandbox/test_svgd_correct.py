from phasic import Graph
import numpy as np

g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

# Starting edge: constant
v0.add_edge(v1, 1.0)
# Parameterized edge
v1.add_edge(v2, [1.0])

g.update_weights([3.0])
np.random.seed(42)
samples = np.random.exponential(1.0/3.0, (100, 2)).sum(axis=1)

print(f'Generated {len(samples)} samples, mean={samples.mean():.3f}')

try:
    # CORRECT: Use pmf_and_moments_from_graph for SVGD
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
    print('✓ Model created with pmf_and_moments_from_graph')
    
    from phasic import SVGD
    svgd = SVGD(
        model=model,
        observed_data=samples,
        theta_dim=1,
        n_particles=10,
        n_iterations=10,
        nr_moments=2
    )
    print('✓ SVGD object created')
    results = svgd.optimize()
    print(f'✓ SVGD completed successfully!')
    print(f'  Posterior mean: {results["theta_mean"]}')
    print(f'  Expected: ~3.0')
except Exception as e:
    print(f'❌ SVGD failed: {e}')
    import traceback
    traceback.print_exc()
