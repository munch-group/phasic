from phasic import Graph
import numpy as np

g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

# Starting edge: constant (NOT scaled)
v0.add_edge(v1, 1.0)
# Parameterized edge (scaled by theta)
v1.add_edge(v2, [1.0])

g.update_weights([3.0])
np.random.seed(42)
# Generate samples from Erlang(2, 3.0)
samples = np.random.exponential(1.0/3.0, (100, 2)).sum(axis=1)

print(f'Generated {len(samples)} samples, mean={samples.mean():.3f} (expected: {2.0/3.0:.3f})')

try:
    # Use pmf_and_moments_from_graph but with nr_moments=0 (no moment computation)
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=0, discrete=False)
    print('✓ Model created with pmf_and_moments_from_graph (nr_moments=0)')
    
    from phasic import SVGD
    svgd = SVGD(
        model=model,
        observed_data=samples,
        theta_dim=1,
        n_particles=50,
        n_iterations=200,
        regularization=0.0  # Explicitly no regularization
    )
    print('✓ SVGD object created')
    results = svgd.optimize()
    print(f'\n✓ SVGD completed successfully!')
    print(f'  Posterior mean: {svgd.theta_mean}')
    print(f'  Posterior std:  {svgd.theta_std}')
    print(f'  True value: 3.0')
    print(f'  Relative error: {abs(svgd.theta_mean[0] - 3.0) / 3.0 * 100:.1f}%')
except Exception as e:
    print(f'\n❌ SVGD failed: {e}')
    import traceback
    traceback.print_exc()
