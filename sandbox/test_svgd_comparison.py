from phasic import Graph
import numpy as np

g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)
v1.add_edge(v2, [1.0])

g.update_weights([3.0])
np.random.seed(42)
samples = np.random.exponential(1.0/3.0, (100, 2)).sum(axis=1)

print(f'Generated {len(samples)} samples, mean={samples.mean():.3f}')

try:
    model = Graph.pmf_from_graph(g, discrete=False)
    print('Model created')
    
    from phasic import SVGD
    svgd = SVGD(
        model=model,
        observed_data=samples,
        theta_dim=1,
        n_particles=10,
        n_iterations=10
    )
    print('SVGD object created')
    results = svgd.fit()
    print(f'✓ SVGD completed')
    print(f'  Posterior mean: {results["theta_mean"]}')
except Exception as e:
    print(f'❌ SVGD failed: {e}')
    import traceback
    traceback.print_exc()
