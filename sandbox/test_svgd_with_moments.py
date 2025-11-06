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

print(f'Generated {len(samples)} samples')

# WITH nr_moments=2
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
print('Model created with nr_moments=2')

from phasic import SVGD
svgd = SVGD(
    model=model,
    observed_data=samples,
    theta_dim=1,
    n_particles=20,
    n_iterations=20,
    nr_moments=2,
    regularization=0.0
)
print('SVGD initialized')

results = svgd.optimize()
print(f'✓ SVGD completed!')
print(f'Posterior mean: {svgd.theta_mean}')
