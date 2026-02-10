# Hex Grid Graph Implementation

## Overview

Created a complete implementation for constructing phasic graphs on hexagonal grids constrained by shapefile boundaries, with comprehensive visualization and animation capabilities.

## Files Created

### 1. `hex_grid_graph.py` (Main Module)
Complete implementation of the `HexGridGraph` class with:
- **Hex grid generation**: Evenly spaced hexagonal grid within shapefile boundary
- **Vertex mapping**: Bidirectional dictionaries `vertex_to_coords` and `coords_to_vertex`
- **phasic Graph integration**: Build graphs with custom transition callbacks
- **Static visualization**: Plot hex grids with customizable coloring
- **Cumulated occupancy**: Color vertices by `graph.cumulated_occupancy()`
- **State probability animation**: Animate temporal development using `graph.state_probability()`

**Key Methods:**
- `__init__(shapefile_path, hex_size)` - Load shapefile and generate hex grid
- `build_graph(callback, ...)` - Create phasic Graph with hex vertices
- `get_neighbors(vertex_idx, max_distance)` - Find adjacent hex cells
- `plot_hex_grid(values, ...)` - General plotting with value coloring
- `plot_cumulated_occupancy(time, ...)` - Visualize cumulated occupancy
- `animate_state_probability(initial_state, times, ...)` - Create animations

### 2. `examples/hex_grid_example.py` (Full Examples)
Five comprehensive examples demonstrating:
1. **Basic hex grid** - Simple grid visualization
2. **Random walk** - Symmetric diffusion with cumulated occupancy over time
3. **State probability animation** - GIF export with temporal evolution
4. **Directional bias** - Asymmetric movement (northward bias)
5. **Connectivity analysis** - Visualize number of neighbors per vertex

### 3. `examples/test_hex_grid_simple.py` (Quick Test)
Minimal test script to verify installation and basic functionality.

## Usage Examples

### Basic Setup

```python
from hex_grid_graph import HexGridGraph
import numpy as np

# Create hex grid from shapefile
hex_graph = HexGridGraph(
    shapefile_path='boundary.shp',
    hex_size=0.05  # Adjust based on your CRS units
)

print(f"Created {len(hex_graph.hex_centers)} vertices")
```

### Building a Graph with Custom Transitions

```python
def random_walk_callback(state, hex_graph):
    """Simple random walk."""
    vertex_idx = int(state[0])
    neighbors = hex_graph.get_neighbors(vertex_idx)

    rate = 1.0
    transitions = []
    for neighbor_idx in neighbors:
        next_state = state.copy()
        next_state[0] = neighbor_idx
        transitions.append((next_state, rate, []))

    return transitions

# Build graph
hex_graph.build_graph(
    callback=random_walk_callback,
    state_length=1,
    parameterized=False
)
```

### Plotting Cumulated Occupancy

```python
# Plot at specific time
fig, ax = hex_graph.plot_cumulated_occupancy(
    time=1.0,
    cmap='viridis',
    title='Cumulated Occupancy at t=1.0'
)
plt.savefig('occupancy.png')
```

### Creating State Probability Animation

```python
# Animate from center vertex
initial_state = np.array([0])  # Start at vertex 0
times = np.linspace(0.1, 2.0, 30)

anim = hex_graph.animate_state_probability(
    initial_state=initial_state,
    times=times,
    interval=200,  # ms per frame
    save_path='diffusion.gif'
)
plt.show()
```

### Parameterized Movement (for SVGD)

```python
def drift_callback(state, hex_graph):
    """Movement with parameterized drift toward center."""
    vertex_idx = int(state[0])
    neighbors = hex_graph.get_neighbors(vertex_idx)

    center_x, center_y = hex_graph.boundary.centroid.coords[0]
    current_x, current_y = hex_graph.hex_centers[vertex_idx]

    transitions = []
    for neighbor_idx in neighbors:
        next_x, next_y = hex_graph.hex_centers[neighbor_idx]

        # Distance before/after
        dist_before = (current_x - center_x)**2 + (current_y - center_y)**2
        dist_after = (next_x - center_x)**2 + (next_y - center_y)**2

        # Parameterized edge: weight = base_weight + theta[0] * 1.0 + theta[1] * drift_coef
        drift_coef = 1.0 if dist_after < dist_before else -1.0

        next_state = state.copy()
        next_state[0] = neighbor_idx
        transitions.append((
            next_state,
            0.1,  # base_weight
            [1.0, drift_coef]  # coefficients for theta[0], theta[1]
        ))

    return transitions

hex_graph.build_graph(
    callback=drift_callback,
    state_length=1,
    parameterized=True
)

# Use with SVGD
from phasic import SVGD

observed_times = np.array([5.0, 7.5, 10.0])
svgd = SVGD(
    model=hex_graph.graph,
    observed_data=observed_times,
    theta_dim=2,
    n_particles=50,
    n_iterations=500
)
results = svgd.optimize()
```

## Architecture Details

### Hex Grid Generation Algorithm

1. **Compute bounding box** from shapefile boundary
2. **Generate regular hexagonal lattice** using offset rows
   - Horizontal spacing: `1.5 * hex_size`
   - Vertical spacing: `sqrt(3) * hex_size`
   - Every other row offset by `1.5 * hex_size`
3. **Filter vertices** to only include those inside shapefile boundary
4. **Create bidirectional mappings** between vertex indices and (lat, lon) coordinates

### State Representation

- `state[0]` = vertex index (position in hex grid)
- `state[1:]` = optional additional state variables (e.g., internal states)

### Callback Function Pattern

```python
def callback(state, hex_graph):
    """
    Define transitions from current state.

    Args:
        state: numpy array with state[0] = vertex index
        hex_graph: HexGridGraph instance for accessing grid structure

    Returns:
        List of tuples:
        - Non-parameterized: [(next_state, rate, [])]
        - Parameterized: [(next_state, base_weight, [coeff_1, coeff_2, ...])]
    """
    vertex_idx = int(state[0])
    neighbors = hex_graph.get_neighbors(vertex_idx)

    transitions = []
    for neighbor_idx in neighbors:
        next_state = state.copy()
        next_state[0] = neighbor_idx
        transitions.append((next_state, 1.0, []))

    return transitions
```

### Coordinate Mappings

```python
# Vertex index → (latitude, longitude)
lat, lon = hex_graph.vertex_to_coords[vertex_idx]

# (longitude, latitude) → vertex index
vertex_idx = hex_graph.coords_to_vertex[(lon, lat)]

# Direct access to hex centers (lon, lat)
lon, lat = hex_graph.hex_centers[vertex_idx]
```

## Dependencies

- **geopandas** - Shapefile I/O and geometric operations
- **shapely** - Geometric objects and spatial queries
- **matplotlib** - Plotting and visualization
- **numpy** - Numerical computations
- **phasic** - Phase-type distribution graph library

## Installation

```bash
# Install dependencies (with pixi)
pixi add geopandas
pixi install

# Install phasic in development mode
pixi run pip install -e .

# Run examples
pixi run python examples/test_hex_grid_simple.py
pixi run python examples/hex_grid_example.py
```

## Generated Outputs

Running the full example script (`hex_grid_example.py`) generates:

1. `hex_grid_basic.png` - Empty hex grid within boundary
2. `hex_grid_random_walk.png` - Cumulated occupancy at t=0.5, 1.0, 2.0, 5.0
3. `hex_diffusion.gif` - Animated state probability evolution
4. `hex_grid_biased.png` - Northward-biased movement pattern
5. `hex_grid_connectivity.png` - Number of neighbors per vertex

## Performance Characteristics

- **Grid generation**: O(n) where n = number of hex cells
- **Neighbor finding**: O(n) naive implementation (can be optimized with spatial index)
- **Graph construction**: Depends on callback complexity
- **Cumulated occupancy plot**: O(n × vertices_length)
- **State probability animation**: O(n × vertices_length × n_frames)

## Future Enhancements

Potential improvements:
1. **Spatial indexing** for O(1) neighbor lookups (KD-tree or R-tree)
2. **Directed edges** based on geographic features (e.g., terrain slope)
3. **Multi-scale grids** with varying hex sizes
4. **3D visualizations** with topographic data
5. **Integration with raster data** for environmental covariates
6. **Caching** of graph structures for repeated use
7. **Parallel computation** of state probabilities for animations

## Notes

- **Hex grid orientation**: Flat-top hexagons (can be modified for pointy-top)
- **Coordinate reference systems**: Shapefile CRS determines hex size units
- **Edge handling**: Vertices near boundary may have fewer neighbors
- **State space size**: Scales with hex grid density (balance resolution vs computation)

## Testing

```bash
# Quick test
pixi run python examples/test_hex_grid_simple.py

# Full examples (generates all plots)
pixi run python examples/hex_grid_example.py
```

Expected output:
```
=== Testing Basic Hex Grid ===
Created synthetic shapefile: /tmp/test_circle.shp
Created hex grid with 59 vertices
Saved: hex_grid_basic.png
SUCCESS!
```

---

**Implementation Date:** 2025-11-30
**phasic Version:** 0.22.21
