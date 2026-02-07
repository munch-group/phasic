"""
Example script demonstrating hex grid graph functionality.
Creates a synthetic circular boundary and demonstrates:
1. Hex grid generation within boundary
2. Random walk simulation
3. Cumulated occupancy visualization
4. State probability animation
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure phasic to not use FFI (simpler for examples)
import phasic
phasic.configure(ffi=False)

from hex_grid_graph import HexGridGraph


def create_synthetic_shapefile(path: str, radius: float = 1.0, center: tuple = (0, 0)):
    """
    Create a synthetic circular shapefile for testing.

    Args:
        path: Output path for shapefile
        radius: Radius of circle
        center: (lon, lat) center coordinates
    """
    # Create circular polygon
    circle = Point(center).buffer(radius)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [circle]}, crs='EPSG:4326')

    # Save to shapefile
    gdf.to_file(path)
    print(f"Created synthetic shapefile: {path}")
    return path


def example_1_basic_hex_grid():
    """Example 1: Basic hex grid visualization."""
    print("\n=== Example 1: Basic Hex Grid ===")

    # Create synthetic shapefile
    shapefile_path = '/tmp/test_circle.shp'
    create_synthetic_shapefile(shapefile_path, radius=0.5)

    # Create hex grid
    hex_graph = HexGridGraph(shapefile_path, hex_size=0.05)

    print(f"Created hex grid with {len(hex_graph.hex_centers)} vertices")

    # Plot empty grid
    fig, ax = hex_graph.plot_hex_grid(title='Basic Hex Grid')
    plt.tight_layout()
    plt.savefig('hex_grid_basic.png', dpi=150, bbox_inches='tight')
    print("Saved: hex_grid_basic.png")
    plt.close()


def example_2_random_walk():
    """Example 2: Random walk with cumulated occupancy."""
    print("\n=== Example 2: Random Walk with Cumulated Occupancy ===")

    # Create synthetic shapefile
    shapefile_path = '/tmp/test_circle.shp'
    create_synthetic_shapefile(shapefile_path, radius=0.5)

    # Create hex grid
    hex_graph = HexGridGraph(shapefile_path, hex_size=0.05)

    # Define random walk callback
    def random_walk_callback(state, hex_graph):
        """Random walk: equal rate to all neighbors."""
        vertex_idx = int(state[0])
        neighbors = hex_graph.get_neighbors(vertex_idx)

        if not neighbors:
            return []  # Absorbing state (shouldn't happen with reflecting boundaries)

        # Equal rate to each neighbor
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

    print(f"Built graph with {hex_graph.graph.vertices_length()} vertices")

    # Plot cumulated occupancy at different times
    times = [0.5, 1.0, 2.0, 5.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, t in enumerate(times):
        # Get occupancy values for all vertices
        occupancy = np.array(hex_graph.graph.cumulated_occupancy(t))

        # Plot on subplot
        ax = axes[i]

        # Plot boundary
        hex_graph.gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)

        # Create hexagon patches
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection

        patches = []
        for lon, lat in hex_graph.hex_centers:
            hex_patch = RegularPolygon(
                (lon, lat),
                numVertices=6,
                radius=hex_graph.hex_size,
                orientation=0  # Pointy-top hexagon
            )
            patches.append(hex_patch)

        pc = PatchCollection(patches, cmap='viridis', linewidths=0.3, edgecolors='black')
        pc.set_array(occupancy)
        pc.set_clim(0, occupancy.max())
        ax.add_collection(pc)

        ax.set_aspect('equal')
        ax.set_title(f't = {t:.1f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Colorbar
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label('Cumulated Occupancy', fontsize=10)

    plt.suptitle('Random Walk: Cumulated Occupancy Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hex_grid_random_walk.png', dpi=150, bbox_inches='tight')
    print("Saved: hex_grid_random_walk.png")
    plt.close()


def example_3_state_probability_animation():
    """Example 3: State probability animation."""
    print("\n=== Example 3: State Probability Animation ===")

    # Create synthetic shapefile
    shapefile_path = '/tmp/test_circle.shp'
    create_synthetic_shapefile(shapefile_path, radius=0.5)

    # Create hex grid
    hex_graph = HexGridGraph(shapefile_path, hex_size=0.08)

    # Define random walk callback
    def random_walk_callback(state, hex_graph):
        vertex_idx = int(state[0])
        neighbors = hex_graph.get_neighbors(vertex_idx)

        if not neighbors:
            return []

        rate = 2.0  # Faster diffusion
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

    print(f"Built graph with {hex_graph.graph.vertices_length()} vertices")

    # Find center vertex (starting point)
    center_x, center_y = hex_graph.boundary.centroid.coords[0]
    min_dist = float('inf')
    center_vertex = 0

    for idx, (x, y) in enumerate(hex_graph.hex_centers):
        dist = (x - center_x)**2 + (y - center_y)**2
        if dist < min_dist:
            min_dist = dist
            center_vertex = idx

    print(f"Starting from center vertex: {center_vertex}")

    # Create animation
    initial_state = np.array([center_vertex])
    times = np.linspace(0.1, 2.0, 30)

    anim = hex_graph.animate_state_probability(
        initial_state=initial_state,
        times=times,
        interval=200,
        save_path='hex_diffusion.gif'
    )

    print("Saved: hex_diffusion.gif")
    plt.show()


def example_4_directional_bias():
    """Example 4: Movement with directional bias."""
    print("\n=== Example 4: Movement with Directional Bias ===")

    # Create synthetic shapefile
    shapefile_path = '/tmp/test_circle.shp'
    create_synthetic_shapefile(shapefile_path, radius=0.5)

    # Create hex grid
    hex_graph = HexGridGraph(shapefile_path, hex_size=0.06)

    # Define biased walk callback (bias toward north/positive y)
    def biased_walk_callback(state, hex_graph):
        vertex_idx = int(state[0])
        neighbors = hex_graph.get_neighbors(vertex_idx)

        if not neighbors:
            return []

        current_x, current_y = hex_graph.hex_centers[vertex_idx]
        transitions = []

        for neighbor_idx in neighbors:
            next_x, next_y = hex_graph.hex_centers[neighbor_idx]

            # Higher rate for northward movement
            if next_y > current_y:
                rate = 3.0  # Northward
            elif next_y < current_y:
                rate = 0.5  # Southward
            else:
                rate = 1.0  # East/West

            next_state = state.copy()
            next_state[0] = neighbor_idx
            transitions.append((next_state, rate, []))

        return transitions

    # Build graph
    hex_graph.build_graph(
        callback=biased_walk_callback,
        state_length=1,
        parameterized=False
    )

    print(f"Built graph with {hex_graph.graph.vertices_length()} vertices")

    # Plot cumulated occupancy
    fig, ax = hex_graph.plot_cumulated_occupancy(
        time=1.0,
        cmap='coolwarm',
        title='Northward-Biased Movement: Cumulated Occupancy at t=1.0'
    )
    plt.tight_layout()
    plt.savefig('hex_grid_biased.png', dpi=150, bbox_inches='tight')
    print("Saved: hex_grid_biased.png")
    plt.close()


def example_5_neighbor_analysis():
    """Example 5: Analyze hex grid connectivity."""
    print("\n=== Example 5: Hex Grid Connectivity Analysis ===")

    # Create synthetic shapefile
    shapefile_path = '/tmp/test_circle.shp'
    create_synthetic_shapefile(shapefile_path, radius=0.5)

    # Create hex grid
    hex_graph = HexGridGraph(shapefile_path, hex_size=0.07)

    # Analyze connectivity
    n_vertices = len(hex_graph.hex_centers)
    neighbor_counts = np.zeros(n_vertices)

    for vertex_idx in range(n_vertices):
        neighbors = hex_graph.get_neighbors(vertex_idx)
        neighbor_counts[vertex_idx] = len(neighbors)

    print(f"Total vertices: {n_vertices}")
    print(f"Average neighbors: {neighbor_counts.mean():.2f}")
    print(f"Min neighbors: {int(neighbor_counts.min())}")
    print(f"Max neighbors: {int(neighbor_counts.max())}")

    # Plot connectivity
    fig, ax = hex_graph.plot_hex_grid(
        values=neighbor_counts,
        cmap='plasma',
        title='Hex Grid Connectivity (Number of Neighbors)',
        colorbar_label='Number of Neighbors'
    )
    plt.tight_layout()
    plt.savefig('hex_grid_connectivity.png', dpi=150, bbox_inches='tight')
    print("Saved: hex_grid_connectivity.png")
    plt.close()


if __name__ == '__main__':
    # Run selected examples (skip animation for speed)
    example_1_basic_hex_grid()
    example_2_random_walk()
    # example_3_state_probability_animation()  # Skip animation for quick test
    example_4_directional_bias()
    example_5_neighbor_analysis()

    print("\n=== All examples completed successfully! ===")
