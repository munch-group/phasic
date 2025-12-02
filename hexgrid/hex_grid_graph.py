"""
Hex Grid Graph Generation and Visualization
Creates a phasic Graph where vertices represent hexagonal grid cells
within a geographic boundary defined by a shapefile.
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from shapely.geometry import Point, Polygon
from typing import Dict, Tuple, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from phasic import Graph


class HexGridGraph:
    """
    Manages hexagonal grid graph construction and visualization.

    Attributes:
        graph: phasic Graph object
        vertex_to_coords: dict mapping vertex index to (lat, lon)
        coords_to_vertex: dict mapping (lat, lon) to vertex index
        hex_size: size of hexagonal cells
        boundary: shapely Polygon defining the area
    """

    def __init__(self, shapefile_path: str, hex_size: float):
        """
        Initialize hex grid graph from shapefile.

        Args:
            shapefile_path: Path to shapefile defining boundary
            hex_size: Size of hexagonal cells (in same units as shapefile CRS)
        """
        self.hex_size = hex_size
        self.gdf = gpd.read_file(shapefile_path)
        self.boundary = self.gdf.unary_union

        # Generate hex grid
        self.hex_centers = self._generate_hex_grid()

        # Initialize mappings
        self.vertex_to_coords: Dict[int, Tuple[float, float]] = {}
        self.coords_to_vertex: Dict[Tuple[float, float], int] = {}

        # Populate mappings immediately (needed for get_neighbors)
        for idx, (lon, lat) in enumerate(self.hex_centers):
            self.vertex_to_coords[idx] = (lat, lon)
            self.coords_to_vertex[(lon, lat)] = idx

        # Create graph
        self.graph = None

    def _generate_hex_grid(self) -> List[Tuple[float, float]]:
        """
        Generate hexagonal grid centers within boundary.

        Uses pointy-top hexagon orientation where:
        - Horizontal distance between adjacent centers: sqrt(3) * hex_size
        - Vertical distance between rows: 1.5 * hex_size
        - Odd rows offset by sqrt(3)/2 * hex_size

        Returns:
            List of (lon, lat) coordinates for hex centers
        """
        minx, miny, maxx, maxy = self.boundary.bounds

        # Hexagon geometry (pointy-top orientation)
        # For pointy-top hexagons with circumradius r:
        # - Width (flat edge to flat edge) = sqrt(3) * r
        # - Height (vertex to vertex) = 2 * r
        # Distance between adjacent hex centers horizontally
        horiz_dist = np.sqrt(3) * self.hex_size
        # Distance between rows vertically
        vert_dist = 1.5 * self.hex_size

        hex_centers = []
        row = 0
        y = miny

        while y <= maxy:
            # Offset every other row by half the horizontal distance
            x_offset = (np.sqrt(3) / 2) * self.hex_size if row % 2 == 1 else 0
            x = minx + x_offset

            while x <= maxx:
                point = Point(x, y)
                if self.boundary.contains(point):
                    hex_centers.append((x, y))
                x += horiz_dist

            y += vert_dist
            row += 1

        return hex_centers

    def build_graph(self,
                    callback: Callable,
                    state_length: int = 1,
                    parameterized: bool = False,
                    start_vertex: int = 0,
                    **graph_kwargs) -> 'Graph':
        """
        Build phasic Graph with hex grid vertices.

        The callback function receives a state array where:
        - state[0] = vertex index (position in hex grid)
        - state[1:] = additional state variables (optional)

        Args:
            callback: Function defining transitions between vertices
            state_length: Length of state vector (min 1 for vertex index)
            parameterized: Whether to use parameterized edges
            start_vertex: Index of hex cell where probability starts (default 0)
            **graph_kwargs: Additional arguments for Graph constructor

        Returns:
            Configured phasic Graph object
        """
        # Import Graph here to allow phasic configuration before import
        from phasic import Graph

        # Create empty graph
        self.graph = Graph(state_length)

        # Get the starting vertex and connect it to the specified hex cell
        # This is essential - probability mass starts at the starting vertex
        starting_vertex = self.graph.starting_vertex()
        start_hex_state = [start_vertex] + [0] * (state_length - 1)
        start_hex_vertex = self.graph.find_or_create_vertex(start_hex_state)
        starting_vertex.add_edge(start_hex_vertex, 1.0)  # Instant transition to start cell

        # Manually create all hex grid vertices and add edges based on callback
        n_vertices = len(self.hex_centers)

        for vertex_idx in range(n_vertices):
            # Create state for this vertex
            state = [vertex_idx] + [0] * (state_length - 1)

            # Find or create vertex in graph
            vertex = self.graph.find_or_create_vertex(state)

            # Get transitions from callback
            transitions = callback(np.array(state), self)

            # Add edges based on transitions
            for transition in transitions:
                if parameterized:
                    # Parameterized: (next_state, base_weight, coefficients)
                    next_state, base_weight, coefficients = transition
                    next_vertex = self.graph.find_or_create_vertex(list(next_state))
                    vertex.add_edge_parameterized(
                        next_vertex,
                        base_weight=base_weight,
                        edge_state=coefficients
                    )
                else:
                    # Non-parameterized: (next_state, rate, [])
                    next_state, rate, _ = transition
                    next_vertex = self.graph.find_or_create_vertex(list(next_state))
                    vertex.add_edge(next_vertex, rate)

        # Build mapping from hex cell index to graph vertex index
        self._hex_to_graph_vertex = {}
        for i in range(1, self.graph.vertices_length()):  # Skip starting vertex at 0
            v = self.graph.vertex_at(i)
            state = list(v.state())
            if len(state) >= 1:
                hex_idx = int(state[0])
                if 0 <= hex_idx < len(self.hex_centers):
                    self._hex_to_graph_vertex[hex_idx] = i

        return self.graph

    def get_state_probabilities(self, time: float) -> np.ndarray:
        """
        Get state probabilities mapped to hex cell indices.

        Args:
            time: Time point for probability calculation

        Returns:
            Array of probabilities indexed by hex cell index
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        # Get raw probabilities from graph
        raw_probs = np.array(self.graph.state_probability(time))

        # Map to hex cell indices
        n_hex = len(self.hex_centers)
        hex_probs = np.zeros(n_hex)

        for hex_idx, graph_idx in self._hex_to_graph_vertex.items():
            if graph_idx < len(raw_probs):
                hex_probs[hex_idx] = raw_probs[graph_idx]

        return hex_probs
    
    def get_cumulated_occupancies(self, time: float) -> np.ndarray:
        """
        Get state probabilities mapped to hex cell indices.

        Args:
            time: Time point for probability calculation

        Returns:
            Array of probabilities indexed by hex cell index
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        # Get raw probabilities from graph
        raw_vals = np.array(self.graph.cumulated_occupancy(time))

        # Map to hex cell indices
        n_hex = len(self.hex_centers)
        hex_vals = np.zeros(n_hex)

        for hex_idx, graph_idx in self._hex_to_graph_vertex.items():
            if graph_idx < len(raw_vals):
                hex_vals[hex_idx] = raw_vals[graph_idx]

        return hex_vals    

    def get_center_vertex(self) -> int:
        """
        Find the hex vertex closest to the center of the boundary.

        Returns:
            Index of the center vertex
        """
        center_x, center_y = self.boundary.centroid.coords[0]
        min_dist = float('inf')
        center_vertex = 0

        for idx, (x, y) in enumerate(self.hex_centers):
            dist = (x - center_x)**2 + (y - center_y)**2
            if dist < min_dist:
                min_dist = dist
                center_vertex = idx

        return center_vertex

    def get_neighbors(self, vertex_idx: int, max_distance: int = 1) -> List[int]:
        """
        Get neighboring vertices within specified distance.

        Args:
            vertex_idx: Index of center vertex
            max_distance: Maximum distance in hex grid steps

        Returns:
            List of neighbor vertex indices
        """
        if vertex_idx not in self.vertex_to_coords:
            return []

        center_x, center_y = self.hex_centers[vertex_idx]
        neighbors = []

        # Check all vertices
        for idx, (x, y) in enumerate(self.hex_centers):
            if idx == vertex_idx:
                continue

            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # For flat-top hex grid, all immediate neighbors are sqrt(3) * hex_size apart
            # Use sqrt(3) * hex_size * 1.1 as threshold (10% tolerance)
            threshold = self.hex_size * np.sqrt(3) * 1.1 * max_distance

            if dist <= threshold:
                neighbors.append(idx)

        return neighbors

    def plot_hex_grid(self,
                      values: Optional[np.ndarray] = None,
                      cmap: str = 'viridis',
                      title: str = 'Hex Grid',
                      figsize: Tuple[int, int] = (5, 4),
                      show_boundary: bool = True,
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      colorbar_label: str = 'Value') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot hex grid with optional value coloring.

        Args:
            values: Array of values for each vertex (for coloring)
            cmap: Matplotlib colormap name
            title: Plot title
            figsize: Figure size
            show_boundary: Whether to show shapefile boundary
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            colorbar_label: Label for colorbar

        Returns:
            (figure, axes) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot boundary
        if show_boundary:
            self.gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)

        # Create hexagon patches (pointy-top orientation = 0)
        patches = []
        for lon, lat in self.hex_centers:
            hex_patch = RegularPolygon(
                (lon, lat),
                numVertices=6,
                radius=self.hex_size,
                orientation=0,  # Pointy-top hexagon
            )
            patches.append(hex_patch)

        # Create collection with thin borders
        pc = PatchCollection(patches, match_original=False, linewidths=0.3,
                            edgecolors='black', cmap=cmap)

        # Set colors
        if values is not None:
            pc.set_array(values)
            if vmin is not None or vmax is not None:
                pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            cbar = plt.colorbar(pc, ax=ax)
            cbar.set_label(colorbar_label)
        else:
            pc.set_facecolor('lightblue')
            ax.add_collection(pc)

        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        return fig, ax

    def plot_cumulated_occupancy(self,
                                 time: float,
                                 **plot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot hex grid colored by cumulated occupancy.

        Args:
            time: Time point for cumulated occupancy calculation
            **plot_kwargs: Additional arguments for plot_hex_grid

        Returns:
            (figure, axes) tuple
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        # Get cumulated occupancy for all vertices at once
        occupancy = np.array(self.graph.cumulated_occupancy(time))

        # Set defaults that can be overridden
        plot_kwargs.setdefault('title', f'Cumulated Occupancy at t={time:.2f}')
        plot_kwargs.setdefault('colorbar_label', 'Cumulated Occupancy')

        # Plot
        return self.plot_hex_grid(values=occupancy, **plot_kwargs)
    

    def animate_state_probability(self,
                                  initial_state: np.ndarray,
                                  times: np.ndarray,
                                  interval: int = 100,
                                  figsize: Tuple[int, int] = (5, 4),
                                  save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animation of state probability development over time.

        Args:
            initial_state: Starting state vector
            times: Array of time points for animation
            interval: Milliseconds between frames
            figsize: Figure size
            save_path: Optional path to save animation (e.g., 'animation.gif')

        Returns:
            FuncAnimation object
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        n_vertices = len(self.hex_centers)

        # Compute probabilities for all times
        prob_over_time = np.zeros((len(times), n_vertices))

        for t_idx, time in enumerate(times):
            for vertex_idx in range(n_vertices):
                state = np.zeros(self.graph.state_length())
                state[0] = vertex_idx
                prob_over_time[t_idx, vertex_idx] = self.graph.state_probability(
                    initial_state, state, time
                )

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create hexagon patches (pointy-top orientation = 0)
        patches = []
        for lon, lat in self.hex_centers:
            hex_patch = RegularPolygon(
                (lon, lat),
                numVertices=6,
                radius=self.hex_size,
                orientation=0  # Pointy-top hexagon
            )
            patches.append(hex_patch)

        pc = PatchCollection(patches, cmap='hot', linewidths=0.3, edgecolors='black')
        pc.set_clim(0, prob_over_time.max())
        ax.add_collection(pc)

        # Plot boundary
        self.gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Colorbar
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label('State Probability')

        # Title
        title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                       ha='center', fontsize=14, fontweight='bold')

        def update(frame):
            pc.set_array(prob_over_time[frame])
            title.set_text(f'State Probability at t={times[frame]:.2f}')
            return pc, title

        anim = FuncAnimation(fig, update, frames=len(times),
                           interval=interval, blit=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=1000/interval)

        return anim


# Example usage
def example_random_walk():
    """Example: Random walk on hex grid with reflecting boundaries."""

    # Define callback for random walk
    def random_walk_callback(state, hex_graph):
        """Random walk: equal probability to all neighbors."""
        vertex_idx = int(state[0])
        neighbors = hex_graph.get_neighbors(vertex_idx)

        if not neighbors:
            return []  # Absorbing state

        # Equal probability to each neighbor
        rate = 1.0 / len(neighbors)
        transitions = []

        for neighbor_idx in neighbors:
            next_state = state.copy()
            next_state[0] = neighbor_idx
            transitions.append((next_state, rate, []))

        return transitions

    # Create hex grid graph
    hex_graph = HexGridGraph(
        shapefile_path='path/to/shapefile.shp',
        hex_size=0.01  # Adjust based on your CRS units
    )

    # Build graph
    hex_graph.build_graph(
        callback=random_walk_callback,
        state_length=1,
        parameterized=False
    )

    # Plot static cumulated occupancy
    fig, ax = hex_graph.plot_cumulated_occupancy(time=10.0)
    plt.show()

    # Create animation
    initial_state = np.array([0])  # Start at vertex 0
    times = np.linspace(0, 20, 50)

    anim = hex_graph.animate_state_probability(
        initial_state=initial_state,
        times=times,
        interval=200,
        save_path='hex_diffusion.gif'
    )
    plt.show()


def example_parameterized_movement():
    """Example: Parameterized movement with drift toward center."""

    def drift_callback(state, hex_graph):
        """Movement with drift toward center."""
        vertex_idx = int(state[0])
        neighbors = hex_graph.get_neighbors(vertex_idx)

        if not neighbors:
            return []

        # Calculate center of boundary
        center_x, center_y = hex_graph.boundary.centroid.coords[0]
        current_x, current_y = hex_graph.hex_centers[vertex_idx]

        transitions = []
        for neighbor_idx in neighbors:
            next_x, next_y = hex_graph.hex_centers[neighbor_idx]

            # Distance to center before and after move
            dist_before = (current_x - center_x)**2 + (current_y - center_y)**2
            dist_after = (next_x - center_x)**2 + (next_y - center_y)**2

            # Parameterized: theta[0] = base rate, theta[1] = drift strength
            next_state = state.copy()
            next_state[0] = neighbor_idx

            # Coefficient for drift parameter
            drift_coef = 1.0 if dist_after < dist_before else -1.0

            transitions.append((
                next_state,
                0.1,  # base_weight
                [1.0, drift_coef]  # coefficients for [theta[0], theta[1]]
            ))

        return transitions

    hex_graph = HexGridGraph('path/to/shapefile.shp', hex_size=0.01)
    hex_graph.build_graph(
        callback=drift_callback,
        state_length=1,
        parameterized=True
    )

    # Use with SVGD for inference
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
    print(f"Estimated theta: {results['theta_mean']}")


if __name__ == '__main__':
    # Run example (requires shapefile)
    example_random_walk()
