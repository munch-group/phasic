"""
Hex Grid Graph Visualization

Application-level wrapper combining phasic HexGrid with matplotlib
visualization for hex grid graphs on geographic boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple, List, Optional, Callable, TYPE_CHECKING

from phasic import HexGrid, Property, StateIndexer

if TYPE_CHECKING:
    from phasic import Graph


class HexGridGraph:
    """
    Manages hexagonal grid graph construction and visualization.

    Wraps :class:`phasic.HexGrid` for grid topology and neighbor computation,
    and adds matplotlib-based plotting and animation methods.

    Attributes
    ----------
    hex_grid : HexGrid
        Core hex grid object providing topology and neighbor computation.
    graph : Graph or None
        phasic Graph object (after build_graph is called).
    hex_size : float
        Size of hexagonal cells.
    """

    def __init__(self, shapefile_path: str, hex_size: float):
        """
        Initialize hex grid graph from shapefile.

        Parameters
        ----------
        shapefile_path : str
            Path to shapefile defining boundary.
        hex_size : float
            Circumradius of hexagonal cells (in shapefile CRS units).
        """
        self.hex_grid = HexGrid.from_shapefile(shapefile_path, hex_size)
        self.hex_size = hex_size

        # Backward compatibility: list of (lon, lat) center coordinates
        self.hex_centers = [
            self.hex_grid.rowcol_to_coords(r, c)
            for r, c in self.hex_grid.valid_cells()
        ]

        # Mappings between valid cell sequential index and (row, col)
        self._valid_cells = self.hex_grid.valid_cells()
        self._seqidx_to_rowcol = {i: rc for i, rc in enumerate(self._valid_cells)}
        self._rowcol_to_seqidx = {rc: i for i, rc in enumerate(self._valid_cells)}

        # Backward compat: vertex_to_coords and coords_to_vertex
        self.vertex_to_coords: Dict[int, Tuple[float, float]] = {}
        self.coords_to_vertex: Dict[Tuple[float, float], int] = {}
        for idx, (r, c) in enumerate(self._valid_cells):
            coords = self.hex_grid.rowcol_to_coords(r, c)
            self.vertex_to_coords[idx] = coords
            self.coords_to_vertex[coords] = idx

        # Store gdf for boundary plotting
        self.gdf = self.hex_grid._gdf

        self.graph = None
        self._property_set = None

    def build_graph(self,
                    callback: Callable,
                    state_length: int = 1,
                    parameterized: bool = False,
                    start_vertex: int = 0,
                    **graph_kwargs) -> 'Graph':
        """
        Build phasic Graph with hex grid vertices.

        The callback function receives a state array and this HexGridGraph:
        ``callback(state, hex_graph)`` where ``state[0]`` is the sequential
        vertex index (position in hex grid).

        Parameters
        ----------
        callback : callable
            Function ``(state, hex_graph) -> transitions``.
        state_length : int
            Length of state vector (min 1 for vertex index).
        parameterized : bool
            Whether to use parameterized edges.
        start_vertex : int
            Sequential index of hex cell where probability starts.
        **graph_kwargs
            Additional arguments for Graph constructor.

        Returns
        -------
        Graph
        """
        from phasic import Graph

        # Create a simple PropertySet for sequential indexing
        n_valid = len(self._valid_cells)
        pset = StateIndexer(
            grid=[Property('cell', max_value=n_valid - 1)]
        ).grid
        self._property_set = pset

        # Build graph manually (using sequential indices for backward compat)
        self.graph = Graph(state_length, **graph_kwargs)

        sv = self.graph.starting_vertex()
        start_state = [start_vertex] + [0] * (state_length - 1)
        start_v = self.graph.find_or_create_vertex(start_state)
        sv.add_edge(start_v, 1.0)

        for seq_idx in range(n_valid):
            state = np.array([seq_idx] + [0] * (state_length - 1))
            vertex = self.graph.find_or_create_vertex(list(state))

            transitions = callback(state, self)

            for transition in transitions:
                if parameterized:
                    next_state, coefficients = transition
                    next_vertex = self.graph.find_or_create_vertex(list(next_state))
                    vertex.add_edge(next_vertex, coefficients)
                else:
                    next_state, rate, _ = transition
                    next_vertex = self.graph.find_or_create_vertex(list(next_state))
                    vertex.add_edge(next_vertex, rate)

        # Build mapping from sequential index to graph vertex index
        self._seq_to_graph_vertex = {}
        for i in range(1, self.graph.vertices_length()):
            v = self.graph.vertex_at(i)
            state = list(v.state())
            if len(state) >= 1:
                seq_idx = int(state[0])
                if 0 <= seq_idx < n_valid:
                    self._seq_to_graph_vertex[seq_idx] = i

        return self.graph

    def get_state_probabilities(self, time: float) -> np.ndarray:
        """
        Get state probabilities mapped to sequential vertex indices.

        Parameters
        ----------
        time : float
            Time point for probability calculation.

        Returns
        -------
        np.ndarray
            Probabilities indexed by sequential vertex index.
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        raw_probs = np.array(self.graph.state_probability(time))
        n_valid = len(self._valid_cells)
        hex_probs = np.zeros(n_valid)

        for seq_idx, graph_idx in self._seq_to_graph_vertex.items():
            if graph_idx < len(raw_probs):
                hex_probs[seq_idx] = raw_probs[graph_idx]

        return hex_probs

    def get_accumulated_occupancies(self, time: float) -> np.ndarray:
        """
        Get accumulated occupancies mapped to sequential vertex indices.

        Parameters
        ----------
        time : float
            Time point for calculation.

        Returns
        -------
        np.ndarray
            Accumulated occupancies indexed by sequential vertex index.
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        raw_vals = np.array(self.graph.accumulated_occupancy(time))
        n_valid = len(self._valid_cells)
        hex_vals = np.zeros(n_valid)

        for seq_idx, graph_idx in self._seq_to_graph_vertex.items():
            if graph_idx < len(raw_vals):
                hex_vals[seq_idx] = raw_vals[graph_idx]

        return hex_vals

    def get_center_vertex(self) -> int:
        """
        Find the sequential vertex index closest to the boundary center.

        Returns
        -------
        int
            Sequential index of the center vertex.
        """
        center_rc = self.hex_grid.center_cell()
        return self._rowcol_to_seqidx.get(center_rc, 0)

    def get_neighbors(self, vertex_idx: int, max_distance: int = 1) -> List[int]:
        """
        Get neighboring vertices by sequential index.

        Parameters
        ----------
        vertex_idx : int
            Sequential index of center vertex.
        max_distance : int
            Maximum distance in hex grid steps.

        Returns
        -------
        list of int
            Neighbor sequential indices.
        """
        if vertex_idx not in self._seqidx_to_rowcol:
            return []

        r, c = self._seqidx_to_rowcol[vertex_idx]

        if max_distance == 1:
            neighbor_rcs = self.hex_grid.neighbors(r, c)
        else:
            neighbor_rcs = self.hex_grid.neighbors_within(r, c, max_distance)

        return [
            self._rowcol_to_seqidx[rc]
            for rc in neighbor_rcs
            if rc in self._rowcol_to_seqidx
        ]

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

        Parameters
        ----------
        values : np.ndarray, optional
            Array of values for each valid vertex (for coloring).
        cmap : str
            Matplotlib colormap name.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        show_boundary : bool
            Whether to show shapefile boundary.
        vmin, vmax : float, optional
            Color scale limits.
        colorbar_label : str
            Label for colorbar.

        Returns
        -------
        tuple of (Figure, Axes)
        """
        fig, ax = plt.subplots(figsize=figsize)

        if show_boundary:
            self.gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)

        orientation = 0 if self.hex_grid.orientation == "pointy" else np.pi / 6
        patches = []
        for x, y in self.hex_centers:
            hex_patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=self.hex_size,
                orientation=orientation,
            )
            patches.append(hex_patch)

        pc = PatchCollection(patches, match_original=False, linewidths=0.3,
                            edgecolors='black', cmap=cmap)

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

    def plot_accumulated_occupancy(self,
                                 time: float,
                                 **plot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot hex grid colored by accumulated occupancy.

        Parameters
        ----------
        time : float
            Time point for calculation.
        **plot_kwargs
            Additional arguments for :meth:`plot_hex_grid`.

        Returns
        -------
        tuple of (Figure, Axes)
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        occupancy = self.get_accumulated_occupancies(time)

        plot_kwargs.setdefault('title', f'Accumulated Occupancy at t={time:.2f}')
        plot_kwargs.setdefault('colorbar_label', 'Accumulated Occupancy')

        return self.plot_hex_grid(values=occupancy, **plot_kwargs)

    def animate_state_probability(self,
                                  initial_state: np.ndarray,
                                  times: np.ndarray,
                                  interval: int = 100,
                                  figsize: Tuple[int, int] = (5, 4),
                                  save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animation of state probability over time.

        Parameters
        ----------
        initial_state : np.ndarray
            Starting state vector.
        times : np.ndarray
            Array of time points for animation.
        interval : int
            Milliseconds between frames.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save animation (e.g., 'animation.gif').

        Returns
        -------
        FuncAnimation
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        n_valid = len(self._valid_cells)

        prob_over_time = np.zeros((len(times), n_valid))
        for t_idx, time in enumerate(times):
            prob_over_time[t_idx] = self.get_state_probabilities(time)

        fig, ax = plt.subplots(figsize=figsize)

        orientation = 0 if self.hex_grid.orientation == "pointy" else np.pi / 6
        patches = []
        for x, y in self.hex_centers:
            hex_patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=self.hex_size,
                orientation=orientation,
            )
            patches.append(hex_patch)

        pc = PatchCollection(patches, cmap='hot', linewidths=0.3, edgecolors='black')
        pc.set_clim(0, prob_over_time.max())
        ax.add_collection(pc)

        self.gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label('State Probability')

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
