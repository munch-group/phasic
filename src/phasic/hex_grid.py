"""
Hexagonal grid support for phasic.

Provides HexGrid for creating hexagonal grids within geographic boundaries,
with neighbor computation, coordinate conversion, and Graph construction.
Grid position is stored as regular row/col Properties that compose with
StateIndexer alongside other properties.

Example
-------
>>> from phasic import HexGrid, StateIndexer, Property
>>>
>>> grid = HexGrid.from_shapefile('africa.shp', hex_size=1.5)
>>>
>>> indexer = StateIndexer(
...     lineage=grid.properties() + [Property('population', max_value=3, min_value=1)]
... )
>>>
>>> def migration(state, grid):
...     props = indexer.lineage.index_to_props(int(state[0]), as_dict=True)
...     transitions = []
...     for nr, nc in grid.neighbors(props['row'], props['col']):
...         next_idx = indexer.lineage.props_to_index(
...             row=nr, col=nc, population=props['population']
...         )
...         transitions.append(([next_idx], 1.0))
...     return transitions
>>>
>>> g = grid.build_graph(migration, indexer.lineage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .state_indexing import Property

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapely.geometry.base import BaseGeometry

    from .state_indexing import PropertySet


# Hex neighbor (drow, dcol) offsets by parity. These are derived directly
# from rowcol_to_coords: for "pointy" the layout is odd-r (odd rows shoved
# right by half a cell), for "flat" it is odd-q (odd columns shoved down).
# A true first-order neighbor is exactly hex-spacing (sqrt(3)*hex_size)
# from the cell center; each offset below satisfies that (verified in
# test_hex_grid_adjacency). The even/odd tables were previously swapped,
# which connected cells two hexes apart and dropped real neighbors on
# alternating rows/columns.
#
# Pointy-top (odd-r): same-row +/-1, plus the two upper and two lower cells.
# For even rows (row % 2 == 0) the diagonals lean left (dcol in {0, -1}):
_EVEN_ROW_OFFSETS = [(0, -1), (0, 1), (-1, 0), (-1, -1), (1, 0), (1, -1)]
# For odd rows (row % 2 == 1) the diagonals lean right (dcol in {0, +1}):
_ODD_ROW_OFFSETS = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]

# Flat-top (odd-q): same-column +/-1, plus the two left and two right cells.
# For even columns (col % 2 == 0) the diagonals lean up (drow in {0, -1}):
_EVEN_COL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (-1, -1), (0, 1), (-1, 1)]
# For odd columns (col % 2 == 1) the diagonals lean down (drow in {0, +1}):
_ODD_COL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]


@dataclass
class HexCell:
    """Lightweight view of a hex grid cell with neighbor access.

    Parameters
    ----------
    row : int
        Row index in the grid.
    col : int
        Column index in the grid.

    Attributes
    ----------
    coords : tuple[float, float]
        Real-world (x, y) coordinates of the cell center.
    is_valid : bool
        Whether the cell is inside the boundary.
    """

    row: int
    col: int
    _grid: HexGrid

    def __repr__(self) -> str:
        x, y = self.coords
        return f"HexCell(row={self.row}, col={self.col}, coords=({x:.4f}, {y:.4f}), valid={self.is_valid})"

    @property
    def coords(self) -> tuple[float, float]:
        """Real-world (x, y) coordinates of cell center."""
        return self._grid.rowcol_to_coords(self.row, self.col)

    @property
    def is_valid(self) -> bool:
        """Whether cell is inside the boundary."""
        return self._grid.is_valid(self.row, self.col)

    def neighbors(self) -> list[HexCell]:
        """First-order valid neighbors as HexCell objects."""
        return [
            HexCell(r, c, self._grid)
            for r, c in self._grid.neighbors(self.row, self.col)
        ]

    def neighbors_within(self, distance: int) -> list[HexCell]:
        """All valid neighbors within ``distance`` hex steps."""
        return [
            HexCell(r, c, self._grid)
            for r, c in self._grid.neighbors_within(self.row, self.col, distance)
        ]


class HexGrid:
    """Hexagonal grid within a geographic boundary.

    Creates a hex grid sized to cover a boundary geometry. Provides
    ``row_property`` and ``col_property`` as regular :class:`Property`
    objects for composition into a :class:`StateIndexer`.

    Parameters
    ----------
    boundary : BaseGeometry
        Shapely geometry defining the area to cover.
    hex_size : float
        Circumradius of hexagonal cells (in boundary CRS units).
    orientation : str, optional
        ``"pointy"`` (pointy-top, default) or ``"flat"`` (flat-top).

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> boundary = Point(0, 0).buffer(5)
    >>> grid = HexGrid(boundary, hex_size=1.0)
    >>> grid.n_rows, grid.n_cols
    (8, 7)
    >>> grid.row_property
    Property(name='row', max_value=7, min_value=0, offset=0)
    """

    def __init__(
        self,
        boundary: BaseGeometry,
        hex_size: float,
        orientation: str = "pointy",
    ) -> None:
        self._boundary = boundary
        self._hex_size = hex_size
        self._orientation = orientation

        # Compute grid dimensions from boundary bounds
        minx, miny, maxx, maxy = boundary.bounds

        if orientation == "pointy":
            horiz = np.sqrt(3) * hex_size
            vert = 1.5 * hex_size
        elif orientation == "flat":
            horiz = 1.5 * hex_size
            vert = np.sqrt(3) * hex_size
        else:
            raise ValueError(f"orientation must be 'pointy' or 'flat', got {orientation!r}")

        self._n_cols = int(np.ceil((maxx - minx) / horiz)) + 1
        self._n_rows = int(np.ceil((maxy - miny) / vert)) + 1
        self._origin = (minx, miny)

        # Create Property objects for StateIndexer composition
        self._row_property = Property("row", max_value=self._n_rows - 1)
        self._col_property = Property("col", max_value=self._n_cols - 1)

        # Precompute validity mask and adjacency
        self._valid_mask = self._compute_valid_mask()
        self._adjacency = self._compute_adjacency()

    @classmethod
    def from_shapefile(
        cls,
        path: str,
        hex_size: float,
        **kwargs,
    ) -> HexGrid:
        """Create hex grid from a shapefile.

        Requires ``geopandas`` (imported lazily).

        Parameters
        ----------
        path : str
            Path to shapefile.
        hex_size : float
            Circumradius of hexagonal cells.
        **kwargs
            Passed to :class:`HexGrid` constructor.

        Returns
        -------
        HexGrid
        """
        import geopandas as gpd

        gdf = gpd.read_file(path)
        boundary = gdf.unary_union
        grid = cls(boundary, hex_size, **kwargs)
        grid._gdf = gdf
        return grid

    # --- Properties for StateIndexer composition ---

    @property
    def row_property(self) -> Property:
        """Row Property for use in a StateIndexer PropertySet."""
        return self._row_property

    @property
    def col_property(self) -> Property:
        """Column Property for use in a StateIndexer PropertySet."""
        return self._col_property

    def properties(self) -> list[Property]:
        """Return ``[row_property, col_property]`` for easy composition.

        Example
        -------
        >>> indexer = StateIndexer(
        ...     lineage=grid.properties() + [Property('allele', max_value=1)]
        ... )
        """
        return [self._row_property, self._col_property]

    # --- Grid info ---

    @property
    def n_rows(self) -> int:
        """Number of rows in the grid."""
        return self._n_rows

    @property
    def n_cols(self) -> int:
        """Number of columns in the grid."""
        return self._n_cols

    @property
    def hex_size(self) -> float:
        """Circumradius of hexagonal cells."""
        return self._hex_size

    @property
    def boundary(self) -> BaseGeometry:
        """Shapely boundary geometry."""
        return self._boundary

    @property
    def orientation(self) -> str:
        """Grid orientation: ``"pointy"`` or ``"flat"``."""
        return self._orientation

    # --- Cell access ---

    def cell(self, row: int, col: int) -> HexCell:
        """Get a :class:`HexCell` for the given (row, col).

        Parameters
        ----------
        row, col : int
            Grid coordinates.

        Returns
        -------
        HexCell
        """
        return HexCell(row, col, self)

    # --- Neighbors ---

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """First-order valid neighbor ``(row, col)`` pairs.

        Parameters
        ----------
        row, col : int
            Grid coordinates of the cell.

        Returns
        -------
        list of (int, int)
            Neighboring cells that are within bounds and inside the boundary.
        """
        return self._adjacency.get((row, col), [])

    def neighbors_within(
        self, row: int, col: int, distance: int
    ) -> list[tuple[int, int]]:
        """All valid neighbors within ``distance`` hex steps (BFS).

        Parameters
        ----------
        row, col : int
            Grid coordinates of the center cell.
        distance : int
            Maximum number of hex steps.

        Returns
        -------
        list of (int, int)
            All reachable cells within ``distance`` steps.
        """
        visited = {(row, col)}
        frontier = {(row, col)}
        for _ in range(distance):
            next_frontier = set()
            for r, c in frontier:
                for nr, nc in self._adjacency.get((r, c), []):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        next_frontier.add((nr, nc))
            frontier = next_frontier
        visited.discard((row, col))
        return sorted(visited)

    # --- Coordinate conversion ---

    def rowcol_to_coords(self, row: int, col: int) -> tuple[float, float]:
        """Convert (row, col) to real-world (x, y) coordinates.

        Parameters
        ----------
        row, col : int
            Grid coordinates.

        Returns
        -------
        tuple of (float, float)
            (x, y) coordinates of the cell center.
        """
        ox, oy = self._origin
        s = self._hex_size

        if self._orientation == "pointy":
            x = ox + col * np.sqrt(3) * s + (row % 2) * np.sqrt(3) / 2 * s
            y = oy + row * 1.5 * s
        else:  # flat
            x = ox + col * 1.5 * s
            y = oy + row * np.sqrt(3) * s + (col % 2) * np.sqrt(3) / 2 * s

        return (float(x), float(y))

    def coords_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Find the nearest valid cell to real-world coordinates.

        Parameters
        ----------
        x, y : float
            Real-world coordinates.

        Returns
        -------
        tuple of (int, int)
            (row, col) of the nearest valid cell.

        Raises
        ------
        ValueError
            If no valid cells exist in the grid.
        """
        min_dist = float("inf")
        best = None
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                if self._valid_mask[r, c]:
                    cx, cy = self.rowcol_to_coords(r, c)
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    if d < min_dist:
                        min_dist = d
                        best = (r, c)
        if best is None:
            raise ValueError("No valid cells in grid")
        return best

    def center_cell(self) -> tuple[int, int]:
        """(row, col) of the cell nearest to the boundary centroid.

        Returns
        -------
        tuple of (int, int)
        """
        cx, cy = self._boundary.centroid.coords[0]
        return self.coords_to_rowcol(cx, cy)

    def valid_cells(self) -> list[tuple[int, int]]:
        """All ``(row, col)`` pairs inside the boundary.

        Returns
        -------
        list of (int, int)
        """
        cells = []
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                if self._valid_mask[r, c]:
                    cells.append((r, c))
        return cells

    def is_valid(self, row: int, col: int) -> bool:
        """Whether the cell at (row, col) is inside the boundary.

        Parameters
        ----------
        row, col : int
            Grid coordinates.

        Returns
        -------
        bool
        """
        if 0 <= row < self._n_rows and 0 <= col < self._n_cols:
            return bool(self._valid_mask[row, col])
        return False

    def all_coords(self) -> np.ndarray:
        """Coordinates for all grid cells.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_cols, 2)`` array of ``(x, y)`` coordinates.
        """
        coords = np.zeros((self._n_rows, self._n_cols, 2))
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                coords[r, c] = self.rowcol_to_coords(r, c)
        return coords

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid cells, shape ``(n_rows, n_cols)``."""
        return self._valid_mask

    # --- Graph construction ---

    def build_graph(
        self,
        transition_fn: Callable,
        property_set: PropertySet,
        start_cell: tuple[int, int] | None = None,
        parameterized: bool = False,
        **graph_kwargs,
    ):
        """Build a phasic Graph by iterating valid hex cells.

        The resulting Graph is a standard phasic Graph, fully compatible
        with serialize, FFI, JAX (jit/grad/vmap/pmap), SVGD, etc.

        Parameters
        ----------
        transition_fn : callable
            Pure function ``(state, grid) -> transitions`` where ``state``
            is a numpy array (PropertySet flat index) and ``grid`` is this
            HexGrid. Returns list of transitions:

            - Non-parameterized: ``[(next_state, rate), ...]``
            - Parameterized: ``[(next_state, coefficients), ...]``
        property_set : PropertySet
            The PropertySet containing ``row``/``col`` properties (and
            possibly other properties). Used to determine ``state_length``.
        start_cell : tuple of (int, int), optional
            ``(row, col)`` of starting cell. Defaults to :meth:`center_cell`.
        parameterized : bool, optional
            Whether edges are parameterized (default False).
        **graph_kwargs
            Passed to ``Graph`` constructor.

        Returns
        -------
        Graph
            Standard phasic Graph.
        """
        from phasic import Graph

        # Graph state_length = 1: each vertex state is a single flat index
        # into the PropertySet's combinatorial space.
        graph = Graph(1, **graph_kwargs)

        if start_cell is None:
            start_cell = self.center_cell()

        # Connect starting vertex to start cell
        # The start cell index depends on the property_set composition.
        # For a grid-only PropertySet, it's just props_to_index(row=r, col=c).
        # For multi-property sets, we need to iterate over all combinations
        # of the other properties too. Here we connect to the start cell
        # with all other properties at their minimum values.
        start_idx = property_set.props_to_index(
            row=start_cell[0], col=start_cell[1]
        )
        sv = graph.starting_vertex()

        if isinstance(start_idx, np.ndarray):
            # Partial specification: multiple indices (other props vary).
            # Connect to first one (all other props at min values).
            start_idx = int(start_idx[0])

        start_state = [start_idx]
        start_v = graph.find_or_create_vertex(start_state)
        sv.add_edge(start_v, 1.0)

        # Iterate all valid cells
        for row, col in self.valid_cells():
            # Get all flat indices for this (row, col) with all combinations
            # of other properties
            indices = property_set.props_to_index(row=row, col=col)

            if isinstance(indices, (int, np.integer)):
                indices = [int(indices)]
            else:
                indices = [int(i) for i in indices]

            for idx in indices:
                state = np.array([idx])
                vertex = graph.find_or_create_vertex([idx])

                transitions = transition_fn(state, self)

                for t in transitions:
                    if parameterized:
                        next_state, coeffs = t
                        nv = graph.find_or_create_vertex(list(next_state))
                        vertex.add_edge(nv, coeffs)
                    else:
                        next_state, rate = t[0], t[1]
                        nv = graph.find_or_create_vertex(list(next_state))
                        vertex.add_edge(nv, rate)

        return graph

    # --- Result mapping ---

    def map_to_grid(
        self,
        graph,
        property_set: PropertySet,
        values: np.ndarray,
        prop_filter: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Map graph vertex values to a ``(n_rows, n_cols)`` grid.

        Parameters
        ----------
        graph : Graph
            The phasic Graph (vertices carry PropertySet flat indices
            as state vectors).
        property_set : PropertySet
            The PropertySet containing row/col properties.
        values : np.ndarray
            Values indexed by graph vertex index (e.g., from
            ``graph.state_probability()``).
        prop_filter : dict, optional
            Fixed non-grid property values to filter on, e.g.,
            ``{'population': 1}``. If ``None`` and extra properties
            exist, sums over all combinations.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_cols)`` array.
        """
        grid_values = np.zeros((self._n_rows, self._n_cols))

        row_name = self._row_property.name
        col_name = self._col_property.name

        for i in range(graph.vertices_length()):
            if i >= len(values):
                break
            v = graph.vertex_at(i)
            state = list(v.state())
            if not state:
                continue
            pset_idx = int(state[0])
            if pset_idx < 0 or pset_idx >= property_set.state_length:
                continue

            props = property_set.index_to_props(pset_idx, as_dict=True)
            r, c = props[row_name], props[col_name]

            # Apply filter if specified
            if prop_filter is not None:
                skip = False
                for k, v_filter in prop_filter.items():
                    if props.get(k) != v_filter:
                        skip = True
                        break
                if skip:
                    continue

            if 0 <= r < self._n_rows and 0 <= c < self._n_cols:
                grid_values[r, c] += values[i]

        return grid_values

    # --- Internal ---

    def _compute_valid_mask(self) -> np.ndarray:
        """Compute (n_rows, n_cols) boolean mask of cells inside boundary."""
        from shapely.geometry import Point

        mask = np.zeros((self._n_rows, self._n_cols), dtype=bool)
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                x, y = self.rowcol_to_coords(r, c)
                mask[r, c] = self._boundary.contains(Point(x, y))
        return mask

    def _compute_adjacency(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Precompute neighbor lists using hex coordinate offsets."""
        if self._orientation == "pointy":
            even_offsets = _EVEN_ROW_OFFSETS
            odd_offsets = _ODD_ROW_OFFSETS
        else:
            even_offsets = _EVEN_COL_OFFSETS
            odd_offsets = _ODD_COL_OFFSETS

        adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                if not self._valid_mask[r, c]:
                    adj[(r, c)] = []
                    continue

                if self._orientation == "pointy":
                    offsets = even_offsets if r % 2 == 0 else odd_offsets
                else:
                    offsets = even_offsets if c % 2 == 0 else odd_offsets

                nbrs = []
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self._n_rows
                        and 0 <= nc < self._n_cols
                        and self._valid_mask[nr, nc]
                    ):
                        nbrs.append((nr, nc))
                adj[(r, c)] = nbrs

        return adj
