"""Regression test for the hex-grid even/odd neighbor-offset swap.

The neighbor offset tables were swapped between even/odd parity, so even
rows received odd-row diagonal offsets (and vice versa). The result:
`neighbors()` returned cells two hexes away and omitted the truly adjacent
cells on alternating rows/columns — silently wrong migration/coalescent
adjacency, with no crash (adjacency stayed symmetric).

A true first-order neighbor's center is exactly one hex spacing
(sqrt(3) * hex_size) from the cell center, for BOTH orientations and BOTH
parities. These tests assert that geometric invariant, which fails on the
old swapped tables.
"""
import math

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Point  # noqa: E402

from phasic.hex_grid import HexGrid  # noqa: E402


def _spacing(grid):
    return math.sqrt(3) * grid._hex_size


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_every_neighbor_is_exactly_one_hex_away(orientation):
    boundary = Point(0, 0).buffer(6)
    grid = HexGrid(boundary, hex_size=1.0, orientation=orientation)
    spacing = _spacing(grid)

    checked_even = checked_odd = 0
    for r in range(grid._n_rows):
        for c in range(grid._n_cols):
            if not grid._valid_mask[r, c]:
                continue
            cx, cy = grid.rowcol_to_coords(r, c)
            nbrs = grid.neighbors(r, c)
            assert nbrs, f"cell ({r},{c}) has no neighbors"
            for nr, nc in nbrs:
                nx, ny = grid.rowcol_to_coords(nr, nc)
                d = math.hypot(nx - cx, ny - cy)
                assert d == pytest.approx(spacing, rel=1e-9), (
                    f"{orientation}: ({r},{c})->({nr},{nc}) distance {d:.4f} "
                    f"!= hex spacing {spacing:.4f}"
                )
            # count that we exercised both parities
            parity = r % 2 if orientation == "pointy" else c % 2
            if parity == 0:
                checked_even += 1
            else:
                checked_odd += 1

    assert checked_even > 0 and checked_odd > 0, "must cover both parities"


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_adjacency_is_symmetric(orientation):
    boundary = Point(0, 0).buffer(6)
    grid = HexGrid(boundary, hex_size=1.0, orientation=orientation)
    for r in range(grid._n_rows):
        for c in range(grid._n_cols):
            if not grid._valid_mask[r, c]:
                continue
            for nr, nc in grid.neighbors(r, c):
                back = grid.neighbors(nr, nc)
                assert (r, c) in back, (
                    f"{orientation}: ({r},{c})->({nr},{nc}) not mutual"
                )


def test_interior_pointy_cell_has_six_neighbors():
    """A deep-interior cell must have all six neighbors, all at hex spacing."""
    boundary = Point(0, 0).buffer(10)
    grid = HexGrid(boundary, hex_size=1.0, orientation="pointy")
    # pick a cell near the centroid (guaranteed interior)
    r0, c0 = grid.center_cell()
    nbrs = grid.neighbors(r0, c0)
    assert len(nbrs) == 6, f"interior cell should have 6 neighbors, got {len(nbrs)}"
