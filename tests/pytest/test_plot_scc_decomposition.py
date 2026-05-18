"""Tests for Graph.plot_scc_decomp().

The plot is a structural visualisation only — these tests
verify the API surface (returns an axes, accepts kwargs,
handles the empty-SCC edge case) and check the auto-generated
title against the known SCC structure of toy_b.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # headless backend for CI

import matplotlib.pyplot as plt
import pytest

from toy_model import build_toy_b, BUILDERS


def test_plot_returns_axes():
    g = build_toy_b()
    ax = g.plot_scc_decomp()
    assert ax is not None
    plt.close(ax.figure)


def test_plot_title_matches_decomposition():
    """Title encodes (n_sccs, n_levels, widest_level, total_vertices)."""
    g = build_toy_b()
    scc = g.scc_decomposition()
    from phasic.distributed_scc import compute_scc_levels
    levels = compute_scc_levels(scc)

    ax = g.plot_scc_decomp()
    title = ax.get_title()
    assert f"{len(scc)} SCCs" in title
    assert f"{len(levels)} levels" in title
    assert f"widest {max(len(l) for l in levels)}" in title
    plt.close(ax.figure)


def test_plot_accepts_custom_axes():
    fig, ax = plt.subplots()
    g = build_toy_b()
    returned = g.plot_scc_decomp(ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_accepts_custom_title():
    g = build_toy_b()
    ax = g.plot_scc_decomp(title="my title")
    assert ax.get_title() == "my title"
    plt.close(ax.figure)


def test_plot_hides_axes_decorations():
    """Treemap should have no tick marks and no spines."""
    g = build_toy_b()
    ax = g.plot_scc_decomp()
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    assert not any(s.get_visible() for s in ax.spines.values())
    plt.close(ax.figure)


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_plot_works_for_every_toy(toy_name):
    """Smoke test: every toy model in the regression set
    produces a valid plot."""
    g = BUILDERS[toy_name]()
    ax = g.plot_scc_decomp()
    title = ax.get_title()
    # Title always names at least one SCC.
    assert "SCCs" in title
    plt.close(ax.figure)


def test_plot_show_indices_kwarg():
    """show_indices=False should still produce a plot."""
    g = build_toy_b()
    ax = g.plot_scc_decomp(show_indices=False)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_annotate_sizes_kwarg():
    """annotate_sizes=False suppresses the 'Nv' suffix."""
    g = build_toy_b()
    ax = g.plot_scc_decomp(annotate_sizes=False)
    # We can't easily inspect tile labels via the axes object,
    # but at least the call must succeed.
    assert ax is not None
    plt.close(ax.figure)


def test_plot_handles_single_scc():
    """A graph whose SCC decomposition is a single trivial SCC
    still plots cleanly (no division by zero in colour mapping)."""
    from phasic import Graph
    g = Graph(1)
    v = g.find_or_create_vertex([1])
    g.starting_vertex().add_edge(v, 1.0)
    ax = g.plot_scc_decomp()
    assert ax is not None
    plt.close(ax.figure)
