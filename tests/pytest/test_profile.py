"""Tests for the graph profiler (phasic.profile_graph / Graph.profile).

Assertions are machine-robust: structural metrics (SCC counts, level widths,
max_rate) are deterministic; the parallel_elimination *recommendation* depends on
os.cpu_count(), so it is only asserted conditionally on cores being available.

Run in isolation (the full suite has unrelated hangs):
    pixi run pytest tests/pytest/test_profile.py -q
"""
import os

import numpy as np
import pytest

import phasic
from phasic import Graph, Property, StateIndexer


def _build_twolocus(nr):
    idx = StateIndexer(descendants=[Property('loc1', max_value=nr),
                                    Property('loc2', max_value=nr)])
    init = [0] * idx.state_length
    init[idx.props_to_index(loc1=1, loc2=1)] = nr

    def cb(state, indexer=None):
        t = []
        if state.sum() <= 1:
            return t
        for i in range(indexer.state_length):
            if state[i] == 0:
                continue
            pi = indexer.index_to_props(i)
            for j in range(i, indexer.state_length):
                if state[j] == 0:
                    continue
                pj = indexer.index_to_props(j)
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                c = state.copy(); c[i] -= 1; c[j] -= 1
                l1 = pi.descendants.loc1 + pj.descendants.loc1
                l2 = pi.descendants.loc2 + pj.descendants.loc2
                if l1 <= nr and l2 <= nr:
                    c[idx.props_to_index(loc1=l1, loc2=l2)] += 1
                    t.append([c, [state[i] * (state[j] - same) / (1 + same), 0]])
            if state[i] > 0 and pi.descendants.loc1 > 0 and pi.descendants.loc2 > 0:
                c = state.copy(); c[i] -= 1
                c[idx.props_to_index(loc1=pi.descendants.loc1, loc2=0)] += 1
                c[idx.props_to_index(loc1=0, loc2=pi.descendants.loc2)] += 1
                t.append([c, [0, 1]])
        return t
    return Graph(cb, ipv=init, indexer=idx)


def _build_coal(n):
    def cb(state, **kw):
        k = int(state[0])
        return [] if k <= 1 else [(np.array([k - 1]), [k * (k - 1) / 2.0])]
    return Graph(cb, ipv=[n])


def test_twolocus_structure_and_parallel_recommendation():
    p = phasic.profile_graph(_build_twolocus(5))
    # Structural facts are deterministic regardless of machine.
    assert p.n_vertices == 340
    assert p.param_length == 2
    assert p.n_sccs > 1
    assert p.max_level_width >= 2          # independent SCCs exist
    assert p.parallelizable_frac > 0.5     # most work is at width>=2 levels
    # Recommendation depends on available cores.
    decided, _why = p.recommendations["parallel_elimination"]
    assert decided is ((os.cpu_count() or 1) > 1)


def test_twolocus_path_is_forward_low_max_rate():
    p = phasic.profile_graph(_build_twolocus(5))
    assert p.recommendations["path"][0] == "forward"
    assert p.auto_granularity < 1e5


def test_coalescent_chain_disables_parallel():
    p = phasic.profile_graph(_build_coal(80))
    assert p.max_level_width == 1          # condensation is a chain
    # A chain has nothing independent => OFF on any machine.
    assert p.recommendations["parallel_elimination"][0] is False


def test_coalescent_large_recommends_sojourn():
    # High max_rate (C(400,2)) => forward-PDF granularity huge => sojourn.
    p = phasic.profile_graph(_build_coal(400))
    assert p.auto_granularity >= 1e5
    assert p.recommendations["path"][0] == "sojourn"


def test_dyn_probe_negligible_for_small_sccs():
    # two-locus SCCs are tiny (<10 ms elimination) => dyn immaterial => OFF.
    p = phasic.profile_graph(_build_twolocus(5), probe_dyn=True)
    assert p.recommendations["dyn_ordering"][0] is False
    assert p.dyn_probe is not None


def test_probe_dyn_false_skips_probe():
    p = phasic.profile_graph(_build_twolocus(5), probe_dyn=False)
    assert p.dyn_probe is None
    assert "not probed" in p.recommendations["dyn_ordering"][1]


def test_method_matches_function():
    g = _build_twolocus(5)
    assert g.profile().recommendations == phasic.profile_graph(g).recommendations


def test_theta_shape_validation():
    g = _build_twolocus(5)            # param_length == 2
    with pytest.raises(ValueError):
        phasic.profile_graph(g, theta=np.ones(3))


def test_report_and_snippet_are_strings():
    p = phasic.profile_graph(_build_twolocus(5))
    assert isinstance(p.report(), str) and "SCC structure" in p.report()
    assert isinstance(p.apply_snippet, str)
