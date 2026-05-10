"""WP-3 tests: format extension for cross-SCC binding (PTD_PCG_PTR_EXTERNAL).

Verifies the new ``_ex`` save/load entry points and the
``ptd_scc_collect_external_anchors`` helper. The basic round trip:

    synth, meta = scc.as_synthetic_graph()
    synth.update_weights(...)
    synth.expected_waiting_time()       # populates the PRC
    synth._save_synthetic_param_compute_graph_ex(path)
    # ... new graph rebuilt from same parent ...
    new_synth._load_synthetic_param_compute_graph_ex(path, external_table)
    new_synth.expected_waiting_time()  # uses external_table values

Also confirms format-revision compatibility: rev-1 files load via
both v1 and v2 loaders; rev-2 files containing EXTERNAL pointers
fail clearly when loaded via the v1 loader.

See wp3-plan.md for the design.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from phasic import Graph
from toy_model import BUILDERS, build_toy_base, build_toy_d


def _scc_with_uc(g):
    """Return the first SCC of g that has at least one
    upstream-connecting vertex (so its synthetic graph has Type A
    placeholder edges to anchor)."""
    scc_decomp = g.scc_decomposition()
    for scc in scc_decomp.sccs_in_topo_order():
        synth, meta = scc.as_synthetic_graph()
        if meta.n_upstream_connecting > 0 or meta.n_downstream_connecting > 0:
            return scc
    raise AssertionError("no SCC with placeholder edges in this graph")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_with_external_anchors():
    """Save a synthetic graph's PRC with EXTERNAL anchors, load it
    against the same synthetic graph topology with a populated
    external_table, replay, verify the result matches direct
    elimination of the synthetic graph with the same theta."""
    g = build_toy_base()
    scc = _scc_with_uc(g)
    synth, meta = scc.as_synthetic_graph()

    # Drive an elimination so the PRC exists.
    synth.update_weights([1.0, 1.0, 1.0, 1.0])
    direct_ewt = np.asarray(synth.expected_waiting_time())

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scc.bin")
        n_anchors = synth._save_synthetic_param_compute_graph_ex(path)
        assert n_anchors > 0, "expected at least one EXTERNAL anchor"
        assert os.path.exists(path)

        # Build a fresh synthetic graph with identical topology (the
        # cache is keyed by content; re-extracting the same SCC
        # gives a structurally-identical graph) and load the saved
        # PRC, supplying external_table = [1.0, 1.0, ...] — the
        # same effective values the placeholder coefficients had at
        # save time.
        synth2, _ = scc.as_synthetic_graph()
        external_table = [1.0] * n_anchors
        synth2._load_synthetic_param_compute_graph_ex(path, external_table)

        # Set theta and replay through the loaded compute graph.
        synth2.update_weights([1.0, 1.0, 1.0, 1.0])
        loaded_ewt = np.asarray(synth2.expected_waiting_time())

        np.testing.assert_allclose(loaded_ewt, direct_ewt, rtol=1e-12, atol=1e-12)


def test_external_table_substitution_changes_result():
    """Load the same v2 file twice with different external_table
    values; the elimination output must differ. Confirms the
    EXTERNAL binding actually flows into the replay."""
    g = build_toy_base()
    scc = _scc_with_uc(g)
    synth, meta = scc.as_synthetic_graph()

    synth.update_weights([1.0, 1.0, 1.0, 1.0])
    synth.expected_waiting_time()  # populate PRC

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scc.bin")
        n_anchors = synth._save_synthetic_param_compute_graph_ex(path)
        if n_anchors == 0:
            pytest.skip("no EXTERNAL anchors for this SCC")

        # Run 1: external_table = ones.
        synth_a, _ = scc.as_synthetic_graph()
        synth_a._load_synthetic_param_compute_graph_ex(path, [1.0] * n_anchors)
        synth_a.update_weights([1.0, 1.0, 1.0, 1.0])
        ewt_ones = np.asarray(synth_a.expected_waiting_time())

        # Run 2: external_table with one slot set to 5.0.
        synth_b, _ = scc.as_synthetic_graph()
        table = [1.0] * n_anchors
        table[0] = 5.0
        synth_b._load_synthetic_param_compute_graph_ex(path, table)
        synth_b.update_weights([1.0, 1.0, 1.0, 1.0])
        ewt_modified = np.asarray(synth_b.expected_waiting_time())

        # Results must differ — confirms the EXTERNAL binding is live.
        assert not np.allclose(ewt_ones, ewt_modified), (
            f"external_table substitution had no effect — binding is dead. "
            f"ones-result={ewt_ones}, modified-result={ewt_modified}"
        )


# ---------------------------------------------------------------------------
# Format-revision compatibility
# ---------------------------------------------------------------------------

def test_v1_rev1_file_loads_via_v2_loader():
    """A rev-1 file (no EXTERNAL pointers) loads via the v2 loader
    (passing external_table=[]) without error."""
    # Build a non-synthetic (regular) parameterised graph and
    # populate its Stage A2 cache. This produces a rev-1 file via
    # the existing _save_param_compute_graph helper.
    g = build_toy_base()
    g.update_weights([1.0, 1.0, 1.0, 1.0])
    g.expected_waiting_time()  # populates parameterized_reward_compute_graph

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "stage_a2.bin")
        g._save_param_compute_graph(path)  # writes rev-1

        # Build a fresh graph with same topology and load it via the
        # v2 path with an empty external_table. This should succeed
        # because the rev-1 file contains no EXTERNAL pointers.
        g2 = build_toy_base()
        g2._load_synthetic_param_compute_graph_ex(path, [])
        g2.update_weights([1.0, 1.0, 1.0, 1.0])
        ewt = np.asarray(g2.expected_waiting_time())
        assert np.all(np.isfinite(ewt))


def test_v2_file_rejects_v1_loader():
    """A rev-2 file containing EXTERNAL pointers must fail to
    load via the v1 loader (clear error, no crash)."""
    g = build_toy_base()
    scc = _scc_with_uc(g)
    synth, meta = scc.as_synthetic_graph()
    synth.update_weights([1.0, 1.0, 1.0, 1.0])
    synth.expected_waiting_time()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scc.bin")
        n_anchors = synth._save_synthetic_param_compute_graph_ex(path)
        if n_anchors == 0:
            pytest.skip("no EXTERNAL anchors")

        # Try to load via the v1 loader.
        synth2, _ = scc.as_synthetic_graph()
        with pytest.raises(RuntimeError) as exc_info:
            synth2._load_param_compute_graph(path)
        # Error message should mention rev-2 / EXTERNAL / external_table.
        msg = str(exc_info.value).lower()
        assert "external" in msg or "rev 2" in msg or "r2" in msg, (
            f"expected error mentioning EXTERNAL or rev 2; got: {msg}"
        )


# ---------------------------------------------------------------------------
# Anchor collection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_anchor_count_per_synth_graph(toy_name):
    """For every SCC of every toy variant, verify that the number
    of EXTERNAL anchors equals the number of placeholder edges
    (synth source out-edges + count of edges into synth absorbing)."""
    g = BUILDERS[toy_name]()
    scc_decomp = g.scc_decomposition()
    for scc in scc_decomp.sccs_in_topo_order():
        synth, meta = scc.as_synthetic_graph()
        synth.update_weights([1.0, 1.0, 1.0, 1.0])
        synth.expected_waiting_time()

        # Count expected anchors directly from the synthetic graph.
        src = next(v for v in synth.vertices() if v.index() == 0)
        n_a = len(list(src.edges()))
        abs_idx = synth.vertices_length() - 1
        n_c = 0
        for v in synth.vertices():
            if v.index() == 0 or v.index() == abs_idx:
                continue
            for e in v.edges():
                if e.to().index() == abs_idx:
                    n_c += 1

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scc.bin")
            n_saved = synth._save_synthetic_param_compute_graph_ex(path)
            assert n_saved == n_a + n_c, (
                f"{toy_name}/scc{scc.index()}: anchor count mismatch — "
                f"saved {n_saved}, expected {n_a + n_c} (n_a={n_a}, n_c={n_c})"
            )


# ---------------------------------------------------------------------------
# Toy-D (duplicate-state aux SCC)
# ---------------------------------------------------------------------------

def test_toy_d_aux_scc_roundtrip():
    """Toy-D's aux-containing SCC must round-trip correctly through
    the v2 save/load. The duplicate-state aux vertex shouldn't
    affect the EXTERNAL pointer scheme."""
    g = build_toy_d()
    scc_decomp = g.scc_decomposition()
    aux_scc = None
    for scc in scc_decomp.sccs_in_topo_order():
        if scc.size() == 3:
            aux_scc = scc
            break
    assert aux_scc is not None

    synth, meta = aux_scc.as_synthetic_graph()
    synth.update_weights([1.0, 1.0, 1.0, 1.0])
    direct_ewt = np.asarray(synth.expected_waiting_time())

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "toy_d_aux.bin")
        n_anchors = synth._save_synthetic_param_compute_graph_ex(path)

        synth2, _ = aux_scc.as_synthetic_graph()
        synth2._load_synthetic_param_compute_graph_ex(
            path, [1.0] * n_anchors)
        synth2.update_weights([1.0, 1.0, 1.0, 1.0])
        loaded_ewt = np.asarray(synth2.expected_waiting_time())

        np.testing.assert_allclose(
            loaded_ewt, direct_ewt, rtol=1e-12, atol=1e-12)
