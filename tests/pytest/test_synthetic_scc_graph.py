"""WP-1 tests: synthetic SCC graph constructor.

Verifies SCCVertex.as_synthetic_graph() against the toy variants
(toy_model.py). See wp1-plan.md §7 for the test plan and §11 of
hierar-elimin-cache-reference.md for the toy model design.

These tests do NOT verify cross-path numerical equivalence with
the parent monolithic eliminator — that is WP-5's job. They
verify only that the synthetic graph is structurally valid input
for the existing eliminator and that the metadata is correct.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from phasic import Graph
from phasic import hash as phasic_hash
from toy_model import BUILDERS, build_toy_d, build_toy_c_p, build_toy_c_pprime


# Test parametrisation helpers
def _all_scc_cases() -> Iterable[tuple[str, int]]:
    """Yield (toy_name, scc_index) for every SCC of every toy variant."""
    for toy_name, builder in BUILDERS.items():
        g = builder()
        scc_graph = g.scc_decomposition()
        for i in range(scc_graph.n_sccs()):
            yield (toy_name, i)


SCC_CASES = list(_all_scc_cases())
SCC_CASE_IDS = [f"{name}-scc{idx}" for name, idx in SCC_CASES]


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_graph_has_canonical_layout(toy_name, scc_idx):
    """Vertex count and category counts match the canonical
    layout (per-channel absorbings + phantom):
        1 source + n_uc + n_io + n_dc + n_channels + 1 phantom"""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    synth, meta = scc.as_synthetic_graph()

    expected_n = (
        1  # synthetic source
        + meta.n_upstream_connecting
        + meta.n_internal_only
        + meta.n_downstream_connecting
        + meta.n_channels  # per-channel absorbing vertices
        + 1  # phantom absorbing
    )
    assert synth.vertices_length() == expected_n, (
        f"{toy_name}/scc{scc_idx}: vertex count mismatch — "
        f"have {synth.vertices_length()}, expected {expected_n}"
    )
    assert meta.n_vertices == expected_n


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_synthetic_graph_state_length_preserved(toy_name):
    """state_length is preserved from parent."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    for scc in scc_graph.sccs_in_topo_order():
        synth, _ = scc.as_synthetic_graph()
        assert synth.state_length() == g.state_length(), (
            f"{toy_name}/scc{scc.index()}: state_length mismatch"
        )


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_source_at_index_0(toy_name, scc_idx):
    """parent_indices[0] == -1 (synthetic source has no parent counterpart)."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    _, meta = scc.as_synthetic_graph()
    assert meta.parent_indices[0] == -1


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_absorbing_at_last_index(toy_name, scc_idx):
    """parent_indices[-1] == -1."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    _, meta = scc.as_synthetic_graph()
    assert meta.parent_indices[-1] == -1


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_internal_synth_indices_have_real_parents(toy_name, scc_idx):
    """Every internal synthetic vertex (uc + io + dc) has a valid
    parent_index pointing to a vertex in the SCC. Per-channel
    absorbing and phantom vertices have parent_index == -1."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    _, meta = scc.as_synthetic_graph()
    internal_pidx = set(scc.internal_vertex_indices())

    n_internal = (meta.n_upstream_connecting + meta.n_internal_only
                  + meta.n_downstream_connecting)
    # Internal vertices live at indices [1, 1 + n_internal).
    for v_synth in range(1, 1 + n_internal):
        pi = meta.parent_indices[v_synth]
        assert pi != -1, (
            f"{toy_name}/scc{scc_idx}: internal vertex {v_synth} "
            f"has -1 parent index"
        )
        assert pi in internal_pidx, (
            f"{toy_name}/scc{scc_idx}: synthetic vertex {v_synth} "
            f"maps to parent index {pi} which is not in this SCC"
        )

    # Per-channel absorbings + phantom should have parent == -1.
    for v_synth in range(1 + n_internal, meta.n_vertices):
        pi = meta.parent_indices[v_synth]
        assert pi == -1, (
            f"{toy_name}/scc{scc_idx}: vertex {v_synth} "
            f"(per-channel absorbing or phantom) has parent_index {pi}, expected -1"
        )


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_graph_self_contained(toy_name, scc_idx):
    """No edge in the synthetic graph points to a vertex outside
    the synthetic graph."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    synth, _ = scc.as_synthetic_graph()

    valid_indices = set(range(synth.vertices_length()))
    for v in synth.vertices():
        for e in v.edges():
            assert e.to().index() in valid_indices, (
                f"{toy_name}/scc{scc_idx}: edge to out-of-range "
                f"index {e.to().index()}"
            )


# ---------------------------------------------------------------------------
# Edge handling
# ---------------------------------------------------------------------------

def _find_vertex_by_synth_idx(synth, idx):
    """Return the synthetic vertex at synthetic-graph index `idx`."""
    for v in synth.vertices():
        if v.index() == idx:
            return v
    raise AssertionError(f"No vertex with synthetic index {idx}")


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_internal_edges_count_matches_parent(toy_name, scc_idx):
    """Edge counts in the synthetic graph:
       Type A:   n_upstream_connecting (synth_source out-edges)
       Type B:   parent's internal-internal edges (verbatim)
       Type C:   n_channels (one per external out-channel)
       Phantom:  n_channels (one per per-channel absorbing)"""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    internal_pidx = set(scc.internal_vertex_indices())

    expected_internal = 0
    for v_p in scc.internal_vertex_indices():
        for e in g.vertex_at(v_p).edges():
            if e.to().index() in internal_pidx:
                expected_internal += 1

    synth, meta = scc.as_synthetic_graph()
    n_uc = meta.n_upstream_connecting
    n_channels = meta.n_channels

    # Type A: synth source's out-edges == n_uc.
    src = _find_vertex_by_synth_idx(synth, 0)
    type_a = sum(1 for _ in src.edges())
    assert type_a == n_uc, (
        f"{toy_name}/scc{scc_idx}: synthetic source has "
        f"{type_a} edges, expected {n_uc} (n_upstream_connecting)"
    )

    # Type C: each per-channel absorbing has exactly one
    # incoming edge from a downstream-connecting vertex.
    # Phantom: each per-channel absorbing has exactly one
    # outgoing edge to phantom.
    phantom_idx = meta.n_vertices - 1
    n_internal = (meta.n_upstream_connecting + meta.n_internal_only
                  + meta.n_downstream_connecting)
    channel_first = 1 + n_internal

    # Count Type C: edges from internal vertices to per-channel absorbings.
    type_c = 0
    for v in synth.vertices():
        if v.index() == 0 or v.index() >= channel_first:
            continue
        for e in v.edges():
            target = e.to().index()
            if channel_first <= target < phantom_idx:
                type_c += 1

    # Count phantom edges: edges from per-channel absorbings to phantom.
    n_phantom = 0
    for v in synth.vertices():
        if not (channel_first <= v.index() < phantom_idx):
            continue
        for e in v.edges():
            if e.to().index() == phantom_idx:
                n_phantom += 1

    assert type_c == n_channels, (
        f"{toy_name}/scc{scc_idx}: Type C count {type_c} != n_channels {n_channels}"
    )
    assert n_phantom == n_channels, (
        f"{toy_name}/scc{scc_idx}: phantom count {n_phantom} != n_channels {n_channels}"
    )

    # Total minus Type A + Type C + phantom edges == expected_internal.
    total = sum(len(list(v.edges())) for v in synth.vertices())
    actual_internal = total - type_a - type_c - n_phantom
    assert actual_internal == expected_internal, (
        f"{toy_name}/scc{scc_idx}: internal edge count mismatch — "
        f"have {actual_internal}, expected {expected_internal}"
    )


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_source_edges_are_placeholder(toy_name, scc_idx):
    """Type A edges (synthetic source → upstream-connecting) all
    have weight 1.0 and a coefficient vector of length
    parent.param_length with first slot = 1.0."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    synth, meta = scc.as_synthetic_graph()

    src = _find_vertex_by_synth_idx(synth, 0)
    expected_len = g.param_length()
    # parameterised_edges() exposes coefficient access; use that path.
    for e in src.parameterized_edges():
        assert e.weight() == 1.0, (
            f"{toy_name}/scc{scc_idx}: synthetic source edge weight {e.weight()} != 1.0"
        )
        coeffs = list(e.edge_state(expected_len))
        assert len(coeffs) == expected_len
        assert coeffs[0] == 1.0
        for c in coeffs[1:]:
            assert c == 0.0


# ---------------------------------------------------------------------------
# Toy-D specific: duplicate-state aux vertex handling
# ---------------------------------------------------------------------------

def test_toy_d_aux_scc_has_distinct_zero_state_vertices():
    """Toy-D's SCC containing the aux vertex X has multiple
    all-zero-state vertices in its synthetic graph:
       - Synthetic source (index 0).
       - The aux vertex X (state [0,0]).
       - One per-channel absorbing per external out-channel.
       - The phantom absorbing.
    All must have distinct synthetic indices (no merging via
    state-based lookup). Specifically X must NOT be merged with
    the synthetic source, even though both have all-zero state."""
    g = build_toy_d()
    scc_graph = g.scc_decomposition()

    s = g.starting_vertex()
    aux_pidx = None
    for v in g.vertices():
        if list(v.state()) == list(s.state()) and v.index() != s.index():
            aux_pidx = v.index()
            break
    assert aux_pidx is not None

    aux_scc_idx = None
    for i in range(scc_graph.n_sccs()):
        if aux_pidx in scc_graph.scc_at(i).internal_vertex_indices():
            aux_scc_idx = i
            break
    assert aux_scc_idx is not None

    scc = scc_graph.scc_at(aux_scc_idx)
    synth, meta = scc.as_synthetic_graph()

    zero_state_indices = []
    for v in synth.vertices():
        if all(x == 0 for x in v.state()):
            zero_state_indices.append(v.index())
    zero_state_indices.sort()

    # Expected: source + aux + n_channels per-channel absorbings + phantom
    expected_min = 2 + meta.n_channels + 1  # source + aux + channels + phantom
    assert len(zero_state_indices) >= expected_min, (
        f"Toy-D aux SCC: expected >= {expected_min} zero-state vertices "
        f"(source + aux X + {meta.n_channels} per-channel absorbings + phantom), "
        f"got {len(zero_state_indices)} at indices {zero_state_indices}"
    )
    # Source at 0, phantom at last.
    assert 0 in zero_state_indices
    assert (meta.n_vertices - 1) in zero_state_indices
    # Aux X must be at an internal index (1..1+n_uc+n_io+n_dc).
    n_internal = (meta.n_upstream_connecting + meta.n_internal_only
                  + meta.n_downstream_connecting)
    aux_synth_indices = [
        i for i in zero_state_indices
        if 1 <= i < 1 + n_internal
    ]
    assert len(aux_synth_indices) >= 1, (
        f"Toy-D aux SCC: aux vertex not found in internal range "
        f"[1, {1 + n_internal}); zero-state indices: {zero_state_indices}"
    )
    assert (meta.n_vertices - 1) in zero_state_indices


def test_toy_d_aux_scc_eliminates():
    """Toy-D's aux-containing SCC must pass through the eliminator
    successfully — this is the core regression for the
    duplicate-state invariant (§6.5 of the reference doc)."""
    g = build_toy_d()
    scc_graph = g.scc_decomposition()
    for scc in scc_graph.sccs_in_topo_order():
        synth, _ = scc.as_synthetic_graph()
        synth.update_weights([1.0, 1.0, 1.0, 1.0])
        ewt = np.asarray(synth.expected_waiting_time())
        assert np.all(np.isfinite(ewt)), (
            f"Toy-D scc{scc.index()} expected_waiting_time non-finite: {ewt}"
        )


# ---------------------------------------------------------------------------
# Smoke test: synthetic graph must be eliminable (the most
# important single test in WP-1).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_synthetic_graph_is_eliminable(toy_name, scc_idx):
    """For every SCC of every toy variant: build the synthetic
    graph, set theta, compute expected_waiting_time, and verify
    finite results.

    This does not check numerical equivalence with the parent
    (WP-5's job) — only that the synthetic graph is well-formed
    input for the existing eliminator."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    synth, _ = scc.as_synthetic_graph()

    # Use a non-trivial theta to exercise more of the
    # parameterisation surface than identity.
    synth.update_weights([0.5, 2.0, 1.0, 0.7])

    ewt = np.asarray(synth.expected_waiting_time())
    assert np.all(np.isfinite(ewt)), (
        f"{toy_name}/scc{scc_idx}: expected_waiting_time non-finite: {ewt}"
    )
    assert len(ewt) == synth.vertices_length()


# ---------------------------------------------------------------------------
# Cross-graph SCC hash invariance — the cross-parent reuse invariant.
# ---------------------------------------------------------------------------

def test_toy_c_shared_scc_synthetic_hash_invariant():
    """Toy-C P and Toy-C P' share an SCC ({C, D}) with identical
    internal structure but different external coefficients. Their
    synthetic graphs must produce identical content hashes — this
    is the cross-graph reuse invariant that makes the SCC cache
    hit across parents."""
    p = build_toy_c_p()
    pp = build_toy_c_pprime()
    p_scc = p.scc_decomposition()
    pp_scc = pp.scc_decomposition()

    # The shared SCC is the one containing parent indices [3, 4]
    # in both P and P' (vertices C, D).
    shared_internal = {3, 4}

    p_synth_hash = None
    pp_synth_hash = None
    for scc in p_scc.sccs_in_topo_order():
        if set(scc.internal_vertex_indices()) == shared_internal:
            synth, _ = scc.as_synthetic_graph()
            p_synth_hash = phasic_hash.compute_graph_hash(synth).hash_hex
            break
    for scc in pp_scc.sccs_in_topo_order():
        if set(scc.internal_vertex_indices()) == shared_internal:
            synth, _ = scc.as_synthetic_graph()
            pp_synth_hash = phasic_hash.compute_graph_hash(synth).hash_hex
            break

    assert p_synth_hash is not None
    assert pp_synth_hash is not None
    assert p_synth_hash == pp_synth_hash, (
        f"Synthetic-graph hashes differ across parents for the "
        f"structurally-shared SCC: {p_synth_hash[:16]} vs {pp_synth_hash[:16]}"
    )


def test_toy_c_unshared_scc_synthetic_hash_differs():
    """Symmetric: the SCC ({A, B}) that has DIFFERENT internal
    structure between P and P' must produce different synthetic
    hashes. Otherwise we would falsely cache-hit and produce
    wrong elimination results."""
    p = build_toy_c_p()
    pp = build_toy_c_pprime()
    p_scc = p.scc_decomposition()
    pp_scc = pp.scc_decomposition()

    unshared_internal = {1, 2}

    p_synth_hash = None
    pp_synth_hash = None
    for scc in p_scc.sccs_in_topo_order():
        if set(scc.internal_vertex_indices()) == unshared_internal:
            synth, _ = scc.as_synthetic_graph()
            p_synth_hash = phasic_hash.compute_graph_hash(synth).hash_hex
            break
    for scc in pp_scc.sccs_in_topo_order():
        if set(scc.internal_vertex_indices()) == unshared_internal:
            synth, _ = scc.as_synthetic_graph()
            pp_synth_hash = phasic_hash.compute_graph_hash(synth).hash_hex
            break

    assert p_synth_hash != pp_synth_hash, (
        f"Synthetic-graph hashes match for SCCs with DIFFERENT "
        f"internal structure: {p_synth_hash[:16]} == {pp_synth_hash[:16]} "
        f"— this would cause cache poisoning"
    )


# ---------------------------------------------------------------------------
# Metadata correctness for external edge references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_external_in_edges_are_real_external(toy_name, scc_idx):
    """Every external_in_edge entry references a parent vertex
    that is OUTSIDE the SCC, with an edge index that points to a
    parent edge whose target IS in the SCC (specifically, the
    target should be the synthetic vertex's parent counterpart)."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    internal_pidx = set(scc.internal_vertex_indices())
    _, meta = scc.as_synthetic_graph()

    for v_synth in range(meta.n_vertices):
        edge_refs = meta.external_in_edges[v_synth]
        if not edge_refs:
            continue
        # v_synth must be an internal vertex (not source/absorbing).
        target_pidx = meta.parent_indices[v_synth]
        assert target_pidx != -1
        assert target_pidx in internal_pidx

        for (parent_v, parent_e) in edge_refs:
            assert parent_v not in internal_pidx, (
                f"{toy_name}/scc{scc_idx}: external_in_edges[{v_synth}] "
                f"references parent vertex {parent_v} which IS internal"
            )
            edge = list(g.vertex_at(parent_v).edges())[parent_e]
            assert edge.to().index() == target_pidx, (
                f"{toy_name}/scc{scc_idx}: external_in_edges[{v_synth}] "
                f"edge ref ({parent_v}, {parent_e}) points to "
                f"{edge.to().index()}, expected {target_pidx}"
            )


@pytest.mark.parametrize("toy_name,scc_idx", SCC_CASES, ids=SCC_CASE_IDS)
def test_external_out_edges_are_real_external(toy_name, scc_idx):
    """Symmetric for external_out_edges."""
    g = BUILDERS[toy_name]()
    scc_graph = g.scc_decomposition()
    scc = scc_graph.scc_at(scc_idx)
    internal_pidx = set(scc.internal_vertex_indices())
    _, meta = scc.as_synthetic_graph()

    for v_synth in range(meta.n_vertices):
        edge_refs = meta.external_out_edges[v_synth]
        if not edge_refs:
            continue
        source_pidx = meta.parent_indices[v_synth]
        assert source_pidx != -1
        assert source_pidx in internal_pidx

        for (parent_v, parent_e) in edge_refs:
            assert parent_v == source_pidx, (
                f"{toy_name}/scc{scc_idx}: external_out_edges[{v_synth}] "
                f"references parent vertex {parent_v}, expected {source_pidx}"
            )
            edge = list(g.vertex_at(parent_v).edges())[parent_e]
            assert edge.to().index() not in internal_pidx, (
                f"{toy_name}/scc{scc_idx}: external_out_edges[{v_synth}] "
                f"target {edge.to().index()} is internal — should be external"
            )
