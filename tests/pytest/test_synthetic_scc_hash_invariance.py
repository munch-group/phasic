"""WP-2 tests: canonical hashing invariants for synthetic SCC graphs.

The cross-graph SCC reuse (WP-4) hits the disk cache only if two
parents that share an SCC structurally produce *the same content
hash* for that SCC's synthetic graph. WP-1 already implements a
canonical vertex ordering (lexicographic state vector + out-edge
signature tiebreak + parent-index fallback). These tests lock down
the resulting invariants:

- Determinism: rebuilding the same parent gives the same hash.
- Vertex creation order: building parents that create vertices in
  different orders still gives the same hash for structurally-
  identical SCCs.
- Edge insertion order: same logical SCC built with edges added in
  different orders produces the same hash.
- Cross-parent invariance: two parents with the same SCC structure
  but different external coefficients hash that SCC the same way.
- Cross-parent inverse: two parents with structurally-different
  SCCs hash them differently (no false cache hits).

See §6.2, §6.5 of hierar-elimin-cache-reference.md for the
canonical-ordering design and wp1-experiments.md §1 for the
empirical baseline.
"""

from __future__ import annotations

import pytest

from phasic import Graph
from phasic import hash as phasic_hash
from toy_model import (
    BUILDERS,
    build_toy_base,
    build_toy_c_p,
    build_toy_c_pprime,
    build_toy_d,
)


def _hash_synthetic(scc) -> str:
    synth, _ = scc.as_synthetic_graph()
    return phasic_hash.compute_graph_hash(synth).hash_hex


def _scc_signature(graph, internal_indices):
    """Return a parent-independent signature for an SCC: the
    sorted tuple of state vectors of its internal vertices.

    Used to identify "the same logical SCC" across parents that
    may number vertices differently."""
    return tuple(sorted(
        tuple(graph.vertex_at(i).state())
        for i in internal_indices
    ))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_synthetic_hash_is_deterministic_on_rebuild(toy_name):
    """Building the same toy twice produces the same synthetic-graph
    hash for every SCC. Guards against accidental
    nondeterminism (e.g. memory-address-dependent ordering)."""
    builder = BUILDERS[toy_name]

    g1 = builder()
    g2 = builder()
    scc1 = g1.scc_decomposition()
    scc2 = g2.scc_decomposition()

    assert scc1.n_sccs() == scc2.n_sccs()
    for s1, s2 in zip(scc1.sccs_in_topo_order(), scc2.sccs_in_topo_order()):
        h1 = _hash_synthetic(s1)
        h2 = _hash_synthetic(s2)
        assert h1 == h2, (
            f"{toy_name}: rebuild changed hash for SCC "
            f"{s1.internal_vertex_indices()}: {h1[:16]} vs {h2[:16]}"
        )


# ---------------------------------------------------------------------------
# Invariance under parent vertex creation order
# ---------------------------------------------------------------------------

def _build_toy_base_with_vertex_order(order: str) -> Graph:
    """Build toy-base where {A, B, C, D} are created in the
    requested order (e.g. 'ABCD', 'CDAB', 'DCBA')."""
    name_to_state = {
        'A': [1, 0], 'B': [1, 1],
        'C': [2, 0], 'D': [2, 1],
    }
    g = Graph(2)
    s = g.starting_vertex()
    verts = {name: g.find_or_create_vertex(name_to_state[name]) for name in order}
    omega = g.find_or_create_vertex([3, 0])
    A, B, C, D = verts['A'], verts['B'], verts['C'], verts['D']

    s.add_edge(A, [1.0, 0.0, 0.0, 0.0])
    s.add_edge(C, [1.0, 0.0, 0.0, 0.0])
    A.add_edge(B, [0.0, 2.0, 0.0, 0.0])
    B.add_edge(A, [0.0, 1.0, 0.0, 0.0])
    A.add_edge(C, [0.0, 0.0, 0.5, 0.0])
    B.add_edge(D, [0.0, 0.0, 0.3, 0.0])
    C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
    D.add_edge(C, [0.0, 0.7, 0.0, 0.0])
    C.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    D.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    return g


@pytest.mark.parametrize("order", ['ABCD', 'ACBD', 'CDAB', 'DCBA', 'BACD', 'BDCA'])
def test_synthetic_hash_invariant_under_vertex_creation_order(order):
    """The synthetic-graph hash for each SCC must agree across
    parents that create their internal vertices in different orders.

    This is the core invariant for cross-parent SCC reuse. If a
    parent that creates A,B,C,D in 'CDAB' order produces a different
    SCC₂ hash than one that creates them 'ABCD', the cache never
    hits even though the SCCs are structurally identical.
    """
    reference = _build_toy_base_with_vertex_order('ABCD')
    variant = _build_toy_base_with_vertex_order(order)

    ref_scc = reference.scc_decomposition()
    var_scc = variant.scc_decomposition()

    # Group SCCs by their parent-independent signature.
    ref_by_sig = {}
    for scc in ref_scc.sccs_in_topo_order():
        sig = _scc_signature(reference, scc.internal_vertex_indices())
        ref_by_sig[sig] = _hash_synthetic(scc)

    var_by_sig = {}
    for scc in var_scc.sccs_in_topo_order():
        sig = _scc_signature(variant, scc.internal_vertex_indices())
        var_by_sig[sig] = _hash_synthetic(scc)

    assert set(ref_by_sig) == set(var_by_sig), (
        f"SCC signatures differ between orders: ABCD has "
        f"{set(ref_by_sig)}, {order} has {set(var_by_sig)}"
    )

    for sig in ref_by_sig:
        assert ref_by_sig[sig] == var_by_sig[sig], (
            f"Order {order}: SCC with signature {sig} hashes "
            f"{var_by_sig[sig][:16]}, but ABCD hashes "
            f"{ref_by_sig[sig][:16]} — vertex-order invariance broken"
        )


# ---------------------------------------------------------------------------
# Invariance under parent edge insertion order
# ---------------------------------------------------------------------------

def _build_toy_base_with_internal_edge_order(c_to_d_first: bool) -> Graph:
    """Build toy-base toggling the order of C→D vs D→C insertion."""
    g = Graph(2)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1, 0])
    B = g.find_or_create_vertex([1, 1])
    C = g.find_or_create_vertex([2, 0])
    D = g.find_or_create_vertex([2, 1])
    omega = g.find_or_create_vertex([3, 0])
    s.add_edge(A, [1.0, 0.0, 0.0, 0.0])
    s.add_edge(C, [1.0, 0.0, 0.0, 0.0])
    A.add_edge(B, [0.0, 2.0, 0.0, 0.0])
    B.add_edge(A, [0.0, 1.0, 0.0, 0.0])
    A.add_edge(C, [0.0, 0.0, 0.5, 0.0])
    B.add_edge(D, [0.0, 0.0, 0.3, 0.0])
    if c_to_d_first:
        C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
        D.add_edge(C, [0.0, 0.7, 0.0, 0.0])
    else:
        D.add_edge(C, [0.0, 0.7, 0.0, 0.0])
        C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
    C.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    D.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    return g


def test_synthetic_hash_invariant_under_edge_insertion_order():
    """Same SCC structure built with internal cycle edges added in
    different orders produces the same hash. The graph content
    hasher already sorts edges canonically; this test locks that
    in for synthetic-graph hashing too."""
    g1 = _build_toy_base_with_internal_edge_order(c_to_d_first=True)
    g2 = _build_toy_base_with_internal_edge_order(c_to_d_first=False)
    scc1 = g1.scc_decomposition()
    scc2 = g2.scc_decomposition()
    for s1, s2 in zip(scc1.sccs_in_topo_order(), scc2.sccs_in_topo_order()):
        assert _hash_synthetic(s1) == _hash_synthetic(s2), (
            f"SCC {s1.internal_vertex_indices()}: edge-order variant "
            "produced different hash"
        )


# ---------------------------------------------------------------------------
# Cross-parent invariance and its inverse
# ---------------------------------------------------------------------------

def test_shared_scc_hashes_match_across_toy_c_pair():
    """The Toy-C pair shares SCC₂ ({C, D}) structurally. Their
    synthetic-graph hashes must agree."""
    p = build_toy_c_p()
    pp = build_toy_c_pprime()
    p_scc = p.scc_decomposition()
    pp_scc = pp.scc_decomposition()

    target_states = {(2, 0), (2, 1)}

    p_hash = None
    pp_hash = None
    for scc in p_scc.sccs_in_topo_order():
        states = {tuple(int(x) for x in p.vertex_at(i).state())
                  for i in scc.internal_vertex_indices()}
        if states == target_states:
            p_hash = _hash_synthetic(scc)
    for scc in pp_scc.sccs_in_topo_order():
        states = {tuple(int(x) for x in pp.vertex_at(i).state())
                  for i in scc.internal_vertex_indices()}
        if states == target_states:
            pp_hash = _hash_synthetic(scc)

    assert p_hash is not None
    assert pp_hash is not None
    assert p_hash == pp_hash


def test_unshared_scc_hashes_differ_across_toy_c_pair():
    """SCC₁ ({A, B}) has different internal coefficients between
    Toy-C P and P'. Their hashes must NOT match — a false match
    would cache-poison."""
    p = build_toy_c_p()
    pp = build_toy_c_pprime()
    p_scc = p.scc_decomposition()
    pp_scc = pp.scc_decomposition()

    target_states = {(1, 0), (1, 1)}

    p_hash = None
    pp_hash = None
    for scc in p_scc.sccs_in_topo_order():
        states = {tuple(int(x) for x in p.vertex_at(i).state())
                  for i in scc.internal_vertex_indices()}
        if states == target_states:
            p_hash = _hash_synthetic(scc)
    for scc in pp_scc.sccs_in_topo_order():
        states = {tuple(int(x) for x in pp.vertex_at(i).state())
                  for i in scc.internal_vertex_indices()}
        if states == target_states:
            pp_hash = _hash_synthetic(scc)

    assert p_hash != pp_hash


# ---------------------------------------------------------------------------
# Toy-D specific: aux SCC hashes deterministically
# ---------------------------------------------------------------------------

def test_toy_d_aux_scc_hashes_deterministic():
    """Toy-D's SCC containing the aux vertex (duplicate-state case)
    must produce a stable hash on rebuild — within tier 2 of the
    canonical comparator (out-edge signature). If the aux vertex
    in Toy-D ever needed tier 3 (parent-index fallback), this
    test would expose that nondeterminism."""
    g1 = build_toy_d()
    g2 = build_toy_d()

    def aux_scc(g):
        scc_decomp = g.scc_decomposition()
        for scc in scc_decomp.sccs_in_topo_order():
            if scc.size() == 3:
                return scc
        raise AssertionError("Toy-D should have one size-3 SCC")

    h1 = _hash_synthetic(aux_scc(g1))
    h2 = _hash_synthetic(aux_scc(g2))
    assert h1 == h2


# ---------------------------------------------------------------------------
# Toy-base SCC₂ hashes the same as Toy-B SCC₂ (both contain {C, D}
# with identical internal coefficients; their parents differ in the
# parallel SCC E,F that toy-B adds).
# ---------------------------------------------------------------------------

def test_toy_b_shares_scc2_hash_with_toy_base():
    """Toy-B is toy-base + a parallel SCC₃ ({E,F}). SCC₂ ({C,D})
    is structurally identical in both. Synthetic graph hashes
    must agree, so the SCC cache hits when going from one toy to
    the other."""
    from toy_model import build_toy_b
    base = build_toy_base()
    bee = build_toy_b()

    target_states = {(2, 0), (2, 1)}

    def find_scc_by_states(g, scc_decomp, target):
        for scc in scc_decomp.sccs_in_topo_order():
            states = {tuple(int(x) for x in g.vertex_at(i).state())
                      for i in scc.internal_vertex_indices()}
            if states == target:
                return scc
        return None

    base_scc = find_scc_by_states(base, base.scc_decomposition(), target_states)
    bee_scc = find_scc_by_states(bee, bee.scc_decomposition(), target_states)

    assert base_scc is not None
    assert bee_scc is not None
    assert _hash_synthetic(base_scc) == _hash_synthetic(bee_scc)
