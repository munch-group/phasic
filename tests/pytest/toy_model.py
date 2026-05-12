"""Toy reference graphs for the hierar-elimin-cache branch.

Defined in §11 of hierar-elimin-cache-reference.md. These are the
canonical small graphs used to debug, regression-test, and verify
each work package on the SCC-decomposed elimination pipeline.

Five variants:

- ``build_toy_base``  — two SCCs in a chain, each a 2-cycle. Main
  variant exercising upstream/downstream-connecting,
  cross-SCC parameter dependence, and chained composition.
- ``build_toy_a``     — singleton SCCs only (no cycles). Sanity
  check for the trivial-SCC code path.
- ``build_toy_b``     — three SCCs, two of which are parallel
  (siblings in the condensation DAG). Exercises level-set
  parallelism and joint composition of independent SCCs.
- ``build_toy_c_p``   — same as ``build_toy_base``. Pair with
  ``build_toy_c_pprime`` for the cross-graph SCC-reuse test.
- ``build_toy_c_pprime`` — different SCC1 internals + different
  external coefficients into/out of SCC2, but SCC2's internal
  edges have IDENTICAL coefficients to ``build_toy_c_p``. The
  invariant: SCC2's synthetic-graph content hash must agree
  between P and P'. Without that, cross-graph SCC reuse never
  hits.
- ``build_toy_d``     — toy-base + one auxiliary vertex inside
  SCC₂ with all-zero state (colliding with the starting vertex's
  state). The hard regression case for the duplicate-state
  invariant (§6.5 of the reference doc): every WP that traverses
  graph structure or computes hashes must produce correct
  results here. WPs that use ``find_or_create_vertex(state)``
  to identify existing vertices (instead of by pointer/index)
  will fail on this variant.

State vectors are chosen distinct across all vertices so canonical
state-vector ordering (§6.2) suffices; no automorphism tie-breaking
is needed for these toys.

Coefficient layout (parameterised, 4-dim θ):

    θ[0] — IPV scaling (starting-vertex edges)
    θ[1] — internal SCC dynamics (within-SCC edges)
    θ[2] — inter-SCC dynamics (SCC1 → SCC2 / parallel)
    θ[3] — absorption (edges to Ω)

Each parameterised edge has a 4-element coefficient vector with
exactly one non-zero entry, so weight = c_k · θ_k for the active
slot.

Vertex naming convention:

- Starting vertex ``s`` (state ``[0, 0]``) — auto-created by phasic.
- ``A``, ``B``    — SCC₁ internals (states ``[1, 0]``, ``[1, 1]``).
- ``C``, ``D``    — SCC₂ internals (states ``[2, 0]``, ``[2, 1]``).
- ``E``, ``F``    — SCC₃ internals (Toy-B only; states ``[4, 0]``, ``[4, 1]``).
- ``Ω``           — absorbing vertex (state ``[3, 0]``).

The starting vertex's default state is ``[0, 0]``; we never call
``find_or_create_vertex([0, 0])``, which would create a duplicate
state collision.
"""

from __future__ import annotations

from phasic import Graph


# ----------------------------------------------------------------------
# Toy-base — the canonical reference: SCC₁ → SCC₂, each a 2-cycle.
# ----------------------------------------------------------------------

def build_toy_base() -> Graph:
    """Two SCCs in a chain, each a 2-cycle.

    Edges:
        s ──[θ₀]──> A
        s ──[θ₀]──> C        (IPV)
        A ──[2·θ₁]──> B      ┐
        B ──[1·θ₁]──> A      ┘ SCC₁ internal
        A ──[0.5·θ₂]──> C    ┐
        B ──[0.3·θ₂]──> D    ┘ SCC₁ → SCC₂
        C ──[1.5·θ₁]──> D    ┐
        D ──[0.7·θ₁]──> C    ┘ SCC₂ internal
        C ──[θ₃]──> Ω        ┐
        D ──[θ₃]──> Ω        ┘ absorption
    """
    g = Graph(2)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1, 0])
    B = g.find_or_create_vertex([1, 1])
    C = g.find_or_create_vertex([2, 0])
    D = g.find_or_create_vertex([2, 1])
    omega = g.find_or_create_vertex([3, 0])

    # IPV
    s.add_edge(A, [1.0, 0.0, 0.0, 0.0])
    s.add_edge(C, [1.0, 0.0, 0.0, 0.0])

    # SCC₁ internal cycle
    A.add_edge(B, [0.0, 2.0, 0.0, 0.0])
    B.add_edge(A, [0.0, 1.0, 0.0, 0.0])

    # SCC₁ → SCC₂
    A.add_edge(C, [0.0, 0.0, 0.5, 0.0])
    B.add_edge(D, [0.0, 0.0, 0.3, 0.0])

    # SCC₂ internal cycle
    C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
    D.add_edge(C, [0.0, 0.7, 0.0, 0.0])

    # Absorption
    C.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    D.add_edge(omega, [0.0, 0.0, 0.0, 1.0])

    return g


# ----------------------------------------------------------------------
# Toy-A — singleton SCCs (no cycles).
# ----------------------------------------------------------------------

def build_toy_a() -> Graph:
    """Singleton-only DAG: s → A → C → Ω.

    Every SCC has exactly one vertex. Tests that the trivial-SCC
    code path produces equivalent results to the cyclic toys.
    """
    g = Graph(2)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1, 0])
    C = g.find_or_create_vertex([2, 0])
    omega = g.find_or_create_vertex([3, 0])

    s.add_edge(A, [1.0, 0.0, 0.0, 0.0])
    A.add_edge(C, [0.0, 0.0, 0.5, 0.0])
    C.add_edge(omega, [0.0, 0.0, 0.0, 1.0])

    return g


# ----------------------------------------------------------------------
# Toy-B — three SCCs, two of which are parallel siblings.
# ----------------------------------------------------------------------

def build_toy_b() -> Graph:
    """Three SCCs: SCC₁ → SCC₂, plus SCC₃ parallel to SCC₂.

    SCC₂ ({C, D}) and SCC₃ ({E, F}) are siblings in the
    condensation DAG. SCC₁ feeds SCC₂; ``s`` directly feeds SCC₃.
    Both SCC₂ and SCC₃ absorb into Ω.

    Edges (in addition to toy-base):
        s ──[θ₀]──> E         (IPV into SCC₃)
        E ──[1.5·θ₁]──> F    ┐
        F ──[0.7·θ₁]──> E    ┘ SCC₃ internal cycle
        E ──[θ₃]──> Ω        ┐
        F ──[θ₃]──> Ω        ┘ SCC₃ absorption
    """
    g = Graph(2)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1, 0])
    B = g.find_or_create_vertex([1, 1])
    C = g.find_or_create_vertex([2, 0])
    D = g.find_or_create_vertex([2, 1])
    E = g.find_or_create_vertex([4, 0])
    F = g.find_or_create_vertex([4, 1])
    omega = g.find_or_create_vertex([3, 0])

    # IPV
    s.add_edge(A, [1.0, 0.0, 0.0, 0.0])
    s.add_edge(C, [1.0, 0.0, 0.0, 0.0])
    s.add_edge(E, [1.0, 0.0, 0.0, 0.0])

    # SCC₁ internal cycle
    A.add_edge(B, [0.0, 2.0, 0.0, 0.0])
    B.add_edge(A, [0.0, 1.0, 0.0, 0.0])

    # SCC₁ → SCC₂
    A.add_edge(C, [0.0, 0.0, 0.5, 0.0])
    B.add_edge(D, [0.0, 0.0, 0.3, 0.0])

    # SCC₂ internal cycle
    C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
    D.add_edge(C, [0.0, 0.7, 0.0, 0.0])

    # SCC₂ absorption
    C.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    D.add_edge(omega, [0.0, 0.0, 0.0, 1.0])

    # SCC₃ internal cycle (parallel to SCC₂)
    E.add_edge(F, [0.0, 1.5, 0.0, 0.0])
    F.add_edge(E, [0.0, 0.7, 0.0, 0.0])

    # SCC₃ absorption
    E.add_edge(omega, [0.0, 0.0, 0.0, 1.0])
    F.add_edge(omega, [0.0, 0.0, 0.0, 1.0])

    return g


# ----------------------------------------------------------------------
# Toy-C — the cross-graph SCC-reuse pair.
# ----------------------------------------------------------------------

def build_toy_c_p() -> Graph:
    """Toy-C parent ``P`` — identical to ``build_toy_base``.

    Pair with ``build_toy_c_pprime`` to test cross-graph SCC reuse:
    SCC₂ in P and SCC₂ in P' must produce the same synthetic-graph
    content hash even though everything around SCC₂ differs.
    """
    return build_toy_base()


def build_toy_c_pprime() -> Graph:
    """Toy-C parent ``P'`` — same SCC₂ internals as P, everything else differs.

    Differences from ``build_toy_c_p``:

    - IPV coefficients: ``s → A`` is ``2·θ₀`` (not ``1·θ₀``);
      ``s → C`` is ``3·θ₀`` (not ``1·θ₀``).
    - SCC₁ internal coefficients: ``A → B`` is ``4·θ₁``,
      ``B → A`` is ``5·θ₁`` (was 2 and 1 respectively).
    - SCC₁ → SCC₂ coefficients: ``A → C`` is ``0.9·θ₂``,
      ``B → D`` is ``1.1·θ₂`` (was 0.5 and 0.3).
    - Absorption coefficients: ``C → Ω`` is ``2·θ₃``,
      ``D → Ω`` is ``2.5·θ₃`` (was 1 and 1).

    UNCHANGED (this is the invariant):

    - SCC₂ internal cycle: ``C → D`` is ``1.5·θ₁``,
      ``D → C`` is ``0.7·θ₁`` (same as P).
    - SCC₂ vertex states (``[2, 0]`` and ``[2, 1]``).

    Because the SCC₂ subgraph has identical internal structure,
    both parents must hash SCC₂ the same way (§6.2 / WP-2). That
    is the cross-graph-reuse correctness invariant.
    """
    g = Graph(2)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1, 0])
    B = g.find_or_create_vertex([1, 1])
    C = g.find_or_create_vertex([2, 0])
    D = g.find_or_create_vertex([2, 1])
    omega = g.find_or_create_vertex([3, 0])

    # Different IPV
    s.add_edge(A, [2.0, 0.0, 0.0, 0.0])
    s.add_edge(C, [3.0, 0.0, 0.0, 0.0])

    # Different SCC₁ internals
    A.add_edge(B, [0.0, 4.0, 0.0, 0.0])
    B.add_edge(A, [0.0, 5.0, 0.0, 0.0])

    # Different SCC₁ → SCC₂ weights
    A.add_edge(C, [0.0, 0.0, 0.9, 0.0])
    B.add_edge(D, [0.0, 0.0, 1.1, 0.0])

    # SCC₂ internal — IDENTICAL to P (this is the invariant)
    C.add_edge(D, [0.0, 1.5, 0.0, 0.0])
    D.add_edge(C, [0.0, 0.7, 0.0, 0.0])

    # Different absorption weights
    C.add_edge(omega, [0.0, 0.0, 0.0, 2.0])
    D.add_edge(omega, [0.0, 0.0, 0.0, 2.5])

    return g


# ----------------------------------------------------------------------
# Toy-D — aux vertex with duplicate all-zero state.
# ----------------------------------------------------------------------

def build_toy_d() -> Graph:
    """Toy-base + one auxiliary vertex inside SCC₂ with all-zero state.

    The aux vertex ``X`` is added via ``add_aux_vertex()`` on ``C``,
    so ``X`` has state ``[0, 0]`` — the same as the starting vertex.
    ``add_aux_vertex`` automatically creates two edges:

    - ``X → C`` with constant weight 1.0 (no coefficients,
      ``coefficients_length = 0``).
    - ``C → X`` with the supplied coefficient vector.

    This brings ``X`` into SCC₂'s strongly-connected component
    (the cycle ``C → X → C`` keeps ``X`` strongly connected to ``C``,
    and the existing ``C ↔ D`` cycle keeps ``D`` strongly connected
    to ``C``, so SCC₂ becomes ``{X, C, D}``).

    The aux vertex shares the all-zero state with the starting
    vertex. Any code that identifies vertices by state (via
    ``find_or_create_vertex(state)`` or equivalent) will malfunction
    on this graph: it will either return ``s`` when ``X`` was
    intended, or merge ``X`` and ``s`` into a single vertex during
    synthetic-graph construction.

    The constant aux→parent edge has ``coefficients_length = 0``;
    code that iterates ``edge->coefficients[k]`` blindly must
    handle this.

    Use Toy-D wherever a WP is verified — the duplicate-state
    invariant is load-bearing and broken implementations may
    silently pass other variants.
    """
    g = build_toy_base()

    # Find C (state [2, 0]) — the only vertex with that state.
    # We use index lookup, not state lookup, because we already know
    # the construction order.
    # In build_toy_base: indices are 0=s, 1=A, 2=B, 3=C, 4=D, 5=Ω.
    C = None
    for v in g.vertices():
        if list(v.state()) == [2, 0]:
            C = v
            break
    assert C is not None, "C not found in toy-base graph"

    # Add aux vertex X on C with rate coeffs 0.5·θ[1] (parameterised
    # graph requires array-valued rate).
    X = C.add_aux_vertex([0.0, 0.5, 0.0, 0.0])

    # Sanity check: X has all-zero state and is distinct from s.
    assert list(X.state()) == [0, 0]
    assert X.index() != g.starting_vertex().index()

    return g


# ----------------------------------------------------------------------
# Registry — name → builder.
# ----------------------------------------------------------------------

BUILDERS = {
    'toy_base':     build_toy_base,
    'toy_a':        build_toy_a,
    'toy_b':        build_toy_b,
    'toy_c_p':      build_toy_c_p,
    'toy_c_pprime': build_toy_c_pprime,
    'toy_d':        build_toy_d,
}


# Reference θ vectors used by the regression test.
THETAS = {
    'unit':  [1.0, 1.0, 1.0, 1.0],
    'mixed': [0.5, 2.0, 1.0, 0.7],
}
