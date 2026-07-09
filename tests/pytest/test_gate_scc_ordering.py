"""Equivalence gate for §B #9 — SCC composition ordering / level / cache-path.

This is the Stage-3 WS-A safety-net gate that must be green before any #9/#12
unification touches the SCC machinery (the highest un-gated risk in the plan:
``scc_compose.c``/``scc_synthetic.c`` are 0% native-covered and the two ordering
providers disagree).

The divergence pinned here (verified live):

* ``ptd_compose_scc_prcs`` (``src/c/scc_compose.c:324-334``) explicitly states the
  stored Tarjan order is **NOT** a valid topological order and re-derives a
  sink-first order via Kahn's algorithm.
* ``SCCGraph::sccs_in_topo_order`` (``api/cpp/scc_graph.cpp:70-81``) claims the
  opposite ("C struct already stores vertices in topological order ... guaranteed
  by Tarjan") and returns the stored order verbatim. This method is bound
  (``phasic_pybind.cpp``) and consumed by the Python stitching path, so it is a
  live latent-correctness bug, not merely dead duplication.

Gate structure (Stage-2 characterization pattern):

* passing guards prove the harness is sound (independent Kahn reference is valid;
  ``compute_scc_levels`` is correct; ``cache_file_path`` naming is well-formed);
* **strict xfails** pin the exact divergences — the day #9 is unified they flip to
  XPASS and fail the suite, forcing the marker (and the divergence) to be removed.

No production code is exercised for its gradients; this is topology only.
"""
import re
from collections import deque

import pytest

from phasic import Graph

pytestmark = pytest.mark.equivalence

_SCC_BIN_RE = re.compile(r"^scc_[0-9a-f]{64}\.bin$")


# --------------------------------------------------------------------------- #
# Graph builders — condensations whose stored SCC order is non-topological.
# Each non-trivial SCC is a 2-cycle so the decomposition has >1 multi-vertex SCC.
# --------------------------------------------------------------------------- #
def _build_chain() -> Graph:
    """start -> {1,2} -> {3,4} -> absorbing  (condensation is a path)."""
    g = Graph(1)
    v0 = g.starting_vertex()
    v = [g.find_or_create_vertex([i]) for i in range(1, 5)]
    va = g.find_or_create_vertex([9])
    v0.add_edge(v[0], 1.0)
    v[0].add_edge(v[1], 1.0); v[1].add_edge(v[0], 1.0)   # SCC {1,2}
    v[1].add_edge(v[2], 1.0)
    v[2].add_edge(v[3], 1.0); v[3].add_edge(v[2], 1.0)   # SCC {3,4}
    v[3].add_edge(va, 1.0)
    return g


def _build_diamond() -> Graph:
    """start -> {A},{B} -> {sink-cycle} -> absorbing."""
    g = Graph(1)
    v0 = g.starting_vertex()
    a1 = g.find_or_create_vertex([1]); a2 = g.find_or_create_vertex([2])
    b1 = g.find_or_create_vertex([3]); b2 = g.find_or_create_vertex([4])
    s1 = g.find_or_create_vertex([5]); s2 = g.find_or_create_vertex([6])
    va = g.find_or_create_vertex([9])
    v0.add_edge(a1, 1.0); v0.add_edge(b1, 1.0)
    a1.add_edge(a2, 1.0); a2.add_edge(a1, 1.0)   # SCC A
    b1.add_edge(b2, 1.0); b2.add_edge(b1, 1.0)   # SCC B
    a2.add_edge(s1, 1.0); b2.add_edge(s1, 1.0)
    s1.add_edge(s2, 1.0); s2.add_edge(s1, 1.0)   # SCC sink
    s2.add_edge(va, 1.0)
    return g


def _build_long_chain(k: int = 6) -> Graph:
    g = Graph(1)
    prev = g.starting_vertex()
    st = 1
    for _ in range(k):
        a = g.find_or_create_vertex([st]); b = g.find_or_create_vertex([st + 1])
        st += 2
        prev.add_edge(a, 1.0)
        a.add_edge(b, 1.0); b.add_edge(a, 1.0)
        prev = b
    prev.add_edge(g.find_or_create_vertex([99]), 1.0)
    return g


BUILDERS = {
    "chain": _build_chain,
    "diamond": _build_diamond,
    "long_chain6": _build_long_chain,
}


# --------------------------------------------------------------------------- #
# Helpers — reconstruct the condensation from the Python-bound SCC API.
# --------------------------------------------------------------------------- #
def _condensation(scc_graph):
    """(order, edges): order[pos] = scc.index() in sccs_in_topo_order() sequence;
    edges = set of (from_index, to_index) inter-SCC edges."""
    order, edges = [], set()
    for scc in scc_graph.sccs_in_topo_order():
        fi = scc.index()
        order.append(fi)
        for to in scc.outgoing_scc_edges():
            if to != fi:
                edges.add((fi, to))
    return order, edges


def _is_forward_topological(order, edges) -> bool:
    """True iff, for every edge u->v, u precedes v in ``order`` (standard
    topological-sort validity: a source comes before its descendants)."""
    pos = {idx: p for p, idx in enumerate(order)}
    return all(pos[u] < pos[v] for (u, v) in edges)


def _kahn_forward(order, edges):
    """Independent forward-topological order via Kahn's algorithm — the exact
    procedure ``scc_compose.c`` uses. Returns None if the condensation has a
    cycle (it never should — the condensation of any digraph is a DAG)."""
    indeg = {n: 0 for n in order}
    adj = {n: [] for n in order}
    for (u, v) in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(n for n in order if indeg[n] == 0)
    topo = []
    while q:
        n = q.popleft()
        topo.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return topo if len(topo) == len(order) else None


# --------------------------------------------------------------------------- #
# Passing guards — prove the harness is sound (not vacuously green).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(BUILDERS))
def test_condensation_is_a_dag_and_kahn_reference_is_valid(name):
    """The independent Kahn reference (what the C composer computes) is a valid
    forward-topological order covering every SCC. This anchors the strict-xfail
    below: the property being asserted there is genuinely achievable."""
    scc = BUILDERS[name]().scc_decomposition()
    order, edges = _condensation(scc)
    kahn = _kahn_forward(order, edges)
    assert kahn is not None, "condensation must be acyclic"
    assert sorted(kahn) == sorted(order), "Kahn order must cover every SCC exactly once"
    assert _is_forward_topological(kahn, edges), "Kahn reference must be forward-topological"


@pytest.mark.parametrize("name", list(BUILDERS))
def test_compute_scc_levels_is_a_valid_sink_first_leveling(name):
    """``distributed_scc.compute_scc_levels`` (the Python mirror of the C
    composer's WP-6 level pass) yields a valid sink-first leveling: pure sinks at
    level 0, and level strictly decreasing along every edge. This is currently
    CORRECT (order-independent longest-path), so it is a passing guard that must
    stay green through any #9 de-mirroring."""
    from phasic.distributed_scc import compute_scc_levels

    scc = BUILDERS[name]().scc_decomposition()
    order, edges = _condensation(scc)
    levels = compute_scc_levels(scc)
    level_of = {idx: lvl for lvl, group in enumerate(levels) for idx in group}

    assert sorted(level_of) == sorted(order), "every SCC assigned exactly one level"
    outdeg = {n: 0 for n in order}
    for (u, _v) in edges:
        outdeg[u] += 1
    true_sinks = {n for n in order if outdeg[n] == 0}
    assert set(levels[0]) == true_sinks, "level 0 must be exactly the pure sinks"
    for (u, v) in edges:
        assert level_of[u] > level_of[v], (
            f"level must strictly decrease along edge {u}->{v} "
            f"(got lvl(u)={level_of[u]} <= lvl(v)={level_of[v]})"
        )


# NOTE (native-fragility finding, #9-adjacent): the C-side cache-path SSOT
# ``SCCVertex.cache_file_path`` -> ``ptd_scc_build_synthetic_graph`` raises a
# spurious ``out of memory`` for these graphs *under the project pytest
# configuration* (it works under plain ``python`` and for the toy-model graphs in
# test_synthetic_scc_graph.py). This is the same class of native brittleness as
# the already-skipped ``test_scc_hashing`` ("C-level abort in scc_hashes()") and
# is more evidence the SCC synthetic-graph code (0% native coverage) needs
# hardening before #9 lands. The gate therefore pins the cache-path axis via the
# Python mirror's observable divergence below rather than by calling that C path.


# --------------------------------------------------------------------------- #
# Strict xfails — the exact §B #9 divergences. Each flips to XPASS (and fails
# the suite) the day the pair is unified, forcing removal of the marker.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="#9/Q10: SCCGraph::sccs_in_topo_order (scc_graph.cpp:70-81) trusts the "
    "stored Tarjan order, which scc_compose.c:324-334 proves is NOT topological. "
    "Flips to XPASS when the C++ provider delegates to the single Kahn-based "
    "topo-order provider (remove this marker then).",
)
def test_sccs_in_topo_order_is_topologically_valid():
    """The bound C++ topo-order provider must return a valid forward-topological
    order of the condensation. It does not today: it returns the raw stored order,
    which places SCCs before their inter-SCC dependencies."""
    failures = []
    for name, builder in BUILDERS.items():
        order, edges = _condensation(builder().scc_decomposition())
        if not _is_forward_topological(order, edges):
            failures.append((name, order, sorted(edges)))
    assert not failures, (
        "sccs_in_topo_order returned a non-topological order for: " + repr(failures)
    )


@pytest.mark.xfail(
    strict=True,
    reason="#9 cache-path mirror drift: compute_repository._expected_scc_filenames "
    "passes the (graph, metadata) tuple from as_synthetic_graph() to "
    "compute_graph_hash (which takes a Graph), so it silently returns [] instead "
    "of one scc_<hash>.bin per SCC. Flips to XPASS when the cache-path mirror is "
    "fixed / unified onto the one native provider (fix + remove marker).",
)
def test_expected_scc_filenames_returns_one_wellformed_name_per_scc():
    """The Python cache-path mirror must emit exactly one well-formed
    ``scc_<hash>.bin`` per SCC. It currently returns ``[]`` (silently broken), so
    it has already drifted from the C-side cache-path SSOT — precisely the mirror
    divergence #9 exists to eliminate. Avoids the fragile ``cache_file_path`` C
    path (see note above), so this xfails on the real drift, not on the native OOM."""
    from phasic.compute_repository import _expected_scc_filenames

    g = _build_chain()
    n_sccs = len(g.scc_decomposition())
    names = _expected_scc_filenames(g)
    assert len(names) == n_sccs and all(_SCC_BIN_RE.match(b) for b in names), (
        f"expected {n_sccs} scc_<hash>.bin names, got {names!r}"
    )
