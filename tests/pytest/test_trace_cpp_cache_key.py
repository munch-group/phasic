"""Regression test for the critical trace-library cache-key collision.

The C++ log-likelihood path keyed its compiled-library cache (/tmp/
trace_log_lik_<hash>.so) on trace *topology* only: n_vertices, the state
matrix, and the vertex_rates *index* array. Two models with identical graph
structure but different parameterized-edge coefficients (or constant edge
weights) hashed identically, so _compile_trace_library's skip-if-exists
returned a STALE .so and every likelihood/gradient for the second model was
silently computed with the first model's coefficients.

The fix hashes the generated C++ source, which embeds the coefficients and
constants. These tests assert that two traces differing only in coefficients
(a) are topologically identical (so the OLD key collided) and (b) now produce
different source and different cache hashes.
"""
import hashlib
import json

import numpy as np

from phasic import Graph, _generate_cpp_from_trace
from phasic.trace_elimination import (
    record_elimination_trace,
    _cpp_trace_source_and_hash,
)

OBS = np.array([0.5, 1.0, 2.0])
GRAN = 100


def _graph(coeffs):
    """3-state parameterized graph; identical topology for any coeffs."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, list(coeffs))
    return g


def _old_topology_only_key(trace):
    """The pre-fix cache key (topology only) — reproduced to show it collides."""
    d = {
        "n_vertices": trace.n_vertices,
        "param_length": trace.param_length,
        "state_length": trace.state_length,
        "is_discrete": trace.is_discrete,
        "n_operations": len(trace.operations),
        "states_hash": hashlib.sha256(trace.states.tobytes()).hexdigest()[:8],
        "vertex_rates_hash": hashlib.sha256(trace.vertex_rates.tobytes()).hexdigest()[:8],
    }
    key = f"{json.dumps(d, sort_keys=True)}_{OBS.tobytes()}_{GRAN}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def test_reparameterized_model_gets_distinct_cache_hash():
    trace_a = record_elimination_trace(_graph([2.0, 0.5]))
    trace_b = record_elimination_trace(_graph([3.0, 0.5]))  # only c0 differs

    # Topology is identical -> the OLD key collided (demonstrates the bug).
    assert trace_a.states.tobytes() == trace_b.states.tobytes()
    assert trace_a.vertex_rates.tobytes() == trace_b.vertex_rates.tobytes()
    assert len(trace_a.operations) == len(trace_b.operations)
    assert _old_topology_only_key(trace_a) == _old_topology_only_key(trace_b)

    # The FIX: source and cache hash now distinguish the two models.
    cpp_a, hash_a = _cpp_trace_source_and_hash(trace_a, OBS, GRAN)
    cpp_b, hash_b = _cpp_trace_source_and_hash(trace_b, OBS, GRAN)
    assert cpp_a != cpp_b, "generated C++ must differ when coefficients differ"
    assert hash_a != hash_b, "cache hash must differ -> no stale .so reuse"


def test_hash_matches_sha256_of_generated_source():
    """The cache key is exactly sha256(cpp_code)[:16] — the whole point."""
    trace = record_elimination_trace(_graph([2.0, 0.5]))
    cpp, h = _cpp_trace_source_and_hash(trace, OBS, GRAN)
    expected = hashlib.sha256(cpp.encode()).hexdigest()[:16]
    assert h == expected
    # And it equals a freshly generated source's hash (deterministic).
    cpp2 = _generate_cpp_from_trace(trace, OBS, GRAN)
    assert hashlib.sha256(cpp2.encode()).hexdigest()[:16] == h


def test_different_observations_change_hash():
    trace = record_elimination_trace(_graph([2.0, 0.5]))
    _, h1 = _cpp_trace_source_and_hash(trace, OBS, GRAN)
    _, h2 = _cpp_trace_source_and_hash(trace, OBS + 0.1, GRAN)
    assert h1 != h2
