"""GATE G2 — Moment recurrence E[T^k], 3-way (Stage-3 duplication #5).

Three implementations of the E[T^k] recurrence exist:
  (a) pybind free function ``_moments`` -> ``Graph.moments()``      (phasic_pybind.cpp:212)
  (b) ``phasic::Graph::moments`` "faithful ports"                   (phasiccpp.h:510)
      -> NATIVE-ONLY / unbound.  NOT covered here; must be covered by a
      tests/cpp CTest (see the skip marker below).
  (c) ``GraphBuilder::compute_moments_impl`` -> ``GraphBuilder.compute_moments``
      (graph_builder.cpp:479); the same core also drives the FFI target
      ``ptd_compute_moments``.

This gate pins the two Python-reachable implementations (a) and (c) to
bit-identical output, and transitively pins the FFI numeric core, before
Stage-3 unifies them.

Known divergence: ``Graph.moments(discrete=True)`` calls ``super().moments_discrete``
which has no pybind binding -> AttributeError -> xfail(strict) referencing Q5.
"""
from __future__ import annotations

import numpy as np
import pytest

import phasic
from phasic import Graph
import phasic.phasic_pybind as pb
from phasic.ffi_wrappers import _ensure_json_string
from _gate_backend import requires_ffi, assert_ffi_target  # noqa: E402

pytestmark = [pytest.mark.equivalence]

THETA = np.array([2.0, 3.0])
NR = 4


def _build() -> Graph:
    """Two-stage parameterized Erlang: start -> [3] -> [2] -> [1], rates theta[0],theta[1]."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    return g


def test_g2_moments_a_vs_c_bit_identical():
    # ---- path (a): pybind free function _moments via Graph.moments ----
    ga = _build()
    assert ga.cache_trace is False and ga.is_discrete is False
    ga.update_weights(THETA.tolist())
    a = np.asarray(ga.moments(NR))
    # backend assertion (a): compiled pybind base, not a Python/trace fallback
    assert phasic._Graph.__module__ == "phasic.phasic_pybind"
    assert "moments" in phasic._Graph.__dict__

    # ---- path (c): GraphBuilder.compute_moments (compute_moments_impl) ----
    sj = _ensure_json_string(_build().serialize())
    builder = pb.parameterized.GraphBuilder(sj)
    c = np.asarray(builder.compute_moments(THETA, nr_moments=NR))
    assert type(builder).__module__ == "phasic.phasic_pybind.parameterized"

    # bit-identity (verified diff == 0.0). A nonzero diff is an ordering
    # regression to escalate as Q5-ordering, NOT to relax to allclose.
    np.testing.assert_array_equal(a, c)


@requires_ffi
def test_g2_ffi_core_matches_direct_and_proves_ffi_path():
    import jax.numpy as jnp
    from phasic.ffi_wrappers import compute_moments_ffi
    assert hasattr(pb.parameterized, "get_compute_moments_ffi_capsule")
    assert phasic.get_config()._use_ffi is True

    sj = _ensure_json_string(_build().serialize())
    builder = pb.parameterized.GraphBuilder(sj)
    c = np.asarray(builder.compute_moments(THETA, nr_moments=NR))
    f = np.asarray(compute_moments_ffi(sj, jnp.asarray(THETA), NR))
    np.testing.assert_array_equal(c, f)                       # FFI core == direct core

    # PROVE the FFI custom-call ran (not pure_callback)
    assert_ffi_target(lambda t: compute_moments_ffi(sj, t, NR),
                      jnp.asarray(THETA), target="ptd_compute_moments")


def test_g2_moments_discrete_now_bound():
    """Q5 resolved: Graph.moments(discrete=True) no longer raises (super().moments_discrete
    was unbound). No native discrete-moment routine exists, so it converts the continuous
    waiting-time moments to discrete raw moments. The 1st discrete moment must equal
    expectation_discrete() (an independent existing path)."""
    g = _build().discretize(0.5)          # is_discrete=True; bypasses the continuous guard
    assert g.is_discrete
    g.update_weights([1.0] * g.param_length())   # discretize adds a coeff slot
    m = np.asarray(g.moments(2, discrete=True))
    assert m.shape == (2,) and np.all(np.isfinite(m))
    assert abs(m[0] - g.expectation_discrete()) < 1e-9


@pytest.mark.skip(reason=(
    "Impl (b) phasic::Graph::moments (api/cpp/phasiccpp.h:510) is native-only / "
    "unbound from Python; cover via a C++ test in tests/cpp/testcpp.cpp under CTest, "
    "not pytest. Recorded as a coverage gap in the Stage-2 handoff."))
def test_g2_impl_b_native_only_placeholder():
    pass
