"""Shared backend-detection primitives for the Stage-2 equivalence gates.

These gates pin *duplicate* implementations to identical output BEFORE the
Stage-3 refactor collapses them.  A bit-identity gate is worthless unless it can
prove *which* implementation path it exercised — in particular whether a JAX
call really lowered to the compiled XLA-FFI custom-call (``ptd_compute_*``) or
silently degraded to the ``jax.pure_callback`` host-callback fallback.  This
module centralises that proof plus a couple of tiny graph builders.

CRITICAL import-order note: this module imports ``phasic`` BEFORE ``jax`` so that
``JAX_ENABLE_X64=1`` (set in ``phasic/__init__.py`` at import) is in effect when
JAX initialises — the C FFI handlers require F64 buffers.  Every gate file that
uses JAX must import this module (or ``phasic``) before importing ``jax``.

Not a test module (filename starts with ``_``); pytest will not collect it, and
its ``python_files = test_*.py`` config excludes it too.
"""
from __future__ import annotations

import json

import phasic  # MUST precede jax
import numpy as np
import pytest

from phasic import Graph
import phasic.ffi_wrappers as _fw
from phasic.ffi_wrappers import _make_json_serializable

import jax  # noqa: E402  (import order is deliberate — see module docstring)

# Hard guarantee: F64 buffers, else every FFI call is wrong / truncated.
assert jax.config.jax_enable_x64 is True, (
    "jax_enable_x64 is False — import order broke it. Ensure `import phasic` "
    "(or this module) runs before any `import jax` in this process."
)

# The FFI target names each handler registers with XLA (ffi_wrappers.py).
FFI_TARGETS = {
    "ptd_compute_pmf",
    "ptd_compute_moments",
    "ptd_compute_pmf_and_moments",
    "ptd_compute_pmf_multivariate",
    "ptd_sample_path_conditioned",
}


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def ffi_compiled() -> bool:
    """True iff this build compiled the XLA-FFI handlers (HAVE_XLA_FFI)."""
    return hasattr(phasic.phasic_pybind.parameterized, "get_compute_pmf_ffi_capsule")


def ffi_registered() -> bool:
    """True once any ``*_ffi`` wrapper has registered targets this process."""
    return bool(_fw._FFI_REGISTERED)


def ensure_ffi_registered() -> bool:
    """Force registration now; raises PTDConfigError if FFI is config-disabled."""
    return _fw._register_ffi_targets()


# skip decorator for gates that require the compiled FFI fast path
requires_ffi = pytest.mark.skipif(
    not ffi_compiled(),
    reason="build has no XLA FFI (HAVE_XLA_FFI undefined) — FFI gate is vacuous",
)


# --------------------------------------------------------------------------- #
# Backend assertion via the jaxpr (structural; no HLO-text dependency)
# --------------------------------------------------------------------------- #
def _walk_eqns(jaxpr):
    """Yield every equation in a jaxpr, recursing into nested sub-jaxprs."""
    for eqn in jaxpr.eqns:
        yield eqn
        for v in eqn.params.values():
            sub = getattr(v, "jaxpr", None)
            if sub is not None:
                yield from _walk_eqns(sub)
            elif isinstance(v, (tuple, list)):
                for item in v:
                    sub = getattr(item, "jaxpr", None)
                    if sub is not None:
                        yield from _walk_eqns(sub)


def call_primitives(fn, *args):
    """Return (set of ffi_call target_names, set of all primitive names) for fn.

    ``fn`` is traced with ``jax.make_jaxpr`` at ``*args``.  Any ``structure_json``
    the wrapper closes over must be a closure constant of ``fn`` (not a traced
    positional), which is the natural way the gates call these wrappers.
    """
    closed = jax.make_jaxpr(fn)(*args)
    targets, prims = set(), set()
    for eqn in _walk_eqns(closed.jaxpr):
        name = eqn.primitive.name
        prims.add(name)
        if name == "ffi_call":
            t = eqn.params.get("target_name")
            if t is not None:
                targets.add(t)
    return targets, prims


def assert_ffi_target(fn, *args, target: str = "ptd_compute_pmf"):
    """Assert ``fn`` lowered to the FFI custom-call ``target`` and NOT a callback."""
    targets, prims = call_primitives(fn, *args)
    assert target in targets, (
        f"expected FFI target {target!r}; jaxpr had ffi targets {targets} "
        f"and primitives {sorted(prims)}"
    )
    assert "pure_callback" not in prims and "io_callback" not in prims, (
        f"unexpected host callback in what should be the FFI path: {sorted(prims)}"
    )


def assert_callback_not_ffi(fn, *args):
    """Assert ``fn`` used a host callback (pure_callback), NOT any FFI target."""
    targets, prims = call_primitives(fn, *args)
    assert not (targets & FFI_TARGETS), f"expected callback path, found FFI targets {targets}"
    assert "pure_callback" in prims or "io_callback" in prims, (
        f"expected a host callback primitive, got {sorted(prims)}"
    )


def assert_pybind_direct(out):
    """Assert ``out`` came from the pybind-direct GraphBuilder (a raw numpy array,
    never entered JAX)."""
    assert type(out).__module__.startswith("numpy"), (
        f"expected a numpy result from the pybind-direct path, got {type(out)!r}"
    )


# --------------------------------------------------------------------------- #
# Tiny graph builders + serialization (shared, self-contained; no test_utils)
# --------------------------------------------------------------------------- #
def coalescent_graph(n: int = 4) -> Graph:
    """1-parameter coalescent chain n -> n-1 -> ... -> 1.  Continuous only
    (outgoing rates exceed 1, so it is NOT a valid discrete PH graph)."""
    def _coal(state, **kwargs):
        k = int(state[0])
        if k <= 1:
            return []
        return [([k - 1], [k * (k - 1) / 2.0])]
    return Graph(_coal, ipv=[n])


def dph_graph(n: int = 3, p: float = 0.2) -> Graph:
    """1-parameter chain with per-edge coeff ``p`` so weight = p*theta stays <= 1
    for theta <= 1 — a valid discrete PH graph."""
    def _dph(state, **kwargs):
        k = int(state[0])
        if k <= 1:
            return []
        return [([k - 1], [p])]
    return Graph(_dph, ipv=[n])


def structure_str(g: Graph) -> str:
    """serialize() -> JSON string (GraphBuilder ctor needs a str, not a dict)."""
    return json.dumps(_make_json_serializable(g.serialize()))


def graph_builder(g: Graph):
    """Fresh pybind-direct GraphBuilder for ``g`` (isolated persistent-graph cache)."""
    return phasic.parameterized.GraphBuilder(structure_str(g))


# --------------------------------------------------------------------------- #
# JAX cache reset — importable autouse fixture for bit-identity gates
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_jax_state():
    """Clear JAX's order-sensitive JIT/compilation cache around every test so an
    unrelated test compiling first can't flip a cached program and flake a
    bit-identity assertion in full-suite order.  Mirrors the fixture in
    tests/pytest/inference/test_log_prob_unified_bit_identity.py.

    Import this name into a gate module (``from _gate_backend import
    reset_jax_state``) to activate it there.
    """
    jax.clear_caches()
    yield
    jax.clear_caches()
