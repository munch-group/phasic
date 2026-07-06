"""GATE G6 — conditioned path samplers (Stage-3 duplication #11).

Two near-line-for-line C samplers differing only in RNG and output form:
  Impl A  ptd_random_sample_path_conditioned        (phasic.c:10619, rand())
          via Graph.sample_path_conditioned          (pybind, __init__.py:2629)
  Impl B  ptd_random_sample_path_conditioned_fixed  (phasic.c:10708, rand_r())
          via sample_path_conditioned_ffi            (FFI capsule
          get_sample_path_conditioned_ffi_capsule)

``rand()`` and ``rand_r()`` are different glibc generators, so per-draw
seed-stable identity is impossible; this gate asserts DISTRIBUTIONAL equivalence
(the two draw i.i.d. samples from the same conditioned distribution) and pins the
per-draw divergence as xfail(strict) Q11a.  Each side asserts its own backend via
its unique output form.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import phasic
from phasic import Graph
import phasic.phasic_pybind as cpp
from _gate_backend import requires_ffi  # noqa: E402

pytestmark = [pytest.mark.equivalence, requires_ffi]

N = 20000
MAX_LEN = 12
SEED_PY = 20260705


@pytest.fixture(scope="function")
def fixed_param_graph():
    """Linear-mode parameterized graph, theta=[1.0] => weight == coeff.  Every
    intermediate vertex has a trash escape so its backward-prob h<1 (no premature
    stop at the loop guard backward_probs[idx] < 1.0)."""
    g = Graph(1)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([1])
    S = g.find_or_create_vertex([2])
    L = g.find_or_create_vertex([3])
    M = g.find_or_create_vertex([4])
    d = g.find_or_create_vertex([5])
    T = g.find_or_create_vertex([6])
    s.add_edge(A, [1.0])
    A.add_edge(S, [1.0]); A.add_edge(L, [1.0]); A.add_edge(T, [1.0])
    S.add_edge(d, [1.0]); S.add_edge(T, [1.0])
    L.add_edge(M, [1.0]); L.add_edge(T, [1.0])
    M.add_edge(d, [1.0]); M.add_edge(T, [1.0])
    g.update_weights([1.0])
    # index == enum-order == FFI-rebuild order (required for target-index parity)
    assert [v.index() for v in g.vertices()] == list(range(7))
    return g, d.index()


def _draw_pybind(g, target_idx, n):
    random.seed(SEED_PY)                       # deterministic set_c_seed -> srand
    paths = g.sample_path_conditioned([target_idx], n=n)   # Impl A (list of dict)
    last = np.array([int(p["vertex_indices"][-1]) for p in paths])
    length = np.array([len(p["vertex_indices"]) for p in paths])
    absn = np.array([float(p["entry_times"][-1]) for p in paths])
    return paths, last, length, absn


def _draw_ffi(g, target_idx, n):
    import jax.numpy as jnp
    from phasic.ffi_wrappers import sample_path_conditioned_ffi
    struct = g.serialize()
    theta = jnp.array([1.0])                   # float64 (phasic forces x64 at import)
    last = np.empty(n, int); length = np.empty(n, int); absn = np.empty(n)
    for i, sd in enumerate(range(1, n + 1)):   # explicit per-draw seeds (Impl B)
        vi, et = sample_path_conditioned_ffi(
            struct, theta,
            jnp.array([target_idx], dtype=jnp.int32),
            jnp.array([sd], dtype=jnp.int32), MAX_LEN)
        vi = np.asarray(vi); et = np.asarray(et)
        k = int((vi >= 0).sum())
        length[i] = k; last[i] = int(vi[k - 1]); absn[i] = float(et[k - 1])
    return last, length, absn


def test_g6_backend_is_pybind_rand(fixed_param_graph):
    g, target_idx = fixed_param_graph
    assert cpp.Graph in type(g).__mro__
    p = g.sample_path_conditioned([target_idx], n=1)     # Impl A
    assert isinstance(p, dict) and set(p) == {"vertex_indices", "entry_times"}
    assert (np.asarray(p["vertex_indices"]) >= 0).all()  # no FFI sentinel = Impl A form


def test_g6_backend_is_ffi_capsule(fixed_param_graph):
    import jax.numpy as jnp
    from phasic.ffi_wrappers import sample_path_conditioned_ffi
    g, target_idx = fixed_param_graph
    assert hasattr(cpp.parameterized, "get_sample_path_conditioned_ffi_capsule")
    cap = cpp.parameterized.get_sample_path_conditioned_ffi_capsule()
    assert type(cap).__name__ == "PyCapsule"
    vi, et = sample_path_conditioned_ffi(
        g.serialize(), jnp.array([1.0]),
        jnp.array([target_idx], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32), MAX_LEN)      # Impl B
    vi = np.asarray(vi)
    assert vi.dtype == np.int32 and vi.shape == (MAX_LEN,)   # fixed-size = FFI form
    assert (vi == -1).any()                                   # padding present = FFI form
    k = int((vi >= 0).sum())
    assert int(vi[0]) == 0 and int(vi[k - 1]) == target_idx


def test_g6_distributional_equivalence(fixed_param_graph):
    g, target_idx = fixed_param_graph
    _, a_last, a_len, a_abs = _draw_pybind(g, target_idx, N)
    b_last, b_len, b_abs = _draw_ffi(g, target_idx, N)

    # (1) hard invariants: every conditioned path reaches the target
    assert (a_last == target_idx).all()
    assert (b_last == target_idx).all()
    assert b_len.max() < MAX_LEN                     # guard against silent FFI truncation

    # (2) path-length distribution (guided-selection fractions from identical bp)
    a_p5 = (a_len == 5).mean(); b_p5 = (b_len == 5).mean()
    assert abs(a_p5 - 1 / 3) < 0.03 and abs(b_p5 - 1 / 3) < 0.03
    assert abs(a_p5 - b_p5) < 0.03

    # (3) continuous absorption-time equivalence, ~6 SEM band (cannot flake under H0)
    sem = a_abs.std() / np.sqrt(N)
    assert abs(a_abs.mean() - b_abs.mean()) < 6 * sem


@pytest.mark.xfail(strict=True, reason=(
    "Q11a: Impl A rand() (phasic.c:10641) has no seed argument (set_c_seed, "
    "phasic_pybind.cpp:2089) vs Impl B rand_r() (phasic.c:10732) with an explicit "
    "seed; per-draw seed-stable identity is impossible until Stage-3 unifies both "
    "onto a single seeded rand_r RNG."))
def test_g6_per_draw_seed_identity_xfail(fixed_param_graph):
    import jax.numpy as jnp
    from phasic.ffi_wrappers import sample_path_conditioned_ffi
    g, target_idx = fixed_param_graph
    random.seed(1)
    a = g.sample_path_conditioned([target_idx], n=1)
    vi, et = sample_path_conditioned_ffi(
        g.serialize(), jnp.array([1.0]),
        jnp.array([target_idx], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32), MAX_LEN)
    vi = np.asarray(vi); et = np.asarray(et)
    k = int((vi >= 0).sum())
    assert float(a["entry_times"][-1]) == float(et[k - 1])   # reliably False -> xfailed
