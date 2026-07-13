"""Weight-mode correctness: the FD probe rule, and the guards that stop a
non-linear weight_mode from being silently linearised.

Provenance: audit of the numerical refactor, findings F1-F3
(see audit-phase1-forward-parity.md).

The bugs these pin:

NOTE: this file used to also test `_fd_probe_points` (the relative FD step).
That step has been ROLLED BACK -- see the audit reports -- so those tests are gone.
The weight_mode guards below are independent of the FD step and still apply.

F2  `moments_from_graph` JIT-generates a `build_model()` whose weight computation
    is a hardcoded linear dot product, so a log/callback/formula graph silently
    received LINEAR moments -- E[T] = 0.325 where the truth was 0.75. A wrong
    number, with no error.

F3  `pmf_from_graph_joint_index` / `daisy_chain_joint_probs` reach FFI handlers
    that call `ptd_graph_update_weights(..., use_log=false)` directly, so a 'log'
    graph silently received LINEAR weights.

These assert VALUES against closed forms, never `isfinite`/`> 0` -- the whole point
is that the broken versions produced finite, plausible, WRONG numbers.
"""
import os
import pathlib

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")


# `moments_from_graph` JIT-compiles C++ from the source tree, so any test that
# reaches its VALUE needs the sources on disk (a non-editable install is a copy
# under site-packages, where no `src/` exists). The GUARD tests below do NOT need
# them -- the weight_mode guard fires before the compile -- so only the
# value test is skipped.
_pkg_dir = phasic._get_package_dir()
_HAS_SOURCES = (_pkg_dir / "src" / "cpp" / "phasiccpp.cpp").exists()
requires_sources = pytest.mark.skipif(
    not _HAS_SOURCES,
    reason=(
        f"phasic C/C++ sources not on disk (package root: {_pkg_dir}); "
        "JIT compilation requires `pip install -e .` or PHASIC_SOURCE_DIR "
        "pointing at a source checkout."
    ),
)


# s -> v3 -[2,3]-> v2 -[1,2]-> v1(absorbing).  theta = [1, 2]:
#   linear     rates (2*1+3*2, 1*1+2*2) = (8, 5)      -> E[T] = 1/8 + 1/5 = 0.325
#   log        rates ((2*1)*(3*2), (1*1)*(2*2)) = (12, 4) -> E[T] = 1/12 + 1/4 = 0.3333
#   c0*t0*t1   rates (2*1*2, 1*1*2) = (4, 2)          -> E[T] = 1/4 + 1/2 = 0.75
THETA = np.array([1.0, 2.0])
E_T_LINEAR = 1 / 8 + 1 / 5
E_T_LOG = 1 / 12 + 1 / 4
E_T_NONLINEAR = 1 / 4 + 1 / 2


def _graph(mode="linear"):
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [2.0, 3.0])
    v2.add_edge(v1, [1.0, 2.0])
    if mode == "log":
        g.weight_mode = "log"
    elif mode == "callback":
        g.weight_callback = lambda th, co: float(co[0] * th[0] * th[1])
    elif mode == "formula":
        g.weight_formula = "c0*t0*t1"
    return g


class TestMomentsFromGraphRejectsNonLinearWeightMode:

    @requires_sources
    def test_linear_is_correct_and_unchanged(self):
        # Needs the source tree: this is the only test here that reaches the JIT
        # compile. It SKIPS in a default (non-editable) run -- set
        # PHASIC_SOURCE_DIR to exercise it.
        m = Graph.moments_from_graph(_graph("linear"), nr_moments=1)
        got = float(np.asarray(m(jnp.asarray(THETA)))[0])
        assert got == pytest.approx(E_T_LINEAR, rel=1e-12)

    @pytest.mark.parametrize("mode", ["log", "callback", "formula"])
    def test_non_linear_modes_raise_instead_of_returning_linear(self, mode):
        # Before the guard this returned E[T] = 0.325 -- the LINEAR answer -- for
        # every one of these, where the truth is 0.75 (callback/formula) or
        # 0.3333 (log). Finite, plausible, and wrong.
        with pytest.raises(ValueError, match=r"weight_mode='linear' only"):
            Graph.moments_from_graph(_graph(mode), nr_moments=2)

    def test_the_wrong_answer_it_used_to_give_is_actually_wrong(self):
        # Pins WHY the guard exists: the non-linear rule really does imply a
        # different E[T], so silently returning the linear one was a real error.
        assert E_T_NONLINEAR != pytest.approx(E_T_LINEAR)
        assert E_T_LOG != pytest.approx(E_T_LINEAR)


# ---------------------------------------------------------------------------
# F3 -- joint_index honours callback/formula, and rejects 'log'
# ---------------------------------------------------------------------------

class TestJointIndexWeightMode:

    def test_rejects_log(self):
        with pytest.raises(ValueError, match=r"does not support weight_mode='log'"):
            Graph.pmf_from_graph_joint_index(_graph("log"))

    @pytest.mark.parametrize(
        "mode,expected",
        [("linear", 0.625), ("callback", 0.5), ("formula", 0.5)],
    )
    def test_callback_and_formula_are_honoured(self, mode, expected):
        # The guard must reject ONLY 'log'. callback/formula go through a real
        # branch and must keep working -- and must NOT collapse to the linear
        # value (0.625), which is what a silent linearisation would give.
        model = Graph.pmf_from_graph_joint_index(_graph(mode))
        out = model(jnp.asarray(THETA), jnp.asarray([1.0, 2.0]))
        got = float(np.asarray(out[0])[0])
        assert got == pytest.approx(expected, rel=1e-9)
