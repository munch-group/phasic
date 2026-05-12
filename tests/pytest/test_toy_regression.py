"""Regression test for the toy reference graphs.

Locks down the numerical output of ``Graph.expected_waiting_time`` and
``Graph.expectation`` on the toy variants defined in
``toy_model.py``. This is the regression anchor used throughout the
``hierar-elimin-cache`` branch — every WP must produce output that
agrees with the values captured here to bitwise precision (the
current monolithic eliminator is deterministic, so exact equality is
the right gate; we use a small relative tolerance only as a defence
against floating-point reorderings introduced by future eliminator
refactors).

The reference values are committed alongside the test in
``toy_model_reference.json``. To regenerate (after a deliberate
change to the toy graphs or to the eliminator's expected output),
run ``python -m tests.pytest.update_toy_reference``.

See §11 of ``hierar-elimin-cache-reference.md`` for the design of
the toy graphs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from toy_model import BUILDERS, THETAS


REFERENCE_PATH = Path(__file__).parent / "toy_model_reference.json"


def _load_reference() -> dict:
    if not REFERENCE_PATH.exists():
        pytest.skip(
            f"Reference values not found at {REFERENCE_PATH}; "
            "run update_toy_reference.py to regenerate."
        )
    with open(REFERENCE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reference():
    return _load_reference()


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
@pytest.mark.parametrize("theta_name", list(THETAS.keys()))
def test_toy_expected_waiting_time(toy_name, theta_name, reference):
    """Verify ``expected_waiting_time`` agrees with the captured reference."""
    builder = BUILDERS[toy_name]
    theta = THETAS[theta_name]
    expected = reference[toy_name][theta_name]

    g = builder()
    g.update_weights(theta)
    actual_ewt = np.asarray(g.expected_waiting_time())

    expected_ewt = np.asarray(expected["expected_waiting_time"])
    assert actual_ewt.shape == expected_ewt.shape, (
        f"{toy_name}/{theta_name}: shape mismatch — "
        f"got {actual_ewt.shape}, expected {expected_ewt.shape}"
    )
    np.testing.assert_allclose(
        actual_ewt, expected_ewt, rtol=1e-12, atol=1e-12,
        err_msg=f"{toy_name}/{theta_name}: expected_waiting_time disagrees",
    )


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
@pytest.mark.parametrize("theta_name", list(THETAS.keys()))
def test_toy_expectation(toy_name, theta_name, reference):
    """Verify ``expectation`` (== ``expected_waiting_time[start]``) agrees."""
    builder = BUILDERS[toy_name]
    theta = THETAS[theta_name]
    expected = reference[toy_name][theta_name]

    g = builder()
    g.update_weights(theta)
    actual_exp = float(g.expectation())
    expected_exp = expected["expectation"]

    assert math.isclose(actual_exp, expected_exp, rel_tol=1e-12, abs_tol=1e-12), (
        f"{toy_name}/{theta_name}: expectation disagrees — "
        f"got {actual_exp!r}, expected {expected_exp!r}"
    )


def test_toy_d_has_duplicate_zero_state():
    """Toy-D's invariant: starting vertex and aux vertex both have all-zero state.

    This guards against accidental loss of the duplicate-state condition
    if the toy is refactored — without it, Toy-D no longer exercises the
    invariant it was designed to test.
    """
    g = BUILDERS['toy_d']()
    starting = g.starting_vertex()
    starting_state = list(starting.state())
    starting_idx = starting.index()

    zero_state_vertices = [
        v for v in g.vertices()
        if list(v.state()) == starting_state
    ]
    zero_state_indices = sorted(v.index() for v in zero_state_vertices)

    assert len(zero_state_vertices) >= 2, (
        f"Toy-D should have >= 2 vertices with all-zero state "
        f"(starting + aux); got {len(zero_state_vertices)}: indices {zero_state_indices}"
    )
    assert starting_idx in zero_state_indices, (
        f"starting vertex (index {starting_idx}) not among "
        f"zero-state vertices {zero_state_indices}"
    )


def test_all_toys_reproducible():
    """Building the same toy twice must yield identical expected_waiting_time."""
    theta = [1.0, 1.0, 1.0, 1.0]
    for toy_name, builder in BUILDERS.items():
        g1 = builder()
        g1.update_weights(theta)
        ewt1 = np.asarray(g1.expected_waiting_time())

        g2 = builder()
        g2.update_weights(theta)
        ewt2 = np.asarray(g2.expected_waiting_time())

        np.testing.assert_array_equal(
            ewt1, ewt2,
            err_msg=f"{toy_name}: rebuild produced different ewt",
        )
