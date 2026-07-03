"""Regression test: FFI handlers must broadcast a size-1 (unmapped) theta.

Under vmap_method="expand_dims" an unmapped operand arrives with a size-1
leading axis and must be BROADCAST across the batch. The pmf handlers indexed
theta unconditionally as theta_data + b*theta_len, so when TIMES (or rewards)
was the mapped operand and THETA was the size-1 broadcast operand, they
over-read theta past its single row for b>=1 — feeding garbage weights to the
builder (NaN/garbage PDFs or a crash). BackwardProbabilities had no batch loop
at all and wrote graph-vertices-length regardless of the declared buffer.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402

from phasic import Graph  # noqa: E402
from phasic.test_utils import coalescent_callback_parameterized  # noqa: E402
from phasic.ffi_wrappers import compute_pmf_ffi  # noqa: E402


def _graph():
    return Graph(coalescent_callback_parameterized, ipv=[4, 0, 0, 0])


def test_vmap_over_times_with_fixed_theta():
    """The previously-broken direction: map TIMES, broadcast THETA."""
    sd = _graph().serialize()
    theta = jnp.array([2.0])  # fixed -> size-1 broadcast operand under vmap
    times_batch = jnp.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])  # (3, 2)

    batched = vmap(
        lambda t: compute_pmf_ffi(sd, theta, t, discrete=False, granularity=100)
    )(times_batch)

    assert batched.shape == (3, 2)
    assert np.all(np.isfinite(np.asarray(batched))), "over-read theta -> non-finite pdf"

    # Each batch row must equal the un-vmapped computation with the SAME theta.
    for i in range(times_batch.shape[0]):
        indiv = compute_pmf_ffi(sd, theta, times_batch[i], discrete=False, granularity=100)
        np.testing.assert_allclose(
            np.asarray(batched[i]), np.asarray(indiv), rtol=1e-10, atol=1e-12
        )


def test_vmap_over_theta_still_correct():
    """The already-working direction must be unchanged."""
    sd = _graph().serialize()
    times = jnp.array([1.0, 2.0])
    theta_batch = jnp.array([[1.0], [2.0], [3.0]])

    batched = vmap(
        lambda th: compute_pmf_ffi(sd, th, times, discrete=False, granularity=100)
    )(theta_batch)

    assert batched.shape == (3, 2)
    for i in range(theta_batch.shape[0]):
        indiv = compute_pmf_ffi(sd, theta_batch[i], times, discrete=False, granularity=100)
        np.testing.assert_allclose(
            np.asarray(batched[i]), np.asarray(indiv), rtol=1e-10, atol=1e-12
        )
