"""An exception must never escape an OpenMP structured block.

THE BUG
-------
The batched FFI handlers run `#pragma omp parallel for` over the batch. Their loop
bodies can throw -- GraphBuilder::build() (theta-length mismatch; a non-positive
'log' product; a negative/non-finite 'formula' weight), compute_moments_impl(), the
rewards parsing, and the C layer (a DPH row sum > 1.0001 makes pdf()/dph_pmf() throw).

The OpenMP spec requires a throw inside a parallel region to be caught by the same
thread INSIDE that region. Letting one escape calls std::terminate:

    libc++abi: terminating due to uncaught exception of type std::invalid_argument
    Abort trap: 6

That HARD-KILLS the interpreter: uncatchable, no Python traceback, no checkpoint.
SVGD vmaps over its particles, so batch_size > 1 is the DEFAULT path -- one particle
straying past a weight boundary would kill an entire run.

It fires even on ONE thread: `if(batch_size > 1)` still *opens* a region with a team
of one, so OMP_NUM_THREADS=1 is not a workaround.

Measured before the fix, these four handlers aborted (the others already caught
per-iteration):

    ComputePmfFfiImpl            (log / formula / DPH-row-sum triggers)
    ComputeMomentsFfiImpl        (log)
    ComputeSojournTimesFfiImpl   (formula)   -- reached via pmf_from_graph_joint_index
    SamplePathConditionedFfiImpl (log)

WHY THESE TESTS SPAWN SUBPROCESSES
----------------------------------
An abort kills the process. A regression therefore CANNOT be observed from inside the
test -- pytest itself would die. Each case runs in a subprocess and we assert on the
exit status: SIGABRT (returncode -6, or 134 via a shell) means the bug is back.
"""
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")


_PRELUDE = """
import phasic
import numpy as np
import jax
import jax.numpy as jnp
from phasic import Graph

def chain(mode):
    g = phasic.Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0, 3.0])
    b.add_edge(c, [1.0, 2.0])
    if mode == 'log':
        g.weight_mode = 'log'
    elif mode == 'formula':
        g.weight_formula = 'c0*t0 - c1*t1'
    return g

T = jnp.asarray([0.3, 0.6])
"""


def _run(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a subprocess. An OpenMP abort shows up as SIGABRT."""
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=600,
    )


def _assert_no_abort(res: subprocess.CompletedProcess, label: str) -> None:
    aborted = (
        res.returncode in (-6, 134)
        or "Abort trap" in (res.stderr or "")
        or "terminating due to uncaught exception" in (res.stderr or "")
    )
    assert not aborted, (
        f"{label}: an exception escaped an OpenMP structured block and "
        f"ABORTED the process (rc={res.returncode}).\n"
        f"stderr tail:\n{(res.stderr or '')[-400:]}"
    )
    # It must still FAIL -- loudly and catchably -- not silently return a number.
    assert "CAUGHT" in (res.stdout or ""), (
        f"{label}: expected a catchable exception, got:\n"
        f"stdout={res.stdout!r}\nstderr tail={(res.stderr or '')[-300:]!r}"
    )


# ---------------------------------------------------------------------------
# One case per handler that used to abort. Each uses batch_size == 2, which is
# what jax.vmap (and therefore SVGD, over its particles) produces.
# ---------------------------------------------------------------------------

def test_compute_pmf_handler_does_not_abort():
    res = _run("""
        m = Graph.pmf_from_graph(chain('log'))
        bad = jnp.asarray([1.0, -2.0])      # non-positive product -> build() throws
        try:
            jax.vmap(lambda t: m(t, T))(jnp.stack([bad, bad]))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputePmfFfiImpl")


def test_compute_pmf_handler_does_not_abort_on_dph_row_sum():
    """A different throw site: the C layer rejects a DPH row sum > 1.0001, so the
    throw comes from dph_pmf(), not from build()."""
    res = _run("""
        m = Graph.pmf_from_graph(chain('linear'), discrete=True)
        bad = jnp.asarray([3.0, 3.0])       # row sums >> 1 -> C throws
        try:
            jax.vmap(lambda t: m(t, jnp.asarray([2.0, 3.0])))(jnp.stack([bad, bad]))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputePmfFfiImpl (DPH row sum)")


def test_compute_moments_handler_does_not_abort():
    res = _run("""
        import json
        from phasic import ffi_wrappers as fw
        g = chain('log')
        sj = json.dumps(fw._make_json_serializable(g.serialize()))
        bad = jnp.asarray([1.0, -2.0])
        try:
            jax.vmap(lambda t: fw.compute_moments_ffi(sj, t, 2))(jnp.stack([bad, bad]))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputeMomentsFfiImpl")


def test_sojourn_handler_does_not_abort():
    """Reached via pmf_from_graph_joint_index. 'log' is rejected earlier by a Python
    guard, so the trigger here is a 'formula' that produces a negative weight."""
    res = _run("""
        m = Graph.pmf_from_graph_joint_index(chain('formula'))
        bad = jnp.asarray([1.0, 5.0])       # c0*t0 - c1*t1 < 0 -> build() throws
        try:
            jax.vmap(lambda t: m(t, jnp.asarray([1.0, 2.0]))[0])(jnp.stack([bad, bad]))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputeSojournTimesFfiImpl")


def test_sample_path_handler_does_not_abort():
    res = _run("""
        import json
        from phasic import ffi_wrappers as fw
        g = chain('log')
        sj = json.dumps(fw._make_json_serializable(g.serialize()))
        bad = jnp.asarray([1.0, -2.0])
        try:
            jax.vmap(lambda t: fw.sample_path_conditioned_ffi(
                sj, t, jnp.asarray(3, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.uint32), 16)[0])(jnp.stack([bad, bad]))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "SamplePathConditionedFfiImpl")


# ---------------------------------------------------------------------------
# The fix must not break anything, and must not fail SILENTLY.
# ---------------------------------------------------------------------------

def test_a_partial_batch_failure_still_raises():
    """7 good elements and 1 bad one must RAISE -- not quietly return NaN for the
    bad row and plausible numbers for the rest. A silently-NaN'd particle is exactly
    the kind of failure this whole audit exists to prevent."""
    res = _run("""
        m = Graph.pmf_from_graph(chain('log'))
        good = jnp.asarray([1.0, 2.0])
        bad  = jnp.asarray([1.0, -2.0])
        try:
            out = jax.vmap(lambda t: m(t, T))(jnp.stack([good]*7 + [bad]))
            print('NO_RAISE', np.asarray(out))
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "partial batch failure")


def test_pmf_and_moments_handler_raises_not_silently_nans():
    """ComputePmfAndMomentsFfiImpl (use_ffi=True) had a PRE-EXISTING silent
    failure: its per-element catch NaN-filled the moments buffer, left the pmf
    buffer as a plausible 0.0, and the handler returned Success(). A throwing
    particle silently poisoned the batch. It must RAISE.

    Needs PHASIC_SOURCE_DIR: use_ffi=True routes through the FFI handler.
    """
    res = _run("""
        from phasic import Graph
        m = Graph.pmf_and_moments_from_graph(chain('log'), nr_moments=2, use_ffi=True)
        good = jnp.asarray([1.0, 2.0]); bad = jnp.asarray([1.0, -2.0])
        try:
            jax.vmap(lambda t: m(t, T))(jnp.stack([good, bad]))
            print('NO_RAISE')          # <-- the bug: silent NaN + Success
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputePmfAndMomentsFfiImpl (silent-NaN)")
    assert "NO_RAISE" not in (res.stdout or ""), (
        "ComputePmfAndMomentsFfiImpl silently returned instead of raising:\n"
        f"{res.stdout}"
    )


def test_pmf_multivariate_handler_raises_not_silently_nans():
    """ComputePmfMultivariateFfiImpl -- same pre-existing silent swallow, probed
    through the raw wrapper (the high-level API may route elsewhere)."""
    res = _run("""
        import json
        from phasic import ffi_wrappers as fw
        sj = json.dumps(fw._make_json_serializable(chain('log').serialize()))
        good = jnp.asarray([1.0, 2.0]); bad = jnp.asarray([1.0, -2.0])
        t2 = jnp.asarray([[0.3, 0.6]]); r2 = jnp.asarray([[0.0, 1.0, 1.0, 0.0]])
        try:
            fw.compute_pmf_multivariate_ffi(sj, jnp.stack([good, bad]), t2, r2,
                                            discrete=False)
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', type(e).__name__)
    """)
    _assert_no_abort(res, "ComputePmfMultivariateFfiImpl (silent-NaN)")
    assert "NO_RAISE" not in (res.stdout or ""), (
        "ComputePmfMultivariateFfiImpl silently returned instead of raising:\n"
        f"{res.stdout}"
    )


def test_valid_batches_are_unaffected():
    """The try/catch must be transparent: a batched result must stay BIT-IDENTICAL
    to the unbatched one, and the moments must still match the closed form."""
    res = _run("""
        m  = Graph.pmf_from_graph(chain('linear'))
        mm = Graph.pmf_and_moments_from_graph(chain('linear'), nr_moments=1)
        th = jnp.asarray([1.0, 2.0])

        batched  = np.asarray(jax.vmap(lambda t: m(t, T))(jnp.stack([th]*8)))
        unbatched = np.asarray(m(th, T))
        assert np.array_equal(batched[0], unbatched), 'batched != unbatched'
        assert np.all(batched == batched[0]), 'rows differ'

        mom = np.asarray(jax.vmap(lambda t: mm(t, T)[1])(jnp.stack([th]*8)))
        truth = 1/8 + 1/5                      # hypoexponential(8, 5)
        assert np.allclose(mom[:, 0], truth, rtol=1e-12), (mom[:, 0], truth)
        print('CAUGHT')                        # sentinel: reached the end cleanly
    """)
    _assert_no_abort(res, "valid batch")
