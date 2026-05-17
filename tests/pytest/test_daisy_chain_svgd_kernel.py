"""Kernel-context regression tests for no-exposure daisy-chain SVGD.

The original bug fixed today (2026-05-16) was:
``Graph.svgd(epoch_starts=..., exposure=None)`` raised
``TypeError: No constant handler for type: DynamicJaxprTracer``
when called from inside an IPython kernel (e.g. a Jupyter notebook
running under nbconvert), even though the same code ran cleanly in
a standalone ``python script.py`` invocation. The kernel and the
plain interpreter take different paths through JAX's
``_pjit_batcher``; only the kernel path exercised the
trace-stack shape that exposed a tracer leak in the (then-)
``custom_vmap``-decorated ``_forward`` inside
``Graph.daisy_chain_joint_probs``.

These tests pin the regression by driving an ipykernel from
``jupyter_client``: each test spawns a fresh kernel, sends a
self-contained cell with the failing-pattern SVGD call, and asserts
the cell's status is ``ok`` (i.e. no exception in the kernel) and
the cell's stdout includes the ``SUCCESS`` sentinel.

Skipped automatically on hosts without ``jupyter_client``.
"""
from __future__ import annotations

import textwrap

import pytest


jupyter_client = pytest.importorskip('jupyter_client')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Self-contained model fixture. Built deterministically from np.random.seed(17)
# so the cell stays a single triple-quoted string with no file I/O.
_CELL_PREAMBLE = textwrap.dedent('''
    import numpy as np
    from functools import partial
    from itertools import combinations_with_replacement
    from phasic import (
        Graph, with_ipv, LogGaussPrior, Adamelia,
        StateIndexer, Property, set_log_level,
    )
    set_log_level('WARNING')
    np.random.seed(17)
    all_pairs = partial(combinations_with_replacement, r=2)

    nr_samples = 4
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=nr_samples)]
    )

    @with_ipv([nr_samples] + [0]*(nr_samples-1))
    def coalescent_1param(state, indexer=None):
        transitions = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair_count = state[i] * (state[j] - same) / (1 + same)
            transitions.append([new, [pair_count]])
        return transitions

    graph = Graph(coalescent_1param, indexer=indexer)
    reward_limit = 5
    mutation_rate = 1e-4
    theta = 1/10_000

    # Sample observations from the DISCRETE joint-prob table so the
    # cell has realistic observation vectors to drive SVGD with.
    jpg_disc = graph.joint_prob_graph(
        indexer, reward_limit=reward_limit, mutation_rate=mutation_rate,
    )
    jpg_disc.update_weights([theta, mutation_rate])
    jpt = jpg_disc.joint_prob_table()
    p = jpt['prob'] / jpt['prob'].sum()
    sample = np.random.choice(jpt.index.values, 1000, p=p.to_numpy())
    observations = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
''')


_SVGD_NO_EXPOSURE_CELL = textwrap.dedent('''
    jpg_cont = graph.joint_prob_graph(
        indexer,
        reward_limit=reward_limit,
        mutation_rate=mutation_rate,
        discrete=False,
    )
    svgd = jpg_cont.svgd(
        observations,
        fixed=[(1, mutation_rate)],
        prior=LogGaussPrior(ci=[1/50_000, 1/5000]),
        n_iterations=2,
        n_particles=8,
        optimizer=Adamelia(learning_rate=0.2),
        epoch_starts=[0, 0.5],
    )
    print('SUCCESS')
''')


_SVGD_DISC_FIRST_CELL = textwrap.dedent('''
    # Mirrors the notebook ordering that originally triggered the bug
    # (docs/pages/tutorial/svgd-joint-prob.ipynb cell 26 → cell 32):
    # run a discrete-joint-prob SVGD first, then the continuous one
    # with epoch_starts.
    svgd_disc = jpg_disc.svgd(
        observations,
        fixed=[(1, mutation_rate)],
        prior=LogGaussPrior(ci=[1/50_000, 1/5000]),
        n_iterations=2,
        n_particles=8,
        optimizer=Adamelia(learning_rate=0.2),
    )
''')


# Plan: typed-riding-truffle, batch 5. Mirrors _SVGD_NO_EXPOSURE_CELL
# but adds tied=[(0, [0, 1])] so the no-exposure daisy-chain path
# also exercises _apply_tying in the kernel-context trace stack.
_SVGD_TIED_CELL = textwrap.dedent('''
    jpg_cont = graph.joint_prob_graph(
        indexer,
        reward_limit=reward_limit,
        mutation_rate=mutation_rate,
        discrete=False,
    )
    svgd = jpg_cont.svgd(
        observations,
        fixed=[(1, mutation_rate)],
        tied=[(0, [0, 1])],
        prior=LogGaussPrior(ci=[1/50_000, 1/5000]),
        n_iterations=2,
        n_particles=8,
        optimizer=Adamelia(learning_rate=0.2),
        epoch_starts=[0, 0.5],
    )
    print('SUCCESS')
''')


def _run_cell(kc, code, timeout=600):
    """Send ``code`` to the kernel client ``kc`` and return
    ``(status, stdout, stderr)`` where ``stdout``/``stderr`` are the
    concatenated stream outputs and ``status`` is ``'ok'`` or
    ``'error'``.
    """
    import time

    msg_id = kc.execute(code)
    status = None
    while True:
        msg = kc.get_shell_msg(timeout=timeout)
        if msg['parent_header'].get('msg_id') == msg_id:
            status = msg['content'].get('status')
            break

    # Drain iopub for stream outputs.
    time.sleep(0.5)
    stdout_parts = []
    stderr_parts = []
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=0.5)
        except Exception:
            break
        if msg['msg_type'] == 'stream':
            name = msg['content']['name']
            text = msg['content']['text']
            if name == 'stdout':
                stdout_parts.append(text)
            elif name == 'stderr':
                stderr_parts.append(text)
    return status, ''.join(stdout_parts), ''.join(stderr_parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_exposure_daisy_chain_svgd_runs_in_kernel():
    """No-exposure daisy-chain SVGD must run cleanly inside an
    ipykernel — this is the canonical regression for the
    `DynamicJaxprTracer` const-leak that was reverted on 2026-05-16."""
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name='python3')
    try:
        # Build the model fixture.
        status, _, stderr = _run_cell(kc, _CELL_PREAMBLE)
        assert status == 'ok', f"preamble failed: stderr={stderr!r}"

        # Run the no-exposure daisy-chain SVGD cell.
        status, stdout, stderr = _run_cell(kc, _SVGD_NO_EXPOSURE_CELL)
        assert status == 'ok', (
            f"no-exposure daisy-chain SVGD failed in kernel — "
            f"this is the regression. stdout={stdout!r} stderr={stderr!r}"
        )
        assert 'SUCCESS' in stdout, (
            f"SVGD ran without raising but did not reach the "
            f"SUCCESS print. stdout={stdout!r}"
        )
    finally:
        km.shutdown_kernel()


def test_no_exposure_daisy_chain_svgd_runs_after_discrete_svgd():
    """Same regression, but with a discrete-joint-prob SVGD run
    first — mirrors the original failing notebook ordering
    (cell 26 then cell 32 in
    ``docs/pages/tutorial/svgd-joint-prob.ipynb``). Catches any
    regression where compiled-cache pollution between two SVGD
    runs in the same kernel re-triggers the leak."""
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name='python3')
    try:
        status, _, stderr = _run_cell(kc, _CELL_PREAMBLE)
        assert status == 'ok', f"preamble failed: stderr={stderr!r}"

        status, _, stderr = _run_cell(kc, _SVGD_DISC_FIRST_CELL)
        assert status == 'ok', f"discrete SVGD failed: stderr={stderr!r}"

        status, stdout, stderr = _run_cell(kc, _SVGD_NO_EXPOSURE_CELL)
        assert status == 'ok', (
            f"no-exposure daisy-chain SVGD failed AFTER discrete "
            f"SVGD ran in same kernel — regression. "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
        assert 'SUCCESS' in stdout
    finally:
        km.shutdown_kernel()


def test_no_exposure_daisy_chain_svgd_runs_with_tied_in_kernel():
    """Tied-parameter SVGD must run cleanly inside an ipykernel.
    Mirrors `test_no_exposure_daisy_chain_svgd_runs_in_kernel` but
    adds ``tied=[(0, [0, 1])]`` so the `_apply_tying` scatter inside
    the model wrapper is exercised in the kernel trace stack —
    catches any regression where the tying machinery re-introduces
    a `DynamicJaxprTracer` const-leak under `vmap(jit(grad(...)))`.
    Plan: typed-riding-truffle, batch 5.
    """
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name='python3')
    try:
        status, _, stderr = _run_cell(kc, _CELL_PREAMBLE)
        assert status == 'ok', f"preamble failed: stderr={stderr!r}"

        status, stdout, stderr = _run_cell(kc, _SVGD_TIED_CELL)
        assert status == 'ok', (
            f"tied daisy-chain SVGD failed in kernel — regression. "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
        assert 'SUCCESS' in stdout, (
            f"tied SVGD ran without raising but did not reach the "
            f"SUCCESS print. stdout={stdout!r}"
        )
    finally:
        km.shutdown_kernel()
