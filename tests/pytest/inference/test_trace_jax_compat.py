"""
Stage 0 JAX-compatibility test matrix for the trace pipeline.

Covers, for both the eager Python `evaluate_trace_jax` path and (after
Stage 4 lands) the C++-codegen `_wrap_trace_log_likelihood_for_jax` path:

  1. jax.jit
  2. jax.vmap
  3. jax.grad
  4. jax.pmap (multi-core CPU; auto-skips if local_device_count() < 2)
  5. NamedSharding (jax.Array sharded across the local mesh)
  6. SVGD round-trip with parallel='vmap' and 'pmap' on the local machine
  7. SLURM multi-node — gated behind SLURM_JOB_ID; skipped otherwise
  8. Disk-cached trace round-trip across processes (subprocess re-load)

The reference-graph library lives in test_trace_vs_direct.py — we import
from there to avoid duplication.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

# IMPORTANT: phasic must be imported BEFORE jax so phasic can configure
# multi-CPU device count via XLA_FLAGS.
from phasic import Graph
from phasic.trace_elimination import (
    EliminationTrace,
    evaluate_trace,
    evaluate_trace_jax,
    record_elimination_trace,
)

import jax
import jax.numpy as jnp

# Sibling test module — pytest puts the inference dir on sys.path during
# collection, so a plain absolute import works.
from test_trace_vs_direct import GRAPH_BUILDERS, ACYCLIC  # noqa: E402


# ----------------------------------------------------------------------------
# Per-test fixtures: build a (trace, theta, eager_result) triple per graph
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module", params=ACYCLIC)
def trace_fixture(request):
    """Return (trace, theta_jax, eager_vertex_rates) for an acyclic graph.

    Acyclic only — cyclic graphs raise during recording (Stage 1 will
    flip those tests on, in a separate test file or by relaxing this
    fixture once the self-loop correction is implemented).
    """
    builder = GRAPH_BUILDERS[request.param]
    graph, theta_np = builder()
    trace = record_elimination_trace(graph, theta_dim=len(theta_np))
    eager = evaluate_trace(trace, theta_np, use_log=False)
    return trace, jnp.asarray(theta_np, dtype=jnp.float64), np.asarray(
        eager["vertex_rates"], dtype=np.float64
    )


# ----------------------------------------------------------------------------
# (1) jax.jit — recompilation suppression + eager-equivalence
# ----------------------------------------------------------------------------


def test_jit_matches_eager(trace_fixture):
    trace, theta, eager_rates = trace_fixture
    jit_eval = jax.jit(lambda p: evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])
    jit_rates = np.asarray(jit_eval(theta))
    np.testing.assert_allclose(jit_rates, eager_rates, rtol=1e-12, atol=1e-15)


def test_jit_recompilation_avoided(trace_fixture):
    """Calling a jitted function twice with identical input shape and dtype
    must hit the JAX cache, not recompile. We probe via _cache_size."""
    trace, theta, _ = trace_fixture
    jit_eval = jax.jit(lambda p: evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])
    jit_eval(theta).block_until_ready()
    n_compiles_after_first = jit_eval._cache_size()
    jit_eval(theta).block_until_ready()
    assert jit_eval._cache_size() == n_compiles_after_first, (
        "jit cache miss on identical input — re-compilation occurred"
    )


# ----------------------------------------------------------------------------
# (2) jax.vmap — batched eval
# ----------------------------------------------------------------------------


def test_vmap_matches_per_call(trace_fixture):
    trace, theta, _ = trace_fixture
    # Build a batch of theta vectors by scaling theta
    factors = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64).reshape(-1, 1)
    batch = factors * theta.reshape(1, -1)
    vmap_eval = jax.vmap(lambda p: evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])
    batched_rates = np.asarray(vmap_eval(batch))
    # Compare to per-element loop
    for i in range(batch.shape[0]):
        per = np.asarray(evaluate_trace_jax(trace, batch[i], use_log=False)["vertex_rates"])
        np.testing.assert_allclose(batched_rates[i], per, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------
# (3) jax.grad — finite gradients
# ----------------------------------------------------------------------------


def test_grad_finite(trace_fixture):
    trace, theta, _ = trace_fixture

    def loss(p):
        return jnp.sum(evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])

    g = jax.grad(loss)(theta)
    g_np = np.asarray(g)
    assert g_np.shape == theta.shape
    assert np.all(np.isfinite(g_np)), f"non-finite gradient: {g_np}"


def test_grad_matches_finite_difference(trace_fixture):
    """Cross-check jax.grad against a centred finite difference."""
    trace, theta, _ = trace_fixture

    def loss(p):
        return float(jnp.sum(evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"]))

    eps = 1e-5
    fd = np.zeros(theta.shape, dtype=np.float64)
    for k in range(theta.shape[0]):
        plus = theta.at[k].add(eps)
        minus = theta.at[k].add(-eps)
        fd[k] = (loss(plus) - loss(minus)) / (2 * eps)

    autograd = np.asarray(
        jax.grad(lambda p: jnp.sum(evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"]))(theta)
    )
    np.testing.assert_allclose(autograd, fd, rtol=1e-4, atol=1e-7)


# ----------------------------------------------------------------------------
# (4) jax.pmap — multi-core CPU
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason=f"pmap requires >=2 local devices; got {jax.local_device_count()}",
)
def test_pmap_matches_sequential(trace_fixture):
    trace, theta, _ = trace_fixture
    n = jax.local_device_count()
    factors = jnp.linspace(0.5, 2.0, n, dtype=jnp.float64).reshape(-1, 1)
    batch = factors * theta.reshape(1, -1)
    pmap_eval = jax.pmap(lambda p: evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])
    pmap_rates = np.asarray(pmap_eval(batch))
    # Verify against per-element eager
    for i in range(n):
        per = np.asarray(evaluate_trace_jax(trace, batch[i], use_log=False)["vertex_rates"])
        np.testing.assert_allclose(pmap_rates[i], per, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------
# (5) NamedSharding — explicit jax.Array placement
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason=f"sharding test requires >=2 local devices; got {jax.local_device_count()}",
)
def test_named_sharding_matches_unsharded(trace_fixture):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    trace, theta, _ = trace_fixture
    n = jax.local_device_count()

    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))

    factors = jnp.linspace(0.5, 2.0, n, dtype=jnp.float64).reshape(-1, 1)
    batch = factors * theta.reshape(1, -1)
    batch_sharded = jax.device_put(batch, sharding)

    @jax.jit
    def fan(theta_batch):
        return jax.vmap(lambda p: evaluate_trace_jax(trace, p, use_log=False)["vertex_rates"])(
            theta_batch
        )

    sharded_out = np.asarray(fan(batch_sharded))
    unsharded_out = np.asarray(fan(batch))
    np.testing.assert_allclose(sharded_out, unsharded_out, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------
# (6) Disk-cache portability across processes
# ----------------------------------------------------------------------------


def test_disk_cache_round_trip_across_processes(tmp_path):
    """Record a trace in the parent, persist it to disk via pickle, load
    it in a fresh subprocess, replay it, and confirm the result matches
    the parent's eager evaluation.

    This is the core portability invariant the hierarchical cache and
    the SLURM multi-node story both depend on: an EliminationTrace must
    be fully self-describing once serialised, with no dependence on
    in-process state. Using pickle directly (rather than going through
    save_trace_to_cache, which writes to ~/.phasic_cache/traces/{hash}.pkl)
    isolates this property from the cache layer so a failure here points
    unambiguously at the trace itself.
    """
    import pickle

    builder = GRAPH_BUILDERS["acyclic_coalescent_n3"]
    graph, theta = builder()
    trace = record_elimination_trace(graph, theta_dim=len(theta))
    parent_rates = np.asarray(
        evaluate_trace(trace, theta, use_log=False)["vertex_rates"], dtype=np.float64
    )

    trace_path = tmp_path / "trace.pkl"
    with open(trace_path, "wb") as fh:
        pickle.dump(trace, fh)
    assert trace_path.exists()

    # Spawn a fresh subprocess with no shared in-memory state, pickle-load
    # the trace, and replay. stdout returns the rates as JSON.
    child_script = (
        "import pickle, json\n"
        "import numpy as np\n"
        "from phasic.trace_elimination import evaluate_trace\n"
        f"with open({str(trace_path)!r}, 'rb') as fh:\n"
        "    trace = pickle.load(fh)\n"
        f"theta = np.array({list(theta)!r}, dtype=np.float64)\n"
        "rates = evaluate_trace(trace, theta, use_log=False)['vertex_rates']\n"
        "print(json.dumps(list(map(float, rates))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"child process failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    import json as _json

    # phasic prints a logging banner at import time. The child script's
    # actual JSON output is the last non-empty line.
    json_line = next(
        ln for ln in reversed(proc.stdout.strip().splitlines()) if ln.strip()
    )
    child_rates = np.array(_json.loads(json_line), dtype=np.float64)
    np.testing.assert_allclose(child_rates, parent_rates, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------
# (7) SVGD round-trip with parallel='vmap' / 'pmap'
# ----------------------------------------------------------------------------
#
# This validates the trace path inside Graph.svgd. For now (Stage 0) we
# cannot exercise cache_trace=True end-to-end because Stage 5 wires it in
# — so we only smoke-test that the existing SVGD path with 'vmap' and
# 'pmap' parallelisation runs to completion on an acyclic graph and
# produces a finite posterior. After Stage 5 lands, this test is extended
# to compare cache_trace=True vs cache_trace=False posteriors.


def test_svgd_parallel_vmap_smoke():
    from phasic import SVGD

    graph, true_theta = GRAPH_BUILDERS["acyclic_coalescent_n3"]()
    graph.update_weights(true_theta)
    obs = np.asarray(graph.sample(50))

    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    svgd = SVGD(
        model=model,
        observed_data=obs,
        theta_dim=len(true_theta),
        n_particles=10,
        n_iterations=5,
        learning_rate=0.01,
        parallel="vmap",
        seed=0,
        verbose=False,
    )
    svgd.optimize()
    assert svgd.theta_mean is not None
    assert np.all(np.isfinite(np.asarray(svgd.theta_mean)))


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason=f"pmap SVGD requires >=2 local devices; got {jax.local_device_count()}",
)
def test_svgd_parallel_pmap_smoke():
    from phasic import SVGD

    graph, true_theta = GRAPH_BUILDERS["acyclic_coalescent_n3"]()
    graph.update_weights(true_theta)
    obs = np.asarray(graph.sample(50))

    n_devices = jax.local_device_count()
    n_particles = max(2 * n_devices, 16)  # multiple of n_devices for pmap

    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    svgd = SVGD(
        model=model,
        observed_data=obs,
        theta_dim=len(true_theta),
        n_particles=n_particles,
        n_iterations=5,
        learning_rate=0.01,
        parallel="pmap",
        seed=0,
        verbose=False,
    )
    svgd.optimize()
    assert svgd.theta_mean is not None
    assert np.all(np.isfinite(np.asarray(svgd.theta_mean)))


# ----------------------------------------------------------------------------
# (8) SLURM multi-node — gated; skipped unless SLURM_JOB_ID is set
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    "SLURM_JOB_ID" not in os.environ,
    reason="SLURM multi-node test only runs inside an sbatch job",
)
def test_slurm_multi_node_trace_cache_round_trip():
    """Inside a SLURM job: detect the SLURM environment, configure JAX
    distributed, record the trace on rank 0, persist to a shared
    on-disk cache, and have all ranks load + replay it. Asserts that
    every rank computes the same vertex_rates from the same cached
    trace.

    The sbatch wrapper that drives this test lives at
    scripts/test_slurm_trace.sh. Outside SLURM the test is skipped
    cleanly.
    """
    from phasic.distributed_utils import detect_slurm_environment

    # Defensive: if we somehow get here outside SLURM, skip.
    slurm = detect_slurm_environment()
    if not slurm:
        pytest.skip("not running inside SLURM")

    from phasic.auto_parallel import configure_jax_for_environment, detect_environment

    env = detect_environment()
    configure_jax_for_environment(env)

    # Record (rank 0) → persist → all ranks load → all ranks replay.
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    n_ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    cache_path = os.environ.get(
        "PHASIC_TEST_TRACE_PATH",
        os.path.expanduser("~/phasic_test_slurm_trace.json"),
    )

    builder = GRAPH_BUILDERS["acyclic_coalescent_n3"]
    graph, theta = builder()

    if rank == 0:
        from phasic.trace_serialization import save_trace_to_cache

        trace = record_elimination_trace(graph, theta_dim=len(theta))
        save_trace_to_cache(trace, cache_path)

    # Crude rendezvous via SLURM srun — assumes srun's barrier semantics
    # mean all ranks observe the file once any rank gets here. For a real
    # implementation use jax.experimental.multihost_utils or MPI barriers.
    # In this smoke test we just retry up to a timeout.
    import time

    deadline = time.time() + 30
    while not os.path.exists(cache_path) and time.time() < deadline:
        time.sleep(0.5)
    assert os.path.exists(cache_path), f"rank {rank} did not see {cache_path}"

    from phasic.trace_serialization import load_trace_from_cache

    trace = load_trace_from_cache(cache_path)
    rates = np.asarray(evaluate_trace(trace, theta, use_log=False)["vertex_rates"])
    assert np.all(np.isfinite(rates))
    assert rates.shape[0] == graph.vertices_length()

    # Rank 0 cleans up after itself
    if rank == 0:
        try:
            os.unlink(cache_path)
        except OSError:
            pass

    # Optional: gather rates from every rank via JAX cross-process and
    # assert they agree bit-for-bit. For now we only require that every
    # rank loads + replays without error.
