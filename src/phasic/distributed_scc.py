"""Distributed SCC computation utilities.

Provides serialisation, deserialisation, and worker-side cache
population for per-SCC PRCs. The full distributed pipeline:

1. Orchestrator computes the parent graph's SCC condensation,
   walks SCCs in level-set order (sink-first), and identifies
   missing cache entries (SLURM-WP-2).
2. For each missing SCC, orchestrator serialises just the
   synthetic SCC graph (NOT the full parent) via
   :func:`serialize_scc_synth` (this module, SLURM-WP-4).
3. Worker scripts (SLURM-WP-3) load the serialised synth,
   reconstruct it, and call :func:`cache_synth_prc` which uses
   the shared on-disk cache (SLURM-WP-1's PHASIC_CACHE_DIR
   override).
4. After all workers complete, the orchestrator runs local
   composition — every SCC is now in the cache and composition
   is fast (SLURM-WP-6).

Workers and orchestrator must agree on PHASIC_CACHE_DIR so they
share cache entries. On a typical SLURM cluster, this points to
a shared filesystem location.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def _serialize_pybind_graph(g) -> dict[str, Any]:
    """Serialise a raw pybind Graph (no Python wrapper).

    The Python ``Graph`` class's ``serialize`` method reads
    ``self._weight_mode`` which only the wrapper sets. Pybind
    objects don't allow dynamic attributes, so we cannot stub it
    on ``g`` directly. Use a small shim that forwards attribute
    access to ``g`` and provides ``_weight_mode``.

    Synthetic SCC graphs built by ``ptd_scc_build_synthetic_graph``
    are always linear-mode at the C level — the constructor does
    not propagate the parent's weight mode (and the synth's
    edges all use linear coefficients regardless).
    """
    from phasic import Graph

    class _Shim:
        _weight_mode = "linear"

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    shim = _Shim(g)
    return Graph.serialize(shim)


def serialize_scc_synth(scc_vertex) -> dict[str, Any]:
    """Serialise the synthetic graph of a single SCC.

    The result is a plain dict containing only Python primitives
    and lists — JSON-serialisable, suitable for writing to a work
    unit file the worker reads.

    Parameters
    ----------
    scc_vertex
        An SCCVertex from ``Graph.scc_decomposition()[i]``.

    Returns
    -------
    dict
        Contains everything needed to reconstruct the synthetic
        SCC graph in a worker process. Specifically:

        - ``graph``: serialised graph dict (from
          :meth:`Graph.serialize`), with arrays converted to
          plain lists for JSON friendliness.
        - ``scc_index``: original index in the parent's
          condensation, for orchestrator bookkeeping.

    Notes
    -----
    The serialisation does NOT include parent-graph context —
    composition is always done locally on the orchestrator after
    workers populate the cache. Workers compute only the per-SCC
    PRC artefact.
    """
    synth, _meta = scc_vertex.as_synthetic_graph()
    serialised = _serialize_pybind_graph(synth)

    # Convert numpy arrays to plain lists for JSON.
    graph_dict = {}
    for k, v in serialised.items():
        if isinstance(v, np.ndarray):
            graph_dict[k] = v.tolist()
        else:
            graph_dict[k] = v

    return {
        "graph": graph_dict,
        "scc_index": int(scc_vertex.index()),
    }


def deserialize_scc_synth(data: dict[str, Any]):
    """Reconstruct a synthetic SCC graph from a work-unit dict.

    Inverse of :func:`serialize_scc_synth` — the worker calls
    this on the dict it loaded from disk.

    Parameters
    ----------
    data : dict
        The dict returned by :func:`serialize_scc_synth`.

    Returns
    -------
    Graph
        The reconstructed synthetic SCC graph. Suitable for
        passing to :func:`cache_synth_prc`.

    Notes
    -----
    Synthetic SCC graphs frequently contain auxiliary vertices
    with the same state (e.g. all zero-state phantom/per-channel
    absorbing vertices). The standard ``Graph.from_serialized``
    uses ``find_or_create_vertex`` and would collapse these into
    a single vertex, which breaks the topology. This deserialiser
    uses ``create_vertex`` to preserve every vertex distinctly,
    matching the synth's original layout.
    """
    from phasic import Graph

    graph_data = data["graph"]
    states = np.asarray(graph_data["states"], dtype=np.int32)
    n_vertices = int(graph_data["n_vertices"])
    state_length = int(graph_data["state_length"])
    param_length = int(graph_data["param_length"])

    edges_arr = np.asarray(graph_data.get("edges", []), dtype=np.float64)
    start_edges_arr = np.asarray(
            graph_data.get("start_edges", []), dtype=np.float64)
    param_edges_arr = np.asarray(
            graph_data.get("param_edges", []), dtype=np.float64)
    start_param_edges_arr = np.asarray(
            graph_data.get("start_param_edges", []), dtype=np.float64)

    g = Graph(state_length)
    if param_length > 0:
        g.set_param_length(param_length)

    # Always create vertices via create_vertex so duplicates
    # (which exist among synth aux vertices) don't collapse.
    # The serialised vertex at index 0 IS the starting vertex
    # (Graph.serialize includes it in vertices()), so we map
    # serialised index 0 to g.starting_vertex() and create the
    # rest via create_vertex.
    idx_to_vertex = {0: g.starting_vertex()}
    for i in range(1, n_vertices):
        state = states[i].tolist()
        v = g.create_vertex(state)
        idx_to_vertex[i] = v

    # Add parameterised edges between vertices.
    if param_edges_arr.size > 0:
        for row in param_edges_arr:
            from_idx = int(row[0])
            to_idx = int(row[1])
            coeffs = row[2:].tolist()
            idx_to_vertex[from_idx].add_edge(idx_to_vertex[to_idx], coeffs)

    # Add regular (non-parameterised) edges between vertices. In
    # parameterised graphs these are weight=constant edges with
    # coefficient vector = [weight, 0, 0, ..., 0] under linear
    # mode. Just use add_edge with a scalar weight.
    if edges_arr.size > 0:
        for row in edges_arr:
            from_idx = int(row[0])
            to_idx = int(row[1])
            weight = float(row[2])
            idx_to_vertex[from_idx].add_edge(idx_to_vertex[to_idx], weight)

    # Starting-vertex edges. Graph.serialize puts every start
    # edge in start_edges (with scalar weight), even when the
    # graph is parameterised — there is a hardcoded
    # ``if False:`` branch in Graph.serialize that skips
    # extracting parameterised start edges. To preserve the
    # original synth's hash (which is sensitive to edge
    # coefficient structure), we must re-emit start edges as
    # parameterised when the graph has param_length > 0,
    # padding the coefficient vector with zeros so the linear
    # weight (Σ c_k θ_k) at θ=[1,1,...,1] equals the original
    # scalar weight.
    start = g.starting_vertex()
    if start_edges_arr.size > 0:
        for row in start_edges_arr:
            to_idx = int(row[0])
            weight = float(row[1])
            if param_length > 0:
                # Encode as [weight, 0, 0, ..., 0] so the linear
                # mode at any θ multiplies the first coefficient
                # by θ[0] only. This matches what the C-level
                # synth builder does for Type A / Type C / phantom
                # edges starting from the synthetic source.
                coeffs = [weight] + [0.0] * (param_length - 1)
                start.add_edge(idx_to_vertex[to_idx], coeffs)
            else:
                start.add_edge(idx_to_vertex[to_idx], weight)
    if start_param_edges_arr.size > 0:
        for row in start_param_edges_arr:
            to_idx = int(row[0])
            coeffs = row[1:].tolist()
            start.add_edge(idx_to_vertex[to_idx], coeffs)

    return g


def cache_synth_prc(synth) -> None:
    """Compute and cache the PRC for a synthetic SCC graph.

    Calls into the SLURM-WP-4 C helper
    ``ptd_synth_get_or_compute_prc``: builds the cache path from
    the synth's content hash, loads if cached, otherwise runs
    the parameterised eliminator and writes the result to disk.

    Parameters
    ----------
    synth
        A synthetic SCC graph (typically from
        :func:`deserialize_scc_synth`).

    Notes
    -----
    The side effect is the cache file write. The function
    returns nothing because workers don't consume the PRC —
    they just populate the shared cache.

    Cache location respects ``PHASIC_CACHE_DIR``
    (SLURM-WP-1). Set it on workers and the orchestrator alike.
    """
    from phasic.phasic_pybind import _synth_get_or_compute_prc
    _synth_get_or_compute_prc(synth)


def write_work_unit(scc_vertex, path: str) -> str:
    """Serialise an SCC's synth and write it to a JSON file.

    Parameters
    ----------
    scc_vertex
        An SCCVertex.
    path : str
        Path to write the JSON file. Parent directory must exist.

    Returns
    -------
    str
        The path written to (same as input, for chaining).
    """
    data = serialize_scc_synth(scc_vertex)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def read_work_unit(path: str) -> dict[str, Any]:
    """Load a work-unit JSON file written by :func:`write_work_unit`.

    Parameters
    ----------
    path : str
        Path to the work-unit JSON.

    Returns
    -------
    dict
        Dict suitable for passing to :func:`deserialize_scc_synth`.
    """
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------
# SLURM-WP-2: orchestrator — level-set computation
# ---------------------------------------------------------------


def compute_scc_levels(scc_graph) -> list[list[int]]:
    """Group SCCs by sink-first level (longest path to a sink).

    SCCs at the same level are mutually independent — they have
    no edges to each other in the condensation — so they can be
    computed in parallel. Level 0 contains pure sinks (SCCs with
    no outgoing edges), level 1 contains SCCs whose only
    descendants are at level 0, and so on.

    This mirrors the level computation inside the C composer
    (``ptd_compose_scc_prcs``, WP-6) so the orchestrator's view
    of parallelisable groups matches what the composer actually
    schedules.

    Parameters
    ----------
    scc_graph
        An SCCGraph from ``Graph.scc_decomposition()``.

    Returns
    -------
    list of list of int
        ``levels[l]`` is the list of SCC indices at level ``l``.
        Levels are returned in compute order (level 0 first), so
        a worker pool can iterate over the outer list and submit
        each inner list as a parallel batch.

    Examples
    --------
    >>> scc_decomp = graph.scc_decomposition()
    >>> levels = compute_scc_levels(scc_decomp)
    >>> for level_set in levels:
    ...     # All SCCs in level_set are independent.
    ...     submit_workers_for(level_set)
    """
    n = len(scc_graph)
    # outgoing[i] = list of SCC indices reachable from SCC i.
    outgoing = [list(scc_graph.scc_at(i).outgoing_scc_edges())
                for i in range(n)]

    # level_of[i] = 1 + max(level_of[j] for j in outgoing[i]),
    # or 0 if outgoing is empty. Compute via DFS / memoisation.
    level_of: list[int] = [-1] * n

    def _level(i: int) -> int:
        if level_of[i] >= 0:
            return level_of[i]
        if not outgoing[i]:
            level_of[i] = 0
            return 0
        level_of[i] = 1 + max(_level(j) for j in outgoing[i])
        return level_of[i]

    for i in range(n):
        _level(i)

    max_level = max(level_of) if level_of else 0
    levels: list[list[int]] = [[] for _ in range(max_level + 1)]
    for i, lev in enumerate(level_of):
        levels[lev].append(i)
    return levels


def scc_cache_path_for_synth(synth) -> str:
    """Return the on-disk cache path for a synthetic SCC graph.

    Mirrors the C-level ``ptd_scc_build_cache_path``: the
    filename is ``scc_<content_hash>.bin`` under
    ``<cache_root>/parameterized_reward_compute/``, where
    ``<cache_root>`` honours ``PHASIC_CACHE_DIR``.

    Used by orchestrators to check whether a worker needs to
    run for a given SCC, without going through the C cache
    layer.

    Parameters
    ----------
    synth
        A synthetic SCC graph (from
        ``SCCVertex.as_synthetic_graph()``).

    Returns
    -------
    str
        Absolute path to the cache file (which may or may not
        exist on disk).
    """
    from phasic.cache import _param_compute_cache_dir
    from phasic.phasic_pybind import hash as _hash_mod
    h = _hash_mod.compute_graph_hash(synth)
    return str(_param_compute_cache_dir() / f"scc_{h.hash_hex}.bin")


def find_missing_sccs(scc_graph) -> list[int]:
    """Return the list of SCC indices whose PRC is not cached.

    For each SCC in the condensation, builds the synthetic graph
    and checks whether
    ``<cache_root>/parameterized_reward_compute/scc_<hash>.bin``
    exists. SCCs whose cache file is present are omitted.

    The orchestrator uses this to decide which work units to
    submit to workers — already-cached SCCs are free to compose
    without further work.

    Parameters
    ----------
    scc_graph
        An SCCGraph from ``Graph.scc_decomposition()``.

    Returns
    -------
    list of int
        Indices of SCCs that need (re)computation. Order matches
        the SCCGraph's natural index order.
    """
    import os
    missing = []
    for i in range(len(scc_graph)):
        scc = scc_graph.scc_at(i)
        synth, _meta = scc.as_synthetic_graph()
        path = scc_cache_path_for_synth(synth)
        if not os.path.exists(path):
            missing.append(i)
    return missing


# ---------------------------------------------------------------
# SLURM-WP-5: sbatch job-script generator + work-unit submission
# ---------------------------------------------------------------


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/scc_worker_%A_%a.out
#SBATCH --error={log_dir}/scc_worker_%A_%a.err
#SBATCH --array=0-{max_idx}{concurrency_clause}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
{extra_directives}

set -eu

# Pass cache dir through so workers see the same shared cache.
export PHASIC_CACHE_DIR={cache_dir}

# Each task picks its work unit by SLURM_ARRAY_TASK_ID.
WORK_UNITS=({work_units_quoted})
WORK_UNIT="${{WORK_UNITS[$SLURM_ARRAY_TASK_ID]}}"

{python_exe} -m phasic.scc_worker "$WORK_UNIT"
"""


def generate_sbatch_script(
        work_units: list[str],
        cache_dir: str,
        log_dir: str,
        job_name: str = "phasic_scc",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus: int = 1,
        max_concurrent: int | None = None,
        extra_directives: str = "",
        python_exe: str = "python") -> str:
    """Generate the sbatch array job script for a level-set.

    The orchestrator submits one such script per level-set.
    SLURM runs all array tasks in parallel (subject to scheduler
    limits and the optional ``max_concurrent`` cap), each task
    invoking ``python -m phasic.scc_worker`` on one work unit.

    Parameters
    ----------
    work_units : list of str
        Paths to work-unit JSON files. Length determines the
        SLURM array size. Paths must be visible from the
        compute nodes (typically on a shared filesystem).
    cache_dir : str
        Path the workers should use as ``PHASIC_CACHE_DIR``.
        Must be a shared-filesystem location all workers (and
        the orchestrator) can read and write.
    log_dir : str
        Directory for sbatch stdout/stderr files. Will be
        created by SLURM if not present.
    job_name : str
        SLURM job name (visible in ``squeue``). Default
        ``"phasic_scc"``.
    time : str
        Per-task time limit, in SLURM format (``H:MM:SS`` or
        ``D-HH:MM:SS``). Default 1 hour.
    mem : str
        Per-task memory request (e.g. ``"4G"``). Default 4 GB.
    cpus : int
        ``--cpus-per-task``. Default 1. Set higher if your
        per-SCC eliminations benefit from OpenMP parallelism.
    max_concurrent : int or None
        Cap on simultaneously running array tasks (SLURM
        ``%`` syntax: ``0-N%K``). ``None`` = no cap.
    extra_directives : str
        Free-form additional ``#SBATCH`` lines (e.g. partition,
        account). Each must include the leading ``#SBATCH``.
    python_exe : str
        Python interpreter to invoke. Default ``"python"``;
        useful when the compute node's PATH differs from the
        login node's.

    Returns
    -------
    str
        The sbatch script as a single string.
    """
    if not work_units:
        raise ValueError(
                "generate_sbatch_script: work_units cannot be empty")

    # Quote each work-unit path safely for bash array literal.
    quoted = " ".join(f'"{p}"' for p in work_units)
    concurrency_clause = (
            f"%{max_concurrent}" if max_concurrent is not None else "")

    return SBATCH_TEMPLATE.format(
            job_name=job_name,
            max_idx=len(work_units) - 1,
            concurrency_clause=concurrency_clause,
            time=time,
            mem=mem,
            cpus=cpus,
            extra_directives=extra_directives,
            cache_dir=cache_dir,
            log_dir=log_dir,
            work_units_quoted=quoted,
            python_exe=python_exe)


def write_work_units_for_level(
        scc_graph,
        scc_indices: list[int],
        work_dir: str) -> list[str]:
    """Write one work-unit file per SCC in a level-set.

    Parameters
    ----------
    scc_graph
        SCCGraph from ``Graph.scc_decomposition()``.
    scc_indices : list of int
        SCC indices to write (typically one level from
        :func:`plan_distributed_work`).
    work_dir : str
        Directory where work-unit JSON files are written.
        Must exist (caller's responsibility).

    Returns
    -------
    list of str
        Paths to the written work-unit files, in the same
        order as ``scc_indices``.
    """
    paths = []
    for i in scc_indices:
        scc = scc_graph.scc_at(i)
        path = os.path.join(work_dir, f"scc_{i}.json")
        write_work_unit(scc, path)
        paths.append(path)
    return paths


def submit_sbatch(
        script_path: str,
        sbatch_exe: str = "sbatch") -> str:
    """Submit an sbatch script and return the job ID.

    Parameters
    ----------
    script_path : str
        Path to the script written by
        :func:`generate_sbatch_script`.
    sbatch_exe : str
        sbatch executable. Default ``"sbatch"``.

    Returns
    -------
    str
        SLURM job ID parsed from sbatch's output. Caller can
        poll via ``squeue`` or wait via ``wait_for_jobs``.

    Raises
    ------
    RuntimeError
        If sbatch is not on PATH or returns non-zero.
    """
    import subprocess
    try:
        result = subprocess.run(
                [sbatch_exe, script_path],
                check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
                f"sbatch executable {sbatch_exe!r} not found on PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
                f"sbatch failed (exit {e.returncode}):\n"
                f"  stdout: {e.stdout}\n"
                f"  stderr: {e.stderr}") from e

    # sbatch prints "Submitted batch job <id>".
    out = result.stdout.strip()
    parts = out.split()
    if not parts or not parts[-1].isdigit():
        raise RuntimeError(
                f"Could not parse job ID from sbatch output: {out!r}")
    return parts[-1]


def run_workers_locally(
        work_units: list[str],
        max_workers: int = 1) -> list[int]:
    """Run workers in local subprocesses (no SLURM).

    Test/dev fallback that mirrors the SLURM array submission:
    each work unit is processed by ``python -m phasic.scc_worker``
    in a subprocess. ``max_workers`` parallel processes run
    concurrently (using ``concurrent.futures``).

    Parameters
    ----------
    work_units : list of str
        Paths to work-unit JSON files.
    max_workers : int
        Number of parallel subprocesses. Default 1 (sequential).

    Returns
    -------
    list of int
        Exit codes per work unit, in the same order.
    """
    import concurrent.futures
    import subprocess
    import sys as _sys

    def _run_one(wu: str) -> int:
        result = subprocess.run(
                [_sys.executable, "-m", "phasic.scc_worker", wu],
                env=os.environ.copy(),
                capture_output=True,
                text=True)
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_run_one, work_units))


# ---------------------------------------------------------------
# SLURM-WP-6: top-level distributed-aware precompute wrapper
# ---------------------------------------------------------------


def precompute_distributed(
        graph,
        work_dir: str,
        cache_dir: str | None = None,
        slurm_options: dict[str, Any] | None = None,
        local_max_workers: int = 1,
        wait_for_jobs: bool = True,
        log_dir: str | None = None) -> dict[str, Any]:
    """Populate the per-SCC PRC cache for a parameterised graph.

    Plans level-sets, ships missing-SCC work units to either a
    SLURM cluster (when ``slurm_options`` is provided) or local
    subprocesses (default), and waits for completion so the
    caller can immediately run hierarchical compose against a
    fully-populated cache.

    The function runs entirely on the orchestrator. The
    distributed work is the cache population — composition
    itself is fast and stays local.

    Parameters
    ----------
    graph
        A parameterised :class:`Graph`. The function calls
        :meth:`Graph.scc_decomposition` on it.
    work_dir : str
        Directory where work-unit JSON files are written. Must
        be on a shared filesystem if SLURM is used. Will be
        created if missing.
    cache_dir : str, optional
        Directory used as ``PHASIC_CACHE_DIR`` for workers and
        the orchestrator. If ``None``, the current process's
        existing ``PHASIC_CACHE_DIR`` (or ``$HOME/.phasic_cache``)
        is used. The directory will be created if missing.
    slurm_options : dict, optional
        When provided, work units are submitted via sbatch.
        Recognised keys: ``time``, ``mem``, ``cpus``,
        ``max_concurrent``, ``extra_directives``, ``python_exe``,
        ``sbatch_exe``. Unrecognised keys are passed through to
        :func:`generate_sbatch_script`.
        When ``None``, workers run in local subprocesses.
    local_max_workers : int
        Concurrency for the local fallback. Ignored when
        ``slurm_options`` is set. Default 1.
    wait_for_jobs : bool
        When ``True`` and using SLURM, the function does not
        return until all submitted jobs finish. Implementation
        polls ``squeue``. When ``False``, the function returns
        immediately after submission, leaving the caller to
        wait however they want.
    log_dir : str, optional
        Directory for sbatch stdout/stderr files. Defaults to
        ``<work_dir>/logs``. Ignored for the local fallback.

    Returns
    -------
    dict
        ``{'levels': L, 'work_units_per_level': [N0, N1, ...],
        'job_ids': [...], 'mode': 'slurm' or 'local'}``.

    Notes
    -----
    The current implementation submits one sbatch array job per
    level-set, waiting for each to complete before the next is
    submitted. This is the simplest correct schedule and avoids
    the orchestrator needing to know about SLURM job
    dependencies. For very wide graphs you can chain level-sets
    via ``--dependency=afterok:$prev_jobid`` for a small win;
    that's a future optimisation.
    """
    if log_dir is None:
        log_dir = os.path.join(work_dir, "logs")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["PHASIC_CACHE_DIR"] = cache_dir
    effective_cache = os.environ.get(
            "PHASIC_CACHE_DIR",
            os.path.expanduser("~/.phasic_cache"))

    scc_graph = graph.scc_decomposition()
    plan = plan_distributed_work(scc_graph, only_missing=True)
    n_per_level = [len(lvl) for lvl in plan]

    job_ids: list[str] = []
    mode = "slurm" if slurm_options is not None else "local"

    for level_idx, scc_indices in enumerate(plan):
        if not scc_indices:
            continue
        level_work_dir = os.path.join(
                work_dir, f"level_{level_idx}")
        os.makedirs(level_work_dir, exist_ok=True)

        wu_paths = write_work_units_for_level(
                scc_graph, scc_indices, level_work_dir)

        if slurm_options is not None:
            opts = dict(slurm_options)
            sbatch_exe = opts.pop("sbatch_exe", "sbatch")
            script = generate_sbatch_script(
                    work_units=wu_paths,
                    cache_dir=effective_cache,
                    log_dir=log_dir,
                    job_name=f"phasic_scc_l{level_idx}",
                    **opts)
            script_path = os.path.join(
                    level_work_dir, "submit.sh")
            with open(script_path, "w") as f:
                f.write(script)
            os.chmod(script_path, 0o755)

            job_id = submit_sbatch(script_path, sbatch_exe=sbatch_exe)
            job_ids.append(job_id)

            if wait_for_jobs:
                _wait_for_slurm_job(job_id)
        else:
            rcs = run_workers_locally(
                    wu_paths, max_workers=local_max_workers)
            failures = [(wu, rc) for wu, rc in zip(wu_paths, rcs) if rc != 0]
            if failures:
                raise RuntimeError(
                        f"Local workers failed at level {level_idx}: "
                        f"{failures}")

    return {
        "mode": mode,
        "levels": len(plan),
        "work_units_per_level": n_per_level,
        "job_ids": job_ids,
    }


def _wait_for_slurm_job(job_id: str, poll_seconds: float = 5.0) -> None:
    """Poll squeue until the given job ID is no longer queued."""
    import subprocess
    import time

    while True:
        result = subprocess.run(
                ["squeue", "-h", "-j", job_id],
                capture_output=True, text=True)
        if result.returncode != 0:
            # squeue returns non-zero when job ID is unknown
            # (= already finished). Treat as "done".
            return
        if not result.stdout.strip():
            return
        time.sleep(poll_seconds)


def plan_distributed_work(
        scc_graph,
        only_missing: bool = True) -> list[list[int]]:
    """Plan distributed work as a list of level-sets to submit.

    Combines :func:`compute_scc_levels` and
    :func:`find_missing_sccs`: returns the level-sets from the
    SCC condensation, with already-cached SCCs filtered out
    when ``only_missing=True`` (the default).

    The orchestrator iterates over the returned levels and
    submits one batch of workers per level. SCCs in earlier
    levels (closer to sinks) MUST complete before SCCs in later
    levels are submitted — though within a level, all SCCs are
    independent.

    Parameters
    ----------
    scc_graph
        An SCCGraph.
    only_missing
        When ``True`` (default), drop SCCs that already have a
        cache entry. When ``False``, return all levels including
        cached SCCs (useful for unconditional recomputation).

    Returns
    -------
    list of list of int
        ``plan[l]`` is the list of SCC indices to submit at
        level ``l``. May be empty for levels where all SCCs are
        cached.
    """
    levels = compute_scc_levels(scc_graph)
    if not only_missing:
        return levels
    missing = set(find_missing_sccs(scc_graph))
    return [[i for i in lvl if i in missing] for lvl in levels]
