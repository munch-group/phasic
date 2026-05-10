"""Standalone check that the SCC composer actually uses
multiple CPU cores during elimination.

Strategy: measure user+system CPU time (resource.getrusage)
against wall time during a hierarchical-elimination call. If
elimination is truly parallel, CPU time / wall time should be
substantially > 1 (close to the number of threads OpenMP is
using, in the limit). If parallelism is broken, the ratio
will be ~1 regardless of thread count.

Uses the two-locus ARG from the caching tutorial because its
SCC condensation has many parallel-eliminable SCCs at the
same level (~120 SCCs total for nr_samples=6), which actually
exercises the OpenMP fan-out.

Run:
    pixi run python scripts/check_omp_parallelism.py

You can override the OpenMP thread count for the test via the
usual env var, e.g.:
    OMP_NUM_THREADS=1 pixi run python scripts/check_omp_parallelism.py
    OMP_NUM_THREADS=8 pixi run python scripts/check_omp_parallelism.py
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import tempfile
import time


def cpu_time_used() -> float:
    """Return user+system CPU time consumed by the current
    process AND all its children, in seconds. Includes time
    spent inside OpenMP worker threads."""
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime + self_usage.ru_stime
        + children_usage.ru_utime + children_usage.ru_stime
    )


def build_graph():
    """Construct a two-locus ARG with enough SCCs to exercise
    OpenMP fan-out."""
    import phasic
    from phasic import Graph, Property, StateIndexer

    nr_samples = 6
    indexer = StateIndexer(descendants=[
        Property("loc1", max_value=nr_samples),
        Property("loc2", max_value=nr_samples),
    ])
    initial = [0] * indexer.state_length
    initial[indexer.props_to_index(loc1=1, loc2=1)] = nr_samples

    def two_locus_arg(state, indexer=None):
        transitions = []
        if state.sum() <= 1:
            return transitions
        for i in range(indexer.state_length):
            if state[i] == 0:
                continue
            pi = indexer.index_to_props(i)
            for j in range(i, indexer.state_length):
                if state[j] == 0:
                    continue
                pj = indexer.index_to_props(j)
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                child = state.copy()
                child[i] -= 1
                child[j] -= 1
                loc1 = pi.descendants.loc1 + pj.descendants.loc1
                loc2 = pi.descendants.loc2 + pj.descendants.loc2
                if loc1 <= nr_samples and loc2 <= nr_samples:
                    child[indexer.props_to_index(loc1=loc1, loc2=loc2)] += 1
                    transitions.append([
                        child,
                        [state[i] * (state[j] - same) / (1 + same), 0],
                    ])
            if state[i] > 0 and pi.descendants.loc1 > 0 and pi.descendants.loc2 > 0:
                child = state.copy()
                child[i] -= 1
                child[indexer.props_to_index(loc1=pi.descendants.loc1, loc2=0)] += 1
                child[indexer.props_to_index(loc1=0, loc2=pi.descendants.loc2)] += 1
                transitions.append([child, [0, 1]])
        return transitions

    return Graph(two_locus_arg, ipv=initial, indexer=indexer)


def time_elimination(graph, theta) -> tuple[float, float]:
    """Run a fresh hierarchical-compose call and return
    (wall_seconds, cpu_seconds)."""
    import phasic.cache as cache

    graph.update_weights(theta)

    cpu_before = cpu_time_used()
    wall_before = time.perf_counter()
    _ = graph.expectation()
    wall_after = time.perf_counter()
    cpu_after = cpu_time_used()

    return (wall_after - wall_before, cpu_after - cpu_before)


def main() -> int:
    # Use a fresh temporary cache dir so we always exercise the
    # elimination path, not just disk-cache loads.
    tmp = tempfile.mkdtemp(prefix="phasic_omp_check_")
    os.environ["PHASIC_CACHE_DIR"] = tmp
    # Cache every SCC regardless of size so we observe the full
    # parallel elimination, not bypass for small SCCs.
    os.environ["PHASIC_MIN_SCC_SIZE_TO_CACHE"] = "0"

    try:
        import phasic
        import phasic.cache as cache
        phasic.configure(hierar_elimination=True)

        nthreads_env = os.environ.get("OMP_NUM_THREADS", "(unset)")
        print(f"OMP_NUM_THREADS = {nthreads_env}")
        print(f"os.cpu_count()  = {os.cpu_count()}")
        print(f"PHASIC_CACHE_DIR = {tmp}")
        print()

        print("Building two-locus ARG graph...")
        t0 = time.perf_counter()
        graph = build_graph()
        print(f"  built {graph.vertices_length()} vertices in {time.perf_counter() - t0:.2f}s")

        # First call (cold cache, real elimination work).
        cache.reset_scc_compose_stats()
        wall, cpu = time_elimination(graph, [2.0, 5.0])
        stats = cache.scc_compose_stats()
        ratio = cpu / wall if wall > 0 else float("nan")
        print()
        print(f"=== Cold elimination (cache misses) ===")
        print(f"  wall:                {wall:.3f} s")
        print(f"  cpu (user+sys):      {cpu:.3f} s")
        print(f"  cpu / wall ratio:    {ratio:.2f}")
        print(f"  cache_misses:        {stats['cache_misses']}")
        print(f"  cache_bypassed:      {stats['cache_bypassed']}")
        print(f"  compose total time:  {stats['total_compose_ns'] / 1e9:.3f} s")

        # Second call (warm cache, mostly load + replay).
        cache.reset_scc_compose_stats()
        wall2, cpu2 = time_elimination(graph, [3.0, 7.0])
        stats2 = cache.scc_compose_stats()
        ratio2 = cpu2 / wall2 if wall2 > 0 else float("nan")
        print()
        print(f"=== Warm elimination (cache hits) ===")
        print(f"  wall:                {wall2:.3f} s")
        print(f"  cpu (user+sys):      {cpu2:.3f} s")
        print(f"  cpu / wall ratio:    {ratio2:.2f}")
        print(f"  cache_hits:          {stats2['cache_hits']}")
        print(f"  cache_bypassed:      {stats2['cache_bypassed']}")

        # Verdict.
        #
        # We test the WARM path (cache hits) for the parallelism
        # gate because:
        # - Cold runs are dominated by serial disk writes
        #   (atomic write-then-rename, one .bin file per SCC),
        #   which mask the in-memory compute parallelism.
        # - Warm runs read the cached PRCs and do the actual
        #   parallel multiply-accumulate sweep — cleanly
        #   exercising the parallel-for.
        print()
        print(f"=== Verdict ===")
        threshold = 1.5
        if ratio2 >= threshold:
            print(f"  PASS: warm-elimination cpu/wall = {ratio2:.2f} >= {threshold}")
            print(f"        multiple cores were used during SCC composition.")
            print(f"        (cold-elimination ratio {ratio:.2f} is lower because "
                  f"serial disk I/O on the first miss masks parallelism.)")
            return 0
        else:
            print(f"  FAIL: warm-elimination cpu/wall = {ratio2:.2f} < {threshold}")
            print(f"        elimination appears to be running on a single core.")
            print()
            print(f"  Things to check:")
            print(f"  - Was the build linked against OpenMP_C "
                  f"(not just OpenMP_CXX)? Look for")
            print(f"    '-Xclang -fopenmp' or '-fopenmp' on the "
                  f"scc_compose.c compile line.")
            print(f"  - Is OMP_NUM_THREADS sensible? "
                  f"(current: {nthreads_env})")
            print(f"  - Is PHASIC_MAX_PARALLEL_SCCS set to 1? "
                  f"(current: {os.environ.get('PHASIC_MAX_PARALLEL_SCCS', '(unset)')})")
            print(f"  - For a small model, elimination may be too "
                  f"fast for the parallel-for to pay off; "
                  f"try a larger nr_samples.")
            return 1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
