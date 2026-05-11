"""Sustained-load script: max out N CPU cores running the
hierarchical SCC composer in a tight loop.

Build a synthetic graph whose condensation contains exactly N
tightly-coupled SCCs at the same level (default N=8), each big
enough that its O(k^3) elimination dominates over the
composer's overhead. Run hierarchical compose in a loop for a
fixed duration. The OpenMP parallel-for fans out across the N
sibling SCCs, so every level-1 compose step keeps all N threads
busy. Watch Activity Monitor / `top` while it's running.

The script reports cpu_time / wall_time at the end. On a fully
saturated 8-core machine you should see something close to 8x.
A value near 1.0 means the parallelism isn't kicking in
(probably OpenMP not linked or PHASIC_MAX_PARALLEL_SCCS=1 set).

Run:
    pixi run python scripts/saturate_elimination.py
    pixi run python scripts/saturate_elimination.py --sccs 8 --secs 30
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time


def cpu_time_self() -> float:
    """User+system CPU time for this process (sums all threads)."""
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def build_parallel_friendly_graph(n_sccs: int, scc_size: int):
    """Hand-built parameterised graph: N parallel cliques each
    feeding a shared sink.

    Layout per SCC: a k-vertex clique (every vertex points to
    every other). One vertex in each clique has an additional
    edge to the global sink (the "exit channel"). The starting
    vertex distributes probability evenly across the N cliques.

    All edges are parameterised so the hierarchical SCC composer
    is actually engaged (the C-side path is gated on
    parameterised=True + non-NULL current_params). Coefficient
    vector is length 2: slot 0 = base rate, slot 1 = unused. The
    caller invokes ``update_weights([1.0, 0.0])`` after
    construction.

    Condensation:
      - N source SCCs (one per clique), each at level 1.
      - 1 sink SCC at level 0.
    Total SCCs = N + 1; widest level has N parallel SCCs.
    """
    import phasic

    g = phasic.Graph(2)  # state_length=2: (clique_id, vertex_id_within_clique)
    start = g.starting_vertex()

    # Build the absorbing sink first so we can reference it.
    # Use a state outside any clique's coordinate range.
    sink = g.find_or_create_vertex([n_sccs, 0])

    # Build N cliques.
    inv_n = 1.0 / n_sccs
    for clique in range(n_sccs):
        verts = [g.find_or_create_vertex([clique, i]) for i in range(scc_size)]
        # Starting vertex -> vertex 0 of this clique (parameterised).
        start.add_edge(verts[0], [inv_n, 0.0])
        # Full clique: every vertex -> every other vertex.
        for i in range(scc_size):
            for j in range(scc_size):
                if i == j:
                    continue
                verts[i].add_edge(verts[j], [1.0, 0.0])
            # Exit channel to the sink.
            verts[i].add_edge(sink, [0.1, 0.0])

    return g


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sccs", type=int, default=8,
                   help="number of parallel SCCs at level 1 (default 8)")
    p.add_argument("--scc-size", type=int, default=30,
                   help="vertices per SCC (default 30)")
    p.add_argument("--secs", type=float, default=30.0,
                   help="duration of the sustained-load loop in seconds "
                        "(default 30)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="override PHASIC_CACHE_DIR (default: a fresh "
                        "temp dir, so we always exercise the elimination "
                        "path on the first call)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Fresh cache dir so the first compose actually eliminates
    # rather than just loading from disk.
    if args.cache_dir is None:
        import tempfile
        cache_dir = tempfile.mkdtemp(prefix="phasic_saturate_")
        cleanup_cache = True
    else:
        cache_dir = args.cache_dir
        cleanup_cache = False
    os.environ["PHASIC_CACHE_DIR"] = cache_dir

    # Always cache every SCC regardless of size — the SCCs here
    # are well above the default threshold but be explicit.
    os.environ["PHASIC_MIN_SCC_SIZE_TO_CACHE"] = "0"

    import phasic
    import phasic.cache as cache

    # Enable the hierarchical SCC composer.
    phasic.configure(parallel_elimination=True)

    n_cpus = os.cpu_count() or 1
    omp = os.environ.get("OMP_NUM_THREADS")
    max_par = os.environ.get("PHASIC_MAX_PARALLEL_SCCS")

    print(f"=== saturation probe ===")
    print(f"  cpu_count()             = {n_cpus}")
    print(f"  OMP_NUM_THREADS         = {omp}")
    print(f"  PHASIC_MAX_PARALLEL_SCCS = {max_par or '(unset)'}")
    print(f"  cache dir               = {cache_dir}")
    print(f"  n SCCs                  = {args.sccs}")
    print(f"  vertices per SCC        = {args.scc_size}")
    print(f"  duration                = {args.secs} s")
    print()

    print("Building graph...")
    t0 = time.perf_counter()
    g = build_parallel_friendly_graph(args.sccs, args.scc_size)
    # Bind concrete theta so the C-side hierarchical path is
    # eligible (it requires current_params != NULL).
    g.update_weights([1.0, 0.0])
    print(f"  {g.vertices_length()} vertices, "
          f"built in {time.perf_counter() - t0:.2f} s")

    # Sanity-check the SCC condensation.
    scc = g.scc_decomposition()
    from phasic.distributed_scc import compute_scc_levels
    levels = compute_scc_levels(scc)
    widest = max(len(lvl) for lvl in levels)
    print(f"  {len(scc)} SCCs across {len(levels)} levels, "
          f"widest row has {widest} parallel SCCs")
    if widest < args.sccs:
        print(f"  WARNING: widest row ({widest}) < requested SCCs "
              f"({args.sccs}); parallelism will be capped at {widest}.")

    # Warm up: a few compose calls to populate the cache and
    # trigger any one-time loader work.
    print("\nWarmup (3 compose calls)...")
    cache.reset_scc_compose_stats()
    for _ in range(3):
        _ = g.expectation()
    warmup_stats = cache.scc_compose_stats()
    print(f"  warmup stats: {warmup_stats}")

    # Sustained loop.
    print(f"\nSustained load for {args.secs} s. Watch Activity Monitor / top now.")
    print(f"  Expected: ~{args.sccs} cores busy. cpu/wall ratio at the end.")
    cache.reset_scc_compose_stats()
    iters = 0
    cpu_before = cpu_time_self()
    wall_before = time.perf_counter()
    end = wall_before + args.secs
    while time.perf_counter() < end:
        _ = g.expectation()
        iters += 1
    wall = time.perf_counter() - wall_before
    cpu = cpu_time_self() - cpu_before
    stats = cache.scc_compose_stats()

    ratio = cpu / wall if wall > 0 else float("nan")
    print()
    print(f"=== results ===")
    print(f"  iterations          : {iters}")
    print(f"  wall time           : {wall:.2f} s")
    print(f"  cpu time (all thr)  : {cpu:.2f} s")
    print(f"  cpu / wall ratio    : {ratio:.2f}")
    print(f"  compose_calls       : {stats['compose_calls']}")
    print(f"  cache_hits          : {stats['cache_hits']}")
    print(f"  cache_misses        : {stats['cache_misses']}")
    print(f"  cache_bypassed      : {stats['cache_bypassed']}")
    print(f"  in-compose time     : {stats['total_compose_ns'] / 1e9:.2f} s "
          f"({100 * stats['total_compose_ns'] / 1e9 / wall:.0f}% of wall)")

    target = min(args.sccs, n_cpus)
    if ratio >= target * 0.5:
        print(f"\n  PASS: cpu/wall {ratio:.2f} >= {target * 0.5:.1f} "
              f"(target {target} cores, ~50%+ utilisation)")
        rc = 0
    else:
        print(f"\n  WARN: cpu/wall {ratio:.2f} < {target * 0.5:.1f} "
              f"(target {target}). Parallelism is under-saturating.")
        print(f"  Check:")
        print(f"   - is OpenMP linked into phasic? "
              f"(rebuild with libomp present)")
        print(f"   - is PHASIC_MAX_PARALLEL_SCCS capping fan-out? "
              f"(currently {max_par or 'unset'})")
        print(f"   - is OMP_NUM_THREADS too low? "
              f"(currently {omp})")
        rc = 1

    if cleanup_cache:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
