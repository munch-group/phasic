"""D1-E0 — scale/necessity measurement for Deferred Unit 1 (hierarchical/SCC adjoint).

Answers activation gate A1 of deferred-1-hierarchical-scc-adjoint-plan.md (§3/§5):
"is hierarchical mode ever REQUIRED (OOM/timeout) vs merely faster, and at what
n — SEPARATELY for the primal and the gradient pipelines".

Fixtures (all weight_mode='linear', §1 scope):
  - the 6 toy_model.BUILDERS fixtures (tests/pytest/toy_model.py:332-339)
  - a production-scale two-locus ARG (coalescent class) built via
    StateIndexer + callback (same model as
    tests/pytest/test_scc_parallelism_smoke.py::parallel_friendly_graph),
    scaled by nr_samples: 6 -> 1_044, 8 -> 8_407, 9 -> 22_653,
    10 -> 59_522, 11 -> ~150k vertices.

Measured, per fixture/size (each measurement in a FRESH subprocess so peak RSS
is per-configuration, not cumulative):
  (a) primal — monolithic expected_waiting_time() vs hierarchical
      (PHASIC_HIERAR_ELIMINATION=1, routed through ptd_compose_scc_prcs;
      verified via phasic.cache.scc_compose_stats): wall-clock + peak RSS.
  (b) the shipped exact-gradient pipeline — Graph._moments_grad_theta(1)
      (the pmf_and_moments_from_graph exact path's C core; builds a PRIVATE
      tape per call + O(L) snapshots, phasic.c ~10740ff): wall-clock +
      peak RSS + declined-or-answered (MPFR conditioning gate).
Config matrix pinned per cell: PHASIC_DYN_ORDERING (unset/on),
Stage-A2 on-disk cache (off / cold / warm; PHASIC_REWARD_COMPUTE_CACHE +
PHASIC_CACHE_DIR pointing into the private workdir — ~/.phasic_cache is never
touched), OMP_NUM_THREADS (auto / 1).
Also recorded per graph: tape length L (parameterized reward-compute command
count, via _save_param_compute_graph header parse) and the SCC count/size
distribution (scc_decomposition) — feeds E1's cost model + Deferred-4's
L-statistics (plan §5-E0, §7).

Usage:
  # orchestrator (spawns one subprocess per measurement):
  pixi run python experiments/dr_d1_e0_scale.py --workdir /tmp/d1e0 \
      [--sizes 6,8,9,10] [--stretch] [--skip-toys]
  # a single measurement cell (spawned by the orchestrator):
  pixi run python experiments/dr_d1_e0_scale.py --cell primal --fixture twolocus:10 \
      --workdir /tmp/d1e0

Every cell prints one machine-readable line:  D1E0_RESULT: {json}
The per-measurement time-box is OP_TIMEOUT_S (120 s) enforced by the
orchestrator's subprocess timeout (a signal.alarm cannot interrupt a
GIL-released C call); phase markers on stdout attribute a timeout to
build vs measured-op.

Branch-only de-risk artifact (feedback_derisk_and_reevaluate); writes nothing
into the repo and modifies nothing in src/.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import struct
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

OP_TIMEOUT_S = 120          # per single measured operation (plan E0 time-box)
CELL_GRACE_S = 120          # import + fixture load/build + serialize slack
META_EXTRA_S = 240          # meta cell additionally saves the tape to disk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY_DIR = os.path.join(REPO_ROOT, "tests", "pytest")

THETA_TOY = [0.5, 2.0, 1.0, 0.7]      # toy_model.THETAS['mixed']
THETA_TWOLOCUS = [2.0, 5.0]           # test_scc_parallelism_smoke.THETA

TOY_NAMES = ["toy_base", "toy_a", "toy_b", "toy_c_p", "toy_c_pprime", "toy_d"]


def _rss_mb() -> float:
    """Peak RSS of this process in MB (ru_maxrss is bytes on darwin, KB on linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def _marker(name: str) -> None:
    print(f"D1E0_PHASE: {name}", flush=True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def build_two_locus(nr_samples: int):
    """Two-locus ARG (coalescence + recombination), StateIndexer callback build.

    Verbatim model from tests/pytest/test_scc_parallelism_smoke.py
    (parallel_friendly_graph), parameterised by nr_samples.
    theta = [coalescence rate, recombination rate]  (P = 2, linear mode).
    """
    from phasic import Graph, Property, StateIndexer

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


def _npz_path(workdir: str, n: int) -> str:
    return os.path.join(workdir, f"d1e0_twolocus_n{n}.npz")


def _npz_scalar(a):
    """np.savez round-trip: unwrap 0-d arrays (ints stay int, strings stay str)."""
    if a.ndim != 0:
        return a
    v = a.item()
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def load_fixture(fixture: str, workdir: str):
    """Return (graph, theta, load_seconds, source_str)."""
    from phasic import Graph

    t0 = time.perf_counter()
    if fixture.startswith("twolocus:"):
        n = int(fixture.split(":")[1])
        npz = _npz_path(workdir, n)
        if os.path.exists(npz):
            import numpy as np
            raw = np.load(npz)
            data = {k: _npz_scalar(raw[k]) for k in raw.files}
            g = Graph.from_serialized(data)
            src = "from_serialized(npz)"
        else:
            g = build_two_locus(n)
            src = "callback build"
        return g, THETA_TWOLOCUS, time.perf_counter() - t0, src

    if fixture in TOY_NAMES:
        sys.path.insert(0, TOY_DIR)
        import toy_model
        g = toy_model.BUILDERS[fixture]()
        return g, THETA_TOY, time.perf_counter() - t0, "toy builder"

    raise SystemExit(f"unknown fixture {fixture!r}")


# ---------------------------------------------------------------------------
# tape length L (parameterized reward-compute command count)
# ---------------------------------------------------------------------------

def read_tape_length(g, workdir: str):
    """Save the populated parameterized_reward_compute_graph and parse
    commands_length (uint64 at byte 24) from the ptd_pcg_disk_header
    (phasic.c:3071-3080: magic[8], u32 version, u32 format_revision,
    u64 graph_hash, u64 commands_length, ...). Caller must have run a
    primal first so the tape exists. Returns (L, file_MB) or (None, reason).
    """
    path = os.path.join(workdir, f"d1e0_tape_{os.getpid()}.bin")
    try:
        g._save_param_compute_graph(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        with open(path, "rb") as fh:
            header = fh.read(32)
        magic = header[:8]
        if not magic.startswith(b"PTDPRMC"):
            return None, f"bad magic {magic!r}"
        (L,) = struct.unpack_from("<Q", header, 24)
        return int(L), size_mb
    except Exception as exc:  # noqa: BLE001 — record, don't crash the cell
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def cell_main(args) -> None:
    _marker("import_start")
    import numpy as np  # noqa: F401
    import phasic
    import phasic.cache as cache
    _marker("import_done")

    g, theta, load_s, src = load_fixture(args.fixture, args.workdir)
    _marker("build_done")
    g.update_weights(theta)
    _marker("weights_done")
    rss_after_build = _rss_mb()

    row = {
        "fixture": args.fixture,
        "cell": args.cell,
        "env": {k: os.environ.get(k) for k in (
            "PHASIC_HIERAR_ELIMINATION", "PHASIC_DYN_ORDERING",
            "PHASIC_REWARD_COMPUTE_CACHE", "PHASIC_CACHE_DIR",
            "PHASIC_MIN_SCC_SIZE_TO_CACHE", "OMP_NUM_THREADS")},
        "n_vertices": g.vertices_length(),
        "load_s": round(load_s, 3),
        "load_src": src,
        "rss_after_build_mb": round(rss_after_build, 1),
        "status": "ok",
    }

    if args.cell == "primal":
        cache.reset_scc_compose_stats()
        _marker("op_start")
        t0 = time.perf_counter()
        res = g.expected_waiting_time()
        t1 = time.perf_counter()
        _marker("op_done")
        # second call in the same process = in-memory-warm replay
        t2 = time.perf_counter()
        _ = g.expected_waiting_time()
        t3 = time.perf_counter()
        stats = cache.scc_compose_stats()
        arr = np.asarray(res, dtype=float).ravel()
        row.update({
            "op_s": round(t1 - t0, 3),
            "op2_s": round(t3 - t2, 3),
            "ewt_start": float(arr[0]),
            "ewt_sum": float(np.nansum(arr)),
            "compose_calls": stats.get("compose_calls"),
            "scc_cache_hits": stats.get("cache_hits"),
            "scc_cache_misses": stats.get("cache_misses"),
            "scc_cache_bypassed": stats.get("cache_bypassed"),
        })

    elif args.cell == "grad":
        _marker("op_start")
        t0 = time.perf_counter()
        jac = g._moments_grad_theta(1)
        t1 = time.perf_counter()
        _marker("op_done")
        declined = len(jac) == 0
        row.update({
            "op_s": round(t1 - t0, 3),
            "grad_declined": declined,
            "jac": [float(x) for x in jac][:8],
        })
        if t1 - t0 < 10.0:  # cheap enough: measure a second private-tape call
            t2 = time.perf_counter()
            _ = g._moments_grad_theta(1)
            row["op2_s"] = round(time.perf_counter() - t2, 3)

    elif args.cell == "meta":
        _marker("op_start")
        t0 = time.perf_counter()
        scc = g.scc_decomposition()
        sizes = sorted((int(s) for s in scc.scc_sizes()), reverse=True)
        t_scc = time.perf_counter() - t0
        n1 = sum(1 for s in sizes if s == 1)
        # populate the tape (monolithic primal), then read L from the header
        t0 = time.perf_counter()
        _ = g.expected_waiting_time()
        t_prime = time.perf_counter() - t0
        t0 = time.perf_counter()
        L, extra = read_tape_length(g, args.workdir)
        t_save = time.perf_counter() - t0
        _marker("op_done")
        row.update({
            "scc_s": round(t_scc, 3),
            "n_sccs": len(sizes),
            "scc_singletons": n1,
            "scc_top5": sizes[:5],
            "prime_s": round(t_prime, 3),
            "tape_L": L,
            "tape_note": (f"{extra:.1f} MB on disk" if L is not None else str(extra)),
            "tape_save_s": round(t_save, 3),
        })

    else:
        raise SystemExit(f"unknown cell {args.cell!r}")

    row["rss_peak_mb"] = round(_rss_mb(), 1)
    print("D1E0_RESULT: " + json.dumps(row), flush=True)


# ---------------------------------------------------------------------------
# prep: build the two-locus npz fixtures once (also round-trip validates)
# ---------------------------------------------------------------------------

def prep_main(args) -> None:
    import numpy as np
    from phasic import Graph

    for n in args.size_list:
        npz = _npz_path(args.workdir, n)
        if os.path.exists(npz):
            print(f"prep: n={n} npz exists, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        g = build_two_locus(n)
        t_build = time.perf_counter() - t0
        data = g.serialize(theta_dim=len(THETA_TWOLOCUS))
        np.savez(npz, **data)
        nv = g.vertices_length()
        ne = (len(data["edges"]) + len(data["start_edges"])
              + len(data["param_edges"]) + len(data["start_param_edges"]))
        print(f"prep: n={n} vertices={nv} edges={ne} build={t_build:.1f}s -> {npz}",
              flush=True)
        if n == args.size_list[0]:
            # round-trip gate: reloaded graph must reproduce E[T] to 1e-12
            raw = np.load(npz)
            d2 = {k: _npz_scalar(raw[k]) for k in raw.files}
            g2 = Graph.from_serialized(d2)
            g.update_weights(THETA_TWOLOCUS)
            g2.update_weights(THETA_TWOLOCUS)
            a = np.asarray(g.expected_waiting_time(), dtype=float).ravel()
            b = np.asarray(g2.expected_waiting_time(), dtype=float).ravel()
            np.testing.assert_allclose(a, b, rtol=1e-12)
            print(f"prep: n={n} serialize round-trip E[T] parity OK "
                  f"(max rel dev {np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)):.2e})",
                  flush=True)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def _spawn(fixture: str, cell: str, workdir: str, env_over: dict, tag: str,
           timeout_s: float):
    env = dict(os.environ)
    # scrub every knob we pin, then apply the cell's overrides
    for k in ("PHASIC_HIERAR_ELIMINATION", "PHASIC_DYN_ORDERING",
              "PHASIC_REWARD_COMPUTE_CACHE", "PHASIC_CACHE_DIR",
              "PHASIC_MIN_SCC_SIZE_TO_CACHE", "PHASIC_MAX_PARALLEL_SCCS",
              "PHASIC_DISABLE_GRAPH_CACHE", "OMP_NUM_THREADS"):
        env.pop(k, None)
    env.update(env_over)
    cmd = [sys.executable, os.path.abspath(__file__),
           "--cell", cell, "--fixture", fixture, "--workdir", workdir]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env, cwd=REPO_ROOT)
        out = proc.stdout or ""
        for line in out.splitlines():
            if line.startswith("D1E0_RESULT: "):
                row = json.loads(line[len("D1E0_RESULT: "):])
                row["tag"] = tag
                row["cell_wall_s"] = round(time.perf_counter() - t0, 1)
                return row
        return {"fixture": fixture, "cell": cell, "tag": tag, "status": "error",
                "note": (proc.stderr or "")[-800:] or out[-800:]}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        phases = [l.split(": ", 1)[1] for l in out.splitlines()
                  if l.startswith("D1E0_PHASE: ")]
        where = phases[-1] if phases else "startup"
        return {"fixture": fixture, "cell": cell, "tag": tag,
                "status": f"timeout>{int(timeout_s)}s", "note": f"last phase: {where}"}


def orchestrate(args) -> None:
    os.makedirs(args.workdir, exist_ok=True)
    results = []

    def run(fixture, cell, tag, env_over, timeout_s=None):
        if timeout_s is None:
            timeout_s = OP_TIMEOUT_S + CELL_GRACE_S + (META_EXTRA_S if cell == "meta" else 0)
        row = _spawn(fixture, cell, args.workdir, env_over, tag, timeout_s)
        results.append(row)
        brief = {k: row.get(k) for k in
                 ("status", "op_s", "op2_s", "rss_peak_mb", "n_sccs", "tape_L",
                  "compose_calls", "grad_declined", "note") if row.get(k) is not None}
        print(f"[{fixture:>14s} | {tag:<22s}] {brief}", flush=True)
        return row

    def cache_dir(name):
        d = os.path.join(args.workdir, "cache", name)
        os.makedirs(d, exist_ok=True)
        return d

    E_MONO = {}
    E_HIER = {"PHASIC_HIERAR_ELIMINATION": "1"}

    # -- prep the two-locus npz fixtures (single subprocess, its own budget) --
    # PHASIC_DISABLE_GRAPH_CACHE=1: the callback build must not read from or
    # write to the user's ~/.phasic_cache/graphs (honest build cost + hygiene).
    prep_env = dict(os.environ, PHASIC_DISABLE_GRAPH_CACHE="1",
                    PHASIC_CACHE_DIR=os.path.join(args.workdir, "cache", "prep"))
    if args.size_list:
        cmd = [sys.executable, os.path.abspath(__file__), "--prep",
               "--workdir", args.workdir,
               "--sizes", ",".join(str(s) for s in args.size_list)]
        print("prep: building two-locus fixtures (one-time)...", flush=True)
        proc = subprocess.run(cmd, text=True, cwd=REPO_ROOT, env=prep_env)
        if proc.returncode != 0:
            raise SystemExit("prep failed")

    # -- toys ---------------------------------------------------------------
    if not args.skip_toys:
        for name in TOY_NAMES:
            run(name, "meta", "meta", E_MONO)
            run(name, "primal", "mono", E_MONO)
            run(name, "primal", "hier", E_HIER)
            run(name, "grad", "grad-exact", E_MONO)
        # Stage-A2 wiring check on one toy: cold then warm, same cache dir
        d = cache_dir("toy_base")
        e = dict(E_HIER, PHASIC_REWARD_COMPUTE_CACHE="1",
                 PHASIC_MIN_SCC_SIZE_TO_CACHE="0", PHASIC_CACHE_DIR=d)
        run("toy_base", "primal", "hier-cacheA2-cold", e)
        run("toy_base", "primal", "hier-cacheA2-warm", e)

    # -- two-locus scaling ladder -------------------------------------------
    for n in args.size_list:
        fx = f"twolocus:{n}"
        run(fx, "meta", "meta", E_MONO)
        run(fx, "primal", "mono", E_MONO)
        run(fx, "primal", "hier", E_HIER)
        d = cache_dir(f"n{n}")
        e = dict(E_HIER, PHASIC_REWARD_COMPUTE_CACHE="1",
                 PHASIC_MIN_SCC_SIZE_TO_CACHE="0", PHASIC_CACHE_DIR=d)
        run(fx, "primal", "hier-cacheA2-cold", e)
        run(fx, "primal", "hier-cacheA2-warm", e)
        run(fx, "grad", "grad-exact", E_MONO)

    # -- config-matrix spot checks at the largest ladder size ---------------
    if args.size_list:
        n = args.size_list[-1]
        fx = f"twolocus:{n}"
        run(fx, "primal", "mono-dyn", dict(E_MONO, PHASIC_DYN_ORDERING="1"))
        run(fx, "primal", "hier-dyn", dict(E_HIER, PHASIC_DYN_ORDERING="1"))
        run(fx, "primal", "hier-omp1", dict(E_HIER, OMP_NUM_THREADS="1"))
        run(fx, "grad", "grad-exact-dyn", dict(E_MONO, PHASIC_DYN_ORDERING="1"))
        d = cache_dir(f"n{n}-mono")
        e = dict(E_MONO, PHASIC_REWARD_COMPUTE_CACHE="1",
                 PHASIC_MIN_SCC_SIZE_TO_CACHE="0", PHASIC_CACHE_DIR=d)
        run(fx, "primal", "mono-cacheA2-cold", e)
        run(fx, "primal", "mono-cacheA2-warm", e)

    # -- stretch size: meta + mono + hier only, relaxed build budget --------
    if args.stretch:
        n = args.stretch
        fx = f"twolocus:{n}"
        cmd = [sys.executable, os.path.abspath(__file__), "--prep",
               "--workdir", args.workdir, "--sizes", str(n)]
        print(f"prep: building stretch fixture n={n} ...", flush=True)
        subprocess.run(cmd, text=True, cwd=REPO_ROOT, env=prep_env)
        big = OP_TIMEOUT_S + 3 * CELL_GRACE_S
        run(fx, "meta", "meta", E_MONO, timeout_s=big + META_EXTRA_S)
        run(fx, "primal", "mono", E_MONO, timeout_s=big)
        run(fx, "primal", "hier", E_HIER, timeout_s=big)
        run(fx, "grad", "grad-exact", E_MONO, timeout_s=big)

    out = os.path.join(args.workdir, "d1e0_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {len(results)} rows -> {out}", flush=True)

    # compact table
    print("\nfixture              tag                    status        op_s     op2_s   rssMB")
    for r in results:
        print(f"{r['fixture']:<20s} {r.get('tag', ''):<22s} {r.get('status', ''):<12s}"
              f" {str(r.get('op_s', '')):>8s} {str(r.get('op2_s', '')):>8s}"
              f" {str(r.get('rss_peak_mb', '')):>7s}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", choices=["primal", "grad", "meta"])
    ap.add_argument("--fixture")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--sizes", default="6,8,9,10",
                    help="two-locus nr_samples ladder ('' = none)")
    ap.add_argument("--stretch", type=int, default=0,
                    help="extra nr_samples run with a relaxed budget (0 = off)")
    ap.add_argument("--skip-toys", action="store_true")
    args = ap.parse_args()
    args.size_list = [int(s) for s in args.sizes.split(",") if s.strip()]

    if args.cell:
        if not args.fixture:
            raise SystemExit("--cell requires --fixture")
        cell_main(args)
    elif args.prep:
        prep_main(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
