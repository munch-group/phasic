"""Where does each shipped exact-gradient family break, relative to the
project's REAL scale targets (20k-60k vertices typical; ~200k-600k
required)?

Motivation: D1-E0 measured the monolithic exact-MOMENTS gradient at
96.5 s / 3.7 GB on an 8,407-vertex two-locus graph, and a 59,522-vertex
cell consumed ~50 GB before the session was killed. That is BELOW and
ACROSS the typical range respectively. But the two B3 gradient families
scale differently and must not be lumped together:

  * moments  (`ptd_moments_grad_theta`)          REVERSE-mode over the
    whole monolithic elimination tape -- cost/memory tied to tape
    length L, which is what blew up;
  * sojourn  (`ptd_sojourn_grad_theta_subset`)   FORWARD-mode over
    `theta_dim` parameters -- explicitly designed for large
    joint-probability graphs (b3-joint-index-plan.md sizes them at
    n~7e5).

This harness measures both against the same ladder, so the honest
statement "family X is usable to n~Y" can be made per family.

MEMORY SAFETY (mandatory -- a subprocess TIME-BOX does not bound
memory, which is exactly how the 50 GB run-away outran everything):
  (a) child sets RLIMIT_DATA and RLIMIT_AS before importing phasic;
  (b) the parent polls the child's RSS and SIGKILLs it at a threshold;
  (c) a kill is recorded as a MEMORY-WALL DATA POINT (that IS the
      measurement), and the remaining ladder is ABORTED for that
      family -- never escalate past a wall;
  (d) staged sizes, smallest first.

Usage:
    python dr_scale_gradient_ceiling.py                  # parent driver
    python dr_scale_gradient_ceiling.py --child <n> <op> # internal
"""
import argparse
import json
import os
import subprocess
import sys
import time

MEM_CAP_GB = 6.0          # child hard limit (16 GB machine)
WATCHDOG_GB = 6.5         # parent kills above this RSS
POLL_S = 0.5
TIME_BOX_S = 900.0
LADDER = [6, 8, 9, 10]    # two-locus nr_samples; vertices 1.0k/8.4k/22.7k/59.5k
OPS = ["moments_grad", "sojourn_grad"]

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------- child
def run_child(n: int, op: str):
    import resource
    cap = int(MEM_CAP_GB * (1 << 30))
    for which in ("RLIMIT_DATA", "RLIMIT_AS"):
        r = getattr(resource, which, None)
        if r is None:
            continue
        try:
            soft, hard = resource.getrlimit(r)
            lim = cap if hard in (resource.RLIM_INFINITY,) else min(cap, hard)
            resource.setrlimit(r, (lim, hard))
        except (ValueError, OSError):
            pass   # best-effort; the parent watchdog is the real defence

    sys.path.insert(0, HERE)
    from dr_d1_e0_scale import build_two_locus, THETA_TWOLOCUS
    from phasic import set_log_level
    set_log_level("ERROR")
    import numpy as np

    t0 = time.perf_counter()
    g = build_two_locus(n)
    t_build = time.perf_counter() - t0
    nv = g.vertices_length()
    g.update_weights(list(THETA_TWOLOCUS))

    t1 = time.perf_counter()
    if op == "moments_grad":
        J = np.asarray(g._moments_grad_theta(1))
        ok = J.size > 0
        detail = f"J.size={J.size}"
    elif op == "sojourn_grad":
        # a small index subset -- the production joint-index shape
        idx = list(range(min(4, nv)))
        J = g._sojourn_grad_theta_subset(idx, skip_condition_gate=True)
        ok = len(J) > 0
        detail = f"len={len(J)}"
    else:
        raise SystemExit(f"unknown op {op}")
    t_op = time.perf_counter() - t1

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    print("RESULT " + json.dumps(dict(
        n=n, op=op, vertices=nv, build_s=round(t_build, 2),
        op_s=round(t_op, 2), peak_rss_gb=round(rss_mb / 1024, 2),
        accepted=bool(ok), detail=detail)))


# -------------------------------------------------------------- parent
def run_cell(n: int, op: str):
    import psutil
    cmd = [sys.executable, os.path.abspath(__file__), "--child", str(n), op]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    proc = psutil.Process(p.pid)
    peak = 0.0
    t0 = time.time()
    killed = None
    while p.poll() is None:
        try:
            rss = proc.memory_info().rss / (1 << 30)
            peak = max(peak, rss)
            for c in proc.children(recursive=True):
                try:
                    peak = max(peak, rss + c.memory_info().rss / (1 << 30))
                except psutil.Error:
                    pass
            if peak > WATCHDOG_GB:
                p.kill(); killed = "MEMORY-WALL"; break
        except psutil.Error:
            break
        if time.time() - t0 > TIME_BOX_S:
            p.kill(); killed = "TIME-WALL"; break
        time.sleep(POLL_S)
    out, _ = p.communicate()
    if killed:
        return dict(n=n, op=op, wall=killed, peak_rss_gb=round(peak, 2),
                    elapsed_s=round(time.time() - t0, 1))
    for line in (out or "").splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[len("RESULT "):])
            r["wall"] = None
            r["observed_peak_gb"] = round(peak, 2)
            return r
    return dict(n=n, op=op, wall="NO-RESULT (child exited without output; "
                                 "likely the RLIMIT MemoryError)",
                peak_rss_gb=round(peak, 2),
                elapsed_s=round(time.time() - t0, 1))


def main():
    print("== gradient-family scale ceiling (memory-capped) ==")
    print(f"   child RLIMIT {MEM_CAP_GB} GB | watchdog {WATCHDOG_GB} GB | "
          f"time-box {TIME_BOX_S:.0f}s | ladder {LADDER}")
    print("   TARGETS: 20k-60k vertices TYPICAL; ~200k-600k required\n")
    rows = []
    for op in OPS:
        print(f"-- {op} --")
        for n in LADDER:
            r = run_cell(n, op)
            rows.append(r)
            if r.get("wall"):
                print(f"   nr={n}: *** {r['wall']} *** peak "
                      f"{r.get('peak_rss_gb')} GB after "
                      f"{r.get('elapsed_s')}s -- ABORTING this family's ladder")
                break
            print(f"   nr={n}: vertices={r['vertices']:>7} "
                  f"build={r['build_s']:>7.2f}s op={r['op_s']:>8.2f}s "
                  f"peak={r['peak_rss_gb']:>5.2f} GB accepted={r['accepted']}")
        print()
    with open(os.path.join(HERE, "dr_scale_ceiling_results.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote dr_scale_ceiling_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("N", "OP"))
    a = ap.parse_args()
    if a.child:
        run_child(int(a.child[0]), a.child[1])
    else:
        main()
