"""Summarise s1_*.jsonl: per family x regime x rule, the median relative error
for each s, the best s, and the plan's acceptance pick: the SMALLEST s whose
median is within 2x of the best s on the grid (plan §6 Batch 0)."""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
rows = []
for p in sorted(glob.glob(os.path.join(HERE, "s1_*.jsonl"))):
    rows += [json.loads(l) for l in open(p) if l.strip()]

RULES = ["full", "cap", "ufull", "ucap"]
out = []
fams = sorted({r["family"] for r in rows})
for fam in fams:
    for regime in ("ordinary", "popgen"):
        R = [r for r in rows if r["family"] == fam and r["regime"] == regime]
        if not R:
            continue
        models = sorted({r["model"] for r in R})
        today = np.median([r["today"] for r in R])
        refs = sorted({r["ref"] for r in R})
        spread = max(r["spread"] for r in R)
        line = [f"\n## {fam} / {regime}  (models {', '.join(models)}; n={len(R)} slot-points; "
                f"reference {'+'.join(refs)}; max Richardson spread {spread:.1e}; today's rule median {today:.1e})"]
        for rule in RULES:
            if f"{rule}_{S[0]:g}" not in R[0]:
                continue
            med = [float(np.median([r[f"{rule}_{s:g}"] for r in R])) for s in S]
            mx = [float(np.max([r[f"{rule}_{s:g}"] for r in R])) for s in S]
            best = min(med)
            ok = [s for s, m in zip(S, med) if m <= 2 * best]
            pick = min(ok)
            line.append(f"  {rule:5s} median: " + "  ".join(f"{s:g}:{m:.1e}" for s, m in zip(S, med))
                        + f"\n        max:    " + "  ".join(f"{s:g}:{m:.1e}" for s, m in zip(S, mx))
                        + f"\n        best {best:.1e} at s={S[med.index(best)]:g}; PICK (smallest s within 2x of best) = {pick:g}"
                        + f"; s=1e-4 median {med[S.index(1e-4)]:.1e} ({med[S.index(1e-4)]/best:.1f}x best)")
            out.append(dict(family=fam, regime=regime, rule=rule, median=dict(zip(map(str, S), med)),
                            best=best, pick=pick, s1e4_ratio=med[S.index(1e-4)] / best))
        print("\n".join(line))
# exact-vs-Richardson agreement where both exist
ex = [r for r in rows if "exact_vs_rich" in r]
if ex:
    print("\n## Reference check: exact vs Richardson where both exist")
    for m in sorted({r["model"] for r in ex}):
        v = [r["exact_vs_rich"] for r in ex if r["model"] == m]
        print(f"  {m}: median {np.median(v):.1e}, max {np.max(v):.1e} (n={len(v)})")
json.dump(out, open(os.path.join(HERE, "s2_summary.json"), "w"), indent=1)
