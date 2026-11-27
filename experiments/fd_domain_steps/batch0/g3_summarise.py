"""Summarise the refusal grid (g1, g2): per cell, whether the forward refuses a
negative probe (N columns, primal valid), compared with the plan's §4.4
prediction; plus today's gradient outcome per slot value (G columns)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in ("g1_refusal_grid.jsonl", "g2_refusal_joint.jsonl"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        rows += [json.loads(l) for l in open(p) if l.strip()]

PROBES = ["-1e-12", "-1e-9", "-5e-8", "-1e-7"]
VALS = ["1", "5e-8", "0", "1-5e-8"]
mism = []
for r in rows:
    head = f"{r['kind']:8s} {r['mode']:13s} {r['site']:18s} s{r['slot']}"
    if r.get("build", "ok") != "ok" or "P1" not in r:
        print(f"{head} BUILD {r.get('build', '?')}")
        continue
    neg = [r["N" + k] for k in PROBES]
    ref = [x.startswith("RAISE") for x in neg[1:]]   # -1e-9 .. -1e-7
    measured = "refuses" if all(ref) else ("accepts" if not any(ref) else "partial")
    pred = "refuses" if r["pred_refuse"] else "accepts"
    flag = "" if measured == pred else "  <-- MISMATCH"
    if not r["P1"].startswith("ok"):
        flag = "  (primal at 1 invalid)"
    elif measured != pred:
        mism.append((head, measured, pred, neg))
    print(f"{head} measured={measured:8s} pred={pred:8s}{flag}")
    print("     P: " + "  ".join(f"{v}={r['P'+v]}" for v in VALS))
    print("     N: " + "  ".join(f"{k}={x}" for k, x in zip(PROBES, neg)))
    if "G1" in r:
        print("     G: " + "  ".join(f"{v}={r['G'+v]}" for v in VALS))
print("\nMISMATCHES (primal valid at slot = 1):")
for m in mism:
    print("  ", *m)
print(f"{len(mism)} mismatches; {len(rows)} cells")
