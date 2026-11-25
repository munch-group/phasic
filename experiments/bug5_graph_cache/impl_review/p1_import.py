import time, sys, os
t0 = time.perf_counter()
import phasic
t1 = time.perf_counter()
from phasic import graph_cache as gc
out = {"import_s": round(t1 - t0, 4), "has_cache_key": hasattr(gc, "clear_all_graph_caches") and not hasattr(gc, "is_graph_cache_disabled")}
try:
    from phasic import callback_hash as ch
    if hasattr(ch, "_compute_record"):
        rec = ch._get_record()
        out["record_error"] = rec.get("error") if rec else "MISSING"
        out["n_files"] = len(rec["stats"]) if rec and rec.get("stats") else None
        s0 = {"root": rec["root"], "files": dict(rec["stats"])}
        ts = []
        for _ in range(5):
            a = time.perf_counter(); r = ch._compute_record(s0); ts.append(time.perf_counter() - a)
        out["compute_record_ms_min"] = round(min(ts) * 1e3, 2)
        ts = []
        for _ in range(20):
            a = time.perf_counter(); ch._phasic_build_id(); ts.append(time.perf_counter() - a)
        out["build_id_check_ms_median"] = round(sorted(ts)[10] * 1e3, 3)
except Exception as e:
    out["err"] = repr(e)
print("RESULT", out)
