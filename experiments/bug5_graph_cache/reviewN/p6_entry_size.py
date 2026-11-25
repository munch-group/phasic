"""Entry file size and write-side peak memory: master's json.dump(indent=2)
versus compact separators, and dumps()+encode (a natural os.write
implementation) versus streaming json.dump to the fd."""
import os, sys, json, time, tracemalloc, tempfile
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
from phasic import Graph
from phasic.graph_cache import _serialize_value


def cb(state, **kw):
    a, b = int(state[0]), int(state[1])
    out = []
    if a > 0:
        out.append(([a - 1, b + 1], [float(a), 0.0]))
    if b > 0:
        out.append(([a, b - 1], [0.0, float(b)]))
    if a > 1:
        out.append(([a - 2, b], [float(a * (a - 1)), 1.0]))
    return out


g = Graph(cb, ipv=[[[120, 0], 1.0]], graph_cache=False)
print("vertices", g.vertices_length())
entry = {"format": "2", "graph_data": _serialize_value(g.serialize())}

d = tempfile.mkdtemp(dir=os.environ["HOME"])
for label, kw in [("indent=2 (master)", dict(indent=2)), ("compact", dict(separators=(",", ":")))]:
    p = os.path.join(d, "e.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(entry, f, **kw)
    print(f"{label}: {os.path.getsize(p)/1e6:.2f} MB")
    os.remove(p)

for label, fn in [
    ("dumps+encode", lambda: json.dumps(entry, separators=(",", ":")).encode()),
    ("stream dump", lambda: json.dump(entry, open(os.path.join(d, "s.json"), "w", encoding="utf-8"),
                                      separators=(",", ":"))),
]:
    tracemalloc.start()
    fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label}: python-heap peak {peak/1e6:.1f} MB")
for f in os.listdir(d):
    os.remove(os.path.join(d, f))
os.rmdir(d)
