"""Review A: JSON entry save/load cost vs build at larger sizes (uses master's
_serialize_value/_deserialize_value + from_serialized = the format the plan keeps).
Writes only to a scratch dir given as argv[2]; deletes its file."""
import os, sys, time, json, resource
import numpy as np
assert "scratchpad" in os.environ["HOME"]
from phasic import Graph, with_ipv
from phasic.graph_cache import _serialize_value, _deserialize_value
n = int(sys.argv[1]); out = os.path.join(sys.argv[2], f"entry_{n}.json")
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS bytes -> MB
@with_ipv([n] + [0] * (n - 1))
def coal(state):
    t = []
    for i in range(state.size):
        if state[i] == 0: continue
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t
t0 = time.perf_counter(); g = Graph(coal); tb = time.perf_counter() - t0
print(f"n={n} vertices={g.vertices_length()} build={tb:.2f}s peakRSS={rss():.0f}MB")
t0 = time.perf_counter(); d = g.serialize(); ser = _serialize_value(d)
with open(out, "w") as f: json.dump({"graph_data": ser}, f, indent=2)
ts = time.perf_counter() - t0; size = os.path.getsize(out) / 2**20
del ser
print(f"save={ts:.2f}s size={size:.1f}MB peakRSS={rss():.0f}MB")
t0 = time.perf_counter()
with open(out) as f: e = json.load(f)
g2 = Graph.from_serialized(_deserialize_value(e["graph_data"]))
tl = time.perf_counter() - t0
print(f"load={tl:.2f}s vertices={g2.vertices_length()} peakRSS={rss():.0f}MB")
os.remove(out)
