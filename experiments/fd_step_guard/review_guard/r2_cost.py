"""Cost per step of pdf vs vertices and edges: chain (E/V=1) vs fan (E/V=k).
Also: incremental pdf calls (ascending times) vs one call."""
import time, numpy as np
from phasic import Graph
def chain(V, k, rate):
    g = Graph(1)
    vs = [g.find_or_create_vertex([i+1]) for i in range(V)]
    g.starting_vertex().add_edge(vs[0], 1.0)
    for i in range(V-1):
        # k edges forward (to i+1..i+k, clipped), total exit rate = rate
        tgts = [min(i+j, V-1) for j in range(1, k+1)]
        tgts = sorted(set(tgts))
        for t in tgts:
            vs[i].add_edge(vs[t], rate/len(tgts))
    return g, sum(len(v.edges()) for v in [g.vertex_at(i) for i in range(g.vertices_length())])
for V, k in [(4000, 1), (4000, 4), (4000, 16), (1000, 16)]:
    rate = 5e4  # granularity 1e5
    g, E = chain(V, k, rate)
    t = 0.5
    t0 = time.time(); g.pdf(t); dt = time.time() - t0
    steps = 2*rate*t
    print(f'V={V} k={k} E={E}: {dt:.3f}s  steps={steps:.3g}  ns/edge-step={dt/(steps*E)*1e9:.2f}  ns/vertex-step={dt/(steps*V)*1e9:.2f}', flush=True)
# incremental: ascending times, each call small
g, E = chain(1000, 4, 5e4)
ts = np.linspace(0.005, 0.5, 100)
t0 = time.time(); g.pdf(ts); dt = time.time() - t0
print(f'vectorized ascending 100 times to 0.5: {dt:.3f}s total; per-call steps_to_go ~{2*5e4*0.005:.3g}, total steps {2*5e4*0.5:.3g}')
