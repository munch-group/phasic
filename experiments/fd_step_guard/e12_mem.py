import resource, time
from phasic import Graph
for rate in (1e6, 5e6, 2e7):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, rate)
    t = time.time(); g.pdf(1.0); dt = time.time()-t
    print(f'rate={rate:g} steps~{2*rate:.2g} time={dt:.2f}s maxRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB', flush=True)
