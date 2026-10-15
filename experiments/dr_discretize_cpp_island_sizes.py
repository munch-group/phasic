import sys, time
sys.path.insert(0, 'tests/pytest')
from test_notebook_model_recovery import two_island_graph
for n in [int(a) for a in sys.argv[1:]]:
    t = time.perf_counter(); g = two_island_graph(n); dt = time.perf_counter() - t
    print(f"two_island n={n}: {g.vertices_length()} vertices, param={g.parameterized()} P={g.param_length()}, built in {dt:.1f}s", flush=True)
