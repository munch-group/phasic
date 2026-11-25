"""Review Q: (1) test-26 feasibility cost of the large plain tables on the
prototype; (2) frozenset-constant seed independence (run under several
PYTHONHASHSEED values by the driver); (3) repeated ndarray references are
not memoised by the plan (cost scales with references, not objects)."""
import sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from subdigest_proto import digest, Walker

mode = sys.argv[1]
if mode == "seed":
    src = '''
def make():
    def cb(state, key=None):
        if key in {("a", 1), ("b", 2), ("c", 3)}:
            if key == ("a", 1):
                return 1
        return 0
    return cb
'''
    ns = {}
    exec(src, ns)
    d1 = digest(ns["make"]())[0]
    ns2 = {}
    exec(src.replace('key == ("a", 1)', 'key == ("c", 3)'), ns2)
    d2 = digest(ns2["make"]())[0]
    print(f"seed={os.environ.get('PYTHONHASHSEED')} d={d1[:16]} edited={d2[:16]} differ={d1 != d2}")
elif mode == "cost":
    import itertools
    t = time.perf_counter()
    big = {(i % 7, i // 7 % 11, i): i for i in range(600_000)}
    w = Walker(); w.value(big)
    print(f"600k dict of 3-tuples: {time.perf_counter() - t:.2f}s visits={w.visits}")
    keys = [(i, i + 1) for i in range(60_000)]
    adj = {k: tuple(keys[(j * 7919 + 13 * n) % 60_000] for n in range(20)) for j, k in enumerate(keys)}
    t = time.perf_counter()
    w = Walker(); w.value(adj)
    print(f"60k x 20 shared-tuple adjacency: {time.perf_counter() - t:.2f}s visits={w.visits}")
    # repeated array references: plan memoises containers/functions only
    import numpy as np
    a = np.zeros(1_000_000)  # 8 MB
    lst = [a] * 200
    t = time.perf_counter()
    h = hashlib.sha256()
    for x in lst:
        h.update(memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8)))
    print(f"200 refs to one 8 MB array, hashed per occurrence: {time.perf_counter() - t:.2f}s (1.6 GB)")
