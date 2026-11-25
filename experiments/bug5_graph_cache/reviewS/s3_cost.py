"""reviewS s3: cost of a straightforward v4.4 sub-digest walk (prototype from plan text,
containers + scalars only) on the test-26 inputs. No phasic import, no files."""
import hashlib, time, resource, sys

class K:
    def __init__(self):
        self.memo = {}      # id -> (obj, digest)
        self.path = set()
        self.steps = 0
    def tok(self, tag, payload: bytes):
        return b"%s:%d:" % (tag, len(payload)) + payload
    def scalar(self, x):
        t = type(x)
        if t is int:   return self.tok(b"i", hex(x).encode())
        if t is float: return self.tok(b"f", x.hex().encode())
        if t is str:   return self.tok(b"s", repr(x).encode())
        if t is bool or x is None: return self.tok(b"c", repr(x).encode())
        return None
    def value(self, x, depth=0):
        s = self.scalar(x)
        if s is not None:
            return s
        t = type(x)
        if t not in (list, tuple, dict):
            raise TypeError(t)
        i = id(x)
        if i in self.path: raise ValueError("cycle")
        m = self.memo.get(i)
        if m is not None: return m[1]
        if depth > 100: raise ValueError("depth")
        self.path.add(i); self.steps += 1
        h = hashlib.sha256()
        if t is dict:
            h.update(b"KIND:dict;" + self.tok(b"n", str(len(x)).encode()))
            for k, v in x.items():
                h.update(self.value(k, depth + 1)); h.update(self.value(v, depth + 1))
        else:
            h.update((b"KIND:list;" if t is list else b"KIND:tuple;") + self.tok(b"n", str(len(x)).encode()))
            for e in x:
                h.update(self.value(e, depth + 1))
        self.path.discard(i)
        d = self.tok(b"D", h.digest())
        self.memo[i] = (x, d)
        return d

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

import random
random.seed(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
L = 20
states = [tuple(random.randrange(0, 6) for _ in range(L)) for _ in range(n)]
d = {s: i for i, s in enumerate(states)}
print(f"built {n} state dict, rss {rss_mb():.0f} MB")
k = K(); t0 = time.perf_counter(); k.value(d); t1 = time.perf_counter()
print(f"600k dict: {t1-t0:.2f} s, steps {k.steps}, memo {len(k.memo)}, rss {rss_mb():.0f} MB")
del k
m = 60_000
keys = states[:m]
adj = {s: [keys[(j * 7919 + q) % m] for q in range(20)] for j, s in enumerate(keys)}
k = K(); t0 = time.perf_counter(); k.value(adj); t1 = time.perf_counter()
print(f"60k x 20 adjacency: {t1-t0:.2f} s, steps {k.steps}, rss {rss_mb():.0f} MB")
