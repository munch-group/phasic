"""Node budget (10^7 'visited objects') vs a production-scale lookup table
captured by a callback: dict state-tuple -> int for N states of length L.
Minimal walker in the plan's style (tagged, length-prefixed, counted
composites; every visited object counted against the budget)."""
import hashlib, time, sys

BUDGET = 10**7

class Over(Exception):
    pass

def walk(x, h, cnt):
    cnt[0] += 1
    if cnt[0] > BUDGET:
        raise Over(cnt[0])
    t = type(x)
    if t is int:
        p = hex(x).encode(); h.update(b"i:%d:" % len(p)); h.update(p)
    elif t is float:
        p = x.hex().encode(); h.update(b"f:%d:" % len(p)); h.update(p)
    elif t is tuple or t is list:
        h.update(b"T:%d:" % len(x))
        for e in x:
            walk(e, h, cnt)
    elif t is dict:
        h.update(b"D:%d:" % len(x))
        for k, v in x.items():
            walk(k, h, cnt); walk(v, h, cnt)
    else:
        raise TypeError(t)

for N, L in [(60_000, 20), (200_000, 20), (600_000, 12), (600_000, 20)]:
    table = {tuple((i >> j) & 7 for j in range(L)): i for i in range(N)}
    h = hashlib.sha256(); cnt = [0]
    t0 = time.perf_counter()
    try:
        walk(table, h, cnt)
        res = "ok"
    except Over as e:
        res = f"BUDGET EXCEEDED at {e.args[0]:,} nodes"
    print(f"N={N:,} L={L}: nodes={cnt[0]:,} time={time.perf_counter()-t0:.2f}s -> {res}")
    del table
