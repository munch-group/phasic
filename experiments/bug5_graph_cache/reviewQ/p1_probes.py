"""Review Q probes on the prototype (no phasic import)."""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
from subdigest_proto import digest, Walker, Refused


def clique(k, extra_self=False):
    src = ["def factory():"]
    for i in range(k):
        others = [f"f{j}" for j in range(k) if j != i]
        src.append(f"    def f{i}(s, d=0):")
        src.append(f"        return d if d > 3 else ({' + '.join(o + '(s, d+1)' for o in others)})")
    src.append("    return f0")
    ns = {}
    exec("\n".join(src), ns)
    return ns["factory"]()


print("== P1 clique of k mutually recursive closures (each calls all others)")
for k in range(3, 11):
    f = clique(k)
    t = time.perf_counter()
    d, w = digest(f)
    dt = time.perf_counter() - t
    print(f"k={k:2d} composite walks={w.visits:9d} time={dt:8.3f}s digest={d[:12]}")
    if dt > 60:
        break

print("== P2 dispatch-table recursion through a dict in a closure")
def make_dispatch():
    table = {}
    def a(s):
        return table["b"](s)
    def b(s):
        return table["a"](s) if s else 0
    table["a"] = a
    table["b"] = b
    def cb(state):
        return table["a"](state)
    return cb
try:
    print("digest:", digest(make_dispatch())[0][:12])
except Refused as e:
    print("REFUSED:", e)

print("== P3 closed digest depends on walk/memo order (plan: never)")
def make_fg():
    def f(s):
        return g(s)
    def g(s):
        return f(s)
    return f, g
f, g = make_fg()
w1 = Walker(); tg1 = w1.value(g)[0]
w2 = Walker(); w2.value(f); tg2 = w2.memo[id(g)][1] if id(g) in w2.memo else None
# in w2, walking f first: f closed (memoised), g open (not memoised); walk g now
tg2 = w2.value(g)[0]
print("g as root            :", tg1[-8:].hex())
print("g after f was walked :", tg2[-8:].hex(), "equal" if tg1 == tg2 else "DIFFERENT")

print("== P4 same input, same digest across repeated computations")
d1 = digest(make_fg()[0])[0]; d2 = digest(make_fg()[0])[0]
print("fresh closures equal:", d1 == d2)

print("== P5 recursion through a partial (plan: functions only get back-refs)")
import functools
def make_p():
    holder = []
    def f(s, p=None):
        return holder[0](s)
    p = functools.partial(f, p=1)
    holder.append(p)
    return p
try:
    print("partial root digest:", digest(make_p())[0][:12])
except Refused as e:
    print("REFUSED:", e)
