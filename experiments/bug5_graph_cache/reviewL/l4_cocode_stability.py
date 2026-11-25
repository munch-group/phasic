"""reviewL l4: is co_code / co_consts stable under specialization (warm-up)
and under sys.monitoring instrumentation (debuggers, coverage on 3.12+)?"""
import sys, hashlib


def cb(state, rate=2.0):
    n = state[0]
    out = []
    for k in range(n):
        out.append(([n - 1], rate * n * (n - 1) / 2 + k))
    return out


def fp(f):
    c = f.__code__
    return hashlib.sha256(c.co_code + repr(c.co_consts).encode()
                          + getattr(c, "co_exceptiontable", b"")).hexdigest()[:16]

cold = fp(cb)
for _ in range(20000):
    cb([5])
warm = fp(cb)
print("cold == warm:", cold == warm)

mon = sys.monitoring
TOOL = 3
mon.use_tool_id(TOOL, "probe")
E = mon.events
mon.set_events(TOOL, E.LINE | E.PY_START | E.BRANCH | E.JUMP)
mon.register_callback(TOOL, E.LINE, lambda code, line: None)
mon.register_callback(TOOL, E.PY_START, lambda code, off: None)
cb([5])
instr = fp(cb)
print("cold == instrumented (sys.monitoring LINE/BRANCH):", cold == instr)
mon.set_events(TOOL, 0)
mon.free_tool_id(TOOL)

sys.settrace(lambda *a: None)
cb([5])
traced = fp(cb)
sys.settrace(None)
print("cold == settrace:", cold == traced)
