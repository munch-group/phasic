"""reviewAJ p3: is co_code stable across specialization and sys.monitoring instrumentation (3.13)?"""
import sys, hashlib

def cb(state, rate=1.0):
    k = state[0]
    out = []
    for i in range(k):
        if k > 1:
            out.append(([k - 1], k * (k - 1) / 2 * rate))
    return out

def h(f):
    c = f.__code__
    return hashlib.sha256(c.co_code + c.co_exceptiontable).hexdigest()[:16]

h0 = h(cb)
for _ in range(20000):
    cb([5])
h1 = h(cb)
print("before", h0, "after specialization", h1, "equal", h0 == h1)

mon = sys.monitoring
TOOL = 3
mon.use_tool_id(TOOL, "probe")
mon.register_callback(TOOL, mon.events.LINE, lambda code, line: None)
mon.set_events(TOOL, mon.events.LINE | mon.events.INSTRUCTION)
cb([5])
h2 = h(cb)
mon.set_events(TOOL, 0)
mon.free_tool_id(TOOL)
print("under sys.monitoring", h2, "equal", h0 == h2)

# settrace (coverage.py's classic tracer)
sys.settrace(lambda *a: None)
cb([5])
h3 = h(cb)
sys.settrace(None)
print("under settrace", h3, "equal", h0 == h3)
