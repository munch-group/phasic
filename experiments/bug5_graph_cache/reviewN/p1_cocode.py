"""Is co_code stable across quickening (adaptive specialisation), sys.settrace
and sys.monitoring instrumentation? If not, the key of the SAME callback
would differ between first and later calls in one process."""
import sys, hashlib

def make():
    def cb(state, rate=1.0):
        out = []
        for i in range(len(state)):
            if state[i] > 0:
                s = list(state); s[i] -= 1
                out.append((s, rate * state[i]))
        return out
    return cb

def h(f):
    c = f.__code__
    return hashlib.sha256(c.co_code + repr(c.co_consts).encode()).hexdigest()[:16]

cb = make()
h0 = h(cb)
for _ in range(20000):
    cb([3, 2, 1, 0])
h1 = h(cb)
print("before/after quickening:", h0, h1, "equal" if h0 == h1 else "DIFFER")

# settrace
def tr(frame, ev, arg):
    return tr
sys.settrace(tr)
cb([1, 1])
h2 = h(cb)
sys.settrace(None)
print("under settrace:", h2, "equal" if h0 == h2 else "DIFFER")

# sys.monitoring (3.12+)
if hasattr(sys, "monitoring"):
    m = sys.monitoring
    TOOL = 3
    m.use_tool_id(TOOL, "probe")
    m.register_callback(TOOL, m.events.LINE, lambda code, line: None)
    m.set_events(TOOL, m.events.LINE | m.events.INSTRUCTION)
    cb([1, 1])
    h3 = h(cb)
    m.set_events(TOOL, 0)
    m.free_tool_id(TOOL)
    print("under sys.monitoring:", h3, "equal" if h0 == h3 else "DIFFER")
print(sys.version)
