"""reviewR p4: is co_code (the plan's CODE FINGERPRINT) stable when the code
has been specialised by running, or instrumented by sys.settrace /
sys.monitoring (debuggers, coverage)?  A difference would only cause misses."""
import sys, hashlib

def cb(state, rate=1.0):
    n = state[0]
    out = []
    for k in range(n):
        out.append(([n - 1], rate * n * (n - 1) / 2 + k))
    return out

h = lambda: hashlib.sha256(cb.__code__.co_code).hexdigest()[:16]
print("fresh             ", h())
for i in range(5000):
    cb([5])
print("after 5000 calls  ", h())

def tracer(frame, event, arg):
    return tracer
sys.settrace(tracer)
cb([5])
print("under settrace    ", h())
sys.settrace(None)

mon = sys.monitoring
mon.use_tool_id(3, "probe")
mon.set_local_events(3, cb.__code__, mon.events.LINE | mon.events.INSTRUCTION)
mon.register_callback(3, mon.events.LINE, lambda *a: None)
mon.register_callback(3, mon.events.INSTRUCTION, lambda *a: None)
cb([5])
print("under monitoring  ", h())
