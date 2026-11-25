"""reviewY y6: is co_code (the plan's CODE FINGERPRINT input) stable under
specialisation, sys.monitoring instrumentation (coverage / debuggers on 3.12+)
and sys.settrace? A difference would make keyed callbacks miss under a
debugger or coverage run (a false miss, never a stale hit)."""
import sys

def cb(state, k=2.0):
    s = 0.0
    for i in range(len(state)):
        s += state[i] * k
    return s

c0 = cb.__code__.co_code
for _ in range(5000):          # warm up: adaptive specialisation
    cb([1, 2, 3])
c1 = cb.__code__.co_code
print("after specialisation equal:", c0 == c1)

mon = sys.monitoring
TOOL = 3
mon.use_tool_id(TOOL, "reviewY")
mon.register_callback(TOOL, mon.events.LINE, lambda code, line: None)
mon.set_local_events(TOOL, cb.__code__, mon.events.LINE)
cb([1, 2, 3])
c2 = cb.__code__.co_code
print("under sys.monitoring LINE events equal:", c0 == c2)
mon.set_local_events(TOOL, cb.__code__, 0)
mon.free_tool_id(TOOL)

sys.settrace(lambda *a: None)
cb([1, 2, 3])
c3 = cb.__code__.co_code
sys.settrace(None)
print("under settrace equal:", c0 == c3)
