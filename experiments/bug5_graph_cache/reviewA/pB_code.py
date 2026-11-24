"""Review A: co_code stability under whitespace edits, tracing, monitoring, specialization."""
import sys
def mk(src, name="cb"):
    ns = {}; exec(compile(src, "<m>", "exec"), ns); return ns[name]
def fp(f):
    c = f.__code__
    return (c.co_code, c.co_consts, c.co_names, c.co_exceptiontable)

a = mk("def cb(s):\n    x = 1 + 2\n    return [s, x]\n")
b = mk("def cb(s):\n    x = (1 +\n         2)\n    return [s, x]\n")
c = mk("def cb(s):\n    # comment\n\n    x = 1 + 2   # trailing\n    return [s, x]\n")
d = mk("def cb(s):\n    if s:\n        pass\n    return [s]\n")
e = mk("def cb(s):\n    if s: pass\n    return [s]\n")
print("[1] single-line vs constant split over two lines: same fp =", fp(a) == fp(b))
print("    co_code a =", a.__code__.co_code.hex()); print("    co_code b =", b.__code__.co_code.hex())
print("[2] comment/blank/trailing-comment edit: same fp =", fp(a) == fp(c))
print("[3] 'if s:\\n pass' vs 'if s: pass': same fp =", fp(d) == fp(e))

f = mk("def cb(s):\n    t = 0\n    for i in range(s):\n        t += i * 2.0\n    return t\n")
base = fp(f)
for _ in range(5000): f(10)
print("[4] after 5000 calls (specialization): same fp =", fp(f) == base)
def tr(frame, ev, arg): return tr
sys.settrace(tr); f(3); same_trace = fp(f) == base; sys.settrace(None)
print("[5] under sys.settrace: same fp =", same_trace)
mon = sys.monitoring
mon.use_tool_id(3, "probe")
mon.set_events(3, mon.events.LINE | mon.events.INSTRUCTION)
mon.register_callback(3, mon.events.LINE, lambda *a: None)
mon.register_callback(3, mon.events.INSTRUCTION, lambda *a: None)
f(3); same_mon = fp(f) == base
mon.set_events(3, 0); mon.free_tool_id(3)
print("[6] under sys.monitoring LINE+INSTRUCTION: same fp =", same_mon)
print("python", sys.version.split()[0])
