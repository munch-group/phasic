"""Review I probe 5: a functools.partial captured in a closure carries a
__dict__ that the callback can read. v3.4 keys a partial by (func, args,
keywords) only (section 2.2.1), while it keys a plain function's __dict__.
Two factories below differ ONLY in the partial's attribute: every component
v3.4 lists is identical, the graphs differ. Built with graph_cache=False."""
import functools, types, numpy as np
from phasic import Graph

def rate(n, scale):
    return n * (n - 1) / 2 * scale

def make(s):
    r = functools.partial(rate, scale=1.0)
    r.extra = s                      # attribute on the partial object
    def cb(state, **kw):
        n = state[0]
        if n <= 1:
            return []
        return [[[n - 1], r(n) * r.extra]]
    return cb

g1 = Graph(make(1.0), ipv=[5], graph_cache=False)
g2 = Graph(make(3.0), ipv=[5], graph_cache=False)
c1, c2 = make(1.0), make(3.0)
p1 = c1.__closure__[0].cell_contents; p2 = c2.__closure__[0].cell_contents
print("partial func/args/keywords equal:", p1.func is p2.func, p1.args == p2.args, p1.keywords == p2.keywords)
print("cb code identical:", c1.__code__ == c2.__code__)
print("partial __dict__:", vars(p1), vars(p2))
print("expectation g1 vs g2:", g1.expectation(), g2.expectation())

# Other callables: can attributes be set?
def f(): pass
m = types.MethodType(f, 5)
for name, obj in (("bound method", m), ("np.exp ufunc", np.exp), ("np.sum dispatcher", np.sum), ("len", len)):
    try:
        setattr(obj, "zz_review", 1); ok = True
        delattr(obj, "zz_review")
    except Exception as e:
        ok = f"{type(e).__name__}"
    print(f"setattr on {name}: {ok}")
