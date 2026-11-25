"""Built-in numpy dtypes that pass the plan's rule-2/3 gates
(type(dtype).__module__ == 'numpy.dtypes', not hasobject, fields is None)
but whose raw bytes are not their value, or whose dtype.str collides."""
import numpy as np, sys
print("numpy", np.__version__, sys.platform)

def gates(a):
    d = a.dtype
    return dict(mod=type(d).__module__, str=d.str, hasobject=d.hasobject,
                fields=d.fields, isnative=d.isnative, kind=d.kind)

def plan_bytes(a):
    return bytes(memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8)))

# 1. StringDType (numpy >= 2.0)
if hasattr(np.dtypes, "StringDType"):
    S = np.dtypes.StringDType()
    a = np.array(["x" * 40], dtype=S)
    b = np.array(["x" * 40], dtype=S)
    print("StringDType gates:", gates(a))
    try:
        ba, bb = plan_bytes(a), plan_bytes(b)
        print("StringDType view ok; equal content equal bytes?", ba == bb, "len", len(ba))
        # same-address reuse: free a, allocate c with different content
        addrs = set()
        hits = 0
        for i in range(2000):
            x = np.array([("p%03d" % (i % 997)) * 10], dtype=S)
            bx = plan_bytes(x)
            if bx in addrs:
                hits += 1
            addrs.add(bx)
            del x
        print("StringDType: distinct contents with byte-identical 'plan bytes':", hits)
    except Exception as e:
        print("StringDType plan_bytes raised", type(e).__name__, e)
    s = a[0]
    print("StringDType scalar type:", type(s), "isinstance np.generic:", isinstance(s, np.generic))

# 2. longdouble vs float64; longlong vs int64 (dtype.str collision?)
for t1, t2 in [(np.longdouble, np.float64), (np.longlong, np.int64), (np.intc, np.int32),
               (np.clongdouble, np.complex128)]:
    x1, x2 = t1(1), t2(1)
    print(t1.__name__, t2.__name__, "dtype.str", x1.dtype.str, x2.dtype.str,
          "types differ", type(x1) is not type(x2),
          "exact-type gate both pass", type(x1) is x1.dtype.type and type(x2) is x2.dtype.type,
          "builtin", type(x1.dtype).__module__, "same bytes", x1.tobytes() == x2.tobytes())
    print("   promotion with float32:", (x1 * np.float32(1)).dtype, (x2 * np.float32(1)).dtype)
