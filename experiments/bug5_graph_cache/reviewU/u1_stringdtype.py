# reviewU u1: does numpy StringDType pass the plan's built-in dtype gate
# (type(dtype).__module__ == "numpy.dtypes", not hasobject, fields is None),
# and do arrays with DIFFERENT long strings have IDENTICAL raw bytes?
import numpy as np, hashlib, sys
print("numpy", np.__version__, "python", sys.version.split()[0])
from numpy.dtypes import StringDType
def gate(a):
    d = a.dtype
    return dict(mod=type(d).__module__, hasobject=d.hasobject, fields=d.fields,
                str=d.str, kind=d.kind, char=d.char, flags=d.flags)
def plan_bytes(a):
    return bytes(memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8)))
for s1, s2 in [("a"*40, "b"*40), ("x"*100, "y"*100), ("short1", "short2")]:
    a = np.array([s1], dtype=StringDType())
    b = np.array([s2], dtype=StringDType())
    print("gate", gate(a))
    try:
        ba, bb = plan_bytes(a), plan_bytes(b)
        print(repr(s1[:6]), repr(s2[:6]), "len", len(ba), "bytes equal:", ba == bb,
              hashlib.sha256(ba).hexdigest()[:12], hashlib.sha256(bb).hexdigest()[:12])
    except Exception as e:
        print("view failed:", type(e).__name__, e)
