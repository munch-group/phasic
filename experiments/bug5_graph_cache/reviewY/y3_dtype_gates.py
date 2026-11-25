"""reviewY y3: apply the plan's literal rule-2/rule-3 gates (built-in dtype via
type(dtype).__module__ == 'numpy.dtypes', hasobject, fields) to dtypes the plan
says are refused elsewhere, and check whether the bytes the plan hashes could
collide. Pure numpy, no phasic import."""
import hashlib, sys
import numpy as np
print("numpy", np.__version__, "python", sys.version.split()[0])

def gates(a):
    dt = a.dtype
    builtin = type(dt).__module__ == "numpy.dtypes"
    return dict(dtype_str=dt.str, builtin=builtin, hasobject=dt.hasobject,
                fields=dt.fields is not None, kind=dt.kind, char=dt.char)

def plan_bytes(a):
    return memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8)).tobytes()

def admitted(a):
    g = gates(a)
    return g["builtin"] and not g["hasobject"] and not g["fields"]

try:
    from numpy.dtypes import StringDType
    a = np.array(["a" * 20], dtype=StringDType())
    b = np.array(["b" * 20], dtype=StringDType())
    print("StringDType gates:", gates(a), "admitted by rule 3 text:", admitted(a))
    try:
        ba, bb = plan_bytes(a), plan_bytes(b)
        print("  bytes equal for 'a'*20 vs 'b'*20:", ba == bb, " a==b values:", bool((a == b).all()))
    except Exception as e:
        print("  byte view raises:", type(e).__name__, e)
except ImportError:
    print("no StringDType")

for code in ("g", "G"):
    x = np.zeros(2, dtype=code)
    print(f"long double {code!r} array gates:", gates(x), "admitted by rule 3 text:", admitted(x))

v = np.zeros(2, dtype="V8")
print("unstructured void array gates:", gates(v), "admitted by rule 3 text:", admitted(v))
