"""reviewL l3: numpy StringDType arrays under the v3.7 array rule (rule 3).

Rule 3 admits `type(x) is np.ndarray` with a built-in dtype
(type(dtype).__module__ == 'numpy.dtypes'), not hasobject, fields None, and
hashes dtype.str, shape and memoryview(ascontiguousarray(x).reshape(-1)
.view(np.uint8)).  Does StringDType pass the gates, and do its bytes encode
the string contents?
"""
import numpy as np, hashlib, sys
print("numpy", np.__version__, "python", sys.version.split()[0])
SD = np.dtypes.StringDType()


def rule3(x):
    dt = x.dtype
    gates = dict(exact=type(x) is np.ndarray,
                 builtin=type(dt).__module__ == "numpy.dtypes",
                 hasobject=dt.hasobject, fields=dt.fields, str=dt.str)
    try:
        b = bytes(memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8)))
        h = hashlib.sha256(dt.str.encode() + repr(x.shape).encode() + b).hexdigest()[:16]
    except Exception as e:
        b, h = None, f"{type(e).__name__}: {e}"
    return gates, b, h


def mk(s):
    return np.array([s], dtype=SD)

short_a, short_b = mk("abc"), mk("abd")
print("short a", rule3(short_a)[0], rule3(short_a)[2])
print("short b", rule3(short_b)[2])

# long strings live outside the array buffer
la = mk("x" * 64)
ga, ba, ha = rule3(la)
print("long a gates", ga)
print("long a bytes", ba.hex() if ba else ba, ha)
del la
lb = mk("y" * 64)
gb, bb, hb = rule3(lb)
print("long b bytes", bb.hex() if bb else bb, hb)
print("DIFFERENT contents, SAME fingerprint:", ha == hb)
# equal contents, separate arrays
lc, ld = mk("z" * 64), mk("z" * 64)
print("equal contents, equal fingerprint:", rule3(lc)[2] == rule3(ld)[2])
