"""reviewS s2: probe the numpy built-in dtype gate and the immutable-type gate (no phasic import)."""
import sys, numpy as np, itertools, functools
print("python", sys.version.split()[0], "numpy", np.__version__)
# StringDType gate
try:
    SD = np.dtypes.StringDType()
    a = np.array(["x" * 40, "short"], dtype=SD)
    b = np.array(["y" * 40, "short"], dtype=SD)
    print("StringDType module:", type(a.dtype).__module__, "hasobject:", a.dtype.hasobject,
          "fields:", a.dtype.fields, "str:", a.dtype.str, "itemsize", a.dtype.itemsize)
    ba = memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8)).tobytes() if True else None
except Exception as e:
    print("StringDType bytes path raised:", type(e).__name__, e)
try:
    print("bytes equal (different long strings):",
          np.ascontiguousarray(a).reshape(-1).view(np.uint8).tobytes() ==
          np.ascontiguousarray(b).reshape(-1).view(np.uint8).tobytes())
except Exception as e:
    print("view(uint8) raised:", type(e).__name__, e)
for dt in ["f8", "M8[s]", "m8[ms]", "U3", "S3", "c16", "?", "O"]:
    d = np.dtype(dt)
    print(dt, type(d).__module__, d.hasobject)
# immutable flag
F = 1 << 8
for obj in [itertools.combinations_with_replacement, np.ndarray, np.float64, int, range,
            functools.partial, collections := __import__("collections").Counter]:
    print(getattr(obj, "__qualname__", obj), bool(obj.__flags__ & F))
