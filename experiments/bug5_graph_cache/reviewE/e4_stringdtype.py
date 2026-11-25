"""reviewE e4: does rule 3 (dtype.str + shape + tobytes, refuse hasobject / fields) admit
numpy 2 StringDType arrays, and are their bytes content or pointers?"""
import numpy as np
print("numpy", np.__version__)
if not hasattr(np.dtypes, "StringDType"):
    print("no StringDType"); raise SystemExit
dt = np.dtypes.StringDType()
a = np.array(["x" * 40, "y" * 40], dtype=dt)
b = np.array(["x" * 40, "y" * 40], dtype=dt)
c = np.array(["z" * 40, "w" * 40], dtype=dt)
print("dtype.str", repr(dt.str), "hasobject", dt.hasobject, "fields", dt.fields, "kind", dt.kind)
print("a==b content; tobytes equal:", a.tobytes() == b.tobytes())
print("a vs c differ content; tobytes:", a.tobytes().hex()[:64], c.tobytes().hex()[:64])
s1 = np.array(["ab", "cd"], dtype=dt); s2 = np.array(["ab", "ce"], dtype=dt)
print("short strings tobytes differ:", s1.tobytes() != s2.tobytes())
# id/address reuse: build, take bytes, free, build different content, compare
import gc
keys = set(); coll = 0
for i in range(2000):
    arr = np.array([f"{i:040d}", f"{i:040d}"], dtype=dt)
    k = arr.tobytes()
    if k in keys: coll += 1
    keys.add(k); del arr; gc.collect() if i % 100 == 0 else None
print("distinct-content arrays whose tobytes collided with an earlier one:", coll, "of 2000")
