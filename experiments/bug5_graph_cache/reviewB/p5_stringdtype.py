"""reviewB p5: numpy StringDType (dtype 'T') under §2.2.2 rule 3.

Rule 3 keys an ndarray by dtype.str + shape + C-order bytes and refuses only
dtype.hasobject or dtype.fields is not None. Does a StringDType array pass
that filter, and are its bytes the string CONTENT?  Pure numpy, no phasic.
"""
import numpy as np
from numpy.dtypes import StringDType

print("numpy", np.__version__)
dt = StringDType()
print("dtype.str:", repr(dt.str), "hasobject:", dt.hasobject,
      "fields:", dt.fields, "kind:", dt.kind, "itemsize:", dt.itemsize)

long_a = "a" * 40
long_b = "b" * 40
a = np.array([long_a], dtype=dt)
b = np.array([long_b], dtype=dt)
print("long strings, different content -> same tobytes():",
      a.tobytes() == b.tobytes())
a2 = np.array([long_a], dtype=dt)
print("long strings, same content, new array -> same tobytes():",
      a.tobytes() == a2.tobytes())
s1 = np.array(["x"], dtype=dt); s2 = np.array(["y"], dtype=dt)
print("short strings, different content -> same tobytes():",
      s1.tobytes() == s2.tobytes())
print("example bytes (long a):", a.tobytes().hex())
print("example bytes (long b):", b.tobytes().hex())
