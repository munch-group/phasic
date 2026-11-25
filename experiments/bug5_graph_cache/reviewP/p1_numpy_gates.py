"""Probe: do v4.1 rule-2/3 gates (built-in dtype via numpy.dtypes module,
hasobject, fields) admit StringDType arrays, and are their bytes content?"""
import numpy as np
print("numpy", np.__version__)

def gate(x):
    d = x.dtype
    return dict(mod=type(d).__module__, builtin=type(d).__module__ == "numpy.dtypes",
                hasobject=d.hasobject, fields=d.fields, dstr=d.str)

S = np.dtypes.StringDType()
long1 = "x" * 40 + "A"
long2 = "x" * 40 + "B"
a = np.array([long1], dtype=S)
b = np.array([long2], dtype=S)
print("StringDType gate:", gate(a))
ba = np.ascontiguousarray(a).reshape(-1).view(np.uint8).tobytes() if True else None
try:
    mv = memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8))
    print("uint8 view ok, nbytes", mv.nbytes)
except Exception as e:
    print("uint8 view raised:", type(e).__name__, e)
# Content-equal arrays: do bytes agree?
c = np.array([long1], dtype=S)
try:
    va = np.ascontiguousarray(a).reshape(-1).view(np.uint8).tobytes()
    vc = np.ascontiguousarray(c).reshape(-1).view(np.uint8).tobytes()
    vb = np.ascontiguousarray(b).reshape(-1).view(np.uint8).tobytes()
    print("equal-content arrays same bytes:", va == vc, " different-content same bytes:", va == vb)
except Exception as e:
    print("view raised", type(e).__name__, e)

# Collision attempt: free an array and build another with different long string;
# arena may be reused at same address.
hits = 0
for i in range(2000):
    x = np.array(["y" * 50 + str(i)], dtype=S)
    bx = np.ascontiguousarray(x).reshape(-1).view(np.uint8).tobytes()
    del x
    y = np.array(["z" * 50 + str(i)], dtype=S)
    by = np.ascontiguousarray(y).reshape(-1).view(np.uint8).tobytes()
    if bx == by:
        hits += 1
    del y
print("free/realloc byte collisions between DIFFERENT strings:", hits, "/ 2000")

# scalar side
s = np.str_("abc")
print("np.str_ scalar: generic", isinstance(s, np.generic), "exact", type(s) is s.dtype.type, gate(np.array(s)))
