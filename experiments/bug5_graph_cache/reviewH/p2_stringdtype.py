"""reviewH p2: does rule 3 (type is ndarray, not hasobject, fields is None ->
dtype.str + shape + C-order bytes) give equal fingerprints for different
StringDType arrays?  No phasic import needed."""
import numpy as np
print("numpy", np.__version__)
SD = np.dtypes.StringDType()
a = np.array(["alpha-long-string-number-one", "x"], dtype=SD)
b = np.array(["omega-long-string-number-two", "y"], dtype=SD)
for name, arr in (("a", a), ("b", b)):
    d = arr.dtype
    print(name, "type is ndarray:", type(arr) is np.ndarray,
          "dtype.str:", repr(d.str), "hasobject:", d.hasobject,
          "fields:", d.fields, "kind:", d.kind)
    try:
        print("   tobytes:", arr.tobytes().hex())
    except Exception as e:
        print("   tobytes raised", type(e).__name__, e)

def fp(arr):
    return (arr.dtype.str, arr.shape, arr.tobytes())

try:
    print("rule-3 fingerprints equal for different content:", fp(a) == fp(b))
    print("arrays equal:", bool((a == b).all()))
except Exception as e:
    print("fp raised", type(e).__name__, e)

# same short strings inline?
c = np.array(["s1"], dtype=SD); d = np.array(["s2"], dtype=SD)
print("short strings fp equal:", fp(c) == fp(d))
# numpy scalar from StringDType array
s = a[0]
print("element type:", type(s), "is np.generic:", isinstance(s, np.generic))
