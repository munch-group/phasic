"""reviewR p6: numpy 2 StringDType arrays.  The plan (§2.2.2 rule 9 docs list)
says they are refused, but rule 3's refusal tests are: dtype not built in
(type(dtype).__module__ != 'numpy.dtypes'), dtype.hasobject, dtype.fields.
Which of these refuses StringDType, and what would rule 3 hash otherwise?"""
import numpy as np, hashlib
print("numpy", np.__version__)
SD = np.dtypes.StringDType()
a = np.array(["a" * 40], dtype=SD)
b = np.array(["b" * 40], dtype=SD)
dt = a.dtype
print("type(dtype).__module__:", type(dt).__module__)
print("hasobject:", dt.hasobject, " fields:", dt.fields, " str:", dt.str, " kind:", dt.kind)
admitted = (type(dt).__module__ == "numpy.dtypes") and not dt.hasobject and dt.fields is None
print("admitted by rule-3 gates as written:", admitted)
def plan_digest(x):
    v = np.ascontiguousarray(x).reshape(-1).view(np.uint8)
    return hashlib.sha256(x.dtype.str.encode() + repr(x.shape).encode() + bytes(memoryview(v))).hexdigest()[:16]
for lab, x in (("a", a), ("b", b)):
    try:
        print(lab, "digest", plan_digest(x))
    except Exception as e:
        print(lab, "hash path raises", type(e).__name__, e)
