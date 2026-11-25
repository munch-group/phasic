"""reviewT t3: how do the plan's rule-2/3 gates treat numpy 2 StringDType?
Which gate refuses it (the plan lists it as a documented refusal), and what
would a literal implementation of rule 3's byte hashing see?"""
import numpy as np
print("numpy", np.__version__)
SD = np.dtypes.StringDType()
a = np.array(["x" * 40], dtype=SD)
b = np.array(["y" * 40], dtype=SD)
for name, arr in (("a", a), ("b", b)):
    dt = arr.dtype
    print(name, "dtype.str", dt.str, "| type module", type(dt).__module__,
          "| hasobject", dt.hasobject, "| fields", dt.fields, "| kind", dt.kind,
          "| char", dt.char)
    try:
        mv = memoryview(np.ascontiguousarray(arr).reshape(-1).view(np.uint8))
        print("   uint8 view ok, bytes", bytes(mv).hex())
    except Exception as e:
        print("   uint8 view raises", type(e).__name__, e)
    try:
        print("   tobytes", arr.tobytes().hex())
    except Exception as e:
        print("   tobytes raises", type(e).__name__, e)
s = a[0]
print("scalar type", type(s), "isinstance np.generic", isinstance(s, np.generic))
# other built-in dtypes whose bytes are not content
for dt in ("O",):
    print(dt, np.dtype(dt).hasobject)
