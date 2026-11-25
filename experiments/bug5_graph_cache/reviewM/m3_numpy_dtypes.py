"""reviewM m3: numpy dtypes admitted by v3.8 rule 3 (type(x) is np.ndarray, built-in
dtype = type(dtype).__module__ == 'numpy.dtypes', not hasobject, fields is None).
Pure numpy; no phasic import, no cache."""
import numpy as np, hashlib

def admitted(x):
    d = x.dtype
    return (type(x) is np.ndarray and type(d).__module__ == "numpy.dtypes"
            and not d.hasobject and d.fields is None)

def digest(x):
    mv = memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))
    return x.dtype.str, x.shape, hashlib.sha256(mv).hexdigest()[:16]

print("numpy", np.__version__)
cases = {}
if hasattr(np.dtypes, "StringDType"):
    a = np.array(["short", "x" * 40], dtype=np.dtypes.StringDType())
    b = np.array(["short", "y" * 40], dtype=np.dtypes.StringDType())
    cases["StringDType a"] = a
    cases["StringDType b"] = b
for k, x in cases.items():
    d = x.dtype
    print(k, "module", type(d).__module__, "hasobject", d.hasobject, "fields", d.fields,
          "str", d.str, "admitted", admitted(x))
    try:
        print("   digest", digest(x))
    except Exception as ex:
        print("   digest raises", type(ex).__name__, ex)
# also check a StringDType with na_object / coerce variants: same dtype.str?
if hasattr(np.dtypes, "StringDType"):
    for kw in ({}, {"na_object": None}, {"coerce": False}, {"na_object": np.nan}):
        d = np.dtypes.StringDType(**kw)
        print("StringDType", kw, "str", d.str, "repr", repr(d), "hasobject", d.hasobject)
# Other dtype-parameter cases where dtype.str may hide a parameter
for d in (np.dtype("M8[ns]"), np.dtype("M8[2ns]"), np.dtype("m8[D]"), np.dtype("m8[7D]")):
    print(repr(d), "str", d.str)
