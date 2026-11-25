"""reviewV v2: which numpy dtypes pass the plan's rule-3 gates
(type(dtype).__module__ == 'numpy.dtypes', not hasobject, fields is None,
not long double) and do equal-content arrays always hash equal / unequal
content always differ under 'dtype.str + shape + layout + C-order bytes'?"""
import numpy as np, hashlib
print("numpy", np.__version__)

def gate(a):
    d = a.dtype
    if type(d).__module__ != "numpy.dtypes":
        return f"refused: dtype module {type(d).__module__}"
    if d.hasobject:
        return "refused: hasobject"
    if d.fields is not None:
        return "refused: fields"
    if d.char in "gG":
        return "refused: longdouble"
    return "ADMITTED"

def key(a):
    lay = "C" if a.flags.c_contiguous else ("F" if a.flags.f_contiguous else "S" + repr(a.strides))
    b = memoryview(np.ascontiguousarray(a).reshape(-1).view(np.uint8))
    return (a.dtype.str, a.shape, lay, hashlib.sha256(b).hexdigest()[:16])

cases = {}
try:
    SD = np.dtypes.StringDType()
    cases["StringDType short"] = np.array(["ab", "cd"], dtype=SD)
    cases["StringDType long A"] = np.array(["x" * 40, "y" * 40], dtype=SD)
    cases["StringDType long B"] = np.array(["z" * 40, "w" * 40], dtype=SD)
except Exception as e:
    print("StringDType unavailable:", e)
cases["U"] = np.array(["ab", "cd"])
cases["void unstructured"] = np.zeros(3, "V4")
cases["subarray dtype"] = np.zeros(2, dtype=np.dtype((np.int32, (2,))))
cases["datetime"] = np.array(["2020-01-01"], dtype="M8[D]")
cases["bool"] = np.array([True, False])
cases["f2"] = np.array([1.0], dtype=np.float16)
for k, a in cases.items():
    g = gate(a)
    try:
        kk = key(a) if g == "ADMITTED" else None
    except Exception as e:
        kk = f"key error {type(e).__name__}: {e}"
    print(f"{k:22s} dtype={a.dtype!r:28s} hasobject={a.dtype.hasobject} kind={a.dtype.kind} -> {g}  key={kk}")
if "StringDType long A" in cases:
    A, B = cases["StringDType long A"], cases["StringDType long B"]
    print("long A/B equal content?", (A == B).all(), " equal key?", key(A) == key(B))
    A2 = np.array(["x" * 40, "y" * 40], dtype=SD)
    print("A vs A2 same content: equal key?", key(A) == key(A2))
