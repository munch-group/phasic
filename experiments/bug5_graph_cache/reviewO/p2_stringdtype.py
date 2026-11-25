"""reviewO p2: plan rule 3 (ndarray) read literally, applied to numpy 2's
StringDType (a BUILT-IN dtype in numpy.dtypes). No phasic import needed."""
import hashlib, numpy as np
print("numpy", np.__version__)

def rule3_token(x):
    # literal transcription of v4.0 §2.2.2 rule 3
    assert type(x) is np.ndarray
    dt = x.dtype
    if type(dt).__module__ != "numpy.dtypes":
        return "REFUSED(not built-in)"
    if dt.hasobject:
        return "REFUSED(hasobject)"
    if dt.fields is not None:
        return "REFUSED(fields)"
    mv = memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))
    return (dt.str, x.shape, hashlib.sha256(mv).hexdigest()[:16])

SD = np.dtypes.StringDType()
print("StringDType module:", type(SD).__module__, " str:", SD.str,
      " hasobject:", SD.hasobject, " fields:", SD.fields)
cases = {
    "short a": np.array(["a"], dtype=SD),
    "short b": np.array(["b"], dtype=SD),
    "long A*20": np.array(["A" * 20], dtype=SD),
    "long B*20": np.array(["B" * 20], dtype=SD),
    "long A*20 again": np.array(["A" * 20], dtype=SD),
}
for k, v in cases.items():
    try:
        print(f"{k:16s}", rule3_token(v))
    except Exception as e:
        print(f"{k:16s} raised {type(e).__name__}: {e}")
