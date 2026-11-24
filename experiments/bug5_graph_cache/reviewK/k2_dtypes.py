"""reviewK k2: v3.6 rule 3 (ndarray: dtype.str + shape + C-order bytes via
memoryview, refused iff hasobject or fields) and rule 2 (scalar:
dtype.str + tobytes) on dtypes the plan does not name. No phasic import."""
import hashlib
import numpy as np

print("numpy", np.__version__)


def rule3(a):
    """Literal v3.6 rule 3. Returns the token material or raises."""
    assert type(a) is np.ndarray
    if a.dtype.hasobject or a.dtype.fields is not None:
        return "REFUSED"
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(repr(a.shape).encode())
    h.update(memoryview(np.ascontiguousarray(a)))
    return h.hexdigest()[:16]


def show(tag, a):
    try:
        r = rule3(a)
    except Exception as e:
        r = f"RAISES {type(e).__name__}: {e}"
    print(f"{tag:45s} dtype.str={a.dtype.str!r:10} hasobject={a.dtype.hasobject} "
          f"-> {r}")
    return r

# 1. numpy 2 variable-width StringDType
if hasattr(np.dtypes, "StringDType"):
    S = np.dtypes.StringDType()
    a = np.array(["x" * 40], dtype=S)
    b = np.array(["y" * 40], dtype=S)
    c = np.array(["short"], dtype=S)
    d = np.array(["other"], dtype=S)
    ra, rb = show("StringDType long 'x'*40", a), show("StringDType long 'y'*40", b)
    rc, rd = show("StringDType short 'short'", c), show("StringDType short 'other'", d)
    print("   long strings share a key:", ra == rb and not ra.startswith(("RAISES", "REFUSED")))
    try:
        print("   raw bytes a == b:", a.tobytes() == b.tobytes())
    except Exception as e:
        print("   tobytes raises", type(e).__name__, e)
    # scalar of StringDType: element access gives a Python str (rule 1)
    print("   a[0] type:", type(a[0]).__name__)

# 2. datetime / timedelta arrays (memoryview has no buffer format for them)
show("datetime64[D]", np.array(["2020-01-01"], dtype="M8[D]"))
show("timedelta64[s]", np.array([5], dtype="m8[s]"))
# 3. other formats through hashlib.update(memoryview)
show("complex128", np.array([1 + 2j]))
show("bool", np.array([True, False]))
show("float16", np.array([1.5], dtype=np.float16))
show("longdouble", np.array([1.5], dtype=np.longdouble))
show("unicode U3", np.array(["abc"]))
show("bytes S3", np.array([b"abc"]))
show("0-d float", np.array(3.0))
show("empty (0,3)", np.empty((0, 3)))
show("empty (3,0)", np.empty((3, 0)))
# 4. subarray dtype: fields None, but shape folds in
x = np.zeros(2, dtype=("f8", (2,)))
show("subarray dtype ('f8',(2,)) -> folds", x)
# 5. scalar rule 2
for s in (np.datetime64("2020-01-01"), np.timedelta64(3, "s"), np.str_("ab"),
          np.bytes_(b"ab"), np.float16(1.5)):
    ok = type(s) is s.dtype.type
    print(f"scalar {type(s).__name__:12s} exact={ok} dtype.str={s.dtype.str!r} "
          f"bytes={s.tobytes()!r}")
