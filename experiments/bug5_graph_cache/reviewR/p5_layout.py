"""reviewR p5: two arrays the plan keys identically (rule 3: dtype.str, shape,
C-order bytes) but that are not interchangeable for a callback:
 (a) C- vs Fortran-ordered copies: reductions can differ in the last bits;
 (b) layout-dependent results (ravel(order='K'), tobytes(order='A'));
 (c) dtype.str collisions between distinct numpy dtypes ('l' vs 'q').
The plan promises bit-equality of a hit with a fresh build on the same machine
type (§2.2 item 3)."""
import numpy as np, hashlib

def plan_digest(x):
    b = memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))
    return hashlib.sha256(x.dtype.str.encode() + repr(x.shape).encode() + bytes(b)).hexdigest()[:16]

rng = np.random.default_rng(1)
A = rng.random((300, 301))
F = np.asfortranarray(A)
print("(a) digests equal:", plan_digest(A) == plan_digest(F))
print("    A.sum()  =", repr(A.sum()))
print("    F.sum()  =", repr(F.sum()))
print("    equal bits:", A.sum() == F.sum())
print("    A.sum(axis=1)[:3] == F.sum(axis=1)[:3]:", np.array_equal(A.sum(axis=1), F.sum(axis=1)))
print("    A.mean() == F.mean():", A.mean() == F.mean())
B = np.arange(6).reshape(2, 3)
BF = np.asfortranarray(B)
print("(b) digests equal:", plan_digest(B) == plan_digest(BF),
      " ravel('K'):", B.ravel('K').tolist(), "vs", BF.ravel('K').tolist())
x = np.array([1, 2], dtype='l'); y = np.array([1, 2], dtype='q')
print("(c) dtype 'l' vs 'q': str", x.dtype.str, y.dtype.str, "digest equal:",
      plan_digest(x) == plan_digest(y), " dtype.char", x.dtype.char, y.dtype.char,
      " type(x[0]) is type(y[0]):", type(x[0]) is type(y[0]))
