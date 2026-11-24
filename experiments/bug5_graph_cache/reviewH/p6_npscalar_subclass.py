"""reviewH p6: §2.2.2 rule 2 read literally ("numpy scalar (np.generic
subclass): x.dtype.str + tobytes()") admits a Python subclass of a numpy
scalar type, whose overridden arithmetic changes what the callback computes,
with the same fingerprint as the plain numpy scalar."""
import numpy as np


class Scaled(np.float64):
    def __mul__(self, other):
        return np.float64(float(self) * other * 10.0)
    __rmul__ = __mul__


def rule2(x):
    assert isinstance(x, np.generic)            # literal rule 2 gate
    assert not x.dtype.fields and not x.dtype.hasobject
    return ("npscalar", x.dtype.str, x.tobytes())


a, b = np.float64(0.5), Scaled(0.5)
print("type(a) is float:", type(a) is float, "type(b) is float:", type(b) is float,
      "(rule 1 does not match either)")
print("rule-2 fingerprints equal:", rule2(a) == rule2(b))
print("what a callback computes with rate*n for n=3:", a * 3, "vs", b * 3)
print("guard type(x) is x.dtype.type:", type(a) is a.dtype.type, type(b) is b.dtype.type)
