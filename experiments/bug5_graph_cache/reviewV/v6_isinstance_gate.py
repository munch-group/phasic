"""reviewV v6: the plan gates rule 2 on isinstance(x, np.generic) 'so .dtype
is never read from an arbitrary object', and rule 7 on isinstance(x, type).
isinstance() falls back to x.__class__, so a user object can pass both gates
and have its own attributes read.  Shows (a) the gate passes, (b) .dtype is
then read from the user object, (c) a deliberately spoofed object passes the
exact-type and built-in-dtype checks as the plan words them."""
import numpy as np, types

reads = []

class Spoof:
    @property
    def __class__(self):
        reads.append("__class__")
        return np.float64

class FakeDType:
    pass
FakeDType.__module__ = "numpy.dtypes"

class Spoof2:
    @property
    def __class__(self):
        return np.float64
    @property
    def dtype(self):
        reads.append("dtype")
        d = FakeDType(); d.type = Spoof2; d.str = "<f8"; d.fields = None; d.hasobject = False; d.char = "d"
        return d
    def tobytes(self):
        reads.append("tobytes")
        return b"\0" * 8

x = Spoof()
print("isinstance(Spoof(), np.generic):", isinstance(x, np.generic), "reads:", reads)
y = Spoof2(); reads.clear()
gate = isinstance(y, np.generic)
exact = type(y) is y.dtype.type
builtin = type(y.dtype).__module__ == "numpy.dtypes"
print("Spoof2: isinstance gate", gate, "| exact-type check", exact, "| built-in dtype check", builtin, "| reads", reads)
print("token would be", y.dtype.str, y.tobytes())

class TSpoof:
    @property
    def __class__(self):
        return type
print("isinstance(TSpoof(), type):", isinstance(TSpoof(), type))
print("type-based gates instead:", issubclass(type(y), np.generic), issubclass(type(TSpoof()), type))
