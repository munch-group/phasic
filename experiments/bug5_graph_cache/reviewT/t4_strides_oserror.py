"""reviewT t4: (a) arrays the plan's rule 3 admits whose strides are not a
multiple of the itemsize (layout token 'strides divided by the itemsize');
(b) can an OSError subclass always be rebuilt as type(e)(errno, msg, filename)
with a new message (the load path's 're-raised as the same OSError subclass
with a message naming the file')?"""
import errno
import numpy as np

s = np.zeros(4, dtype=[("a", "f8"), ("b", "i4")])
a = s["a"]
t = np.zeros(4, dtype=[("a", "f8"), ("b", "i4"), ("c", "i1")])
b = t["a"]
for name, x in (("a", a), ("b", b)):
    print(name, "type is ndarray:", type(x) is np.ndarray, "dtype", x.dtype.str,
          "fields", x.dtype.fields, "C", x.flags.c_contiguous, "F", x.flags.f_contiguous,
          "strides", x.strides, "floor(stride/itemsize)",
          tuple(st // x.itemsize for st in x.strides),
          "true ratio", tuple(st / x.itemsize for st in x.strides))

subclasses = [PermissionError, FileNotFoundError, BlockingIOError, InterruptedError,
              IsADirectoryError, TimeoutError, ConnectionResetError, OSError]
for cls in subclasses:
    e = cls(errno.EACCES, "orig", "/x/entry.json")
    try:
        new = type(e)(e.errno, "orig; file /x/entry.json (owner unknown)", e.filename)
        print(cls.__name__, "rebuild ok:", type(new).__name__, new.filename)
    except Exception as ex:
        print(cls.__name__, "rebuild FAILS:", type(ex).__name__, ex)
# an OSError raised with errno that maps to a different subclass
e = OSError(errno.ESTALE, "Stale file handle", "/x/entry.json")
print("OSError(ESTALE) ->", type(e).__name__)
