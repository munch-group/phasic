# What does the plan's rule-2 allow-set contain? Does it change after importing
# jax / ml_dtypes (import-order dependence)?
import numpy as np
def allow():
    s = {t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy"}
    s -= {np.void, np.object_, np.longdouble, np.clongdouble}
    return s
a0 = allow()
print("numpy", np.__version__, "n=", len(a0))
print(sorted(t.__name__ for t in a0))
print("immutable flag on np.float64:", bool(np.float64.__flags__ & (1 << 8)), "np.dtype:", bool(np.dtype.__flags__ & (1<<8)))
try:
    import ml_dtypes
    a1 = allow()
    print("after ml_dtypes import n=", len(a1), "new:", sorted(t.__name__ for t in a1 - a0))
    print("ml float8 module:", ml_dtypes.float8_e4m3fn.__module__)
except Exception as e:
    print("ml_dtypes:", e)
# np.str_ scalar token parts
x = np.str_("ab"); print("str_:", x.dtype.str, x.tobytes())
x = np.datetime64("2020-01-01"); print("dt64:", x.dtype.str, x.tobytes())
