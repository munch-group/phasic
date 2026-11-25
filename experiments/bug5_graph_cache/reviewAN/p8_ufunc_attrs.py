# reviewAN p8: the attributes the builtin/ufunc arm reads, for the objects the plan says are admitted
import sys, math, types, numpy as np
for lab, f in [("np.exp", np.exp), ("np.add", np.add), ("np.sum", np.sum), ("np.linalg.solve", np.linalg.solve), ("math.sqrt", math.sqrt), ("len", len)]:
    mod = getattr(f, "__module__", None); qn = getattr(f, "__qualname__", None); nm = getattr(f, "__name__", None)
    slf = getattr(f, "__self__", None)
    key = qn if type(f) is not np.ufunc else nm
    res = None
    if mod in sys.modules and key is not None:
        o = sys.modules[mod]
        for part in key.split("."):
            o = getattr(o, part, None)
        res = o is f
    print(f"{lab:16s} type={type(f).__name__:28s} module={mod!s:14s} qualname={qn!s:10s} name={nm!s:8s} self={type(slf).__name__:10s} resolves={res}")
