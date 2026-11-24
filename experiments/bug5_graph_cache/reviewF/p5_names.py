"""Review F p5: attributes the v3.1 by-name rules (§2.2.1) rely on."""
import sys, math, itertools, functools
import numpy as np


def info(label, obj):
    g = lambda n: getattr(obj, n, "<MISSING>")
    mod, qn, nm, slf = g("__module__"), g("__qualname__"), g("__name__"), g("__self__")
    res = "<n/a>"
    m = sys.modules.get(mod) if isinstance(mod, str) else None
    if m is not None:
        cur = m
        try:
            for part in (qn if isinstance(qn, str) else nm).split("."):
                cur = getattr(cur, part)
            res = cur is obj
        except AttributeError:
            res = "attr-miss"
    flags = getattr(obj, "__flags__", None)
    imm = None if flags is None else bool(flags & (1 << 8))
    print(f"{label:40s} type={type(obj).__name__:28s} module={mod!s:14s} qualname={qn!s:34s} "
          f"name={nm!s:24s} self={type(slf).__name__:10s} resolves={res} immutable={imm}")


info("np.exp", np.exp)
info("np.sum", np.sum)
info("np.linalg.solve", np.linalg.solve)
info("math.sqrt", math.sqrt)
info("len", len)
info("itertools.combinations_with_replacement", itertools.combinations_with_replacement)
info("functools.partial (class)", functools.partial)
info("np.frompyfunc ufunc", np.frompyfunc(lambda x: x, 1, 1))
info("{}.get", {}.get)
info("np.random.rand", np.random.rand)
info("str.upper (method_descriptor)", str.upper)
try:
    import scipy.special as sp
    info("scipy.special.gamma", sp.gamma)
    info("scipy.special.comb", sp.comb)
except Exception as e:
    print("scipy:", e)
