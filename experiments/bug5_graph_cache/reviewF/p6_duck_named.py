"""Review F p6: objects that are NOT builtins but carry __module__/__qualname__
copied by functools.update_wrapper, resolve to themselves through sys.modules
and have no __self__ -- i.e. pass every CONDITION of §2.2.1's builtin arm.
If that arm is implemented by duck typing (conditions only) instead of an
exact type test, these Python-level functions would be keyed by NAME, and a
redefinition (notebook cell re-run) would hit stale."""
import sys, functools, types
import numpy as np


@functools.lru_cache(maxsize=None)
def rate(n):
    return 0.5 * n


class Wrapped:                      # a callable instance made by a decorator
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f
    def __call__(self, *a):
        return self.f(*a)


@Wrapped
def rate2(n):
    return 0.5 * n


def check(label, obj):
    mod, qn, slf = (getattr(obj, n, None) for n in ("__module__", "__qualname__", "__self__"))
    m = sys.modules.get(mod) if isinstance(mod, str) else None
    resolves = m is not None and getattr(m, qn, None) is obj
    print(f"{label:22s} type={type(obj).__name__:22s} module={mod} qualname={qn} "
          f"__self__={slf!r} resolves_to_itself={resolves} "
          f"is_builtin_type={isinstance(obj, (types.BuiltinFunctionType, np.ufunc))}")


check("lru_cache function", rate)
check("update_wrapper object", rate2)
