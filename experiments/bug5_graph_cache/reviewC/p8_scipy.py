"""Review C p8: scipy.special ufunc attributes (plan 2.2.1 says __module__ is None)."""
import scipy, scipy.special as sp, numpy as np
print("scipy", scipy.__version__)
for f in (sp.gamma, sp.binom, np.exp):
    try:
        m = f.__module__
        print(f.__name__, "f.__module__ =", repr(m))
    except AttributeError as e:
        print(f.__name__, "f.__module__ raises AttributeError:", e)
    print("   getattr default:", repr(getattr(f, "__module__", "<absent>")), "hasattr __qualname__:", hasattr(f, "__qualname__"))
