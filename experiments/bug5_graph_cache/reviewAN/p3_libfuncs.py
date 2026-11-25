# reviewAN p3: what do common Python-level library helpers carry in defaults/kwdefaults/
# closure/__dict__ (all keyed by the plan's function rule)? Anything non-plain => refusal.
import os
assert os.environ["HOME"].startswith("/private/tmp/")
import types, functools, math
import numpy as np, scipy, scipy.special as sp
print("scipy", scipy.__version__, "numpy", np.__version__)
def describe(label, f):
    print(label, type(f).__name__)
    if type(f) is not types.FunctionType:
        return
    print("   __defaults__  =", f.__defaults__)
    print("   __kwdefaults__=", f.__kwdefaults__)
    print("   closure       =", [type(c.cell_contents).__name__ for c in (f.__closure__ or ())])
    print("   __dict__ keys =", {k: type(v).__name__ for k, v in vars(f).items()})
import scipy.stats as st
for lab, f in [("scipy.special.comb", sp.comb), ("scipy.special.perm", sp.perm),
               ("scipy.special.factorial", sp.factorial),
               ("numpy.linalg.matrix_power", np.linalg.matrix_power),
               ("numpy.isclose", np.isclose), ("numpy.round", np.round), ("numpy.zeros_like", np.zeros_like),
               ("numpy.array_equal", np.array_equal), ("numpy.append", np.append),
               ("math.comb", math.comb), ("scipy.stats.binom.pmf (bound)", st.binom.pmf)]:
    describe(lab, f)
# a function decorated by numpy's array_function_dispatch wrapper: what type?
print("np.append type:", type(np.append), " np.isclose type:", type(np.isclose))
