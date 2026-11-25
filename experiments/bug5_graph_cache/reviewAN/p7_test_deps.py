# reviewAN p7: are the objects/packages tests 11/18/26 need available in the env?
import importlib, numpy as np
print("default_rng type:", type(np.random.default_rng).__name__)
for m in ["dill", "ml_dtypes", "pandas", "scipy"]:
    try:
        importlib.import_module(m); print(m, "available")
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
f = np.frompyfunc(lambda x: x, 1, 1)
print("frompyfunc ufunc: type", type(f).__name__, "__module__", getattr(f, "__module__", "<absent>"), "__name__", f.__name__)
