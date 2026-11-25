"""reviewX x6: which commonly captured modules are ModuleType SUBCLASS
instances (relevant to the §2.2.1 module rule's unspecified type gate)?"""
import importlib, types, sys
for name in ["numpy", "numpy.random", "numpy.linalg", "math", "itertools", "scipy", "scipy.special",
             "scipy.stats", "jax", "jax.numpy", "pandas", "phasic", "phasic.state_indexing"]:
    try:
        m = importlib.import_module(name)
    except Exception as e:
        print(f"{name:24s} import failed: {type(e).__name__}"); continue
    t = type(m)
    print(f"{name:24s} type {t.__module__}.{t.__qualname__:20s} exact={t is types.ModuleType} "
          f"sys.modules-identity={sys.modules.get(m.__name__) is m}")
