"""reviewG g4: if the 'Python function' arm were implemented by duck typing
(hasattr(f, '__code__')), would distinct compiled (Cython) functions share a full
CODE FINGERPRINT (all fields the plan keys, incl. co_name)? Scan already-importable
compiled modules of pandas / scipy for cyfunctions and group them by that tuple."""
import sys, types, importlib, pkgutil

FIELDS = ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars",
          "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags", "co_name")
def fp(co):
    return tuple(getattr(co, a, None) for a in FIELDS) + (getattr(co, "co_exceptiontable", b""),)

groups = {}
nmods = 0
for pkgname in ("pandas._libs", "scipy.sparse.csgraph", "scipy.special", "scipy.optimize", "scipy.stats"):
    try:
        pkg = importlib.import_module(pkgname)
    except Exception:
        continue
    for mi in pkgutil.walk_packages(pkg.__path__, pkgname + "."):
        if "test" in mi.name:
            continue
        try:
            m = importlib.import_module(mi.name)
        except Exception:
            continue
        nmods += 1
        for k, v in list(vars(m).items()):
            if type(v).__name__ == "cython_function_or_method" and hasattr(v, "__code__"):
                try:
                    key = fp(v.__code__) + (v.__defaults__, v.__kwdefaults__)
                    hash(key)
                except Exception:
                    continue
                groups.setdefault(key, {})[id(v)] = f"{getattr(v,'__module__',None)}.{k}"
coll = [sorted(set(d.values())) for d in groups.values() if len(d) > 1]
coll = [c for c in coll if len(c) > 1]
print(f"modules scanned: {nmods}; distinct cyfunction fingerprints: {len(groups)}")
print(f"groups of DISTINCT cyfunctions sharing one full code fingerprint + defaults: {len(coll)}")
for c in coll[:8]:
    print("  ", c)
