"""reviewAF p3: can the plan's generated-class check (rule 8 (c)) pass on an ordinary StateIndexer?

Plan v5.7 rule 8(c): for ix._result_class vs c._result_class and each PropertySet's
_props_dataclass vs c's _props_class(): every reference entry that is a function must be an
exact FunctionType with the same FUNCTION FINGERPRINT (code, defaults, kwdefaults, __dict__, closure
cells position by position; a cell holding the reference class must hold the candidate class);
every NON-function entry must be the IDENTICAL object or equal by VALUE FINGERPRINT.

This probe classifies every entry. VALUE-FINGERPRINTABLE is approximated by the plan's plain rules:
exact None/bool/int/float/complex/str/bytes, exact list/tuple/dict of those, Ellipsis,
immutable compiled types resolving by name, exact functions (recursively not checked here).
"""
import sys, types, os
assert os.environ["HOME"].startswith("/private/tmp/")
from phasic.state_indexing import StateIndexer, Property

def is_immutable_named_type(x):
    if type(x) is not type:
        return False
    if not (x.__flags__ & (1 << 8)):
        return False
    m = sys.modules.get(x.__module__)
    o = m
    try:
        for part in x.__qualname__.split("."):
            o = getattr(o, part)
    except Exception:
        return False
    return o is x

def plain(x, depth=0):
    t = type(x)
    if x is None or t in (bool, int, float, complex, str, bytes) or x is Ellipsis:
        return True, ""
    if t in (list, tuple):
        for e in x:
            ok, why = plain(e, depth + 1)
            if not ok:
                return ok, why
        return True, ""
    if t is dict:
        for k, v in x.items():
            for e in (k, v):
                ok, why = plain(e, depth + 1)
                if not ok:
                    return ok, why
        return True, ""
    if t is types.FunctionType:
        return True, "(function)"
    if is_immutable_named_type(x):
        return True, "(immutable type)"
    return False, f"{t.__module__}.{t.__qualname__}"

def classify(ref, cand, label):
    print(f"=== {label}: ref={ref.__qualname__} cand={cand.__qualname__} same_obj={ref is cand}")
    rk, ck = set(vars(ref)), set(vars(cand))
    print("   key sets equal:", rk == ck)
    n_bad = 0
    for k in sorted(rk):
        rv, cv = vars(ref)[k], vars(cand).get(k)
        if type(rv) is types.FunctionType:
            cells = []
            for name, cell in zip(rv.__code__.co_freevars, rv.__closure__ or ()):
                try:
                    c = cell.cell_contents
                except ValueError:
                    cells.append((name, "empty"))
                    continue
                if c is ref:
                    cells.append((name, "REF CLASS (special-cased)"))
                    continue
                ok, why = plain(c)
                cells.append((name, "ok" if ok else f"NOT FINGERPRINTABLE: {why}"))
                if not ok:
                    n_bad += 1
            print(f"   fn  {k}: cells={cells}")
        else:
            ident = rv is cv
            ok, why = plain(rv)
            verdict = "identical" if ident else ("value-fingerprintable" if ok else f"NEITHER identical NOR fingerprintable: {why}")
            if not ident and not ok:
                n_bad += 1
            print(f"   val {k}: {verdict}")
    print(f"   -> entries that fail the plan's rule: {n_bad}")

ix = StateIndexer(lineages=[Property("ton", min_value=1, max_value=4)])
ix.lineages.index_to_props(2)        # creates _props_dataclass (lazy)
ix.index_to_props(2)                 # uses _result_class
c = StateIndexer.from_dict(ix.to_dict())
classify(c._result_class, ix._result_class, "_result_class")
classify(c.lineages._props_class(), ix.lineages._props_dataclass, "_props_dataclass")
print("python", sys.version.split()[0])
