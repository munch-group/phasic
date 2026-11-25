"""Review AI probe 1: lazily added class attributes on generated dataclasses,
numpy str/bytes scalars with trailing NULs, and loaded phasic.* modules."""
import copy, sys, os, pickle
import numpy as np
from dataclasses import make_dataclass

print("python", sys.version.split()[0], "numpy", np.__version__)

# (a) snapshot of a make_dataclass class, then copy an instance
C = make_dataclass("IndexResult", [("lineage", object, None)], frozen=True)
snap = dict(vars(C))
inst = C(lineage=1)
copy.copy(inst)
added = set(vars(C)) - set(snap)
changed = [k for k in snap if vars(C).get(k) is not snap[k]]
print("(a) after copy.copy(instance): added", sorted(added), "changed", changed)
C2 = make_dataclass("IndexResult", [("lineage", object, None)], frozen=True)
snap2 = dict(vars(C2))
try:
    pickle.dumps(C2(lineage=1))
except Exception as e:
    print("(a) pickle raised", type(e).__name__)
print("(a) after failed pickle.dumps(instance): added", sorted(set(vars(C2)) - set(snap2)))

# (b) numpy str_/bytes_ scalars with trailing NUL
for a, b in [(np.str_("a"), np.str_("a\x00")), (np.bytes_(b"a"), np.bytes_(b"a\x00"))]:
    print("(b)", repr(a), repr(b), "py-equal:", str(a) == str(b) if isinstance(a, str) else bytes(a) == bytes(b),
          "dtype.str", a.dtype.str, b.dtype.str, "tobytes equal:", a.tobytes() == b.tobytes(),
          "len", len(a), len(b))
print("(b) np.str_ in sctypeDict values:", np.str_ in set(np.sctypeDict.values()),
      "np.bytes_:", np.bytes_ in set(np.sctypeDict.values()))
allowed = {t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy"}
print("(b) allow-set candidates:", sorted(t.__name__ for t in allowed))
