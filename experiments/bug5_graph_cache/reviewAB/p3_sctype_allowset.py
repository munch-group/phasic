"""reviewAB p3: what does rule 2's allow-set (np.sctypeDict.values() with
__module__ == 'numpy', minus void/object/longdouble/clongdouble) contain,
and what would 'the same gate as rule 2' admit for arrays (rule 3)?"""
import numpy as np
print("numpy", np.__version__)
vals = {t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy"}
allow = vals - {np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set:", sorted(t.__name__ for t in allow))
print("np.record in sctypeDict values:", np.record in set(np.sctypeDict.values()))
# distinct classes sharing dtype.str (encoding = dtype.str + tobytes, no class name)
by_str = {}
for t in allow:
    try:
        s = np.dtype(t).str
    except Exception as e:
        s = f"ERR {e}"
    by_str.setdefault(s, []).append(t.__name__)
print("classes sharing dtype.str:", {k: v for k, v in by_str.items() if len(v) > 1})
# rule 3 via dtype.type in allow-set
for label, arr in [("StringDType", None), ("U", np.array(["ab"])), ("M8", np.array(["2020"], dtype="M8[D]"))]:
    if label == "StringDType":
        try:
            arr = np.array(["ab"], dtype=np.dtypes.StringDType())
        except Exception as e:
            print("StringDType unavailable", e); continue
    print(label, "dtype.type =", arr.dtype.type, "in allow:", arr.dtype.type in allow,
          "type(dtype) =", type(arr.dtype).__name__)
