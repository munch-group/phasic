"""reviewK k3: dtype.str does not identify user-defined numpy dtypes
(ml_dtypes, installed with jax). v3.6 rules 2/3 key dtype.str + bytes."""
import numpy as np
import ml_dtypes as md

names = ["bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3b11fnuz",
         "int4", "uint4"]
for n in names:
    t = getattr(md, n, None)
    if t is None:
        continue
    dt = np.dtype(t)
    a = np.array([1.0, 2.0], dtype=dt) if 'int' not in n else np.array([1, 2], dtype=dt)
    s = a[0]
    print(f"{n:20s} dtype.str={dt.str!r:7} kind={dt.kind!r} hasobject={dt.hasobject} "
          f"fields={dt.fields} bytes={a.tobytes()!r} scalar_isgeneric="
          f"{isinstance(s, np.generic)} scalar_exact={type(s) is s.dtype.type}")

# Same bytes, different dtypes with identical dtype.str -> identical token
a = np.frombuffer(bytes([0x3c, 0x40]), dtype=md.float8_e4m3fn).copy()
b = np.frombuffer(bytes([0x3c, 0x40]), dtype=md.float8_e5m2).copy()
print("e4m3fn values", a.astype(float), "e5m2 values", b.astype(float))
print("same dtype.str:", a.dtype.str == b.dtype.str, " same bytes:",
      a.tobytes() == b.tobytes(), " same values:",
      bool((a.astype(float) == b.astype(float)).all()))
sa, sb = a[0], b[0]
print("scalars: same dtype.str:", sa.dtype.str == sb.dtype.str, " same tobytes:",
      sa.tobytes() == sb.tobytes(), " values", float(sa), float(sb))

# Candidate gate: dtype.isbuiltin (0 flexible/structured, 1 builtin, 2 user-defined)
for t in (np.float64, "U3", "S2", "M8[D]", "V8", md.bfloat16, md.float8_e4m3fn,
          md.int4, np.dtypes.StringDType()):
    dt = np.dtype(t)
    print(f"isbuiltin {str(dt):22s} {dt.isbuiltin}  type(dtype).__module__="
          f"{type(dt).__module__}")

# Pair with IDENTICAL dtype.str AND identical bytes but different values
raw = bytes([0x38, 0x40])
p = np.frombuffer(raw, dtype=md.float8_e4m3fn).copy()
q = np.frombuffer(raw, dtype=md.float8_e4m3b11fnuz).copy()
print("PAIR e4m3fn vs e4m3b11fnuz: dtype.str", p.dtype.str, q.dtype.str,
      "| same bytes", p.tobytes() == q.tobytes(), "| values",
      p.astype(float).tolist(), q.astype(float).tolist())
r = np.array([-1], dtype=md.int4); u = np.array([15], dtype=md.uint4)
print("PAIR int4[-1] vs uint4[15]: dtype.str", r.dtype.str, u.dtype.str,
      "| bytes", r.tobytes(), u.tobytes())
