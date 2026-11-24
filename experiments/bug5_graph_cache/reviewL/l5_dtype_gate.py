"""reviewL l5: dtype gate module names and hasobject/fields for common dtypes."""
import numpy as np
for spec in ["f8", "i4", "?", "U5", "S3", "V8", "M8[s]", "m8[D]", "c16", "g", "O"]:
    dt = np.dtype(spec)
    print(f"{spec:6s} mod={type(dt).__module__:14s} str={dt.str:8s} hasobject={dt.hasobject} fields={dt.fields}")
print("np.float64 scalar dtype mod:", type(np.float64(1).dtype).__module__)
print("longdouble itemsize", np.dtype('g').itemsize)
# same dtype.str, different dtype?
for a, b in [("g", "f8"), ("q", "l"), ("G", "c16")]:
    da, db = np.dtype(a), np.dtype(b)
    print(f"{a} vs {b}: str {da.str}/{db.str} equal_dtype={da == db} type_same={da.type is db.type} char {da.char}/{db.char}",
          "bytes_equal=", np.array([0.1], dtype=da).tobytes() == np.array([0.1], dtype=db).tobytes() if a != 'q' else "")
