"""Review D probe 6: numpy 2 StringDType arrays under plan rule 3
(dtype.str + shape + tobytes(); refused only if hasobject or fields)."""
import numpy as np
from numpy.dtypes import StringDType
print("numpy", np.__version__)
a = np.array(["x" * 40, "y" * 40], dtype=StringDType())
b = np.array(["p" * 40, "q" * 40], dtype=StringDType())
for n, x in (("a", a), ("b", b)):
    print(n, "dtype.str", x.dtype.str, "hasobject", x.dtype.hasobject, "fields", x.dtype.fields,
          "tobytes", x.tobytes().hex())
print("a,b contents differ:", (a != b).all(), " tobytes equal:", a.tobytes() == b.tobytes())
c = np.array(["x" * 40, "y" * 40], dtype=StringDType())
print("a,c contents equal:", (a == c).all(), " tobytes equal:", a.tobytes() == c.tobytes())
s1 = np.array(["ab", "cd"], dtype=StringDType()); s2 = np.array(["ab", "ce"], dtype=StringDType())
print("short strings: tobytes equal:", s1.tobytes() == s2.tobytes())
