"""Review D probe 3: plan rule 8 keys any object with to_dict() by what
to_dict() returns. Is pandas' to_dict() injective on DataFrame/Series content?
No phasic import, no files."""
import warnings
import pandas as pd, numpy as np
print("pandas", pd.__version__)
a = pd.DataFrame({"rate": [1.0, 2.0, 3.0]}, index=["p1", "p1", "p2"])
b = pd.DataFrame({"rate": [9.0, 2.0, 3.0]}, index=["p1", "p1", "p2"])
print("dup index: a.to_dict()", a.to_dict(), " b.to_dict()", b.to_dict(),
      " equal:", a.to_dict() == b.to_dict(), " frames equal:", a.equals(b))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    c = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    d = pd.DataFrame([[7.0, 2.0]], columns=["x", "x"])
    print("dup columns: c", c.to_dict(), " d", d.to_dict(), " equal:", c.to_dict() == d.to_dict(),
          " warnings:", [str(x.message)[:60] for x in w])
s1 = pd.Series([1.0, 5.0], index=[0, 0]); s2 = pd.Series([4.0, 5.0], index=[0, 0])
print("Series dup index:", s1.to_dict(), s2.to_dict(), s1.to_dict() == s2.to_dict())
# row order of a callback iterating df.itertuples() vs to_dict order
e = pd.DataFrame({"r": [1.0, 2.0]}, index=[0, 1]); f = e.iloc[::-1]
print("row order: e", e.to_dict(), " f", f.to_dict(), " dict insertion order lists:",
      list(e.to_dict()["r"].items()), list(f.to_dict()["r"].items()))
