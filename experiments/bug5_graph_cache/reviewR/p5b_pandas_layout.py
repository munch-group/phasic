import numpy as np, pandas as pd
rng = np.random.default_rng(2)
M = rng.random((50, 7))
df = pd.DataFrame(M)
a = df.to_numpy()
print("pandas", pd.__version__, "df.to_numpy() C?", a.flags.c_contiguous, "F?", a.flags.f_contiguous)
c = np.ascontiguousarray(a)
print("same content:", np.array_equal(a, c), " row sums bit-equal:", np.array_equal(a.sum(axis=1), c.sum(axis=1)))
