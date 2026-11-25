# reviewAN p6: does the planned array-byte expression work on every admitted dtype/shape?
import numpy as np
def planned(x):
    return bytes(memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8)))
cases = {
 "dt64": np.array(['2020-01-01'], dtype='M8[D]'),
 "td64": np.array([1, 2], dtype='m8[s]'),
 "0d": np.array(3.5),
 "U0": np.zeros(3, dtype='U0'),
 "S0": np.zeros(3, dtype='S0'),
 "empty2d": np.empty((0, 3)),
 "noncontig": np.arange(12.).reshape(3, 4)[:, ::2],
 "bool": np.array([True, False]),
}
for k, x in cases.items():
    try:
        b = planned(x); print(k, x.dtype.str, x.dtype.type.__name__, x.shape, "ok", len(b))
    except Exception as e:
        print(k, x.dtype.str, x.shape, "RAISES", type(e).__name__, e)
# 'U0' / 'S0' collision check: different shapes give same (empty) bytes; shape is keyed separately
