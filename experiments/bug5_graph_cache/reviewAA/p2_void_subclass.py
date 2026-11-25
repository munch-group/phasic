"""reviewAA p2: does the plan's attribute-free numpy-scalar gate
`type(x) is np.dtype(type(x)).type` admit Python subclasses of np.void
(np.record, user subclasses)?  If so, x.dtype / x.tobytes() (user code) is
read afterwards."""
import numpy as np

calls = []

class V(np.void):
    @property
    def dtype(self):
        calls.append("dtype")
        return np.dtype("<f8")

    def tobytes(self):
        calls.append("tobytes")
        return b"\0" * 8

rec = np.zeros(1, dtype=[("a", "<i8")]).view(np.recarray)[0]
print("np.record instance type:", type(rec))
for name, cls in [("np.record", np.record), ("user V(np.void)", V)]:
    try:
        d = np.dtype(cls)
        print(f"{name}: np.dtype(cls) = {d!r}, .type is cls: {d.type is cls}")
    except Exception as ex:
        print(f"{name}: np.dtype(cls) raises {type(ex).__name__}: {ex}")

# Make a V instance
try:
    v = np.zeros(1, dtype=np.dtype((V, [("a", "<i8")])))[0]
    print("V instance type:", type(v), "gate passes:", type(v) is np.dtype(type(v)).type)
    print("x.dtype ->", v.dtype, "calls:", calls)
except Exception as ex:
    print("V instance creation raises", type(ex).__name__, ex)
print("rec gate passes:", type(rec) is np.dtype(type(rec)).type)
