"""Review D probe 7: functions reached only through a to_dict() result are
temporaries; once the result is dropped, the next to_dict() call can create
a different function at the same id. A visited map keyed by id() without a
strong reference would then emit a back-reference for a different function."""
class M:
    def __init__(self, rate): self.rate = rate
    def to_dict(self):
        r = self.rate
        return {"hazard": lambda s: s * r}
ids = []
for m in (M(1.0), M(5.0)):
    d = m.to_dict()
    f = d["hazard"]
    ids.append(id(f)); cells = [c.cell_contents for c in f.__closure__]
    print("rate", m.rate, "id", hex(id(f)), "closure", cells)
    del d, f
print("same id for two different functions:", ids[0] == ids[1])
