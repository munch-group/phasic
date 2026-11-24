"""Does an id-based cycle check give false positives? Compare a process-global
'seen ids' set (one reading of §2.2.2) with an ancestors-on-the-path set."""
import gc


class M:
    def __init__(self, v): self.v = v
    def to_dict(self): return {"v": [self.v]}      # fresh containers each call


def walk(x, seen, path_mode, path=()):
    if isinstance(x, (list, tuple, dict)) or hasattr(x, "to_dict"):
        key = id(x)
        pool = set(path) if path_mode else seen
        if key in pool:
            raise ValueError("cycle")
        if not path_mode:
            seen.add(key)
        path = path + (key,)
        if hasattr(x, "to_dict") and not isinstance(x, (list, tuple, dict)):
            walk(x.to_dict(), seen, path_mode, path)        # result freed after return
        elif isinstance(x, dict):
            for v in x.values(): walk(v, seen, path_mode, path)
        else:
            for v in x: walk(v, seen, path_mode, path)


cases = {
    "shared list referenced twice": {"a": [1, 2], "b": None},
    "ipv with one state list reused": [[[5, 0], 0.5]],
    "many to_dict objects (id reuse of freed results)": {f"m{i}": M(i) for i in range(200)},
}
s = [5, 0]; cases["ipv with one state list reused"] = [[s, 0.5], [s, 0.5]]
cases["shared list referenced twice"]["b"] = cases["shared list referenced twice"]["a"]
cyc = [1]; cyc.append(cyc); cases["true cycle"] = {"x": cyc}
for name, val in cases.items():
    for mode in (False, True):
        try:
            walk(val, set(), mode); r = "ok"
        except ValueError:
            r = "FLAGGED AS CYCLE"
        print(f"{name:52s} {'path-set' if mode else 'global-set'}: {r}")
