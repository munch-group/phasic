"""reviewZ probes, part 2 (unmodified master; HOME redirected by caller).

z6  with graph_cache's module-level logger neutralised (stub module in
    sys.modules), does anything ELSE import phasic_pybind before
    OMP_NUM_THREADS is set?  (checks the plan's claim that the lazy logger
    fix is sufficient for test 14)
z7  numpy 2 StringDType arrays against the plan's rule 3 gates
"""
import os, sys, subprocess, textwrap


def z6():
    code = textwrap.dedent(r'''
        import sys, os, types, traceback, importlib.abc
        stub = types.ModuleType("phasic.graph_cache")
        class GraphCache: pass
        stub.GraphCache = GraphCache
        stub.get_graph_cache_stats = lambda: None
        stub.print_graph_cache_info = lambda: None
        sys.modules["phasic.graph_cache"] = stub
        class Hook(importlib.abc.MetaPathFinder):
            done = False
            def find_spec(self, name, path=None, target=None):
                if name == "phasic.phasic_pybind" and not Hook.done:
                    Hook.done = True
                    st = traceback.extract_stack()
                    frames = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in st
                              if "phasic" in f.filename and "importlib" not in f.filename]
                    print("OMP set at pybind import:", "OMP_NUM_THREADS" in os.environ)
                    print("phasic frames:", frames[-6:])
                return None
        sys.meta_path.insert(0, Hook())
        try:
            import phasic
        except Exception as e:
            print("import failed (expected with stub?)", type(e).__name__, str(e)[:200])
    ''')
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print("[z6]", r.stdout.strip()[-800:], ("STDERR:" + r.stderr[-600:]) if r.returncode else "")


def z7():
    import numpy as np
    SD = np.dtypes.StringDType
    long1 = "x" * 40 + "A"
    long2 = "x" * 40 + "B"
    a = np.array([long1, "short"], dtype=SD())
    d = a.dtype
    print("[z7] StringDType: type module", type(d).__module__, "str", repr(d.str), "char", d.char,
          "hasobject", d.hasobject, "fields", d.fields, "itemsize", d.itemsize)
    def raw(x):
        try:
            return bytes(memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8)))
        except Exception as e:
            return f"view failed: {type(e).__name__}: {e}"
    ra = raw(a)
    print("[z7] uint8 view of StringDType array:", ra if isinstance(ra, str) else f"{len(ra)} bytes")
    if not isinstance(ra, str):
        b = np.array([long1, "short"], dtype=SD())
        print("[z7] equal arrays, equal raw bytes:", raw(b) == ra)
        # same-process reuse: free a, build a different array of equal shape/lengths
        del a, b
        hits = 0
        for i in range(200):
            p = np.array([long1, "short"], dtype=SD()); rp = raw(p); del p
            q = np.array([long2, "short"], dtype=SD()); rq = raw(q); del q
            if rp == rq:
                hits += 1
        print(f"[z7] DIFFERENT contents with IDENTICAL raw bytes: {hits}/200 trials")
    try:
        print("[z7] tobytes len", len(np.array([long1], dtype=SD()).tobytes()))
    except Exception as e:
        print("[z7] tobytes failed", type(e).__name__, e)


if __name__ == "__main__":
    assert "scratchpad" in os.environ.get("HOME", ""), "HOME must be redirected"
    for w in (sys.argv[1:] or ["z6", "z7"]):
        globals()[w]()
