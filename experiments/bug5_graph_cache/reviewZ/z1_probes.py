"""reviewZ probes on UNMODIFIED master code (HOME redirected by caller).

z1  who imports phasic.phasic_pybind first during `import phasic` (OMP unset)
z2  master cache hit vs fresh build: C-level state/behaviour parity
z3  numpy dtype-class module gate battery
z4  co_code stability under sys.monitoring / settrace
z5  builtin / dispatcher attribute facts the plan relies on
"""
import os, sys, subprocess, json, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

def z1():
    code = textwrap.dedent(r'''
        import sys, os, traceback, importlib.abc
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
        import phasic
    ''')
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print("[z1]", r.stdout.strip()[-800:], ("STDERR:" + r.stderr[-400:]) if r.returncode else "")


def z2():
    import phasic, numpy as np
    from phasic import Graph
    gc_dir = os.path.join(os.environ["HOME"], ".phasic_cache", "graphs")

    def const_cb(state, n=4.0):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], k * (k - 1) / 2]]

    def param_cb(state, n=4.0):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], [k * (k - 1) / 2, 1.0, 0.5]]]

    def facts(g):
        out = {}
        out["V"] = g.vertices_length()
        out["parameterized"] = bool(g.parameterized())
        out["param_length"] = int(g.param_length())
        out["was_dph"] = bool(g.get_was_dph())
        out["dyn"] = bool(g.dyn_ordering)
        out["is_discrete"] = bool(g.is_discrete)
        # behaviour probes
        h = g.clone()
        try:
            v = h.find_or_create_vertex([7])
            h.vertex_at(1).add_edge(v, [1.0, 1.0, 1.0])
            out["add_param_edge"] = "ok"
        except Exception as e:
            out["add_param_edge"] = type(e).__name__
        h = g.clone()
        try:
            v = h.find_or_create_vertex([7])
            h.vertex_at(1).add_edge(v, 2.0)
            out["add_const_edge"] = "ok"
        except Exception as e:
            out["add_const_edge"] = type(e).__name__
        h = g.clone()
        try:
            h.update_weights([1.0, 1.0])
            out["uw2"] = "ok"
        except Exception as e:
            out["uw2"] = type(e).__name__ + ":" + str(e)[:60]
        try:
            out["E"] = float(g.expectation())
        except Exception as e:
            out["E"] = type(e).__name__
        return out

    for name, cb, kw in [("const", const_cb, {}), ("param", param_cb, {}),
                         ("param_td2", param_cb, {"theta_dim": 2})]:
        cold = Graph(cb, ipv=[5], **kw)       # builds and saves (master default cache on)
        nfiles = len(os.listdir(gc_dir)) if os.path.isdir(gc_dir) else 0
        hit = Graph(cb, ipv=[5], **kw)        # should be a hit on master
        fc, fh = facts(cold), facts(hit)
        diff = {k: (fc[k], fh[k]) for k in fc if fc[k] != fh[k]}
        print(f"[z2] {name}: files={nfiles} fresh={fc}")
        print(f"[z2] {name}: hit-vs-fresh differences: {diff or 'none'}")


def z3():
    import numpy as np
    print("[z3] numpy", np.__version__)
    for s in ["f8", ">f8", "i4", "?", "U3", "S3", "M8[s]", "m8[ns]", "V8", "O", "c16", "f2", "g", "G"]:
        d = np.dtype(s)
        print(f"[z3] {s:7} module={type(d).__module__} str={d.str} fields={d.fields is not None} obj={d.hasobject}")
    try:
        d = np.dtypes.StringDType()
        print("[z3] StringDType module", type(d).__module__, d.str)
    except Exception as e:
        print("[z3] StringDType", e)
    try:
        import ml_dtypes
        for t in ["float8_e4m3fn", "bfloat16", "int4"]:
            d = np.dtype(getattr(ml_dtypes, t))
            print(f"[z3] ml_dtypes.{t} module={type(d).__module__} str={d.str}")
    except ImportError:
        print("[z3] ml_dtypes not installed")
    # dtype with metadata
    d = np.dtype("f8", metadata={"a": 1})
    print("[z3] f8+metadata module", type(d).__module__, d.str)


def z4():
    import types
    def f(state, a=1):
        x = state[0] + a
        return [x, x * 2]
    c0 = f.__code__.co_code
    for _ in range(200):
        f([1])  # warm up specialisation
    c1 = f.__code__.co_code
    seen = []
    sys.settrace(lambda *a: None)
    try:
        f([1]); c2 = f.__code__.co_code
    finally:
        sys.settrace(None)
    c3 = None
    if hasattr(sys, "monitoring"):
        m = sys.monitoring
        tid = 3
        m.use_tool_id(tid, "probe")
        m.set_events(tid, m.events.LINE | m.events.CALL)
        m.register_callback(tid, m.events.LINE, lambda *a: None)
        f([1]); c3 = f.__code__.co_code
        m.set_events(tid, 0); m.free_tool_id(tid)
    print("[z4] co_code stable after warmup:", c0 == c1, "under settrace:", c0 == c2,
          "under sys.monitoring:", c3 == c0 if c3 is not None else "n/a")


def z5():
    import math, numpy as np, functools, itertools, types
    def info(o):
        return dict(type=type(o).__name__, module=getattr(o, "__module__", "<none>"),
                    qualname=getattr(o, "__qualname__", "<none>"), name=getattr(o, "__name__", "<none>"),
                    self_type=type(getattr(o, "__self__", None)).__name__)
    for nm, o in [("math.sqrt", math.sqrt), ("len", len), ("np.exp", np.exp), ("np.sum", np.sum),
                  ("np.linalg.solve", np.linalg.solve), ("functools.reduce", functools.reduce)]:
        print("[z5]", nm, info(o))
    print("[z5] type(np.sum) is type(np.linalg.solve):", type(np.sum) is type(np.linalg.solve))
    t = itertools.combinations_with_replacement
    print("[z5] cwr immutable flag:", bool(t.__flags__ & (1 << 8)), "type is type:", type(t) is type)
    print("[z5] np.float64 immutable flag:", bool(np.float64.__flags__ & (1 << 8)))


if __name__ == "__main__":
    which = sys.argv[1:] or ["z1", "z2", "z3", "z4", "z5"]
    assert "scratchpad" in os.environ.get("HOME", ""), "HOME must be redirected"
    for w in which:
        globals()[w]()
