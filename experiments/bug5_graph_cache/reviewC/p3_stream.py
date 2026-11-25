"""Review C p3: prototype of the v2.7 section 2.2 token stream, written
from the plan text only, then attacked for equal streams on different
inputs. Not phasic code."""
import sys, types, functools, random, itertools, hashlib
import numpy as np


class KeyErr(ValueError):
    pass


def tok(out, tag, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    out.append(tag.encode() + b":" + str(len(payload)).encode() + b":" + payload)


class FP:
    def __init__(self):
        self.out = []
        self.funcs = {}          # id -> first-visit index
        self.path = []           # ids on current path

    # ---- values -------------------------------------------------------
    def value(self, x, depth=0):
        if depth > 100:
            raise KeyErr("depth")
        t = type(x)
        if x is None or t in (bool, str, bytes):
            tok(self.out, t.__name__, repr(x)); return
        if t is int:
            tok(self.out, "int", hex(x)); return
        if t is float:
            tok(self.out, "float", x.hex()); return
        if t is complex:
            tok(self.out, "complex", x.real.hex() + "," + x.imag.hex()); return
        if isinstance(x, np.generic):
            if isinstance(x, np.void) or x.dtype.fields is not None or x.dtype.hasobject:
                raise KeyErr("npvoid")
            tok(self.out, "npscalar", x.dtype.str); tok(self.out, "bytes", x.tobytes()); return
        if t is np.ndarray:
            if x.dtype.hasobject or x.dtype.fields is not None:
                raise KeyErr("ndarray dtype")
            tok(self.out, "ndarray", x.dtype.str)
            tok(self.out, "shape", str(x.ndim))
            for d in x.shape:
                tok(self.out, "int", hex(d))
            tok(self.out, "bytes", np.ascontiguousarray(x).tobytes()); return
        if t in (list, tuple, dict):
            if id(x) in self.path:
                raise KeyErr("cycle")
            self.path.append(id(x))
            try:
                if t is dict:
                    tok(self.out, "dict", str(len(x)))
                    for k, v in x.items():
                        if type(k) is not str:
                            raise KeyErr("dict key")
                        tok(self.out, "str", repr(k)); self.value(v, depth + 1)
                else:
                    tok(self.out, t.__name__, str(len(x)))
                    for v in x:
                        self.value(v, depth + 1)
            finally:
                self.path.pop()
            return
        if isinstance(x, types.ModuleType):
            if sys.modules.get(x.__name__) is not x:
                raise KeyErr("module")
            tok(self.out, "module", x.__name__); return
        if callable(x) and not isinstance(x, type) and not callable(getattr(x, "to_dict", None)):
            self.func(x, depth); return
        if isinstance(x, type):
            self.imm_type(x); return
        if callable(getattr(x, "to_dict", None)):
            if id(x) in self.path:
                raise KeyErr("cycle")
            self.path.append(id(x))
            try:
                tok(self.out, "obj", type(x).__module__ + "." + type(x).__qualname__)
                self.value(x.to_dict(), depth + 1)
            finally:
                self.path.pop()
            return
        raise KeyErr("unfingerprintable %s" % t.__name__)

    def imm_type(self, x):
        if not (x.__flags__ & (1 << 8)):
            raise KeyErr("mutable type")
        mod = sys.modules.get(getattr(x, "__module__", None) or "")
        o = mod
        for p in x.__qualname__.split("."):
            o = getattr(o, p, None)
        if o is not x:
            raise KeyErr("type resolve")
        tok(self.out, "type", x.__module__ + "." + x.__qualname__)

    # ---- functions ----------------------------------------------------
    def func(self, f, depth=0):
        if isinstance(f, types.FunctionType):
            if id(f) in self.funcs:
                tok(self.out, "backref", str(self.funcs[id(f)])); return
            self.funcs[id(f)] = len(self.funcs)
            tok(self.out, "function", "")
            self.code(f.__code__)
            self.value(f.__defaults__, depth + 1)
            self.value(f.__kwdefaults__, depth + 1)
            cl = f.__closure__ or ()
            tok(self.out, "closure", str(len(cl)))
            for c in cl:
                try:
                    v = c.cell_contents
                except ValueError:
                    tok(self.out, "empty", ""); continue
                self.value(v, depth + 1)
            return
        if isinstance(f, types.MethodType):
            tok(self.out, "method", "")
            self.func(f.__func__, depth + 1); self.value(f.__self__, depth + 1); return
        if type(f) is functools.partial:
            tok(self.out, "partial", "")
            self.value(f.func, depth + 1)
            tok(self.out, "args", str(len(f.args)))
            for a in f.args:
                self.value(a, depth + 1)
            self.value(dict(f.keywords), depth + 1)
            return
        s = getattr(f, "__self__", None)
        if s is None or isinstance(s, types.ModuleType):
            mod = getattr(f, "__module__", None)
            name = getattr(f, "__qualname__", None) or getattr(f, "__name__", None)
            m = sys.modules.get(mod) if mod else None
            o = m
            if m is not None and name:
                for p in name.split("."):
                    o = getattr(o, p, None)
            if m is not None and o is f:
                tok(self.out, "builtin", mod + "." + name); return
        raise KeyErr("callable %r" % (f,))

    def code(self, co):
        tok(self.out, "code", "")
        tok(self.out, "bytes", co.co_code)
        tok(self.out, "bytes", getattr(co, "co_exceptiontable", b""))
        for fld in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
            v = getattr(co, fld)
            tok(self.out, "tuple", str(len(v)))
            for s in v:
                tok(self.out, "str", s)
        for fld in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags"):
            tok(self.out, "int", hex(getattr(co, fld)))
        tok(self.out, "str", co.co_name)
        tok(self.out, "consts", str(len(co.co_consts)))
        for c in co.co_consts:
            self.const(c)

    def const(self, c):
        if isinstance(c, types.CodeType):
            self.code(c)
        elif type(c) is frozenset:
            subs = []
            for e in c:
                sub = FP(); sub.const(e); subs.append(b"".join(sub.out))
            tok(self.out, "frozenset", str(len(subs)))
            self.out.extend(sorted(subs))
        elif c is Ellipsis:
            tok(self.out, "ellipsis", "")
        elif type(c) is slice:
            tok(self.out, "slice", "")
            for v in (c.start, c.stop, c.step):
                self.const(v)
        elif type(c) is tuple:
            tok(self.out, "tuple", str(len(c)))
            for v in c:
                self.const(v)
        else:
            self.value(c)


def stream(cache_key, cb, **kw):
    fp = FP()
    tok(fp.out, "fmt", "2")
    tok(fp.out, "py", "%d.%d" % sys.version_info[:2])
    tok(fp.out, "build", "X")
    tok(fp.out, "str", cache_key)
    fp.func(cb)
    tok(fp.out, "kwargs", str(len(kw)))
    for k, v in kw.items():
        tok(fp.out, "str", k); fp.value(v)
    return b"".join(fp.out)


def cb(state, **kw):
    return []


# ---------- 1. targeted pairs (expected distinct) -----------------------
pairs = [
    ([[1], 2], [[1, 2]]),
    ({'a': {'b': 1}, 'c': 2}, {'a': {'b': 1, 'c': 2}}),
    ([[[3, 0], 1]], [[[3, 0]], 1]),
    ([1, 2], (1, 2)),
    (1, 1.0), (1, True), (0.0, -0.0),
    ("a:1:b", ["a", "b"]),
    (b"x", "x"),
    (np.array([1, 2]), np.array([[1, 2]])),
    (np.array([1, 2], dtype='<i8'), np.array([1, 2], dtype='<i4')),
    (np.array([], dtype=float).reshape(0, 3), np.array([], dtype=float).reshape(3, 0)),
    (np.int64(1), np.array(1, dtype=np.int64)),
    (np.float64(1.0), 1.0),
    ({'x': 1}, [('x', 1)]),
    ("str:1:a", "str:1:a "),
]
coll = 0
for a, b in pairs:
    if stream("k", cb, v=a) == stream("k", cb, v=b):
        coll += 1; print("COLLISION", repr(a), repr(b))
# kwargs-block splits
if stream("k", cb, a=[1], b=2) == stream("k", cb, a=[1, 2]):
    coll += 1; print("COLLISION kwargs split")
if stream("k:2", cb) == stream("k", cb, **{"2": 0}):
    coll += 1; print("COLLISION key/kwarg")
print("targeted pairs collisions:", coll)

# ---------- 2. closures / code ----------------------------------------
def make(rate):
    def f(state, **kw):
        return rate
    return f
def make2(a, b):
    def f(state, **kw):
        return (a, b)
    return f
print("closure rate 1 vs 5 distinct:", stream("k", make(1.0)) != stream("k", make(5.0)))
print("closure ([1],2) vs ([1,2],?) distinct:", stream("k", make2([1], 2)) != stream("k", make2([1, 2], None)))
l1 = eval("lambda s, **k: 1"); l2 = eval("lambda s, **k: 2")
print("two lambdas distinct:", stream("k", l1) != stream("k", l2))
# frozenset constants
def fs1(s, **k): return s in {1, 2, (3, 4)}
def fs2(s, **k): return s in {1, (2, 3), 4}
print("frozenset consts distinct:", stream("k", fs1) != stream("k", fs2))

# recursion shapes
def mk_rec():
    def a(s, **k): return b(s)
    def b(s, **k): return a(s)
    return a
def mk_rec2():
    def a(s, **k): return b(s)
    def b(s, **k): return b(s)
    return a
print("recursion shapes distinct:", stream("k", mk_rec()) != stream("k", mk_rec2()))

# module in cell
def mk_mod():
    import numpy as np_
    def f(s, **k): return np_.exp(1)
    return f
print("module cell hashes:", len(stream("k", mk_mod())) > 0)

# ---------- 3. random fuzz: stream equality <=> canonical equality -----
def canon(x):
    t = type(x)
    if t is np.ndarray:
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if t in (list, tuple):
        return (t.__name__, tuple(canon(v) for v in x))
    if t is dict:
        return ("dict", tuple((k, canon(v)) for k, v in x.items()))
    return (t.__name__, repr(x) if t is not float else x.hex())

rng = random.Random(0)
atoms = [0, 1, 2, -1, 1.0, 0.5, -0.0, True, False, None, "", "a", "1", "a:1:", b"", b"a",
         np.int8(1), np.float32(1), np.array([1]), np.array([[1]]), np.array([], dtype=int)]
def gen(d):
    r = rng.random()
    if d > 3 or r < 0.4:
        return rng.choice(atoms)
    n = rng.randint(0, 3)
    k = rng.random()
    if k < 0.4:
        return [gen(d + 1) for _ in range(n)]
    if k < 0.7:
        return tuple(gen(d + 1) for _ in range(n))
    return {rng.choice(["a", "b", "c", "ab", ""]) + str(i): gen(d + 1) for i in range(n)}
seen = {}
bad = 0
N = 60000
for _ in range(N):
    nkw = rng.randint(0, 2)
    kw = {rng.choice(["x", "y", "xy"]) + str(i): gen(0) for i in range(nkw)}
    s = hashlib.sha256(stream("k", cb, **kw)).hexdigest()
    c = tuple((k, canon(v)) for k, v in kw.items())
    if s in seen and seen[s] != c:
        bad += 1
        if bad < 4:
            print("FUZZ COLLISION", seen[s], c)
    seen.setdefault(s, c)
print("fuzz:", N, "inputs,", len(seen), "distinct streams, collisions:", bad)
