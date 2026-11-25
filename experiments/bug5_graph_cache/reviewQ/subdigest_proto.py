"""Review Q: prototype of the v4.2 sub-digest scheme (plan section 2.2.2 end),
written from the plan text only. Not phasic code.

Choices where the plan is silent (documented as findings if load-bearing):
- path depth counts every container/function frame;
- back-ref token = distance up the current path to the function;
- a subtree is CLOSED iff no back-ref inside it targets a frame above it;
  closed sub-digests are memoised by id (strong refs), open ones are not;
- constants and values share one memo (plan does not separate them).
"""
import hashlib, types, functools, sys

INF = 10 ** 9


class Refused(Exception):
    pass


def tok(tag, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    return tag.encode() + b":" + str(len(payload)).encode() + b":" + payload


class Walker:
    def __init__(self):
        self.memo = {}      # id -> (obj, token)
        self.onpath = {}    # id -> depth
        self.depth = 0
        self.visits = 0     # composite walks actually performed

    # ---- generic composite handling ---------------------------------
    def _composite(self, x, build, is_func):
        i = id(x)
        if i in self.onpath:
            if is_func:
                d = self.onpath[i]
                return tok("B", str(self.depth - d)), d
            raise Refused(f"cycle through container {type(x).__name__}")
        if i in self.memo:
            return self.memo[i][1], INF
        self.visits += 1
        my = self.depth
        self.onpath[i] = my
        self.depth += 1
        try:
            parts, minref = build()
        finally:
            self.depth -= 1
            del self.onpath[i]
        t = tok("S", hashlib.sha256(b"".join(parts)).digest())
        if minref >= my:          # closed: no back-ref above this frame
            self.memo[i] = (x, t)
            return t, INF
        return t, minref

    def _seq(self, items, mode):
        parts, mr = [tok("n", str(len(items)))], INF
        for e in items:
            t, m = self.value(e, mode=mode)
            parts.append(t)
            mr = min(mr, m)
        return parts, mr

    # ---- values ------------------------------------------------------
    def value(self, x, mode="value", cell=False):
        tx = type(x)
        if x is None or tx in (bool, str, bytes):
            return tok("v" + tx.__name__, repr(x)), INF
        if tx is int:
            return tok("int", hex(x)), INF
        if tx is float:
            return tok("float", x.hex()), INF
        if x is Ellipsis:
            return tok("ellipsis", ""), INF
        if tx in (list, tuple):
            def b():
                p, m = self._seq(list(x), mode)
                return [tok("type", tx.__name__)] + p, m
            return self._composite(x, b, False)
        if tx is dict:
            def b():
                parts, mr = [tok("type", "dict"), tok("n", str(len(x)))], INF
                for k, v in x.items():
                    for e in (k, v):
                        t, m = self.value(e, mode=mode)
                        parts.append(t)
                        mr = min(mr, m)
                return parts, mr
            return self._composite(x, b, False)
        if tx is frozenset and mode == "const":
            # elements' sub-digests sorted (hash-seed independent)
            toks = sorted(self.value(e, mode="const")[0] for e in x)
            return tok("frozenset", b"".join([tok("n", str(len(toks)))] + toks)), INF
        if tx is types.CodeType and mode == "const":
            return self.code(x)
        if tx is types.FunctionType:
            return self._composite(x, lambda: self._func(x), True)
        if tx is functools.partial:
            def b():
                parts, mr = [tok("type", "partial")], INF
                for e in (x.func, x.args, x.keywords, dict(vars(x))):
                    t, m = self.value(e)
                    parts.append(t)
                    mr = min(mr, m)
                return parts, mr
            return self._composite(x, b, False)  # plan: partial is not 'function'
        if tx is types.ModuleType and cell:
            if sys.modules.get(x.__name__) is not x:
                raise Refused("module not in sys.modules")
            return tok("module", x.__name__), INF
        if tx is types.BuiltinFunctionType and (x.__self__ is None or type(x.__self__) is types.ModuleType):
            return tok("builtin", f"{x.__module__}.{x.__qualname__}"), INF
        raise Refused(f"not fingerprintable: {tx.__name__}")

    def _func(self, f):
        mod = f.__module__
        if mod == "__mp_main__":
            mod = "__main__"
        parts, mr = [tok("type", "function")], INF
        for e in (mod,):
            parts.append(self.value(e)[0])
        t, m = self.code(f.__code__); parts.append(t); mr = min(mr, m)
        for e in (f.__defaults__, f.__kwdefaults__, dict(f.__dict__)):
            t, m = self.value(e); parts.append(t); mr = min(mr, m)
        cells = f.__closure__ or ()
        parts.append(tok("n", str(len(cells))))
        for c in cells:
            try:
                content = c.cell_contents
            except ValueError:
                parts.append(tok("empty", ""))
                continue
            t, m = self.value(content, cell=True)
            parts.append(t)
            mr = min(mr, m)
        return parts, mr

    def code(self, co):
        parts = [tok("type", "code")]
        for name in ("co_code", "co_exceptiontable"):
            parts.append(tok(name, getattr(co, name, b"")))
        for name in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
            v = getattr(co, name)
            parts.append(tok(name, b"".join([tok("n", str(len(v)))] + [tok("s", s) for s in v])))
        for name in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags", "co_name"):
            parts.append(tok(name, str(getattr(co, name))))
        parts.append(tok("n", str(len(co.co_consts))))
        for c in co.co_consts:
            parts.append(self.value(c, mode="const")[0])
        return tok("code", hashlib.sha256(b"".join(parts)).digest()), INF


def digest(callback, **kw):
    w = Walker()
    t, _ = w.value(callback)
    kt = [w.value(k)[0] + w.value(v)[0] for k, v in kw.items()]
    return hashlib.sha256(t + b"".join(kt)).hexdigest(), w
