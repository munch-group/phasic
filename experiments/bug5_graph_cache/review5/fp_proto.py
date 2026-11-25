"""Minimal prototype of plan v2.1 §2.2.1/§2.2.2 (code + function + value
fingerprint), written ONLY for review probes. Not phasic code."""
import functools, hashlib, types
import numpy as np


BUILTIN_AS_FUNCTION = False  # plan ambiguity: does a cell holding a builtin take the §2.2.1 builtin arm?


class NotFP(Exception):
    pass


def _tok(out, tag, payload):
    b = payload if isinstance(payload, bytes) else str(payload).encode()
    out.append(f"{tag}:{len(b)}:".encode() + b)


def code_fp(co, out):
    _tok(out, "code", co.co_name)
    for f in ("co_code", "co_exceptiontable"):
        _tok(out, f, getattr(co, f))
    for f in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
        _tok(out, f, repr(getattr(co, f)))
    for f in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags"):
        _tok(out, f, getattr(co, f))
    _tok(out, "nconsts", len(co.co_consts))
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            code_fp(c, out)
        elif type(c) is frozenset:
            subs = []
            for e in c:
                o = []
                value_fp(e, o, set(), 0)
                subs.append(b"".join(o))
            _tok(out, "frozenset", b"|".join(sorted(subs)))
        elif c is Ellipsis:
            _tok(out, "ellipsis", "")
        else:
            value_fp(c, out, set(), 0)


def func_fp(f, out, visited, depth):
    if id(f) in visited:
        _tok(out, "backref", visited[id(f)] if isinstance(visited, dict) else "")
        return
    if isinstance(visited, dict):
        visited[id(f)] = len(visited)
    else:
        visited.add(id(f))
    if isinstance(f, types.FunctionType):
        code_fp(f.__code__, out)
        value_fp(f.__defaults__, out, visited, depth + 1)
        value_fp(f.__kwdefaults__ and dict(f.__kwdefaults__), out, visited, depth + 1)
        for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            _tok(out, "cell", name)
            try:
                v = cell.cell_contents
            except ValueError:
                _tok(out, "empty", "")
                continue
            value_fp(v, out, visited, depth + 1)
    elif isinstance(f, types.MethodType):
        func_fp(f.__func__, out, visited, depth + 1)
        value_fp(f.__self__, out, visited, depth + 1)
    elif isinstance(f, functools.partial):
        func_fp(f.func, out, visited, depth + 1)
        value_fp(f.args, out, visited, depth + 1)
        value_fp(dict(f.keywords), out, visited, depth + 1)
    elif isinstance(f, (types.BuiltinFunctionType, np.ufunc)) or type(f).__name__ == "_ArrayFunctionDispatcher":
        _tok(out, "builtin", f"{getattr(f, '__module__', None)}.{getattr(f, '__qualname__', getattr(f, '__name__', '?'))}")
    else:
        raise NotFP(f"callable {type(f).__name__}")


def value_fp(x, out, visited, depth):
    if depth > 100:
        raise NotFP("depth")
    t = type(x)
    if x is None or t in (bool, int, str, bytes):
        _tok(out, t.__name__, repr(x))
    elif t is float:
        _tok(out, "float", x.hex())
    elif t is complex:
        _tok(out, "complex", x.real.hex() + "," + x.imag.hex())
    elif isinstance(x, np.generic):
        _tok(out, "npscalar", x.dtype.str.encode() + b"/" + x.tobytes())
    elif t is np.ndarray:
        if x.dtype == object:
            raise NotFP("object array")
        _tok(out, "ndarray", f"{x.dtype.str}{x.shape}".encode() + np.ascontiguousarray(x).tobytes())
    elif t in (list, tuple):
        _tok(out, t.__name__, len(x))
        for e in x:
            value_fp(e, out, visited, depth + 1)
    elif t is dict:
        _tok(out, "dict", len(x))
        for k, v in x.items():
            if type(k) is not str:
                raise NotFP("dict key")
            _tok(out, "key", k)
            value_fp(v, out, visited, depth + 1)
    elif isinstance(x, (types.FunctionType, types.MethodType, functools.partial)) or (
            BUILTIN_AS_FUNCTION and (isinstance(x, (types.BuiltinFunctionType, np.ufunc))
                                     or type(x).__name__ == "_ArrayFunctionDispatcher")):
        func_fp(x, out, visited, depth + 1)
    elif callable(getattr(x, "to_dict", None)):
        _tok(out, "obj", f"{t.__module__}.{t.__qualname__}")
        value_fp(x.to_dict(), out, visited, depth + 1)
    else:
        raise NotFP(f"{t.__module__}.{t.__qualname__}")


def fingerprint(f, visited_kind=dict):
    out = []
    func_fp(f, out, visited_kind(), 0)
    return hashlib.sha256(b"".join(out)).hexdigest()[:16]
