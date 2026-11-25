"""reviewS s7: can CPython produce tuple constants holding constants the VALUE rules refuse
(frozenset, code)? If yes, the shared memo's admit/refuse outcome depends on walk order."""
import types, sys
srcs = {
 "in_set": "def f(x):\n    return x in {1, 2}\n",
 "tuple_of_set": "def f(x):\n    return x in ({1, 2}, 3)\n",
 "for_set": "def f(x):\n    for y in {1, 2}: pass\n",
 "match_map": "def f(x):\n    match x:\n        case {'a': 1, 'b': 2}: return 1\n",
 "match_or": "def f(x):\n    match x:\n        case (1 | 2, {3: 4}): return 1\n",
 "nested_in": "def f(x):\n    return (x in {1,2}, x in {(1,2)})\n",
 "default_tuple": "def f(x, idx=(..., 0)):\n    return x[..., 0]\n",
 "slice_c": "def f(M):\n    return M[1:3, 0]\n",
}
def walk(c, out):
    for k in c.co_consts:
        if isinstance(k, types.CodeType): walk(k, out)
        elif type(k) is tuple:
            kinds = {type(e).__name__ for e in k}
            out.append(("tuple", sorted(kinds)))
        else:
            out.append((type(k).__name__,))
print(sys.version.split()[0])
for name, s in srcs.items():
    ns = {}; exec(compile(s, name, "exec"), ns)
    out = []; walk(ns["f"].__code__, out)
    flag = any(t[0] == "tuple" and ("frozenset" in t[1] or "code" in t[1] or "slice" in t[1]) for t in out)
    print(f"{name:14s} {'TUPLE-HOLDS-REFUSED-VALUE ' if flag else ''}{out}")
