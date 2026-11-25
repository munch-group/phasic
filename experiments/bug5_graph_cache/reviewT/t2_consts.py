"""reviewT t2: can a TUPLE code constant contain a frozenset (constant-only rule
that the plan's VALUE rules refuse)? And which constant types appear."""
import sys, types
print(sys.version)

srcs = {
    "in_list_of_sets": "def f(x):\n    return x in [{1, 2}, {3}]\n",
    "for_tuple_of_sets": "def f(x):\n    for s in ({1, 2}, {3}):\n        pass\n",
    "nested_in": "def f(x):\n    return x in ({1,2},)\n",
    "in_set_of_tuples": "def f(x):\n    return x in {(1, 2), (3, 4)}\n",
    "tuple_with_set_rhs": "def f(x, y):\n    return (x in {1, 2}, y)\n",
    "kw_default_tuple": "def g():\n    def f(x, idx=(Ellipsis, 0)):\n        return x[..., 0]\n    return f\n",
}


def walk(co, path, out):
    for i, c in enumerate(co.co_consts):
        p = f"{path}.co_consts[{i}]"
        if isinstance(c, types.CodeType):
            walk(c, p, out)
        else:
            out.append((p, type(c).__name__, c))
            if isinstance(c, tuple):
                for j, e in enumerate(c):
                    if isinstance(e, frozenset):
                        out.append((p + f"[{j}]", "FROZENSET-IN-TUPLE", e))


for name, s in srcs.items():
    ns = {}
    exec(compile(s, name, "exec"), ns)
    out = []
    walk(compile(s, name, "exec"), name, out)
    print(name)
    for row in out:
        print("   ", row)

# Is a function default the same object as a constant of its own code?
ns = {}
exec("def g():\n    def f(x, idx=(Ellipsis, 0)):\n        return x[..., 0]\n    return f\n", ns)
f = ns["g"]()
print("default is own const:", any(c is f.__defaults__[0] for c in f.__code__.co_consts))
