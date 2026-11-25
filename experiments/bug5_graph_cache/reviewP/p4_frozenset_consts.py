"""Probe: (1) CPython merges an equal tuple constant appearing both inside a
frozenset constant and standalone in the same code object, so the two are
ONE object (a container back-reference candidate); (2) the frozenset's
iteration order depends on PYTHONHASHSEED for string-containing tuples.
Together: if frozenset elements are walked in iteration order with the
shared back-reference map, which occurrence gets the first-visit index (and
hence the stream) depends on the hash seed."""
import sys

def cb(state, labels=("a", "b", "c")):
    key = (labels[state[0]], state[1])
    if key in {("a", 1), ("b", 2), ("c", 3)}:
        if key == ("a", 1):
            return 1
        return 2
    return 0

co = cb.__code__
fs = [c for c in co.co_consts if isinstance(c, frozenset)][0]
standalone = [c for c in co.co_consts if c == ("a", 1)][0]
inner = [e for e in fs if e == ("a", 1)][0]
print(sys.version.split()[0])
print("co_consts:", co.co_consts)
print("standalone ('a',1) is the frozenset's element:", standalone is inner)
print("frozenset iteration order:", list(fs))
