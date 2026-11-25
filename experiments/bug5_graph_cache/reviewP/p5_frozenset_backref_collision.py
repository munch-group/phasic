"""Toy model of plan v4.1's CODE FINGERPRINT read literally:
- co_consts element by element; tuple constants element by element;
- frozenset constants "by their elements' fingerprints, sorted";
- any container (list, tuple, dict) whose id() was already walked earlier in
  the same key computation contributes a BACK-REFERENCE token carrying its
  first-visit index (sec. 2.2.2, last paragraph), with a strong-ref map.
Two callbacks that differ ONLY in which tuple literal they compare against
(an ordinary one-constant edit) are fingerprinted in two processes with
different PYTHONHASHSEED values.  Usage: p5 X|Y  (run under a given seed)."""
import hashlib, sys, types

def cb_X(state, labels=("a", "b", "c")):
    key = (labels[state[0]], state[1])
    if key in {("a", 1), ("b", 2), ("c", 3)}:
        if key == ("a", 1):
            return 1
        return 2
    return 0

def cb_Y(state, labels=("a", "b", "c")):
    key = (labels[state[0]], state[1])
    if key in {("a", 1), ("b", 2), ("c", 3)}:
        if key == ("c", 3):
            return 1
        return 2
    return 0

class Walker:
    def __init__(self):
        self.seen = {}      # id -> (index, strong ref)

    def tok(self, tag, payload: bytes) -> bytes:
        return tag.encode() + b":" + str(len(payload)).encode() + b":" + payload

    def const(self, c) -> bytes:
        if c is None or type(c) in (int, str, bool, float):
            return self.tok(type(c).__name__, repr(c).encode())
        if type(c) is tuple:
            if id(c) in self.seen:
                return self.tok("ref", str(self.seen[id(c)][0]).encode())
            self.seen[id(c)] = (len(self.seen), c)
            body = b"".join(self.const(e) for e in c)
            return self.tok("tuple", str(len(c)).encode() + b"|" + body)
        if type(c) is frozenset:
            parts = sorted(self.const(e) for e in c)   # walked in iteration order, then sorted
            return self.tok("frozenset", str(len(c)).encode() + b"|" + b"".join(parts))
        if isinstance(c, types.CodeType):
            return self.code(c)
        raise TypeError(type(c))

    def code(self, co) -> bytes:
        out = [self.tok("co_code", co.co_code),
               self.tok("names", repr(co.co_names).encode()),
               self.tok("varnames", repr(co.co_varnames).encode()),
               self.tok("nconsts", str(len(co.co_consts)).encode())]
        out += [self.const(c) for c in co.co_consts]
        return b"".join(out)

which = sys.argv[1]
f = cb_X if which == "X" else cb_Y
w = Walker()
stream = w.code(f.__code__) + w.const(f.__defaults__)
fs = [c for c in f.__code__.co_consts if type(c) is frozenset][0]
print(f"callback {which}: frozenset iteration order {list(fs)}")
print(f"callback {which}: digest {hashlib.sha256(stream).hexdigest()[:32]}")
print(f"callback {which}: f((0,1)) = {f((0, 1))}, f((2,3)) = {f((2, 3))}")
