"""reviewR p2: minimal prototype of the plan's v4.3 §2.2.2 walk (sub-digests,
id-memo with strong refs, current-path cycle check for containers, relative
back-references for functions, open subtrees not memoised).

Only the parts needed for the questions below are modelled (lists, ints,
plain Python functions with code+cells). Two variants:
  PLAN    : any subtree containing a back-reference that points above it is
            NOT memoised (plan text, 'any sub-digest whose subtree ...').
  NAIVE   : containers are memoised unconditionally, i.e. what an implementer
            would write if they trusted the plan's other sentence 'A cycle that
            passes through a CONTAINER ... hits the container cycle check and
            is refused' (so containers can never be open).
Questions:
  Q1  is a function -> container -> same function cycle refused (as the plan
      says) or keyed with a back-reference crossing the container?
  Q2  under NAIVE, can two different kwargs get equal digests?
"""
import hashlib, types


class KeyErr(Exception):
    pass


def tok(tag, payload: bytes) -> bytes:
    return tag.encode() + b":" + str(len(payload)).encode() + b":" + payload


class Walker:
    def __init__(self, naive):
        self.naive = naive
        self.memo = {}      # id -> (obj, digest)
        self.path = []      # current path (all node kinds)

    def walk(self, x):
        """Return (token_bytes, min_ref_depth); min_ref_depth = shallowest path
        index referenced by a back-reference inside x (inf if none)."""
        INF = 10 ** 9
        if type(x) is int:
            return tok("int", hex(x).encode()), INF
        if type(x) is str:
            return tok("str", repr(x).encode()), INF
        is_fn = type(x) is types.FunctionType
        is_list = type(x) is list
        if not (is_fn or is_list):
            raise KeyErr(f"not fingerprintable: {type(x).__name__}")
        # (1) current-path check
        for depth, node in enumerate(self.path):
            if node is x:
                if is_list:
                    raise KeyErr("container cycle")
                dist = len(self.path) - depth
                return tok("bref", str(dist).encode()), depth
        # (2) memo
        if id(x) in self.memo:
            return tok("sub", self.memo[id(x)][1]), INF
        # (3) walk
        my_depth = len(self.path)
        self.path.append(x)
        h = hashlib.sha256()
        mn = INF
        if is_list:
            h.update(tok("kind", b"list"))
            h.update(tok("n", str(len(x)).encode()))
            for e in x:
                t, m = self.walk(e)
                h.update(t)
                mn = min(mn, m)
        else:
            h.update(tok("kind", b"func"))
            co = x.__code__
            h.update(tok("code", co.co_code))
            h.update(tok("consts", repr(co.co_consts).encode()))
            h.update(tok("names", repr((co.co_names, co.co_freevars)).encode()))
            cells = x.__closure__ or ()
            h.update(tok("n", str(len(cells)).encode()))
            for c in cells:
                t, m = self.walk(c.cell_contents)
                h.update(t)
                mn = min(mn, m)
        self.path.pop()
        d = h.digest()
        closed = mn >= my_depth
        if closed or (self.naive and is_list):
            self.memo[id(x)] = (x, d)
        return tok("sub", d), mn


def key(callback, kwargs, naive):
    w = Walker(naive)
    h = hashlib.sha256()
    t, _ = w.walk(callback)
    h.update(t)
    for k, v in kwargs.items():
        h.update(tok("kw", k.encode()))
        t, _ = w.walk(v)
        h.update(t)
    return h.hexdigest()[:16]


def make_pair():
    """F2 holds C2 = [F2]; F3 holds C3 = [F3]; F2 and F3 have DIFFERENT code."""
    def mk2():
        C = []
        def F2(s):
            return C[0] is F2 and s + 2
        C.append(F2)
        return F2, C
    def mk3():
        C = []
        def F3(s):
            return C[0] is F3 and s * 5
        C.append(F3)
        return F3, C
    F2, C2 = mk2()
    F3, C3 = mk3()
    return F2, C2, F3, C3


F2, C2, F3, C3 = make_pair()

# Q1: callback = F2 (cycle F2 -> C2 -> F2 entered at the function)
for naive in (False, True):
    try:
        print(f"Q1 naive={naive}: key(F2) =", key(F2, {}, naive), "(keyed, not refused)")
    except KeyErr as e:
        print(f"Q1 naive={naive}: refused:", e)

# Q1b: callback captures the container first (cb -> C2 -> F2 -> C2)
def mkcb(C):
    def cb(s):
        return C
    return cb
for naive in (False, True):
    try:
        print(f"Q1b naive={naive}: key(cb->C2) =", key(mkcb(C2), {}, naive))
    except KeyErr as e:
        print(f"Q1b naive={naive}: refused:", e)

# Q2: one callback reaching BOTH F2 and F3 (so both containers get walked and,
# under NAIVE, memoised with a relative back-reference); the kwarg x is C2 in
# input A and C3 in input B.  The callback receives different functions.
def mkroot(a, b):
    def root(s, x=None):
        return a(s) + b(s) + x[0](s)
    return root
root = mkroot(F2, F3)
for naive in (False, True):
    try:
        kA = key(root, {"x": C2}, naive)
        kB = key(root, {"x": C3}, naive)
        print(f"Q2 naive={naive}: A={kA} B={kB} ->",
              "COLLISION (x=[F2] vs x=[F3])" if kA == kB else "distinct")
    except KeyErr as e:
        print(f"Q2 naive={naive}: refused:", e)
print("behaviour differs:", root(1, x=C2), "vs", root(1, x=C3))
