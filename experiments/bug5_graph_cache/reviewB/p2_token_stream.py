"""reviewB p2: §2.2 / §2.2.2 rule 4-5 taken literally.

The plan makes each TOKEN self-delimiting (<tag>:<len>:<payload>), but a
container is specified as "type + element fingerprints in order" with no
element count or end token. Then the mapping value -> token sequence is
not injective, so two different keyword values share an entry key.
Pure Python, no phasic import.
"""
import hashlib


def tok(tag, payload: bytes) -> bytes:
    return f"{tag}:{len(payload)}:".encode() + payload


def vf(x) -> bytes:
    # Literal reading of rules 1, 4, 5 (no counts, no end markers).
    if type(x) is int:
        return tok("int", hex(x).encode())
    if type(x) is str:
        return tok("str", repr(x).encode())
    if type(x) in (list, tuple):
        return tok(type(x).__name__, b"") + b"".join(vf(e) for e in x)
    if type(x) is dict:
        return tok("dict", b"") + b"".join(vf(k) + vf(v) for k, v in x.items())
    raise TypeError(type(x))


def key(**kw):
    s = b"".join(tok("kw", k.encode()) + vf(v) for k, v in kw.items())
    return hashlib.sha256(s).hexdigest()[:32]


pairs = [
    (dict(x=[[1], 2]), dict(x=[[1, 2]])),
    (dict(x={"a": {"b": 1}, "c": 2}), dict(x={"a": {"b": 1, "c": 2}})),
    (dict(ipv=[[[3, 0], 1]], n=[]), dict(ipv=[[[3, 0]], 1], n=[])),
]
for a, b in pairs:
    print(a, "vs", b, "-> same key:", key(**a) == key(**b))

# With a count in each container token the collision disappears:
def vf2(x) -> bytes:
    if type(x) is int:
        return tok("int", hex(x).encode())
    if type(x) is str:
        return tok("str", repr(x).encode())
    if type(x) in (list, tuple):
        return tok(type(x).__name__, str(len(x)).encode()) + b"".join(vf2(e) for e in x)
    if type(x) is dict:
        return tok("dict", str(len(x)).encode()) + b"".join(vf2(k) + vf2(v) for k, v in x.items())
    raise TypeError(type(x))

for a, b in pairs:
    sa = b"".join(tok("kw", k.encode()) + vf2(v) for k, v in a.items())
    sb = b"".join(tok("kw", k.encode()) + vf2(v) for k, v in b.items())
    print("with counts:", sa == sb)
