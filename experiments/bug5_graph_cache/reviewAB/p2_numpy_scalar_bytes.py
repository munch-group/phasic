"""reviewAB p2: does rule 2's encoding (x.dtype.str + x.tobytes()) identify
numpy scalars uniquely?  And can a metaclass make `type(x) in frozenset`
run user code (rule 2's membership test)?  No phasic import needed."""
import numpy as np

def enc(x):
    return (x.dtype.str, x.tobytes())

pairs = [
    ("np.str_('a') vs np.str_('a\\x00')", np.str_('a'), np.str_('a\x00')),
    ("np.bytes_(b'a') vs np.bytes_(b'a\\x00')", np.bytes_(b'a'), np.bytes_(b'a\x00')),
    ("np.str_('') vs np.str_('\\x00')", np.str_(''), np.str_('\x00')),
]
for label, a, b in pairs:
    print(label)
    print("   python-equal:", a == b, " len:", len(a), len(b))
    print("   enc a:", enc(a))
    print("   enc b:", enc(b))
    print("   SAME ENCODING" if enc(a) == enc(b) else "   distinct")

# datetime64 generic unit / NaT
for a, b in [(np.datetime64('NaT'), np.datetime64('NaT', 's'))]:
    print("NaT generic vs s:", enc(a), enc(b))

# Rule-2 membership through a hostile metaclass
calls = []
class Meta(type):
    def __hash__(cls):
        calls.append("hash")
        return hash(np.float64)
    def __eq__(cls, other):
        calls.append("eq")
        return True
class Spoof(metaclass=Meta):
    pass
ALLOW = frozenset({np.float64, np.int64})
s = Spoof()
print("type(s) in ALLOW ->", type(s) in ALLOW, "; user code ran:", calls)
print("any(type(s) is t) ->", any(type(s) is t for t in ALLOW))
