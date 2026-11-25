"""reviewS s8: is float.hex injective on Python floats? (NaN sign / payload)"""
import math, struct
a = float("nan"); b = -float("nan"); c = struct.unpack("<d", struct.pack("<Q", 0x7ff8000000000123))[0]
for x in (a, b, c):
    print(repr(x), x.hex(), struct.pack("<d", x).hex(), "copysign:", math.copysign(1.0, x))
print("a.hex()==b.hex():", a.hex() == b.hex(), " bits equal:", struct.pack("<d", a) == struct.pack("<d", b))
