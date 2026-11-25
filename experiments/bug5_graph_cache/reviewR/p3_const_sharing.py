"""reviewR p3: one object reached both as a CODE CONSTANT and as a VALUE.
The plan memoises sub-digests by id in ONE memo shared by values and code
constants (§2.2.2 'Sub-digests'), while tuples are encoded under two rule
sets (constant rules §2.2.1 vs value rule 4).  Is such sharing real?"""
import numpy as np

def make():
    def cb(state, idx=(Ellipsis, 0), shape=(2, 3)):
        a = state[..., 0]
        return (2, 3), idx, shape, a
    return cb

cb = make()
consts = cb.__code__.co_consts
print("co_consts:", consts)
print("defaults :", cb.__defaults__)
for d in cb.__defaults__:
    shared = [c for c in consts if c is d]
    print(f"  default {d!r}: same object as a constant of cb's code -> {bool(shared)}")

# frozenset constant whose elements are tuples that also appear as defaults
def make2():
    def cb2(state, k=("a", 1)):
        return state in {("a", 1), ("b", 2), ("c", 3)} and k
    return cb2
cb2 = make2()
fs = [c for c in cb2.__code__.co_consts if type(c) is frozenset][0]
print("cb2 default ('a',1) is an element object of the frozenset constant:",
      any(e is cb2.__defaults__[0] for e in fs))
