"""Review C p4: tuple constants holding values outside the VALUE rules."""
import sys
print(sys.version.split()[0])
def f1(state, **kw):
    return state[..., 0]
def f2(state, **kw):
    return state[1:3, 0]
def f3(state, **kw):
    return state[:, 0]
def mk():
    def inner(state, idx=(..., 0), **kw):
        return state[idx]
    return inner
def f4(state, **kw):
    return state in {(1, 2), (3, 4)}
for f in (f1, f2, f3, mk, f4):
    print(f.__name__, [ (type(c).__name__, c) for c in f.__code__.co_consts])
print("inner defaults", mk().__defaults__)
