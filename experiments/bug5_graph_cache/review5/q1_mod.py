"""Fixture module for q1 probes (identical file in every subprocess)."""
import numpy as np

LABELS = ("a", "b")


def cb(state, rate=1.0, *, mode="x"):
    """docstring"""
    k = state[0]
    if k <= 1:
        return []
    names = {"alpha", "beta", "gamma", "delta"}      # frozenset constant (str)
    ints = {3, 1, 2}
    if mode in names or k in ints:
        pass
    sq = [i * i for i in range(k)]                    # comprehension (inlined in 3.12+)
    g = sum(x for x in sq)                            # genexpr: nested code object
    f = lambda z: z + 1                               # nested lambda
    def inner(y):
        return y * rate                               # closure over rate
    return [[[k - 1], float(k) * rate + 0.0 * g + 0.0 * f(1) + 0.0 * inner(1)]]


def make(rate):
    def closure_cb(state):
        k = state[0]
        if k <= 1:
            return []
        return [[[k - 1], k * rate]]
    return closure_cb
