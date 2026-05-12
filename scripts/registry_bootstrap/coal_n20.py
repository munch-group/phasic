"""Kingman coalescent for n=20 — bootstrap registry artifact."""
from __future__ import annotations

import numpy as np

import phasic


def coalescent_callback(state):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), [rate])]


def build_graph():
    g = phasic.Graph(coalescent_callback, ipv=[20])
    g.expectation()
    return g
