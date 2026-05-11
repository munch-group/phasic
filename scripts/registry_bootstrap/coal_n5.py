"""Kingman coalescent for n=5 — bootstrap registry artifact.

Used by `phasic publish-compute` to seed munch-group/phasic-traces. The
callable name `build_graph` is required by the CLI.
"""
from __future__ import annotations

import numpy as np

import phasic


def coalescent_callback(state):
    """Standard Kingman coalescent: n lineages coalesce at rate n(n-1)/2."""
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), [rate])]


def build_graph():
    g = phasic.Graph(coalescent_callback, ipv=[5])
    # Populate the C-side parameterized_reward_compute_graph so the
    # publish CLI can save it.
    g.expectation()
    return g
