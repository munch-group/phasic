"""Regression test: reward_transform() must preserve is_discrete for a DPH.

Both discrete reward-transform paths returned Graph(super().reward_transform_
discrete(rewards)); wrapping the C++ _Graph re-runs Graph.__init__, which
forces is_discrete=False, and nothing restored it. All downstream statistics
dispatch on the Python is_discrete attribute, so the returned DPH was treated
as continuous (wrong pmf/moments).
"""
import numpy as np

from phasic import Graph, with_ipv

nr_samples = 4


@with_ipv([nr_samples] + [0] * (nr_samples - 1))
def coalescent(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i + j + 1] += 1
            transitions.append((new, [state[i] * (state[j] - same) / (1 + same)]))
    return transitions


def _mutation_rate(state):
    return [0, sum(state)]


def test_reward_transform_preserves_discrete_flag():
    g = Graph(coalescent)
    dph = g.discretize(_mutation_rate)
    assert dph.is_discrete is True

    rewards = np.ones(dph.vertices_length(), dtype=int)
    transformed = dph.reward_transform(rewards)
    assert transformed.is_discrete is True, (
        "reward_transform on a discrete graph must stay discrete"
    )
    assert transformed.get_was_dph() is True


def test_reward_transform_discrete_explicit_sets_flag():
    g = Graph(coalescent)
    dph = g.discretize(_mutation_rate)
    rewards = np.ones(dph.vertices_length(), dtype=int)
    transformed = dph.reward_transform_discrete(rewards)
    assert transformed.is_discrete is True
    assert transformed.get_was_dph() is True


def test_continuous_reward_transform_stays_continuous():
    g = Graph(coalescent)
    rewards = np.ones(g.vertices_length(), dtype=float)
    transformed = g.reward_transform(rewards)
    assert transformed.is_discrete is False
