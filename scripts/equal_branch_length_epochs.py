"""Compute epoch boundaries that distribute expected reward equally.

For daisy-chain SVGD setup. Uses the graph's own reward-transform +
stop_probability machinery, so it works for any phasic graph.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from scipy.optimize import brentq


def equal_reward_epoch_starts(
    graph,
    n_epochs: int,
    *,
    reward_fn: Callable[[np.ndarray], float] | None = None,
    indexer=None,
    property_name: str = "descendants",
    property_set: str | None = None,
    descendant_weight: Callable[[int], float] | None = None,
    t_max: float | None = None,
    granularity: int = 0,
) -> np.ndarray:
    """Epoch boundaries that distribute the expected total reward equally.

    Two ways to specify rewards:

    1. **Direct** — pass ``reward_fn(state) -> float`` to compute the
       reward at each vertex from its state vector. Most flexible.

    2. **Via indexer** — pass ``indexer`` and the helper sums over
       slots in ``property_set``: reward at a vertex is
       ``Σ_slot count_slot · descendant_weight(prop_value(slot))``.
       Default ``descendant_weight = lambda _: 1`` makes the reward
       equal to the lineage count at that vertex (correct for
       Kingman-style "branch length per unit time"). Pass
       e.g. ``lambda d: d`` to weight by descendants.

    Algorithm: B(t) = Σ_v rewards[v] · accumulated_visiting_time(t)[v]
    is the expected cumulative reward by real time t. Root-find
    boundaries t_i so that B(t_i) = i/n_epochs · B(∞), where B(∞) is
    computed analytically as Σ_v rewards[v] · expected_visiting_time[v].

    Parameters
    ----------
    graph
        A concrete (weights set) phasic Graph.
    n_epochs
        Number of epochs ``e``. Returns a length-``e`` vector.
    reward_fn
        Direct reward function: ``reward_fn(state_vector) -> float``.
        Mutually exclusive with ``indexer``.
    indexer, property_name, property_set, descendant_weight
        Used together when ``reward_fn`` is None. ``indexer`` is the
        StateIndexer used to construct ``graph``. ``property_name`` is
        the per-slot property whose value feeds ``descendant_weight``.
        ``property_set`` (optional) disambiguates if ``property_name``
        appears in multiple sets. ``descendant_weight`` defaults to
        ``lambda _: 1`` (giving lineage counts as rewards).
    t_max
        Upper bound for the bracketing root-find. Defaults to
        ``20 * rewarded.expectation()``.
    granularity
        Forwarded to ``stop_probability``; 0 = auto.

    Returns
    -------
    np.ndarray of shape (n_epochs,)
        ``epoch_starts`` vector starting at 0.0.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
    if n_epochs == 1:
        return np.array([0.0])

    if reward_fn is not None and indexer is not None:
        raise ValueError("Pass either reward_fn or indexer, not both.")
    if reward_fn is None and indexer is None:
        raise ValueError(
            "Must pass either reward_fn(state) or indexer (with optional "
            "property_name / descendant_weight) to define rewards."
        )

    if reward_fn is not None:
        rewards = np.array(
            [float(reward_fn(np.asarray(v.state()))) for v in graph.vertices()],
            dtype=np.float64,
        )
    else:
        rewards = _vertex_rewards_from_indexer(
            graph, indexer, property_name, property_set,
            descendant_weight or (lambda _: 1.0),
        )

    total_reward = float(np.dot(rewards, np.asarray(graph.expected_sojourn_time())))

    if t_max is None:
        t_max = 20.0 * float(graph.expectation())

    def B(t: float) -> float:
        """Expected cumulative reward by real time t."""
        if t <= 0.0:
            return 0.0
        avt = np.asarray(graph.accumulated_visiting_time(
            float(t), granularity=granularity))
        return float(np.dot(rewards, avt))

    boundaries = [0.0]
    t_lo = 0.0
    for i in range(1, n_epochs):
        target = i * total_reward / n_epochs
        max_doublings = 30
        while B(t_max) < target and max_doublings > 0:
            t_max *= 2.0
            max_doublings -= 1
        if B(t_max) < target:
            raise RuntimeError(
                f"Could not bracket epoch boundary {i}: B(t={t_max}) "
                f"= {B(t_max):.6f} < target {target:.6f}."
            )
        t_i = brentq(lambda t: B(t) - target, t_lo, t_max, xtol=1e-9)
        boundaries.append(float(t_i))
        t_lo = float(t_i)
    return np.array(boundaries)


def _vertex_rewards_from_indexer(
    graph, indexer, property_name, property_set, descendant_weight,
) -> np.ndarray:
    """Sum ``count_slot * descendant_weight(prop_value)`` over slots."""
    pset = _resolve_property_set(indexer, property_name, property_set)
    base_offset, base_length = _property_set_state_slice(indexer, pset)

    rewards = np.zeros(graph.vertices_length(), dtype=np.float64)
    for v in graph.vertices():
        state = np.asarray(v.state())
        if state.size == 0:
            continue
        sub = state[base_offset:base_offset + base_length]
        total = 0.0
        for slot_idx, count in enumerate(sub):
            if count == 0:
                continue
            props = pset.index_to_props(int(slot_idx), as_dict=True)
            val = props[property_name]
            total += float(count) * float(descendant_weight(val))
        rewards[v.index()] = total
    return rewards


def _resolve_property_set(indexer, property_name, property_set):
    """Locate the property set holding ``property_name``."""
    if property_set is not None:
        psets = [ps for ps in indexer.property_sets() if ps.name == property_set]
        if not psets:
            raise ValueError(
                f"property_set {property_set!r} not found in indexer; "
                f"available: {[ps.name for ps in indexer.property_sets()]}"
            )
        pset = psets[0]
        if not any(p.name == property_name for p in pset.properties):
            raise ValueError(
                f"property {property_name!r} not in property set {property_set!r}"
            )
        return pset
    for pset in indexer.property_sets():
        if any(p.name == property_name for p in pset.properties):
            return pset
    raise ValueError(
        f"property {property_name!r} not found in any property set"
    )


def _property_set_state_slice(indexer, target_pset):
    """Return (offset, length) of ``target_pset`` in the full state vector."""
    offset = 0
    for pset in indexer.property_sets():
        if pset.name == target_pset.name:
            return offset, pset.state_length
        offset += pset.state_length
    raise ValueError(f"property set {target_pset.name!r} not found in indexer")


if __name__ == "__main__":

    from phasic import Graph, StateIndexer, Property, with_ipv
    from itertools import combinations_with_replacement
    all_pairs = partial(combinations_with_replacement, r=2)
    np.random.seed(42)

    def coalescent(state, indexer=None):
        transitions = []
        for i, j in all_pairs(indexer.lineage):
            p1 = indexer.lineage.index_to_props(i)
            p2 = indexer.lineage.index_to_props(j)
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue 
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            descendants = p1.descendants + p2.descendants
            k = indexer.lineage.props_to_index(descendants=descendants)
            new[k] += 1
            transitions.append([new, [state[i]*(state[j]-same)/(1+same)]])
        return transitions

    nr_samples = 4
    indexer = StateIndexer(
        lineage=[
            Property('descendants', min_value=1, max_value=nr_samples),
        ]
    )
    ipv = [nr_samples]+[0]*(nr_samples-1)


    graph = Graph(coalescent, ipv=ipv, indexer=indexer)
    graph.update_weights([1])

    # Standard Kingman: equal expected branch length per epoch                                            
    epoch_starts = equal_reward_epoch_starts(graph, n_epochs=3, indexer=indexer)                          
    # → [0.0, 0.273, 0.674, 1.367]   (n=4, c=1)                                                           
                                                                                                        
    # # Or pass a callable directly:                                                                        
    # epoch_starts = equal_reward_epoch_starts(                                                             
    #     graph, n_epochs=4,                                                                              
    #     reward_fn=lambda state: float(state.sum()),  # lineage count                                      
    # )                                                                                                     
                                                                                                        
    # # Equal expected number of pairs / coalescence events:                                                
    # epoch_starts = equal_reward_epoch_starts(                                                           
    #     graph, n_epochs=5,                                                                                
    #     reward_fn=lambda s: float(s.sum() * (s.sum() - 1) / 2),                                         
    # )

    print(epoch_starts)