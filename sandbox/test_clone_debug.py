#!/usr/bin/env python3
"""Test clone with debug logging"""

import phasic
from phasic.state_indexing import Property, StateSpace
from phasic.logging_config import set_log_level
import numpy as np

# Enable debug logging
set_log_level('DEBUG')

def two_locus_arg(state, s=None, N=None, R=None, state_space=None):
    transitions = []
    if state.sum() <= 1: return transitions
    for i in range(state_space.size):
        if state[i] == 0: continue
        conf_i = state_space.index_to_props(i)
        for j in range(i, state_space.size):
            if state[j] == 0: continue
            conf_j = state_space.index_to_props(j)
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            child = state.copy()
            child[i] -= 1
            child[j] -= 1
            L1Des = conf_i['L1Des'] + conf_j['L1Des']
            L2Des = conf_i['L2Des'] + conf_j['L2Des']
            if L1Des <= s and L1Des <= s:
                child[state_space.props_to_index(L1Des=L1Des, L2Des=L2Des)] += 1
                transitions.append([child, [state[i]*(state[j]-same)/(1+same)]])
    return transitions

nr_samples = 7
state_space = StateSpace([Property('L1Des', max_value=nr_samples), Property('L2Des', max_value=nr_samples)])
initial = np.empty(state_space.size+2, dtype=int)
initial.fill(0)
initial[state_space.props_to_index(L1Des=1, L2Des=1)] = nr_samples
ipv = [[initial, 1.0]]
graph = phasic.Graph(two_locus_arg, ipv=ipv, s=nr_samples, N=1, R=1, state_space=state_space)

print(f"Original graph: {graph.vertices_length()} vertices")
print("\nCloning...")
clone = graph.clone()
print(f"Cloned graph: {clone.vertices_length()} vertices")

print("\nChecking vertex 0 state addresses:")
orig_state = list(graph.vertices())[0].state()
clone_state = list(clone.vertices())[0].state()
print(f"  Original: {id(orig_state)} -> {orig_state[:5]}")
print(f"  Clone: {id(clone_state)} -> {clone_state[:5]}")
print(f"  Different? {id(orig_state) != id(clone_state)}")
