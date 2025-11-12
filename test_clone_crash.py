#!/usr/bin/env python3
"""
Reproduce the crash when using clone() in compute_trace()
"""

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np

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
            if L1Des <= s and L2Des <= s:
                child[state_space.props_to_index(L1Des=L1Des, L2Des=L2Des)] += 1
                transitions.append([child, [state[i]*(state[j]-same)/(1+same)]])

        if state[i] > 0 and conf_i['L1Des'] > 0 and conf_i['L2Des'] > 0:
            child = state.copy()
            child[i] -= 1
            child[state_space.props_to_index(L1Des=conf_i['L1Des'], L2Des=0)] += 1
            child[state_space.props_to_index(L1Des=0, L2Des=conf_i['L2Des'])] += 1
            transitions.append([child, [R]])

    return transitions

print("Building two_locus_arg graph (this will crash)...")
nr_samples = 7
state_space = StateSpace([
    Property('L1Des', max_value=nr_samples),
    Property('L2Des', max_value=nr_samples)
])

initial = np.empty(state_space.size+2, dtype=int)
initial.fill(0)
initial[state_space.props_to_index(L1Des=1, L2Des=1)] = nr_samples
ipv = [[initial, 1.0]]
graph = phasic.Graph(two_locus_arg, ipv=ipv, s=nr_samples, N=1, R=1, state_space=state_space)

print(f"Graph has {graph.vertices_length()} vertices")

print("\nCalling compute_trace() (using clone)...")
try:
    trace = graph.compute_trace()
    print(f"Success! Trace: {trace.n_vertices} vertices")
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
