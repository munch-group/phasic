#!/usr/bin/env python3
"""Check if hierarchical trace creates valid graph structure."""

import shutil, os
cache_dir = os.path.expanduser('~/.phasic_cache/traces')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

import phasic
from phasic.state_indexing import Property, StateSpace
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace
import numpy as np

def two_locus_arg(state, s=None, N=None, R=None, state_space=None):
    transitions = []
    if state.sum() <= 1:
        return transitions
    for i in range(state_space.size):
        if state[i] == 0:
            continue
        conf_i = state_space.index_to_props(i)
        for j in range(i, state_space.size):
            if state[j] == 0:
                continue
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

nr_samples = 4
state_space = StateSpace([Property('L1Des', max_value=nr_samples), Property('L2Des', max_value=nr_samples)]
)
initial = np.empty(state_space.size, dtype=int)
initial.fill(0)
initial[state_space.props_to_index(L1Des=1, L2Des=1)] = nr_samples
ipv = [[initial, 1.0]]

graph = phasic.Graph(two_locus_arg, ipv=ipv, s=nr_samples, N=1, R=1, state_space=state_space)

# Record traces
direct_trace = record_elimination_trace(graph, param_length=1)
hier_trace = graph.compute_trace(hierarchical=True, min_size=20)

# Instantiate with concrete parameters
theta = np.array([1.0])
direct_graph = instantiate_from_trace(direct_trace, theta)
hier_graph = instantiate_from_trace(hier_trace, theta)

print(f"Direct graph:")
print(f"  Vertices: {direct_graph.vertices_length()}")
print(f"  Starting vertex index: {direct_graph.starting_vertex().index()}")

# Check first few vertices
for i in range(min(5, direct_graph.vertices_length())):
    v = direct_graph.vertex_at(i)
    print(f"  Vertex {i}: rate={v.rate():.6f}")

print(f"\nHierarchical graph:")
print(f"  Vertices: {hier_graph.vertices_length()}")
print(f"  Starting vertex index: {hier_graph.starting_vertex().index()}")

# Check first few vertices
for i in range(min(5, hier_graph.vertices_length())):
    v = hier_graph.vertex_at(i)
    print(f"  Vertex {i}: rate={v.rate():.6f}")

# Try to compute a simple property
print(f"\n Trying to compute moment 1:")
try:
    direct_mom = direct_graph.moment(1)
    print(f"  Direct moment: {direct_mom}")
except Exception as e:
    print(f"  Direct ERROR: {e}")

try:
    hier_mom = hier_graph.moment(1)
    print(f"  Hierarchical moment: {hier_mom}")
except Exception as e:
    print(f"  Hierarchical ERROR: {e}")
