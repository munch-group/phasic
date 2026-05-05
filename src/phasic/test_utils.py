
from phasic import Graph, with_ipv
import os
import numpy as np
import threading
import psutil
from contextlib import contextmanager

def coalescent_manual_construction(nr_samples):
    state_vector_length = nr_samples
    graph = Graph(state_vector_length)
    initial_state = np.zeros(state_vector_length, dtype=int)
    initial_state[0] = nr_samples
    vertex = graph.find_or_create_vertex(initial_state)
    graph.starting_vertex().add_edge(vertex, 1)
    index = 1
    while index < graph.vertices_length():
        vertex = graph.vertex_at(index)
        state = vertex.state()
        for i in range(nr_samples):
            for j in range(i, nr_samples): 
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue 
                new_state = state.copy()
                new_state[i] -= 1
                new_state[j] -= 1
                new_state[i+j+1] += 1
                new_vertex = graph.find_or_create_vertex(new_state)
                rate = state[i]*(state[j]-same)/(1+same)
                vertex.add_edge(new_vertex, rate)
        index += 1    
    return graph


def coalescent_callback(state):
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
            new[i+j+1] += 1
            transitions.append((new, state[i]*(state[j]-same)/(1+same)))
    return transitions


@with_ipv([([4, 0, 0, 0], 1)])
def coalescent_callback_with_ipv(state):
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
            new[i+j+1] += 1
            transitions.append((new, state[i]*(state[j]-same)/(1+same)))
    return transitions


@with_ipv([4, 0, 0, 0])
def coalescent_callback_with_abbr_ipv(state):
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
            new[i+j+1] += 1
            transitions.append((new, state[i]*(state[j]-same)/(1+same)))
    return transitions


def coalescent_callback_parameterized(state):
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
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions


@np.vectorize
def bin_coef(n):
    return n*(n-1)/2


class ResourceMonitor:
    def __init__(self, interval=0.05, include_children=False, max_peak_mem=None, min_peak_cpu=None, min_peak_cpu_per_core=None):
        self.interval = interval
        self.include_children = include_children
        self.max_cpu = 0.0       # percent (can exceed 100 on multi-core)
        self.max_rss = 0         # bytes
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.max_peak_mem = max_peak_mem
        self.min_peak_cpu = min_peak_cpu
        self.min_peak_cpu_per_core = min_peak_cpu_per_core

    def _sample(self):
        procs = [self._proc]
        if self.include_children:
            procs += self._proc.children(recursive=True)
        cpu = sum(p.cpu_percent(None) for p in procs if p.is_running())
        rss = sum(p.memory_info().rss for p in procs if p.is_running())
        return cpu, rss

    def _run(self):
        self._sample()  # prime cpu_percent baseline
        while not self._stop.wait(self.interval):
            cpu, rss = self._sample()
            if cpu > self.max_cpu: self.max_cpu = cpu
            if rss > self.max_rss: self.max_rss = rss

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

class AssertResources():

    def __init__(self, interval=0.05, include_children=False, max_peak_mem=None, min_peak_cpu=None, min_peak_cpu_per_core=None):
        self.interval = interval
        self.include_children = include_children
        self.max_cpu = 0.0       # percent (can exceed 100 on multi-core)
        self.max_rss = 0         # bytes
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.max_peak_mem = max_peak_mem
        self.min_peak_cpu = min_peak_cpu
        self.min_peak_cpu_per_core = min_peak_cpu_per_core

    def _sample(self):
        procs = [self._proc]
        if self.include_children:
            procs += self._proc.children(recursive=True)
        cpu = sum(p.cpu_percent(None) for p in procs if p.is_running())
        rss = sum(p.memory_info().rss for p in procs if p.is_running())
        return cpu, rss

    def _run(self):
        self._sample()  # prime cpu_percent baseline
        while not self._stop.wait(self.interval):
            cpu, rss = self._sample()
            if cpu > self.max_cpu: self.max_cpu = cpu
            if rss > self.max_rss: self.max_rss = rss

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

        
    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        print(self.max_rss)
        if self.max_peak_mem is not None:
            max_mem = self.max_rss/(1024**2)
            print(f'Peak memory use Mb: {max_mem}')
            assert self.max_rss/(1024**2) < self.max_peak_mem            
        if self.min_peak_cpu is not None:
            print(f'Peak cpu use percent: {self.max_cpu}')
            assert self.max_cpu >= self.min_peak_cpu
        if self.min_peak_cpu_per_core is not None:
            peak_cpu_per_core = self.max_cpu / os.cpu_count()
            print(f'Peak cpu use per core percent: {peak_cpu_per_core}')
            assert peak_cpu_per_core >= self.min_peak_cpu_per_core


# with ResourceMonitor(interval=0.05) as m:
#     do_expensive_thing()

# print(f"peak CPU: {m.max_cpu:.0f}%   peak RSS: {m.max_rss/1e9:.2f} GB")        