"""Shared fixtures and snapshot for the discretize pinning gate.

Phase 0 of discretize-cpp-ssot-plan.md: record exactly what Python
``Graph.discretize`` does on the starting commit, so the C++ port (phases
1-4) can be checked against it. Not collected by pytest (leading
underscore). Regenerate the records with

    python tests/pytest/_discretize_pins.py

only when a behaviour change is intended, and say so in the commit.
"""
import json
import os
from itertools import combinations_with_replacement

import numpy as np

from phasic import Graph, Property, StateIndexer, with_ipv

RECORDS = os.path.join(os.path.dirname(__file__), "data", "discretize_pins.json")
N = 4


# ----------------------------------------------------------------- graphs
def _coalescent_cb(n, as_list):
    @with_ipv([n] + [0] * (n - 1))
    def cb(state):
        out = []
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
                r = state[i] * (state[j] - same) / (1 + same)
                out.append([new, [r] if as_list else r])
        return out
    return cb


def coal_param(n=N):
    return Graph(_coalescent_cb(n, True), graph_cache=False)


def coal_constant(n=N):
    return Graph(_coalescent_cb(n, False), graph_cache=False)


def two_param_chain():
    """start -> [3] -> [2] -> [1]; rates theta0 and theta1."""
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0, 0.0])
    b.add_edge(c, [0.0, 0.5])
    return g


def coal_indexed(n=3, as_list=True):
    indexer = StateIndexer(lineages=[Property('descendants', min_value=1, max_value=n)])
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = n

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            r = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [r] if as_list else r])
        return out
    return Graph(cb, indexer=indexer, graph_cache=False)


def epoch_graph():
    g = coal_param()
    g.update_weights([1.0])
    return g.add_epoch(1.0)


def _inplace_build():
    """The graph used by test_modeling_compose.py's _discretize_inplace
    equivalence tests."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    start.add_edge(v1, 2.0)
    v1.add_edge(v2, 2.0)
    return g


# ----------------------------------------------------------------- cases
def _mutation(state):
    return 0.3 * float(state.sum())


def _singletons(state):
    return 0.3 * float(state[0])


def _slicing(state):
    return 0.1 * float(state[:2].sum()) + 0.05


CASES = {
    # name: (builder, call, theta for distribution values or None)
    "param_fixed": (coal_param, lambda g: g.discretize(0.1), [1.0, 1.0]),
    "param_fixed_theta2": (coal_param, lambda g: g.discretize(0.1), [2.0, 0.5]),
    "param_callback": (coal_param, lambda g: g.discretize(_mutation), [1.0, 1.0]),
    "param_callback_zeros": (coal_param, lambda g: g.discretize(_singletons), [1.0, 1.0]),
    "param_callback_slicing": (coal_param, lambda g: g.discretize(_slicing), [1.0, 1.0]),
    "param_callback_kwargs": (coal_param, lambda g: g.discretize(lambda s, mu: mu * float(s.sum()), mu=0.2), [1.0, 1.0]),
    "param_skip_existing": (coal_param, lambda g: g.discretize(0.1).discretize(0.1, skip_existing=True), [1.0, 1.0]),
    "two_param_chain": (two_param_chain, lambda g: g.discretize(0.5), [0.3, 0.4, 1.0]),
    "constant_fixed": (coal_constant, lambda g: g.discretize(0.1), None),
    "constant_callback": (coal_constant, lambda g: g.discretize(_mutation), None),
    "constant_callback_zeros": (coal_constant, lambda g: g.discretize(_singletons), None),
    "constant_skip_existing": (coal_constant, lambda g: g.discretize(0.1).discretize(0.1, skip_existing=True), None),
    "epoch_then_discretize": (epoch_graph, lambda g: g.discretize(0.1), [1.0, 0.0, 1.0, 1.0, 1.0]),
    "indexed_param": (coal_indexed, lambda g: g.discretize(0.1), [1.0, 1.0]),
    "indexed_constant": (lambda: coal_indexed(as_list=False), lambda g: g.discretize(0.1), None),
}

INPLACE_CASES = {
    "inplace_fixed": (_inplace_build, lambda g: g._discretize_inplace(0.1)),
    "inplace_callable": (_inplace_build, lambda g: g._discretize_inplace(lambda s: 0.2)),
}


def _bad_param_graph_with(mode):
    g = coal_param()
    if mode == 'formula':
        g.weight_formula = "t0*c0"
    elif mode == 'callback':
        g.weight_callback = lambda theta, coeffs: float(theta[0] * coeffs[0])
    else:
        g.weight_mode = mode
    return g


def _ragged():
    g = Graph(1)
    s = g.starting_vertex()
    a, b = g.find_or_create_vertex([2]), g.find_or_create_vertex([1])
    g.set_param_length(1)
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0, 7.0])
    return g


def _constant_edge():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (10, 1, 2))
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0])
    a.add_edge_constant(c, 0.5)
    return g


# error table (discretize-cpp-ssot-plan.md §5e): name -> (call, key phrase)
ERRORS = {
    "rate_string": (lambda: coal_param().discretize("fast"), "rate must be a number or callable"),
    "rate_bool": (lambda: coal_param().discretize(True), "rate must be a number or callable"),
    "rate_zero": (lambda: coal_param().discretize(0.0), "rate must be larger than 0"),
    "rate_negative": (lambda: coal_param().discretize(-1.0), "rate must be larger than 0"),
    "rate_nan": (lambda: coal_param().discretize(float('nan')), "rate must be larger than 0"),
    "rate_inf": (lambda: coal_param().discretize(float('inf')), "rate must be larger than 0"),
    "kwargs_with_fixed_rate": (lambda: coal_param().discretize(0.1, mu=1.0), "keyword arguments"),
    "callback_vector": (lambda: coal_param().discretize(lambda s: [0.1]), "must return ONE number"),
    "callback_string": (lambda: coal_param().discretize(lambda s: "0.1"), "must return ONE number"),
    "callback_bool": (lambda: coal_param().discretize(lambda s: True), "must return ONE number"),
    "callback_negative": (lambda: coal_param().discretize(lambda s: -0.1), "must be finite and >= 0"),
    "callback_nan": (lambda: coal_param().discretize(lambda s: float('nan')), "must be finite and >= 0"),
    "constant_callback_vector": (lambda: coal_constant().discretize(lambda s: [0.1]), "must return ONE number"),
    "mode_formula": (lambda: _bad_param_graph_with('formula').discretize(0.1), "weight_mode='linear'"),
    "mode_callback": (lambda: _bad_param_graph_with('callback').discretize(0.1), "weight_mode='linear'"),
    "mode_log": (lambda: _bad_param_graph_with('log').discretize(0.1), "weight_mode='linear'"),
    "ragged_coefficients": (lambda: _ragged().discretize(0.1), "coefficient vector to have"),
    "constant_edge": (lambda: _constant_edge().discretize(0.1), "coefficient-less constant"),
}


# ----------------------------------------------------------------- snapshot
def _edges(v):
    coef = iter(v.parameterized_edges())
    out = []
    for e in v.edges():
        n = e.coefficients_length()
        coeffs = [] if n == 0 else [float(x) for x in next(coef).edge_state(n)]
        out.append([int(e.to().index()), float(e.weight()), int(n), coeffs])
    return out


_ATTRS = ('_n_epochs', '_epoch_state_index', '_base_param_length')


def snapshot(d, theta=None):
    rec = {
        "vertices": d.vertices_length(),
        "state_length": d.state_length(),
        "states": [[int(x) for x in v.state()] for v in d.vertices()],
        "is_aux": [bool(v.is_aux()) for v in d.vertices()],
        "edges": [_edges(v) for v in d.vertices()],
        "rewards": [int(x) for x in d.rewards],
        "param_length": int(d.param_length()),
        "parameterized": bool(d.parameterized()),
        "was_dph": bool(d.get_was_dph()),
        "is_discrete": bool(d.is_discrete),
        "indexer_state_length": (None if getattr(d, '_indexer', None) is None
                                 else int(d._indexer.state_length)),
        "has_callback": getattr(d, '_callback', None) is not None,
        "attrs": {a: (getattr(d, a) if hasattr(d, a) else None) for a in _ATTRS},
    }
    if theta is not None:
        d.update_weights(theta)
        rec["theta"] = list(theta)
        rec["edges_at_theta"] = _edges_weights(d)
    rec["pdf"] = [float(d.pdf(k)) for k in range(1, 6)]
    rec["expected_aux_visits"] = float(d.expectation_discrete(d.rewards))
    return rec


def _edges_weights(d):
    return [[float(e.weight()) for e in v.edges()] for v in d.vertices()]


def serialize_roundtrip(d):
    """What survives serialize -> from_serialized for a discretized graph."""
    d2 = Graph.from_serialized(d.serialize())
    return {"vertices": d2.vertices_length(),
            "is_aux": [bool(v.is_aux()) for v in d2.vertices()],
            "was_dph": bool(d2.get_was_dph()),
            "is_discrete": bool(d2.is_discrete),
            "edges": [_edges(v) for v in d2.vertices()]}


def error_record(call, phrase):
    try:
        call()
    except Exception as e:  # noqa: BLE001 - recording the type is the point
        return {"type": type(e).__name__, "phrase_in_message": phrase in str(e)}
    return {"type": None, "phrase_in_message": False}


def build_records():
    recs = {"cases": {}, "inplace": {}, "errors": {}, "roundtrip": {}}
    for name, (build, call, theta) in CASES.items():
        recs["cases"][name] = snapshot(call(build()), theta)
    for name, (build, call) in INPLACE_CASES.items():
        g = build()
        rewards = call(g)
        recs["inplace"][name] = {"rewards": [int(x) for x in rewards],
                                 "states": [[int(x) for x in v.state()] for v in g.vertices()],
                                 "edges": [_edges(v) for v in g.vertices()],
                                 "expected_aux_visits": float(g.expectation_discrete(rewards))}
    for name, (call, phrase) in ERRORS.items():
        recs["errors"][name] = error_record(call, phrase)
    recs["roundtrip"]["param_fixed"] = serialize_roundtrip(coal_param().discretize(0.1))
    recs["roundtrip"]["constant_fixed"] = serialize_roundtrip(coal_constant().discretize(0.1))
    return recs


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RECORDS), exist_ok=True)
    with open(RECORDS, "w") as f:
        json.dump(build_records(), f, indent=1, sort_keys=True)
    print("wrote", RECORDS)
