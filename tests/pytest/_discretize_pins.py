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
    """The graph of the former _discretize_inplace equivalence tests (their
    frozen outputs are the "inplace" records)."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    start.add_edge(v1, 2.0)
    v1.add_edge(v2, 2.0)
    return g


# ----------------------------------------------------------------- cases
# Callbacks return what users typically return: numpy scalars from numpy
# arithmetic on the state array (no float() wrapping).
def _mutation(state):
    return 0.3 * state.sum()


def _singletons(state):
    return 0.3 * state[0]


def _slicing(state):
    return 0.1 * state[:2].sum() + 0.05


def native_dph():
    """A native discrete graph (is_discrete, rows sum <= 1)."""
    g = Graph(1)
    s = g.starting_vertex()
    a, b = g.find_or_create_vertex([2]), g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, 0.3)
    g.is_discrete = True
    return g


CALLBACK_ARGS = []


def _recording_callback(state):
    CALLBACK_ARGS.append({"type": type(state).__name__, "dtype": state.dtype.str,
                          "ndim": int(state.ndim), "writeable": bool(state.flags.writeable)})
    return 0.1


def _raising_callback(state):
    raise KeyError("callback-own-error")


CASES = {
    # name: (builder, call, theta for distribution values or None)
    "param_fixed": (coal_param, lambda g: g.discretize(0.1), [1.0, 1.0]),
    "param_fixed_theta2": (coal_param, lambda g: g.discretize(0.1), [2.0, 0.5]),
    "param_callback": (coal_param, lambda g: g.discretize(_mutation), [1.0, 1.0]),
    "param_callback_zeros": (coal_param, lambda g: g.discretize(_singletons), [1.0, 1.0]),
    "param_callback_slicing": (coal_param, lambda g: g.discretize(_slicing), [1.0, 1.0]),
    "param_callback_kwargs": (coal_param, lambda g: g.discretize(lambda s, mu: mu * float(s.sum()), mu=0.2), [1.0, 1.0]),
    "param_skip_existing": (coal_param, lambda g: g.discretize(0.1).discretize(0.1, skip_existing=True), [1.0, 1.0]),
    "param_skip_existing_fresh": (coal_param, lambda g: g.discretize(0.1, skip_existing=True), [1.0, 1.0]),
    "param_two_thetas": (coal_param, lambda g: g.discretize(0.1), [[2.0, 0.5], [0.3, 4.0], [2.0, 0.5]]),
    "param_int_rate": (coal_param, lambda g: g.discretize(1), [1.0, 1.0]),
    "param_callback_np_int": (coal_param, lambda g: g.discretize(lambda s: s[0]), [1.0, 1.0]),
    "param_callback_py_int": (coal_param, lambda g: g.discretize(lambda s: 1), [1.0, 1.0]),
    "param_callback_0d_array": (coal_param, lambda g: g.discretize(lambda s: np.array(0.2)), [1.0, 1.0]),
    "native_dph_input": (native_dph, lambda g: g.discretize(0.1), None),
    "two_param_chain": (two_param_chain, lambda g: g.discretize(0.5), [0.3, 0.4, 1.0]),
    "constant_fixed": (coal_constant, lambda g: g.discretize(0.1), None),
    "constant_callback": (coal_constant, lambda g: g.discretize(_mutation), None),
    "constant_callback_zeros": (coal_constant, lambda g: g.discretize(_singletons), None),
    "constant_skip_existing": (coal_constant, lambda g: g.discretize(0.1).discretize(0.1, skip_existing=True), None),
    "epoch_then_discretize": (epoch_graph, lambda g: g.discretize(0.1), [1.0, 0.0, 1.0, 1.0, 1.0]),
    "indexed_param": (coal_indexed, lambda g: g.discretize(0.1), [1.0, 1.0]),
    "indexed_constant": (lambda: coal_indexed(as_list=False), lambda g: g.discretize(0.1), None),
}

# The legacy in-place discretize (Graph._discretize_inplace, deleted with
# decision 5 of discretize-cpp-ssot-plan.md) was kept only for these two
# equivalence cases; its recorded outputs now pin discretize() itself.
INPLACE_CASES = {
    "inplace_fixed": (_inplace_build, 0.1),
    "inplace_callable": (_inplace_build, lambda s: 0.2),
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


# error table (discretize-cpp-ssot-plan.md §5e): name -> (call, phrase).
# Each phrase is unique to its case, so a check that fires for a different
# reason (or in a different order) does not pass.
ERRORS = {
    "rate_string": (lambda: coal_param().discretize("fast"), "got str"),
    "rate_bool": (lambda: coal_param().discretize(True), "got bool"),
    "rate_zero": (lambda: coal_param().discretize(0.0), "larger than 0 and finite, got 0.0"),
    "rate_negative": (lambda: coal_param().discretize(-1.0), "larger than 0 and finite, got -1.0"),
    "rate_nan": (lambda: coal_param().discretize(float('nan')), "larger than 0 and finite, got nan"),
    "rate_inf": (lambda: coal_param().discretize(float('inf')), "larger than 0 and finite, got inf"),
    "kwargs_with_fixed_rate": (lambda: coal_param().discretize(0.1, mu=1.0), "keyword arguments ['mu']"),
    "callback_vector": (lambda: coal_param().discretize(lambda s: [0.1]), "returned list"),
    "callback_string": (lambda: coal_param().discretize(lambda s: "0.1"), "returned str"),
    "callback_bool": (lambda: coal_param().discretize(lambda s: True), "returned bool"),
    "callback_negative": (lambda: coal_param().discretize(lambda s: -0.1), "callable returned -0.1"),
    "callback_nan": (lambda: coal_param().discretize(lambda s: float('nan')), "callable returned nan"),
    "callback_inf": (lambda: coal_param().discretize(lambda s: float('inf')), "callable returned inf"),
    "callback_own_exception": (lambda: coal_param().discretize(_raising_callback), "callback-own-error"),
    "constant_callback_vector": (lambda: coal_constant().discretize(lambda s: [0.1]), "returned list"),
    "mode_formula": (lambda: _bad_param_graph_with('formula').discretize(0.1), "weight_mode='formula'"),
    "mode_callback": (lambda: _bad_param_graph_with('callback').discretize(0.1), "weight_mode='callback'"),
    "mode_log": (lambda: _bad_param_graph_with('log').discretize(0.1), "weight_mode='log'"),
    "ragged_coefficients": (lambda: _ragged().discretize(0.1), "has an edge with 2 coefficients"),
    "constant_edge": (lambda: _constant_edge().discretize(0.1), "discretize() does not support coefficient-less constant"),
    "add_epoch_on_discrete": (lambda: coal_param().discretize(0.1).add_epoch(1.0), "add_epoch() is not supported on a discrete graph"),
    "add_epoch_on_constant": (lambda: coal_constant().add_epoch(1.0), "add_epoch needs a parameterized model"),
    "add_epoch_formula": (lambda: _bad_param_graph_with('formula').add_epoch(1.0), "add_epoch() adds coefficient slots"),
    # two causes at once: pins which check fires first (frozen precedence)
    "rate_negative_with_kwargs": (lambda: coal_param().discretize(-1.0, mu=1.0), "larger than 0 and finite, got -1.0"),
    "log_graph_zero_rate": (lambda: _bad_param_graph_with('log').discretize(0.0), "larger than 0 and finite, got 0.0"),
    "log_graph_constant_edge": (lambda: _with_mode(_constant_edge(), 'log').discretize(0.1), "discretize() does not support coefficient-less constant"),
}


def _with_mode(g, mode):
    g.weight_mode = mode
    return g

# Fields that an INTENDED behaviour change may alter, per case. Filled in by
# the phase that makes the change (decision 8 carries attributes on the
# constant path), so that change is a targeted edit, not a regeneration.
EXPECTED_CHANGES = {
    # decision 8 (phase 4): constant-graph discretize now carries the same
    # Python attributes as the parameterized path (callback, indexer).
    "constant_callback": ("has_callback",),
    "constant_callback_zeros": ("has_callback",),
    "constant_fixed": ("has_callback",),
    "constant_skip_existing": ("has_callback",),
    "indexed_constant": ("has_callback", "indexer_state_length"),
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


def snapshot_graph(g):
    """Structure of a non-discretized graph (e.g. add_epoch output)."""
    return {"vertices": g.vertices_length(),
            "states": [[int(x) for x in v.state()] for v in g.vertices()],
            "edges": [_edges(v) for v in g.vertices()],
            "param_length": int(g.param_length()),
            "attrs": {a: (getattr(g, a) if hasattr(g, a) else None) for a in _ATTRS},
            "indexer_state_length": (None if getattr(g, '_indexer', None) is None
                                     else int(g._indexer.state_length))}


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
        "last_callback_vertices_length": getattr(d, '_last_callback_vertices_length', None),
        "attrs": {a: (getattr(d, a) if hasattr(d, a) else None) for a in _ATTRS},
    }
    if theta is not None:
        thetas = theta if isinstance(theta[0], list) else [theta]
        rec["thetas"] = [list(t) for t in thetas]
        rec["edges_at_thetas"] = []
        for t in thetas:
            d.update_weights(t)
            rec["edges_at_thetas"].append(_edges_weights(d))
    rec["pdf"] = _guard(lambda: [float(d.pdf(k)) for k in range(1, 6)])
    rec["expected_aux_visits"] = _guard(lambda: float(d.expectation_discrete(d.rewards)))
    return rec


def _guard(f):
    """Value, or the error it raises (some inputs pin a known failure, e.g.
    discretize() of a native DPH returns a rewards vector one short)."""
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - recording the failure is the point
        return {"error": type(e).__name__, "message": str(e)}


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


def callback_arg_record():
    CALLBACK_ARGS.clear()
    coal_param().discretize(_recording_callback)
    seen = {json.dumps(a, sort_keys=True) for a in CALLBACK_ARGS}
    return {"calls": len(CALLBACK_ARGS), "distinct": sorted(seen)}


def build_records():
    recs = {"cases": {}, "inplace": {}, "errors": {}, "roundtrip": {}}
    for name, (build, call, theta) in CASES.items():
        recs["cases"][name] = snapshot(call(build()), theta)
    # The "inplace" records came from the deleted legacy method
    # Graph._discretize_inplace (decision 5): they are frozen and copied over,
    # never regenerated.
    with open(RECORDS) as f:
        recs["inplace"] = json.load(f)["inplace"]
    for name, (call, phrase) in ERRORS.items():
        recs["errors"][name] = error_record(call, phrase)
    recs["roundtrip"]["param_fixed"] = serialize_roundtrip(coal_param().discretize(0.1))
    recs["epoch_output"] = snapshot_graph(epoch_graph())
    recs["callback_arg"] = callback_arg_record()
    recs["roundtrip"]["constant_fixed"] = serialize_roundtrip(coal_constant().discretize(0.1))
    return recs


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RECORDS), exist_ok=True)
    records = build_records()
    with open(RECORDS, "w") as f:
        json.dump(records, f, indent=1, sort_keys=True)
    print("wrote", RECORDS)
