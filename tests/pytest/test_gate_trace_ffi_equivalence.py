"""Gate: TRACE-replay == pybind-C == XLA-FFI for the same quantity (audit gap #1).

The situation-map audit found the single largest cross-path hole: the two production JAX
compute paths -- the elimination TRACE replay (trace_elimination) and the XLA-FFI custom-call
(ComputePmf/Moments handlers) -- were NEVER pinned to each other. G1 pins FFI vs pybind-C; G5
pins trace vs pybind-C; nothing pinned trace vs FFI directly, and both existing gates sit on a
couple of tiny linear acyclic fixtures.

This gate closes that hole and, at the same time, extends cross-backend equivalence into the
regimes no existing gate covered: log weight mode, cyclic graphs, discrete PMF, a larger
(coalescent) graph, and several theta magnitudes.

WHAT IS COMPARED, AND HOW
-------------------------
Three backends for the same parameterized graph + theta:
  * TRACE  : record_elimination_trace -> instantiate_from_trace -> query the rebuilt graph.
  * PYBIND : parameterized.GraphBuilder(json).compute_pmf / compute_moments (the C engine).
  * FFI    : Graph.pmf_from_graph / pmf_and_moments_from_graph(use_ffi=True), proven to lower
             to the XLA custom-call (no pure_callback fallback) via _gate_backend.assert_ffi_target.

Exact quantities (mean, var from moments; discrete PMF) are compared bit-for-bit. The continuous
PDF is a uniformization APPROXIMATION whose value depends on the granularity, so it is compared
at a MATCHED granularity (0 = the auto value the FFI path hardcodes). At matched granularity the
three backends agree bit-for-bit; comparing at mismatched granularities is meaningless (an
early version of this gate did exactly that and saw a spurious ~1e-3 "divergence").

THE use_log FOOTGUN (audit finding F-004)
-----------------------------------------
instantiate_from_trace / evaluate_trace_jax default use_log=False. The trace is recorded with
unit weights; log-mode semantics (weight = prod(c_i*theta_i)) are applied at REPLAY via use_log.
A caller that omits use_log for a log-mode graph silently gets the LINEAR answer (wrong: e.g.
E[T] 0.325 instead of 0.333, and the PDF diverges, exploding with theta). This gate therefore
threads use_log from the graph's weight_mode -- and asserts, in test_trace_requires_use_log,
that omitting it really does diverge, so the footgun stays documented and any future
auto-threading of use_log turns that test's xfail into a forcing function.

Structural non-participation (skipped, not failed): the trace engine refuses CYCLIC graphs
(RuntimeError) and FORMULA mode (NotImplementedError) -- for those, only FFI-vs-pybind is
compared (both C engines handle them).
"""
import sys
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, StateIndexer, Property, with_ipv

jnp = pytest.importorskip("jax.numpy")

sys.path.insert(0, __import__("os").path.dirname(__file__))
import _gate_backend as gb  # noqa: E402
from phasic import trace_elimination as te  # noqa: E402
from phasic import phasic_pybind  # noqa: E402

parameterized = phasic_pybind.parameterized

_pkg = phasic._get_package_dir()
requires_sources = pytest.mark.skipif(
    not (_pkg / "src" / "cpp" / "phasiccpp.cpp").exists(),
    reason="phasic C/C++ sources not on disk; set PHASIC_SOURCE_DIR.",
)

pytestmark = [pytest.mark.equivalence]

TIMES = np.array([0.3, 0.6, 1.0, 2.0])
JUMPS = np.array([2.0, 3.0, 5.0])
TOL = 1e-11   # exact quantities + matched-granularity pdf are bit-identical in practice


# --------------------------------------------------------------------------- graphs
def chain(mode="linear", cyclic=False):
    g = phasic.Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    if cyclic:
        a.add_edge(b, [2.0, 0.0]); b.add_edge(a, [0.5, 0.0]); b.add_edge(c, [1.0, 2.0])
    else:
        a.add_edge(b, [2.0, 3.0]); b.add_edge(c, [1.0, 2.0])
    if mode == "log":
        g.weight_mode = "log"
    elif mode == "formula":
        g.weight_formula = "c0*t0 + c1*t1"
    return g


def coalescent(nn=5, mode="linear"):
    ix = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nn)])
    ipv = [0] * ix.state_length
    ipv[ix.lineages.props_to_index(descendants=1)] = nn

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            nw = list(state); nw[i] -= 1; nw[j] -= 1
            nw[indexer.lineages.props_to_index(descendants=(i + 1) + (j + 1))] += 1
            out.append((nw, [state[i] * (state[j] - same) // (1 + same), 0]))
        return out

    g = Graph(cb, indexer=ix)
    if mode == "log":
        g.weight_mode = "log"
    return g


# --------------------------------------------------------------------------- backends
def _trace(mk, theta):
    g = mk()
    use_log = getattr(g, "_weight_mode", "linear") == "log"   # F-004: trace needs explicit use_log
    tr = te.record_elimination_trace(g, theta_dim=len(theta))
    rb = te.instantiate_from_trace(tr, params=np.asarray(theta, np.float64), use_log=use_log)
    return dict(
        mean=float(rb.expectation()),
        var=float(rb.variance()),
        pdf=np.array([rb.pdf(float(t), granularity=0) for t in TIMES]),
    )


def _pybind(mk, theta, jumps=None):
    b = parameterized.GraphBuilder(gb.structure_str(mk()))
    th = np.asarray(theta, np.float64)
    m = np.asarray(b.compute_moments(th, 2))
    out = dict(
        mean=float(m[0]), var=float(m[1] - m[0] ** 2),
        pdf=np.asarray(b.compute_pmf(th, TIMES, discrete=False, granularity=0)),
    )
    if jumps is not None:
        out["dph"] = np.asarray(b.compute_pmf(th, jumps, discrete=True, granularity=0))
    return out


def _ffi(mk, theta, jumps=None):
    mm = Graph.pmf_and_moments_from_graph(mk(), nr_moments=2, use_ffi=True)
    _, mom = mm(jnp.asarray(theta), jnp.asarray(TIMES))
    mom = np.asarray(mom)
    out = dict(
        mean=float(mom[0]), var=float(mom[1] - mom[0] ** 2),
        pdf=np.asarray(Graph.pmf_from_graph(mk())(jnp.asarray(theta), jnp.asarray(TIMES))),
    )
    if jumps is not None:
        md = Graph.pmf_from_graph(mk(), discrete=True)
        out["dph"] = np.asarray(md(jnp.asarray(theta), jnp.asarray(jumps)))
    return out


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# --------------------------------------------------------------------------- 3-way (trace/pybind/FFI)
_THREE_WAY = [
    pytest.param(lambda: chain("linear"), [1.0, 2.0], id="linear-chain"),
    pytest.param(lambda: chain("linear"), [1e-2, 2.0], id="linear-chain-mixed-theta"),
    pytest.param(lambda: chain("linear"), [5.0, 3.0], id="linear-chain-large-theta"),
    pytest.param(lambda: chain("log"), [1.0, 2.0], id="log-chain"),
    pytest.param(lambda: chain("log"), [1e-2, 2.0], id="log-chain-mixed-theta"),
    pytest.param(lambda: chain("log"), [5.0, 3.0], id="log-chain-large-theta"),
    pytest.param(lambda: coalescent(5, "linear"), [1.0, 2.0], id="linear-coalescent5"),
]


@requires_sources
@pytest.mark.parametrize("mk,theta", _THREE_WAY)
def test_trace_pybind_ffi_agree(mk, theta):
    """mean, var and the matched-granularity PDF are bit-identical across all three backends."""
    # prove the FFI path really lowered to the XLA custom-call (no pure_callback fallback)
    m_ffi = Graph.pmf_from_graph(mk())
    gb.assert_ffi_target(lambda th, t: m_ffi(th, t),
                         jnp.asarray(theta), jnp.asarray(TIMES),
                         target="ptd_compute_pmf")
    t, p, f = _trace(mk, theta), _pybind(mk, theta), _ffi(mk, theta)
    for q in ("mean", "var", "pdf"):
        assert _rel(t[q], p[q]) < TOL, f"trace vs pybind {q}: {t[q]} vs {p[q]}"
        assert _rel(p[q], f[q]) < TOL, f"pybind vs FFI {q}: {p[q]} vs {f[q]}"
        assert _rel(t[q], f[q]) < TOL, f"trace vs FFI {q}: {t[q]} vs {f[q]}"


# --------------------------------------------------------------------------- 2-way (pybind/FFI): trace can't
@requires_sources
@pytest.mark.parametrize(
    "mk,theta",
    [
        pytest.param(lambda: chain("linear", cyclic=True), [1.0, 2.0], id="cyclic"),
        pytest.param(lambda: chain("formula"), [1.0, 2.0], id="formula"),
    ],
)
def test_pybind_ffi_agree_where_trace_cannot(mk, theta):
    """Cyclic + formula: the trace engine refuses these, so only the two C backends compare."""
    p, f = _pybind(mk, theta), _ffi(mk, theta)
    for q in ("mean", "var", "pdf"):
        assert _rel(p[q], f[q]) < TOL, f"pybind vs FFI {q}: {p[q]} vs {f[q]}"


# --------------------------------------------------------------------------- discrete PMF cross-path
@requires_sources
def test_discrete_pmf_pybind_ffi_agree():
    """Discrete PMF (dph_pmf, exact) across pybind and FFI on a valid DPH (row sums <= 1)."""
    mk = lambda: chain("linear")
    theta = [0.1, 0.15]
    p = _pybind(mk, theta, jumps=JUMPS)["dph"]
    f = _ffi(mk, theta, jumps=JUMPS)["dph"]
    assert _rel(p, f) < TOL, f"discrete pmf pybind vs FFI: {p} vs {f}"


# --------------------------------------------------------------------------- the footgun, pinned
@requires_sources
def test_trace_requires_use_log():
    """F-004: replaying a LOG-mode trace WITHOUT use_log silently computes LINEAR.

    This pins the footgun. If a future change makes instantiate_from_trace auto-detect
    weight_mode, this test XPASSes and strict mode forces its removal -- a forcing function.
    """
    g = chain("log")
    theta = [1.0, 2.0]
    tr = te.record_elimination_trace(g, theta_dim=2)
    rb_wrong = te.instantiate_from_trace(tr, params=np.asarray(theta, np.float64))  # use_log defaulted False
    rb_right = te.instantiate_from_trace(tr, params=np.asarray(theta, np.float64), use_log=True)
    # linear rates (8,5) -> E[T]=0.325 ; log rates (12,4) -> E[T]=0.3333
    assert abs(rb_wrong.expectation() - 0.325) < 1e-9, "expected the LINEAR (wrong) answer"
    assert abs(rb_right.expectation() - (1 / 12 + 1 / 4)) < 1e-9, "use_log=True must give the log answer"
    pybind_mean = _pybind(lambda: chain("log"), theta)["mean"]
    assert abs(rb_right.expectation() - pybind_mean) < 1e-9   # the correct path matches pybind
