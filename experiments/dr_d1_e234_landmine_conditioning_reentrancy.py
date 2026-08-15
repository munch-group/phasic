"""Deferred-1 de-risks E2 (landmine + guard design), E3 (per-SCC vs
whole-graph conditioning), E4 (reentrancy survey record).

Plan: deferred-1-hierarchical-scc-adjoint-plan.md §5. Branch-only; no
src/ changes.

E2 demonstrates HSCC §4(b): the shipped exact-gradient entry points
cannot tell a synthetic SCC graph from a real parameterized graph --
they return a full-size, plausible Jacobian whose contraction read the
Type-A/phantom PLACEHOLDER coefficients as real dw/dtheta, and
update_weights() itself silently overwrites the compose-injected
phantom weights with linear placeholder values (corrupting the synth
graph's hierarchical semantics). Guard DESIGN is the deliverable;
implementation touches shipped code and needs user approval (routed
per the plan to a micro-batch or a future approved core touch).

E3 answers master-plan risk 13a: on a fixture engineered to be
ill-conditioned ACROSS SCC boundaries but benign within each, compare
the per-SCC (synthetic-graph) gate condition with the whole-graph gate
condition, both recovered by bisecting PHASIC_CONDITION_THRESHOLD.

E4 is a static-read record (source citations) -- reproduced here as a
docstring so the findings file can cite one artifact.

E4 RECORD (read from src/c/scc_compose.c @ master a31d76cb..7371a369):
- The forward composer parallelizes ptd_compose_scc_one over SCCs
  WITHIN a topological level (omp parallel for, :517); levels are
  serial. Shared state inside the loop: parent_result (each SCC writes
  only its own vertices' slots; reads earlier-level slots for phantom
  weights -- safe by the level barrier); per-worker err_msg slabs +
  iter_status; the __thread TLS reentrancy guard
  ptd_scc_compose_in_progress with a documented per-worker bump
  (ptd_compose_scc_one's head comment); telemetry counters via
  atomic_add_u64.
- DECISION INPUT: an adjoint pass traverses the condensation DAG in
  REVERSE (source-first) order -- the OPPOSITE of the forward loop's
  sink-first levels -- so it CANNOT run inside the existing parallel
  loop at all; it is structurally a separate pass over retained
  per-SCC artifacts. The plan's default (separate serial pass first)
  is therefore not merely prudent but forced; parallelizing the
  adjoint later would reuse the same level machinery in reverse.
- Retention note: ptd_compose_scc_one internally calls
  ptd_expected_waiting_time on the synth graph (see its TLS comment);
  whatever tape that inner call builds is destroyed per-call today --
  the retention design must capture it (feeds §6's implementation
  sketch).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
from dr_d4_exact_oracle import build_structure  # noqa: E402  (reused)

import phasic  # noqa: E402
from phasic import Graph, set_log_level  # noqa: E402

set_log_level("WARNING")
FAILS = []


def check(label, ok, detail=""):
    print(f"  {label}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(label)


def two_scc_graph(cross_ratio=1.0):
    """Two cyclic SCCs in series; cross_ratio scales SCC B's rates
    relative to A's (the E3 knob: benign within, extreme across)."""
    g = Graph(1)
    s = g.starting_vertex()
    a1 = g.find_or_create_vertex([10])
    a2 = g.find_or_create_vertex([11])
    b1 = g.find_or_create_vertex([20])
    b2 = g.find_or_create_vertex([21])
    ab = g.find_or_create_vertex([1])
    s.add_edge(a1, 1.0)
    a1.add_edge(a2, [1.0, 0.0])
    a2.add_edge(a1, [0.5, 0.0])
    a2.add_edge(b1, [1.0, 0.0])
    b1.add_edge(b2, [0.0, 1.0 * cross_ratio])
    b2.add_edge(b1, [0.0, 0.5 * cross_ratio])
    b2.add_edge(ab, [1.0 * cross_ratio, 1.0 * cross_ratio])
    return g


THETA = [1.0, 2.0]

print("== E2: the landmine ==")
g = two_scc_graph()
g.update_weights(THETA)
scc = g.scc_decomposition()
synth = None
meta = None
for i in range(scc.n_sccs()):
    sg, m = scc.scc_at(i).as_synthetic_graph()
    if m.n_channels > 0:
        synth, meta = sg, m
        break
assert synth is not None, "no SCC with channels found"
print(f"  synth graph: {synth.vertices_length()} vertices, "
      f"{meta.n_channels} channels (scc_index={meta.scc_index})")

# (a) the shipped exact-gradient entry points ACCEPT the synth graph
chans = meta.channels
phantom_edges = [(c['d_j_synth_idx'], c.get('phantom_edge_idx'))
                 for c in chans] if isinstance(chans, list) else None
try:
    pre_w = None
    synth.update_weights(THETA)
    J = np.asarray(synth._moments_grad_theta(1))
    accepted = J.size > 0 and np.all(np.isfinite(J))
except Exception as exc:
    accepted = False
    print(f"  (unexpected: raised {type(exc).__name__})")
check("E2(a) shipped exact-grad ACCEPTS a synthetic graph "
      "(full-size plausible J -- the landmine)", accepted,
      f"J={J.tolist() if accepted else None}")

# (b) update_weights silently overwrites compose-injected semantics:
# the phantom edge's TRUE weight under hierarchical semantics is
# 1/parent_result[target] (theta-dependent through the PARENT);
# after update_weights it is the LINEAR placeholder-coefficient dot
# product. Show they differ at this theta.
par_res = None
try:
    # parent expectation vector (monolithic; equals the composed values)
    ewt = np.asarray(g.expected_waiting_time())
    par_res = ewt
except Exception:
    pass
if par_res is not None and isinstance(chans, list) and len(chans):
    c0 = chans[0]
    tgt = int(c0['parent_vertex_idx'])
    true_phantom = 1.0 / par_res[tgt] if par_res[tgt] > 0 else 1e300
    # (the synth graph is a raw pybind Graph -- no serialize(); the
    # structural fact stands from the C source: update_weights
    # re-derives EVERY parameterized edge as c.theta, including the
    # Type-A/phantom placeholders whose compose-time weights are
    # injections, and the shipped J above contracted those placeholder
    # coefficients as real dw/dtheta)
    print(f"  E2(b) true phantom weight under hierarchical semantics = "
          f"1/parent_result[{tgt}] = {true_phantom:.6g} "
          f"(theta-dependent through the PARENT -- not any linear "
          f"c.theta; the shipped J's placeholder-coefficient "
          f"contraction cannot represent it).")
print("""  E2 GUARD DESIGN (deliverable; implementation NEEDS user approval --
  it modifies shipped code): add `bool synthetic` (or a magic marker) to
  struct ptd_graph, set in ptd_scc_vertex_as_synthetic_graph
  (scc_synthetic.c) at creation; ONE check inserted at the top of
  ptd_b3_moments_core (covers linear/log/dph/formula/the binp exit in a
  single site -- the Batch-0 shared-core landing the plan predicted)
  plus ptd_sojourn_grad_theta_subset[_nogate]: decline -1 with a log
  ("synthetic SCC graphs carry placeholder coefficients; exact
  gradients of composed quantities need the two-level adjoint").
  Pinned-test-to-be: this experiment's E2(a) flips to declined.""")

print("== E3: per-SCC vs whole-graph gate condition ==")


def bisect_cond(graph, theta):
    def declines(log10_t):
        os.environ['PHASIC_CONDITION_THRESHOLD'] = f"1e{log10_t:.4f}"
        try:
            graph.update_weights(list(theta))
            return np.asarray(graph._moments_grad_theta(1)).size == 0
        finally:
            os.environ.pop('PHASIC_CONDITION_THRESHOLD', None)
    lo, hi = -2.0, 300.0
    if not declines(lo):
        return 10.0 ** lo
    if declines(hi):
        return float('inf')
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if declines(mid) else (lo, mid)
    return 10.0 ** (0.5 * (lo + hi))


for ratio, tag in ((1.0, "benign"), (1e12, "cross-SCC 1e12"),
                   (1e14, "cross-SCC 1e14")):
    g = two_scc_graph(cross_ratio=ratio)
    whole = bisect_cond(g, THETA)
    g2 = two_scc_graph(cross_ratio=ratio)
    g2.update_weights(THETA)
    scc2 = g2.scc_decomposition()
    per_scc = []
    for i in range(scc2.n_sccs()):
        sg, m = scc2.scc_at(i).as_synthetic_graph()
        try:
            c = bisect_cond(sg, THETA)
        except Exception:
            c = None
        per_scc.append(c)
    per_scc_max = max((c for c in per_scc if c is not None), default=None)
    print(f"  {tag}: whole-graph cond={whole:.3e}, "
          f"max per-SCC cond={per_scc_max:.3e}"
          if per_scc_max is not None else
          f"  {tag}: whole={whole:.3e}, per-SCC unavailable")
    if ratio >= 1e12 and per_scc_max is not None:
        under = per_scc_max < whole / 1e3
        print(f"    -> per-SCC gate under-detects cross-boundary "
              f"conditioning: {under} (ratio {whole/per_scc_max:.1e})")

print()
print("E2/E3/E4 COMPLETE" + ("" if not FAILS else f"; FAILURES: {FAILS}"))
sys.exit(0 if not FAILS else 1)
