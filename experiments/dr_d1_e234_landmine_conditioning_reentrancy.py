"""Deferred-1 de-risks E2 (landmine + guard design), E3 (per-SCC vs
whole-graph conditioning), E4 (reentrancy survey record).

Plan: deferred-1-hierarchical-scc-adjoint-plan.md §5. Branch-only; no
src/ changes.

E2 demonstrated HSCC §4(b): BEFORE the guard shipped, the exact-gradient
entry points could not tell a synthetic SCC graph from a real
parameterized graph -- they returned a full-size, plausible Jacobian
whose contraction read the Type-A/phantom PLACEHOLDER coefficients as
real dw/dtheta (measured J=[-1.0, 0.0] on this fixture), and
update_weights() itself silently overwrites the compose-injected
phantom weights with linear placeholder values (corrupting the synth
graph's hierarchical semantics -- still true, and OUT of the guard's
scope by design). The guard SHIPPED 2026-08-16 (user-approved,
b3-d1-e2-guard-plan.md); E2's checks below now assert the DECLINE.

E3 answered master-plan risk 13a: on a fixture engineered to be
ill-conditioned ACROSS SCC boundaries but benign within each, compare
the per-SCC (synthetic-graph) gate condition with the whole-graph gate
condition, both recovered by bisecting PHASIC_CONDITION_THRESHOLD.
POST-GUARD the per-SCC arm is PRE-GUARD HISTORICAL (the guard declines
every per-SCC call at any threshold, by design); reproduce its
1e23/1e28 numbers at the pre-guard commits bfb737ce / 4229207b. The
whole-graph arm still runs live.

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

# (a) POST-GUARD (b3-d1-e2-guard-plan.md): the exact-gradient cores now
# DECLINE the synth graph. PRE-GUARD HISTORICAL RECORD (2026-08-15, the
# landmine this experiment originally demonstrated): _moments_grad_theta(1)
# ACCEPTED it and returned the full-size plausible J=[-1.0, 0.0]
# contracted from placeholder coefficients; _sojourn_grad_theta_subset
# ([0]) likewise returned a size-2 finite row under BOTH
# skip_condition_gate settings (recorded at the guard plan's R2 review).
chans = meta.channels
phantom_edges = [(c['d_j_synth_idx'], c.get('phantom_edge_idx'))
                 for c in chans] if isinstance(chans, list) else None
try:
    synth.update_weights(THETA)
    J = np.asarray(synth._moments_grad_theta(1))
    declined = J.size == 0
except Exception as exc:
    declined = False
    print(f"  (unexpected: raised {type(exc).__name__})")
check("E2(a) exact moments grad DECLINES the synthetic graph "
      "(guard live; pre-guard it returned J=[-1.0, 0.0])", declined)
soj_declined = all(
    len(synth._sojourn_grad_theta_subset([0], skip_condition_gate=s)) == 0
    for s in (False, True))
check("E2(a') exact sojourn grad DECLINES the synthetic graph "
      "(both gate settings; pre-guard it returned a size-2 finite row)",
      soj_declined)

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
    # (the structural fact stands from the C source: update_weights
    # re-derives EVERY parameterized edge as c.theta, including the
    # Type-A/phantom placeholders whose compose-time weights are
    # injections, and the PRE-GUARD J contracted those placeholder
    # coefficients as real dw/dtheta. NB a synth CAN be serialized --
    # phasic.distributed_scc.serialize_scc_synth, the SLURM route --
    # which is why deserialize_scc_synth re-applies the marker.)
    print(f"  E2(b) true phantom weight under hierarchical semantics = "
          f"1/parent_result[{tgt}] = {true_phantom:.6g} "
          f"(theta-dependent through the PARENT -- not any linear "
          f"c.theta; the PRE-GUARD J's placeholder-coefficient "
          f"contraction could not represent it -- which is why the "
          f"guard declines rather than approximating).")
print("""  E2 GUARD -- SHIPPED (user-approved 2026-08-15; plan
  b3-d1-e2-guard-plan.md). `bool synthetic` on struct ptd_graph, set in
  ptd_scc_build_synthetic_graph at creation, propagated by
  ptd_clone_graph, and re-applied by
  distributed_scc.deserialize_scc_synth across the SLURM boundary; ONE
  decline check at the top of ptd_b3_moments_core (all five contraction
  kinds) plus one in ptd_b3_sojourn_grad_core (both public wrappers),
  each logging at WARNING. Pinned by
  tests/pytest/test_synthetic_scc_guard.py; the two PASS lines above
  ARE the post-guard behaviour.""")

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


# POST-GUARD SCOPING (b3-d1-e2-guard-plan.md, R2 finding 1; the
# reproduction path is the PRE-GUARD commits bfb737ce / 4229207b --
# G4 refuter B, MINOR-7): the
# per-SCC arm of this measurement is PRE-GUARD HISTORICAL. It bisected
# PHASIC_CONDITION_THRESHOLD against _moments_grad_theta on SYNTHETIC
# graphs; post-guard every such call declines at ANY threshold, so the
# bisection degenerates to inf and the printed science would silently
# invert. The recorded pre-guard result stands
# (b3-d1-derisk-findings.md E3): per-SCC gate condition exploded to
# 1e23 (cross-SCC 1e12) / 1e28 (cross-SCC 1e14) through phantom-weight
# scale mixing while the whole-graph condition stayed ~1e1 -- risk 13a
# INVERTED (per-SCC gates OVER-decline). By design that measurement is
# no longer reproducible on a guarded build. The whole-graph arm below
# remains live.
for ratio, tag in ((1.0, "benign"), (1e12, "cross-SCC 1e12"),
                   (1e14, "cross-SCC 1e14")):
    g = two_scc_graph(cross_ratio=ratio)
    whole = bisect_cond(g, THETA)
    print(f"  {tag}: whole-graph cond={whole:.3e} "
          f"(per-SCC arm: PRE-GUARD HISTORICAL, see comment above)")

print()
print("E2/E3/E4 COMPLETE" + ("" if not FAILS else f"; FAILURES: {FAILS}"))
sys.exit(0 if not FAILS else 1)
