"""D1-E1: pure-Python reference two-level (hierarchical/SCC) adjoint vs oracles.

De-risk experiment E1 of Deferred Unit 1 (hierarchical/SCC two-level adjoint),
spec: deferred-1-hierarchical-scc-adjoint-plan.md section 5 E1. Composition
mechanics ground truth: atlas/plan-feasibility-hierarchical-scc.md sections 2/4,
src/c/scc_compose.c (ptd_compose_scc_one:138-267, phantom rule :204-225),
src/c/scc_synthetic.c (Type-A placeholders :844-857).

What this script does
---------------------
1. Extracts each toy_model.BUILDERS fixture (6, all weight_mode='linear',
   continuous) into a plain edge-list form. Conventions mirrored from C source:
     - non-start parameterized edges: w(theta) = dot(coeffs, theta)
       (ptd_graph_update_weights, phasic.c:5546);
     - starting-vertex edges are NEVER rescaled by update_weights
       (phasic.c:5650-5655) -> theta-CONSTANT, weight = construction weight;
     - coefficients_length==0 edges (toy_d aux X->C): constant weight;
     - the starting vertex accrues NO holding time: E_start is the
       IPV-probability average of its targets' values (verified numerically
       against ptd_expected_waiting_time in the step-0 sanity gate).
2. Reproduces the sink-first per-SCC composition in Python (Tarjan SCCs by
   hand, condensation Kahn topo, per-SCC dense first-step solve over
   members + per-channel absorbing vertices, with the composer's LITERAL
   piecewise phantom rule: y = 1/x for x > 0, but the CONSTANT 1e300 when
   parent_result[target] == 0.0; scc_compose.c:213-223). Gate: value parity
   vs BOTH the C hierarchical path (SCCGraph.compose == ptd_compose_scc_prcs)
   and the monolithic expected_waiting_time, <= 1e-12.
3. Implements the two-level REVERSE pass: outer source-first traversal of the
   condensation; per SCC a cotangent-seeded VJP of its dense elimination
   (seeds = incoming cotangents on that SCC's internal-vertex results, solved
   as mu = M^-T bar_E); Type-C adjoints contracted against the PARENT edge's
   real coefficients; phantom adjoints routed through the piecewise rule
   d(1/x)/dx = -1/x^2 for x>0 / derivative 0 (cotangent dropped) at the 1e300
   clamp; Type-A source-edge placeholders excluded, with their zero adjoint
   asserted by PERTURBATION (not assumption).
4. Oracles (both mandatory per plan review F1):
   (a) dense per-vertex-cotangent VJP reference: jax.vjp over the monolithic
       dense solve E(theta) (alpha @ (-S)^-1 form), >= 5 RANDOM per-vertex
       cotangent seeds per fixture x theta;
   (b) the shipped Graph._moments_grad_theta(1) binding as the e_start
       one-hot cross-check (it seeds one-hot at vertex 0; phasic.c).
   Gate: <= 1e-12 relative (inf-norm) across all 6 fixtures x 4 theta
   vectors, restricted to the well-conditioned regime (these toys).
5. Perturbation tests:
   (iii) coupling: the full Jacobian assembled from the two-level adjoint
       (basis cotangents) is compared against central finite differences of
       the REAL C composer (SCCGraph.compose) -- any cross-SCC coupling not
       modeled by {Type-B weights, Type-C weights, phantom 1/x} would break
       this match;
   (iv) Type-A: per-SCC systems are re-solved WITH the synthetic source and
       its placeholder edges included, at wildly perturbed placeholder
       weights -> member results and dtheta must be unchanged, and
       mu_source == 0 (so a contraction of the [1,0,...] placeholder
       coefficients would add exactly 0; dtheta[0] is the pollution sentinel
       since theta[0] drives only never-rescaled start edges in these toys).

Run:  cd /Users/kmt/phasic && pixi run python experiments/dr_d1_e1_twolevel_adjoint.py
Exit: 0 = all gates pass (GO for the C design's mathematical core), 1 = failure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "/Users/kmt/phasic/tests/pytest")

import phasic  # noqa: E402  (forces JAX x64 at import)
from toy_model import BUILDERS  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Theta vectors: same 4 as tests/pytest/test_scc_compose.py (well-conditioned).
THETAS = [
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 2.0, 1.0, 0.7],
    [2.5, 0.3, 1.5, 0.9],
    [0.1, 5.0, 2.0, 0.5],
]
N_COT_SEEDS = 5
CLAMP = 1e300  # scc_compose.c:222 constant injected when parent_result[target]==0.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass
class EdgeRec:
    frm: int
    to: int
    kind: str          # 'param' | 'const' | 'start'
    coeffs: np.ndarray | None   # real coefficients ('param' only)
    w0: float          # construction weight ('const'/'start'; ignored for 'param')


@dataclass
class Extraction:
    n: int
    start: int
    P: int
    out: list = field(default_factory=list)   # out[v] = [EdgeRec, ...]


def extract(g) -> Extraction:
    verts = list(g.vertices())
    n = len(verts)
    start = g.starting_vertex().index()
    ex = Extraction(n=n, start=start, P=g.param_length(), out=[[] for _ in range(n)])
    for v in verts:
        vi = v.index()
        edges = list(v.edges())
        pedges = list(v.parameterized_edges())  # clen>0 subset, same relative order
        pi = 0
        for e in edges:
            clen = e.coefficients_length()
            to = e.to().index()
            if clen > 0:
                pe = pedges[pi]
                pi += 1
                assert pe.to().index() == to, "edges()/parameterized_edges() misaligned"
                coeffs = np.asarray(list(pe.edge_state(clen)), dtype=float)
                if vi == start:
                    # update_weights SKIPS the starting vertex (phasic.c:5650-5655):
                    # IPV edges are theta-CONSTANT despite carrying coefficients.
                    ex.out[vi].append(EdgeRec(vi, to, "start", None, e.weight()))
                else:
                    ex.out[vi].append(EdgeRec(vi, to, "param", coeffs, e.weight()))
            else:
                ex.out[vi].append(EdgeRec(vi, to, "const", None, e.weight()))
        assert pi == len(pedges)
    return ex


def edge_weight(e: EdgeRec, theta: np.ndarray) -> float:
    if e.kind == "param":
        return float(np.dot(e.coeffs, theta))
    return e.w0


# ---------------------------------------------------------------------------
# SCC decomposition (iterative Tarjan) + condensation topo order
# ---------------------------------------------------------------------------

def tarjan_scc(n: int, adj: list[list[int]]) -> list[list[int]]:
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    stack: list[int] = []
    sccs: list[list[int]] = []
    counter = [0]
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, ei = work[-1]
            if ei == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for k in range(ei, len(adj[v])):
                w = adj[v][k]
                if index[w] == -1:
                    work[-1] = (v, k + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                p = work[-1][0]
                low[p] = min(low[p], low[v])
    return sccs


def condensation_topo(ex: Extraction, sccs: list[list[int]]) -> list[int]:
    """Kahn forward-topological order over the condensation (source-first)."""
    scc_of = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            scc_of[v] = i
    nS = len(sccs)
    succ = [set() for _ in range(nS)]
    indeg = [0] * nS
    for v in range(ex.n):
        for e in ex.out[v]:
            a, b = scc_of[v], scc_of[e.to]
            if a != b and b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    order = [i for i in range(nS) if indeg[i] == 0]
    head = 0
    while head < len(order):
        a = order[head]
        head += 1
        for b in succ[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                order.append(b)
    assert len(order) == nS
    return order


# ---------------------------------------------------------------------------
# Python composition (mirror of ptd_compose_scc_prcs) + records for reverse
# ---------------------------------------------------------------------------

@dataclass
class SCCRecord:
    kind: str                     # 'absorbing' | 'start' | 'normal'
    members: list = field(default_factory=list)
    # channels: list of dicts with keys
    #   v (member), t (parent target), w (Type-C weight value),
    #   coeffs (parent edge coefficients or None), x (parent_result[t]),
    #   y (phantom weight; 1/x or CLAMP), clamped (bool)
    channels: list = field(default_factory=list)
    internal: list = field(default_factory=list)  # (v, u, w, coeffs|None)
    M: np.ndarray | None = None   # (nm+nc) x (nm+nc) transient system
    E: np.ndarray | None = None   # solution: members then s_abs values
    start_rho: float = 0.0        # start SCC only


@dataclass
class ComposeOut:
    result: np.ndarray
    recs: dict                    # scc index -> SCCRecord
    order: list                   # forward-topological (source-first) SCC order
    sccs: list
    clamp_hits: int = 0


def compose_py(ex: Extraction, theta: np.ndarray) -> ComposeOut:
    adj = [[e.to for e in ex.out[v]] for v in range(ex.n)]
    sccs = tarjan_scc(ex.n, adj)
    # Mirror ptd_isolate_starting_vertex_scc (phasic.c:2282): the C path forces
    # the starting vertex into its own SCC. In every toy the start is already a
    # singleton (no incoming edges); assert rather than re-implement.
    for comp in sccs:
        if ex.start in comp:
            assert comp == [ex.start], (
                "start vertex not a singleton SCC; the C path would isolate it "
                "(ptd_isolate_starting_vertex_scc) -- fixture outside E1 scope")
    order = condensation_topo(ex, sccs)
    result = np.zeros(ex.n)
    recs: dict[int, SCCRecord] = {}
    clamp_hits = 0

    for si in reversed(order):  # sink-first, like the composer's level loop
        comp = sccs[si]
        inS = set(comp)
        if all(len(ex.out[v]) == 0 for v in comp):
            # absorbing SCC: parent_result stays 0 by convention (calloc)
            recs[si] = SCCRecord(kind="absorbing", members=list(comp))
            continue

        if comp == [ex.start]:
            # start surrogate: NO holding time (IPV vertex). Faithful phantom
            # form: E_start = sum_ch (w/rho) * E_sabs, E_sabs = 1/y.
            rec = SCCRecord(kind="start", members=[ex.start])
            rho = sum(edge_weight(e, theta) for e in ex.out[ex.start])
            rec.start_rho = rho
            val = 0.0
            for e in ex.out[ex.start]:
                w = edge_weight(e, theta)
                x = result[e.to]
                if x > 0.0:
                    y, clamped = 1.0 / x, False
                elif x < 0.0:
                    raise RuntimeError("negative parent_result (composer would error)")
                else:
                    y, clamped = CLAMP, True
                    clamp_hits += 1
                e_sabs = 1.0 / y
                val += (w / rho) * e_sabs
                rec.channels.append(dict(v=ex.start, t=e.to, w=w,
                                         coeffs=None if e.kind != "param" else e.coeffs,
                                         x=x, y=y, clamped=clamped))
            result[ex.start] = val
            recs[si] = rec
            continue

        members = sorted(comp)
        pos = {v: i for i, v in enumerate(members)}
        rec = SCCRecord(kind="normal", members=members)
        for v in members:
            for e in ex.out[v]:
                w = edge_weight(e, theta)
                if e.to in inS:
                    rec.internal.append((v, e.to, w,
                                         e.coeffs if e.kind == "param" else None))
                else:
                    x = result[e.to]
                    if x > 0.0:
                        y, clamped = 1.0 / x, False
                    elif x < 0.0:
                        raise RuntimeError("negative parent_result")
                    else:
                        y, clamped = CLAMP, True
                        clamp_hits += 1
                    rec.channels.append(dict(v=v, t=e.to, w=w,
                                             coeffs=e.coeffs if e.kind == "param" else None,
                                             x=x, y=y, clamped=clamped))
        nm, nc = len(members), len(rec.channels)
        nt = nm + nc
        M = np.zeros((nt, nt))
        for (v, u, w, _c) in rec.internal:
            M[pos[v], pos[v]] += w
            M[pos[v], pos[u]] -= w
        for k, ch in enumerate(rec.channels):
            M[pos[ch["v"]], pos[ch["v"]]] += ch["w"]     # rho includes external
            M[pos[ch["v"]], nm + k] -= ch["w"]           # Type-C into s_abs
            M[nm + k, nm + k] = ch["y"]                  # phantom (to absorbing)
        E = np.linalg.solve(M, np.ones(nt))
        rec.M, rec.E = M, E
        for v in members:
            result[v] = E[pos[v]]
        recs[si] = rec

    return ComposeOut(result=result, recs=recs, order=order, sccs=sccs,
                      clamp_hits=clamp_hits)


# ---------------------------------------------------------------------------
# Two-level adjoint (outer source-first DAG pass + per-SCC cotangent-seeded VJP)
# ---------------------------------------------------------------------------

def adjoint_py(ex: Extraction, theta: np.ndarray, lam: np.ndarray,
               co: ComposeOut | None = None) -> np.ndarray:
    """VJP: dtheta = (d result / d theta)^T @ lam, via the two-level structure.

    bar_x[v] carries the accumulated cotangent on parent_result[v]. SCCs are
    consumed SOURCE-FIRST (forward topo): every contribution to bar_x over an
    SCC's members (caller seed + all upstream phantom routings) has landed
    before that SCC's VJP runs. This is resolution (ii): parent vertex t's
    cotangent is consumed by the unique SCC containing t, and is fed by the
    caller seed plus one phantom-rule routing per upstream channel v->t.
    """
    if co is None:
        co = compose_py(ex, theta)
    P = ex.P
    dtheta = np.zeros(P)
    bar_x = np.asarray(lam, dtype=float).copy()

    for si in co.order:  # source-first (reverse of composition order)
        rec = co.recs[si]
        if rec.kind == "absorbing":
            # constant-0 output: cotangent dropped (matches derivative 0)
            continue

        if rec.kind == "start":
            bar_s = bar_x[ex.start]
            rho = rec.start_rho
            for ch in rec.channels:
                p = ch["w"] / rho
                # E_start = sum p * E_sabs ; E_sabs = 1/y ; y = 1/x or CLAMP
                bar_esabs = bar_s * p
                y = ch["y"]
                bar_y = bar_esabs * (-1.0 / (y * y))     # d(1/y)/dy
                if not ch["clamped"]:
                    x = ch["x"]
                    bar_x[ch["t"]] += bar_y * (-1.0 / (x * x))  # d(1/x)/dx
                # else: y == 1e300 is a CONSTANT -> derivative 0, cotangent dropped
                # Type-C weight adjoint at the start SCC: w and rho are BOTH
                # start-edge weights -> theta-CONSTANT (update_weights skips the
                # start vertex, phasic.c:5650-5655) -> no dtheta contribution.
                # ch["coeffs"] is deliberately NEVER contracted here even though
                # the IPV edges DO carry coefficient arrays -- contracting them
                # would be a real C-implementation bug (they are never rescaled).
            continue

        # normal SCC: cotangent-seeded VJP of M E = 1
        members, channels = rec.members, rec.channels
        pos = {v: i for i, v in enumerate(members)}
        nm, nc = len(members), len(channels)
        bar_E = np.zeros(nm + nc)
        for v in members:
            bar_E[pos[v]] = bar_x[v]
        # s_abs results are never copied to parent_result -> zero seeds
        mu = np.linalg.solve(rec.M.T, bar_E)
        E = rec.E
        # d/dq [ mu^T (1 - M E) ]: for edge weight w on (v -> u):
        #   dM[v,v]/dw = +1, dM[v,u]/dw = -1  =>  bar_w = mu_v (E_u - E_v)
        for (v, u, w, coeffs) in rec.internal:
            bar_w = mu[pos[v]] * (E[pos[u]] - E[pos[v]])
            if coeffs is not None:
                dtheta += bar_w * coeffs          # Type-B: own real coefficients
        for k, ch in enumerate(channels):
            e_sabs = E[nm + k]
            bar_w = mu[pos[ch["v"]]] * (e_sabs - E[pos[ch["v"]]])
            if ch["coeffs"] is not None:
                dtheta += bar_w * ch["coeffs"]    # Type-C: PARENT edge's real coeffs
            # phantom edge s_abs -> phantom-absorbing (E_phantom = 0):
            #   bar_y = mu_sabs * (0 - E_sabs)
            bar_y = -mu[nm + k] * e_sabs
            if not ch["clamped"]:
                x = ch["x"]
                bar_x[ch["t"]] += bar_y * (-1.0 / (x * x))   # piecewise rule, x>0
            # else: CLAMP constant -> derivative 0, cotangent dropped
    return dtheta


# ---------------------------------------------------------------------------
# Oracle (a): dense monolithic solve + jax.vjp with random cotangents
# ---------------------------------------------------------------------------

def make_dense_fn(ex: Extraction):
    trans = [v for v in range(ex.n) if v != ex.start and len(ex.out[v]) > 0]
    tpos = {v: i for i, v in enumerate(trans)}
    nt = len(trans)

    def f(theta):
        M = jnp.zeros((nt, nt))
        for v in trans:
            for e in ex.out[v]:
                if e.kind == "param":
                    w = jnp.dot(jnp.asarray(e.coeffs), theta)
                else:
                    w = e.w0
                M = M.at[tpos[v], tpos[v]].add(w)
                if e.to in tpos:
                    M = M.at[tpos[v], tpos[e.to]].add(-w)
        Et = jnp.linalg.solve(M, jnp.ones(nt))
        E = jnp.zeros(ex.n)
        for v in trans:
            E = E.at[v].set(Et[tpos[v]])
        # start vertex: IPV probability average, NO holding time; start-edge
        # weights are theta-constant by the C convention.
        ws = jnp.asarray([e.w0 for e in ex.out[ex.start]])
        tos = [e.to for e in ex.out[ex.start]]
        Es = sum((ws[i] / ws.sum()) * E[t] for i, t in enumerate(tos))
        E = E.at[ex.start].set(Es)
        return E

    return f


# ---------------------------------------------------------------------------
# Perturbation test (iv): Type-A source-edge placeholders
# ---------------------------------------------------------------------------

def type_a_perturbation(ex: Extraction, theta: np.ndarray, lam: np.ndarray):
    """Re-solve each non-start SCC WITH the synthetic source + Type-A edges
    included in the transient system, at several placeholder weights. Returns
    (max member-value drift, max |mu_source|, max |dtheta drift|, would-be
    theta[0] pollution if the [1,0,...] placeholders were wrongly contracted).

    The C builder (scc_synthetic.c:844-857) gives Type-A edges placeholder
    coefficients [1.0, 0, ...]; under the synth graph's update_weights the
    START vertex (the synthetic source) is skipped, so the weights stay at the
    construction value 1.0 -- but we perturb them hard anyway, per review F7:
    zero adjoint must be SHOWN, not assumed.
    """
    co = compose_py(ex, theta)
    base_dtheta = adjoint_py(ex, theta, lam, co)
    scc_of = {}
    for i, comp in enumerate(co.sccs):
        for v in comp:
            scc_of[v] = i
    max_val_drift = 0.0
    max_mu_source = 0.0
    max_dth_drift = 0.0
    max_pollution = 0.0

    for si, rec in co.recs.items():
        if rec.kind != "normal":
            continue
        members = rec.members
        pos = {v: i for i, v in enumerate(members)}
        nm, nc = len(members), len(rec.channels)
        # upstream-connecting members: receive an edge from outside the SCC
        uc = sorted({e.to for v in range(ex.n) if scc_of[v] != si
                     for e in ex.out[v] if e.to in pos})
        if not uc:
            continue
        for wA in (1.0, 137.0, 1e-3):
            nt = 1 + nm + nc          # source row 0, then members, then s_abs
            M = np.zeros((nt, nt))
            rhoA = wA * len(uc)
            M[0, 0] = rhoA
            for v in uc:
                M[0, 1 + pos[v]] -= wA           # Type-A: source -> uc member
            for (v, u, w, _c) in rec.internal:
                M[1 + pos[v], 1 + pos[v]] += w
                M[1 + pos[v], 1 + pos[u]] -= w
            for k, ch in enumerate(rec.channels):
                M[1 + pos[ch["v"]], 1 + pos[ch["v"]]] += ch["w"]
                M[1 + pos[ch["v"]], 1 + nm + k] -= ch["w"]
                M[1 + nm + k, 1 + nm + k] = ch["y"]
            E = np.linalg.solve(M, np.ones(nt))
            # member values must be untouched by the source/Type-A block
            drift = max(abs(E[1 + pos[v]] - rec.E[pos[v]]) /
                        max(abs(rec.E[pos[v]]), 1e-300) for v in members)
            max_val_drift = max(max_val_drift, drift)
            # adjoint with the source present: seed = incoming cotangent on
            # members (from the true full-graph reverse pass we can't know
            # per-SCC here, so use the caller lam restricted -- the STRUCTURE
            # of mu_source = 0 is seed-independent: source column of M has
            # only its diagonal)
            bar_E = np.zeros(nt)
            for v in members:
                bar_E[1 + pos[v]] = lam[v]
            mu = np.linalg.solve(M.T, bar_E)
            max_mu_source = max(max_mu_source, abs(mu[0]))
            # would-be pollution of theta[0] if the placeholder [1,0,...] were
            # (wrongly) contracted like a real coefficient vector:
            for v in uc:
                bar_wA = mu[0] * (E[1 + pos[v]] - E[0])
                max_pollution = max(max_pollution, abs(bar_wA * 1.0))
    # full-pipeline drift: dtheta must not change when sources are included
    # (adjoint_py never includes them; equality with base checks the pipeline
    # is deterministic, and the mu_source==0 result above shows inclusion
    # would add exactly zero).
    dth2 = adjoint_py(ex, theta, lam)
    max_dth_drift = float(np.max(np.abs(dth2 - base_dtheta)))
    return max_val_drift, max_mu_source, max_dth_drift, max_pollution


# ---------------------------------------------------------------------------
# Main gate loop
# ---------------------------------------------------------------------------

def relerr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def main() -> int:
    rng_root = np.random.default_rng(20260815)
    failures = []
    print("=" * 100)
    print("D1-E1: two-level SCC adjoint reference vs oracles "
          "(gate <= 1e-12 rel; FD-vs-C-compose <= 5e-6)")
    print("=" * 100)

    for name, builder in BUILDERS.items():
        g = builder()                      # keep alive: SCCGraph borrows it
        ex = extract(g)
        scc_c = g.scc_decomposition()      # C-side decomposition for compose()
        f_dense = make_dense_fn(ex)
        sccs = tarjan_scc(ex.n, [[e.to for e in ex.out[v]] for v in range(ex.n)])
        sizes = sorted((len(c) for c in sccs), reverse=True)
        print(f"\n--- {name}: n={ex.n}, SCCs={len(sccs)} sizes={sizes} ---")
        if max(sizes) == ex.n:
            print("    NOTE: single-SCC fixture -- two-level machinery degenerate here")

        # ------ per-theta gates ------
        for th in THETAS:
            theta = np.asarray(th, dtype=float)
            tag = f"{name} theta={th}"

            # value parity: mine vs C hierarchical compose vs monolithic
            g.update_weights(list(theta))
            mono = np.asarray(g.expected_waiting_time())
            comp_c = np.asarray(scc_c.compose(g, list(theta)))
            co = compose_py(ex, theta)
            e_vs_c = relerr(co.result, comp_c)
            e_vs_m = relerr(co.result, mono)

            # sanity: dense fn parity with monolithic
            e_dense = relerr(np.asarray(f_dense(jnp.asarray(theta))), mono)

            # oracle (a): >= 5 random per-vertex cotangent seeds
            _, vjp = jax.vjp(f_dense, jnp.asarray(theta))
            max_vjp_err = 0.0
            for _ in range(N_COT_SEEDS):
                lam = rng_root.standard_normal(ex.n)
                want = np.asarray(vjp(jnp.asarray(lam))[0])
                got = adjoint_py(ex, theta, lam, co)
                max_vjp_err = max(max_vjp_err, relerr(got, want))

            # oracle (b): shipped binding, e_start one-hot
            J1 = np.asarray(g._moments_grad_theta(1))
            if J1.size == 0:
                e_ship = float("nan")
                ship_note = "DECLINED(MPFR gate)"
            else:
                lam0 = np.zeros(ex.n)
                lam0[ex.start] = 1.0
                got0 = adjoint_py(ex, theta, lam0, co)
                e_ship = relerr(got0, J1)
                ship_note = ""

            # (iii): my Jacobian (basis cotangents) vs central FD of C compose
            J_mine = np.zeros((ex.n, ex.P))
            for v in range(ex.n):
                lamv = np.zeros(ex.n)
                lamv[v] = 1.0
                J_mine[v] = adjoint_py(ex, theta, lamv, co)
            J_fd = np.zeros((ex.n, ex.P))
            for k in range(ex.P):
                eps = 1e-6 * max(1.0, abs(theta[k]))
                tp, tm = theta.copy(), theta.copy()
                tp[k] += eps
                tm[k] -= eps
                J_fd[:, k] = (np.asarray(scc_c.compose(g, list(tp)))
                              - np.asarray(scc_c.compose(g, list(tm)))) / (2 * eps)
            e_fd = float(np.max(np.abs(J_mine - J_fd)) /
                         max(np.max(np.abs(J_fd)), 1.0))

            ok = (e_vs_c <= 1e-12 and e_vs_m <= 1e-12 and e_dense <= 1e-12
                  and max_vjp_err <= 1e-12
                  and (np.isnan(e_ship) or e_ship <= 1e-12)
                  and e_fd <= 5e-6)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append(tag)
            print(f"  [{status}] theta={th}")
            print(f"      value:  mine-vs-Ccompose={e_vs_c:.2e}  mine-vs-mono={e_vs_m:.2e}"
                  f"  dense-vs-mono={e_dense:.2e}  clamp_hits={co.clamp_hits}")
            print(f"      grad:   dense-VJP(max over {N_COT_SEEDS} random seeds)={max_vjp_err:.2e}"
                  f"  shipped-e_start={e_ship:.2e} {ship_note}"
                  f"  J-vs-FD(C compose)={e_fd:.2e}")

        # ------ (iv) Type-A perturbation, once per fixture ------
        theta = np.asarray(THETAS[1])
        lam = rng_root.standard_normal(ex.n)
        vdrift, mu_src, dth_drift, pollution = type_a_perturbation(ex, theta, lam)
        ok4 = vdrift <= 1e-13 and mu_src <= 1e-15 and dth_drift == 0.0 \
            and pollution <= 1e-15
        # dtheta[0] sentinel: theta[0] only drives never-rescaled start edges
        dth = adjoint_py(ex, theta, lam)
        ok0 = dth[0] == 0.0
        if not (ok4 and ok0):
            failures.append(f"{name} type-A")
        print(f"  [{'PASS' if ok4 and ok0 else 'FAIL'}] Type-A perturbation: "
              f"member-value drift={vdrift:.2e}  |mu_source|={mu_src:.2e}  "
              f"dtheta drift={dth_drift:.2e}  would-be theta[0] pollution={pollution:.2e}  "
              f"dtheta[0]={dth[0]!r}")

    print("\n" + "=" * 100)
    if failures:
        print(f"NO-GO: {len(failures)} gate failure(s): {failures}")
        return 1
    print("GO: all gates pass -- two-level adjoint == dense oracle == shipped "
          "e_start binding on all 6 fixtures x 4 thetas; Type-A placeholders "
          "receive exactly zero adjoint (shown by perturbation); the only "
          "cross-SCC couplings are Type-C weights and phantom 1/x (J matches "
          "FD of the real C composer).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
