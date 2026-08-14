"""Batch B I1 micro-gates (plan v2 SS-G): formula-mode exact moment gradient.

Modes:
  dump <npz>   -- PRE-B install (main checkout at the branch base):
                  linear/log/dph Jacobians, REWARDLESS and (linear/log)
                  REWARDS-BEARING -- byte-identity goldens for the three
                  untouched kinds (the core is edited in place; v2 SS-G
                  mandates fresh goldens incl. the rewards paths, which
                  Batch A's own gate only checked by tolerance).
  check <npz>  -- B build:
    (a) byte-identity vs the dump (5 golden arrays);
    (b) formula correctness:
        b1 linear-equivalent formula vs the LINEAR exact path on a twin
           graph (FD-INDEPENDENT oracle) at benign AND mixed
           (theta=[1,1e-8]) scales -- the motivating-defect cell;
        b2 pow/exp mix vs FD-of-the-PRIMAL at K=1..3;
        b3 theta-in-exponent (t0**t1) vs FD-of-primal;
        b4 select() with edges on both branches vs FD-of-primal;
        b5 aux-CONSTANT-edge fixture vs FD-of-primal (the skip-set
           discriminator: FD never moves the constant edge; an unskipped
           edge would shift the exact J by a finite wrong term);
    (c) rewards x formula: mixed rewards vs FD-of-primal; all-ones ==
        rewardless BITWISE;
    (d) decline contract: was_dph+formula; no-tape (linear graph via the
        formula wrapper); theta_len mismatch; rewards_len mismatch;
    (e) POW domain edges: d(t0**2), d(t0**1), d(t0**3) at t0=0 against
        the analytic values (0, 1, 0) -- the two-term-rule cells; plus
        the D-B5 sqrt boundary decline at the WRAPPER level (t0=1.0,
        formula sqrt(t0-c0)+0.5: inner gradient inf -> non-finite J ->
        size 0; asserting at the wrapper avoids the FD-minus-probe
        domain trap recorded in b3-batchB-findings.md).
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
MODE, NPZ = sys.argv[1], sys.argv[2]
fails = []


# ---------------------------------------------------------------- fixtures
def _lin_chain4():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for i in range(3):
        v[i].add_edge(v[i + 1], [float(i + 1)])
    v[3].add_edge(va, [2.0])
    return g


def _log2():
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_mode = 'log'
    return g


def _wasdph():
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    return g.discretize(0.5)


THETA1 = [2.0]
THETA2 = [1.0, 2.0]
K = 3

if MODE == "dump":
    out = {}
    g = _lin_chain4(); g.update_weights(THETA1)
    out["lin_plain"] = np.asarray(g._moments_grad_theta(K))
    n = g.vertices_length()
    rw = np.ones(n); rw[1] = 0.0; rw[2] = 2.0; rw[-1] = 0.5
    g.update_weights(THETA1)
    out["lin_rw"] = np.asarray(g._moments_grad_theta(K, rewards=rw.tolist()))
    gl = _log2(); gl.update_weights(THETA2, log=True)
    out["log_plain"] = np.asarray(gl._moments_grad_theta_log(K, THETA2))
    nl = gl.vertices_length()
    rwl = np.ones(nl); rwl[1] = 0.0; rwl[2] = 2.0; rwl[-1] = 0.5
    out["log_rw"] = np.asarray(gl._moments_grad_theta_log(K, THETA2,
                                                          rewards=rwl.tolist()))
    gd = _wasdph()
    thd = [0.3] * gd.param_length()
    out["dph_plain"] = np.asarray(gd._moments_grad_theta_dph(2, thd))
    np.savez(NPZ, **out)
    print("dumped:", {k: v.shape for k, v in out.items()})
    sys.exit(0)

assert MODE == "check"
old = np.load(NPZ)

print("== (a) byte-identity vs pre-B goldens (5 arrays incl. rewards paths) ==")
g = _lin_chain4(); g.update_weights(THETA1)
new = {"lin_plain": np.asarray(g._moments_grad_theta(K))}
n = g.vertices_length()
rw = np.ones(n); rw[1] = 0.0; rw[2] = 2.0; rw[-1] = 0.5
g.update_weights(THETA1)
new["lin_rw"] = np.asarray(g._moments_grad_theta(K, rewards=rw.tolist()))
gl = _log2(); gl.update_weights(THETA2, log=True)
new["log_plain"] = np.asarray(gl._moments_grad_theta_log(K, THETA2))
nl = gl.vertices_length()
rwl = np.ones(nl); rwl[1] = 0.0; rwl[2] = 2.0; rwl[-1] = 0.5
new["log_rw"] = np.asarray(gl._moments_grad_theta_log(K, THETA2,
                                                      rewards=rwl.tolist()))
gd = _wasdph()
new["dph_plain"] = np.asarray(gd._moments_grad_theta_dph(
    2, [0.3] * gd.param_length()))
for key, arr in new.items():
    same = np.array_equal(arr, old[key])
    print(f"  {key}: identical={same}")
    if not same:
        fails.append(f"(a) {key}")


# ---------------------------------------------------------------- helpers
def _fd_primal_jac(mk, theta, rewards, k, rel=1e-6):
    th = np.asarray(theta, float)
    J = np.empty((k, th.size))
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-10)
        tp = th.copy(); tp[j] += h
        tm = th.copy(); tm[j] -= h
        gp = mk(); gp.update_weights(tp.tolist())
        mp = np.asarray(gp.moments(k) if rewards is None
                        else gp.moments(k, list(rewards)))
        gm = mk(); gm.update_weights(tm.tolist())
        mm = np.asarray(gm.moments(k) if rewards is None
                        else gm.moments(k, list(rewards)))
        J[:, j] = (mp - mm) / (2 * h)
    return J


def _formula_jac(g, theta, k, rewards=None):
    g.update_weights(list(theta))
    kw = {} if rewards is None else {"rewards": list(rewards)}
    return np.asarray(g._moments_grad_theta_formula(k, list(theta), **kw))


def _check_cell(label, J, ref, tol):
    if J.size != ref.size:
        print(f"  {label}: DECLINED (size {J.size})")
        fails.append(f"{label} declined")
        return
    J = J.reshape(ref.shape)
    rel = np.max(np.abs(J - ref)) / max(np.max(np.abs(ref)), 1e-300)
    status = "PASS" if rel < tol else "FAIL"
    print(f"  {label}: rel={rel:.2e} {status}")
    if rel >= tol:
        fails.append(label)


print("== (b1) linear-equivalent formula vs LINEAR exact (FD-independent) ==")


def _feq(coeffs_list, param_length=None):
    # formula twin: w = t0*c0 + t1*c1 == the linear dot product. When the
    # coefficient vectors are LONGER than n_theta, param_length must be
    # pinned FIRST (the canonical aligned-decoupled pattern, D-B6.2) --
    # otherwise the graph is the lazy-decoupled class the exact path
    # statically declines (b3-batchB-findings.md).
    g = Graph(1)
    if param_length is not None:
        g.set_param_length(param_length)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    edges = [(v[0], v[1]), (v[1], v[2]), (v[2], va)]
    for (a, b), c in zip(edges, coeffs_list):
        a.add_edge(b, list(c))
    return g


COEFFS = [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]
for theta, tag in ((np.asarray([1.0, 2.0]), "benign"),
                   (np.asarray([1.0, 1e-8]), "MIXED-SCALE")):
    gf = _feq(COEFFS)
    gf.weight_formula = "t0*c0 + t1*c1"
    Jf = _formula_jac(gf, theta, K)
    glin = _feq(COEFFS)
    glin.update_weights(theta.tolist())
    Jl = np.asarray(glin._moments_grad_theta(K))
    _check_cell(f"(b1) {tag}", Jf, Jl.reshape(K, 2), 1e-12)

print("== (b2) pow/exp mix vs primal-FD, K=1..3 ==")


def _pow_fixture():
    g = _feq([(1.0, 0.5, 2.0), (2.0, 0.25, 1.5), (0.5, 1.5, 3.0)],
             param_length=2)
    g.weight_formula = "(t0*c0)**c1 + t1*c2*0.1"
    return g


for kk in (1, 2, 3):
    gf = _pow_fixture()
    Jf = _formula_jac(gf, [1.3, 0.7], kk)
    ref = _fd_primal_jac(_pow_fixture, [1.3, 0.7], None, kk)
    _check_cell(f"(b2) K={kk}", Jf, ref, 1e-5)

print("== (b3) theta-in-exponent t0**t1 vs primal-FD ==")


def _texp_fixture():
    # coefficient vectors padded to param_length (edges need cl >= P);
    # the formula reads only c0
    g = _feq([(1.2, 0.0), (0.8, 0.0), (1.5, 0.0)], param_length=2)
    g.weight_formula = "c0 * t0**t1 + 0.2"
    return g


Jf = _formula_jac(_texp_fixture(), [1.4, 0.6], 2)
ref = _fd_primal_jac(_texp_fixture, [1.4, 0.6], None, 2)
_check_cell("(b3)", Jf, ref, 1e-5)

print("== (b4) select() with edges on both branches vs primal-FD ==")


def _sel_fixture():
    # c0 differs per edge: edges with c0>1.5 use t0*c1, others t1*c1
    g = _feq([(2.0, 1.0), (1.0, 2.0), (3.0, 0.5)])
    g.weight_formula = "select(c0 > 1.5, t0*c1, t1*c1)"
    return g


Jf = _formula_jac(_sel_fixture(), [1.1, 0.9], 2)
ref = _fd_primal_jac(_sel_fixture, [1.1, 0.9], None, 2)
_check_cell("(b4)", Jf, ref, 1e-5)

print("== (b5) aux-CONSTANT-edge fixture (skip-set discriminator) ==")


def _const_fixture():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    v2.add_aux_vertex_constant(0.7)
    g.weight_formula = "t0*c0 + t1*c1"
    return g


Jf = _formula_jac(_const_fixture(), [1.2, 0.8], 2)
ref = _fd_primal_jac(_const_fixture, [1.2, 0.8], None, 2)
_check_cell("(b5)", Jf, ref, 1e-5)

print("== (c) rewards x formula ==")
gf = _feq(COEFFS)
gf.weight_formula = "t0*c0 + t1*c1"
nf = gf.vertices_length()
rwf = np.ones(nf); rwf[1] = 0.0; rwf[2] = 2.0; rwf[-1] = 0.5


def _mk_feq():
    g = _feq(COEFFS)
    g.weight_formula = "t0*c0 + t1*c1"
    return g


Jr = _formula_jac(gf, [1.0, 2.0], K, rewards=rwf)
refr = _fd_primal_jac(_mk_feq, [1.0, 2.0], rwf, K)
_check_cell("(c) mixed rewards", Jr, refr, 1e-5)
g1 = _mk_feq()
Jo = _formula_jac(g1, [1.0, 2.0], K, rewards=np.ones(nf))
g2 = _mk_feq()
J0 = _formula_jac(g2, [1.0, 2.0], K)
same = np.array_equal(Jo, J0)
print(f"  (c) all-ones == rewardless: {same}")
if not same:
    fails.append("(c) ones!=rewardless")

print("== (d) decline contract ==")
gw = _wasdph()
# was_dph graph has no formula tape either, but was_dph fires first; the
# discriminating no-tape cell is the plain-linear graph below.
r1 = np.asarray(gw._moments_grad_theta_formula(
    2, [0.3] * gw.param_length())).size
gl2 = _lin_chain4()
gl2.update_weights(THETA1)
r2 = np.asarray(gl2._moments_grad_theta_formula(2, THETA1)).size
g3 = _mk_feq()
g3.update_weights([1.0, 2.0])
r3 = np.asarray(g3._moments_grad_theta_formula(2, [1.0])).size  # short theta
r4 = np.asarray(g3._moments_grad_theta_formula(
    2, [1.0, 2.0], rewards=[1.0, 2.0])).size  # wrong rewards len
print(f"  was_dph/no-tape/theta-mismatch/rewards-mismatch sizes: "
      f"{r1}/{r2}/{r3}/{r4} (all 0 = declined)")
if r1 or r2 or r3 or r4:
    fails.append("(d) decline contract")

print("== (e) POW domain edges + D-B5 boundary decline ==")
for expo, want in ((2.0, 0.0), (1.0, 1.0), (3.0, 0.0)):
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [expo])
    # w = t0**c0 + 1.0 so the weight stays positive at t0=0
    g.weight_formula = "t0**c0 + 1.0"
    J = _formula_jac(g, [0.0], 1)
    if J.size != 1:
        print(f"  (e) d(t0**{expo}) at 0: DECLINED (size {J.size}) FAIL")
        fails.append(f"(e) pow {expo}")
        continue
    # dm_1/dt0 = dm/dw * dw/dt0; dw/dt0 at 0 = want (0, 1, 0). For
    # want==0 the whole gradient must be exactly 0; for want==1 just
    # finite/nonzero -- the analytic pin is on dw, checked via J==0.
    okc = (J[0] == 0.0) if want == 0.0 else np.isfinite(J[0]) and J[0] != 0.0
    print(f"  (e) d(t0**{expo}) at t0=0: J={J[0]:.6g} "
          f"{'PASS' if okc else 'FAIL'}")
    if not okc:
        fails.append(f"(e) pow {expo}")

gsq = Graph(1)
s = gsq.starting_vertex()
v2 = gsq.find_or_create_vertex([2])
v1 = gsq.find_or_create_vertex([1])
s.add_edge(v2, 1.0)
v2.add_edge(v1, [1.0])
gsq.weight_formula = "sqrt(t0 - c0) + 0.5"
gsq.update_weights([1.0])
Jb = np.asarray(gsq._moments_grad_theta_formula(2, [1.0]))
print(f"  (e) sqrt boundary wrapper decline: size={Jb.size} (0 = declined) "
      f"{'PASS' if Jb.size == 0 else 'FAIL'}")
if Jb.size != 0:
    fails.append("(e) sqrt boundary")

# (e2) G4 fold (B-G4-3): POW at a=0 with b=0.5 -- the POW OPCODE's own
# boundary (the sqrt cell above exercises the SQRT opcode, a different
# adjoint rule): true dw = 0.5*0**(-0.5) = inf -> non-finite J -> decline.
gp5 = Graph(1)
s = gp5.starting_vertex()
v2 = gp5.find_or_create_vertex([2])
v1 = gp5.find_or_create_vertex([1])
s.add_edge(v2, 1.0)
v2.add_edge(v1, [0.5])
gp5.weight_formula = "t0**c0 + 1.0"
gp5.update_weights([0.0])
Jp5 = np.asarray(gp5._moments_grad_theta_formula(2, [0.0]))
print(f"  (e2) d(t0**0.5) at t0=0 declines: size={Jp5.size} (0 = declined) "
      f"{'PASS' if Jp5.size == 0 else 'FAIL'}")
if Jp5.size != 0:
    fails.append("(e2) pow 0.5 boundary")

print()
print("ALL MICRO-GATES PASS" if not fails else f"FAILURES: {fails}")
sys.exit(0 if not fails else 1)
