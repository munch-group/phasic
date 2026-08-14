"""Batch C I1 micro-gates (plan v2): the pre-contraction binp exit.

Modes:
  dump <npz>   -- PRE-C install (main checkout at branch base): 7 goldens
                  (lin/log/formula x {plain, rewards} + dph_plain) --
                  byte-identity for the four existing kinds (the core
                  signature is edited again).
  check <npz>  -- C build:
    (a) byte-identity vs the dump (7 arrays);
    (b1) exit+matmul on a LINEAR-EQUIVALENT callback vs the LINEAR exact
         path on a twin graph (FD-independent oracle), benign AND mixed
         (theta=[1,1e-8]) scale;
    (b2) nonlinear callback (exp/ratio) vs FD-of-the-PRIMAL, K=1..3;
    (b3) LAZY-BUILT aux-coefficients callback (coeff len > n_theta, no
         set_param_length -- the documented primary idiom, SUPPORTED per
         v2 SS-A) vs FD-of-primal;
    (c) rewards x callback vs FD-of-primal + all-ones == rewardless
        BITWISE;
    (d) decline contract: was_dph (load-bearing per D-C5); rewards_len
        mismatch; empty-tuple representation;
    (e) frozen flags: an add_aux_vertex_constant edge is flagged and the
        skip-aware composition still matches FD-of-primal.

The gate contracts J = binp @ W itself (jax.grad rows; frozen rows
skipped) -- the model-level dispatch is I3's surface, gated by pytest.
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


def _formula2():
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_formula = "t0*c0 + t1*c1"
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
    gf = _formula2(); gf.update_weights(THETA2)
    out["formula_plain"] = np.asarray(gf._moments_grad_theta_formula(K, THETA2))
    gf2 = _formula2(); gf2.update_weights(THETA2)
    out["formula_rw"] = np.asarray(gf2._moments_grad_theta_formula(
        K, THETA2, rewards=rwl.tolist()))
    gd = _wasdph()
    out["dph_plain"] = np.asarray(gd._moments_grad_theta_dph(
        2, [0.3] * gd.param_length()))
    np.savez(NPZ, **out)
    print("dumped:", {k: v.shape for k, v in out.items()})
    sys.exit(0)

assert MODE == "check"
import jax
import jax.numpy as jnp

old = np.load(NPZ)

print("== (a) byte-identity vs pre-C goldens (7 arrays) ==")
new = {}
g = _lin_chain4(); g.update_weights(THETA1)
new["lin_plain"] = np.asarray(g._moments_grad_theta(K))
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
gf = _formula2(); gf.update_weights(THETA2)
new["formula_plain"] = np.asarray(gf._moments_grad_theta_formula(K, THETA2))
gf2 = _formula2(); gf2.update_weights(THETA2)
new["formula_rw"] = np.asarray(gf2._moments_grad_theta_formula(
    K, THETA2, rewards=rwl.tolist()))
gd = _wasdph()
new["dph_plain"] = np.asarray(gd._moments_grad_theta_dph(
    2, [0.3] * gd.param_length()))
for key, arr in new.items():
    same = np.array_equal(arr, old[key])
    print(f"  {key}: identical={same}")
    if not same:
        fails.append(f"(a) {key}")


# ---------------------------------------------------------------- helpers
def _exit_contract(g, cb, theta, k, rewards=None):
    """The Batch-C composition the model will use: exit -> W -> matmul."""
    g.update_weights(list(theta), callback=cb)
    kw = {} if rewards is None else {"rewards": list(rewards)}
    binp_flat, coeffs, frozen = g._moments_binp_exit(k, **kw)
    binp_flat = np.asarray(binp_flat)
    if binp_flat.size == 0:
        return None
    ni = len(frozen)
    binp = binp_flat.reshape(k, ni)
    P = len(theta)
    grad_cb = jax.jit(jax.grad(cb, argnums=0))
    W = np.zeros((ni, P))
    for i in range(ni):
        if frozen[i]:
            continue
        W[i] = np.asarray(grad_cb(jnp.asarray(theta, dtype=jnp.float64),
                                  jnp.asarray(coeffs[i], dtype=jnp.float64)))
    return binp @ W


def _fd_primal_jac_cb(mk, cb, theta, rewards, k, rel=1e-6):
    th = np.asarray(theta, float)
    J = np.empty((k, th.size))
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-10)
        tp = th.copy(); tp[j] += h
        tm = th.copy(); tm[j] -= h
        gp = mk(); gp.update_weights(tp.tolist(), callback=cb)
        mp = np.asarray(gp.moments(k) if rewards is None
                        else gp.moments(k, list(rewards)))
        gm = mk(); gm.update_weights(tm.tolist(), callback=cb)
        mm = np.asarray(gm.moments(k) if rewards is None
                        else gm.moments(k, list(rewards)))
        J[:, j] = (mp - mm) / (2 * h)
    return J


def _check_cell(label, J, ref, tol):
    if J is None or J.size != ref.size:
        print(f"  {label}: DECLINED")
        fails.append(f"{label} declined")
        return
    J = J.reshape(ref.shape)
    rel = np.max(np.abs(J - ref)) / max(np.max(np.abs(ref)), 1e-300)
    status = "PASS" if rel < tol else "FAIL"
    print(f"  {label}: rel={rel:.2e} {status}")
    if rel >= tol:
        fails.append(label)


COEFFS = [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]


def _chain3(coeffs_list):
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)], coeffs_list):
        a.add_edge(b, list(c))
    return g


print("== (b1) linear-equivalent CALLBACK exit+matmul vs LINEAR exact ==")
cb_lin = lambda t, c: jnp.dot(t, c[:2])  # noqa: E731
for theta, tag in (([1.0, 2.0], "benign"), ([1.0, 1e-8], "MIXED-SCALE")):
    gcb = _chain3(COEFFS)
    gcb.weight_callback = cb_lin
    J = _exit_contract(gcb, cb_lin, theta, K)
    glin = _chain3(COEFFS)
    glin.update_weights(list(theta))
    Jl = np.asarray(glin._moments_grad_theta(K)).reshape(K, 2)
    _check_cell(f"(b1) {tag}", J, Jl, 1e-9)

print("== (b2) nonlinear callback vs primal-FD, K=1..3 ==")
cb_nl = lambda t, c: jnp.exp(t[0] * c[0] * 0.3) + t[1] * c[1] / (t[0] + 0.5)  # noqa: E731


def _mk_nl():
    g = _chain3(COEFFS)
    g.weight_callback = cb_nl
    return g


for kk in (1, 2, 3):
    J = _exit_contract(_mk_nl(), cb_nl, [1.3, 0.7], kk)
    ref = _fd_primal_jac_cb(_mk_nl, cb_nl, [1.3, 0.7], None, kk)
    _check_cell(f"(b2) K={kk}", J, ref, 1e-5)

print("== (b3) LAZY aux-coefficients callback (len 3 > n_theta 2) ==")
cb_aux = lambda t, c: t[0] * c[0] + t[1] * c[1] + c[2] * 0.1  # noqa: E731


def _mk_aux():
    # lazily built: 3-coeff edges lock param_length=3; theta has len 2 --
    # the documented PSMC-style idiom, SUPPORTED (v2 SS-A)
    g = _chain3([(1.0, 0.5, 2.0), (2.0, 0.25, 1.0), (0.5, 1.5, 0.3)])
    g.weight_callback = cb_aux
    return g


J = _exit_contract(_mk_aux(), cb_aux, [1.1, 0.9], 2)
ref = _fd_primal_jac_cb(_mk_aux, cb_aux, [1.1, 0.9], None, 2)
_check_cell("(b3) lazy aux-coeffs", J, ref, 1e-5)

print("== (c) rewards x callback ==")
gc = _chain3(COEFFS)
gc.weight_callback = cb_lin
nrw = gc.vertices_length()
rwc = np.ones(nrw); rwc[1] = 0.0; rwc[2] = 2.0; rwc[-1] = 0.5


def _mk_lin_cb():
    g = _chain3(COEFFS)
    g.weight_callback = cb_lin
    return g


J = _exit_contract(_mk_lin_cb(), cb_lin, [1.0, 2.0], K, rewards=rwc)
ref = _fd_primal_jac_cb(_mk_lin_cb, cb_lin, [1.0, 2.0], rwc, K)
_check_cell("(c) mixed rewards", J, ref, 1e-5)
J1 = _exit_contract(_mk_lin_cb(), cb_lin, [1.0, 2.0], K,
                    rewards=np.ones(nrw))
J0 = _exit_contract(_mk_lin_cb(), cb_lin, [1.0, 2.0], K)
same = np.array_equal(J1, J0)
print(f"  (c) all-ones == rewardless: {same}")
if not same:
    fails.append("(c) ones!=rewardless")

print("== (d) decline contract ==")
gw = _wasdph()
r1 = gw._moments_binp_exit(2)
d1 = len(np.asarray(r1[0])) == 0
g3 = _mk_lin_cb()
g3.update_weights([1.0, 2.0], callback=cb_lin)
r2 = g3._moments_binp_exit(2, rewards=[1.0, 2.0])  # wrong rewards length
d2 = len(np.asarray(r2[0])) == 0
# G4 fold: the non-parameterized decline (was in the plan's (d) list)
gnp_ = Graph(1)
s_ = gnp_.starting_vertex()
a_ = gnp_.find_or_create_vertex([2])
b_ = gnp_.find_or_create_vertex([1])
s_.add_edge(a_, 1.0)
a_.add_edge(b_, 0.7)  # constant-only graph: not parameterized
r3 = gnp_._moments_binp_exit(2)
d3 = len(np.asarray(r3[0])) == 0
print(f"  was_dph declined: {d1}; rewards-len declined: {d2}; "
      f"non-parameterized declined: {d3}")
if not (d1 and d2 and d3):
    fails.append("(d) decline contract")

print("== (e) frozen flags on an aux-constant edge ==")
ge = Graph(1)
s = ge.starting_vertex()
v2 = ge.find_or_create_vertex([2])
v1 = ge.find_or_create_vertex([1])
s.add_edge(v2, 1.0)
v2.add_edge(v1, [1.0, 0.5])
v2.add_aux_vertex_constant(0.7)
ge.weight_callback = cb_lin
ge.update_weights([1.2, 0.8], callback=cb_lin)
_, coeffs_e, frozen_e = ge._moments_binp_exit(2)
n_frozen = int(np.sum(np.asarray(frozen_e, dtype=int)))
print(f"  frozen inputs: {n_frozen} of {len(frozen_e)} "
      f"(expect >= 1: the aux-constant edge)")
if n_frozen < 1:
    fails.append("(e) no frozen flag")


def _mk_e():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    v2.add_aux_vertex_constant(0.7)
    g.weight_callback = cb_lin
    return g


J = _exit_contract(_mk_e(), cb_lin, [1.2, 0.8], 2)
ref = _fd_primal_jac_cb(_mk_e, cb_lin, [1.2, 0.8], None, 2)
_check_cell("(e) skip-aware composition", J, ref, 1e-5)

print()
print("ALL MICRO-GATES PASS" if not fails else f"FAILURES: {fails}")
sys.exit(0 if not fails else 1)
