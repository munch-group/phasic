"""Deferred-2 activation gate A2 -- the plan §1 value test.

Question: after Batch H (exact FINAL-epoch slots in production svgd),
how much gradient error remains attributable to the FD INTERMEDIATE
epoch terms?

Method (no full E3 reference needed -- and its collapse-modeling risk
avoided): the production `daisy_chain_joint_probs` primitive is its own
oracle via Richardson-extrapolated RELATIVE-step central differences of
the PRIMAL (the shipped custom_vjp backward uses ABSOLUTE eps=1e-7
steps -- the known B3 defect class; Richardson at relative steps h and
h/2 is accurate wherever the primal is smooth, which it is here).
Per-epoch-slot error attribution: the flat theta is (n_epochs x P);
epoch-0 slots are the INTERMEDIATE terms D2 would make exact, epoch-1
slots are what H already covers in production (this primitive itself is
all-FD, so its epoch-1 error also shows what H removed).

Honest limitation (stated per the plan's discipline): Richardson-FD is
still FD -- at a truly pathological theta both estimators fail
together; the mixed-scale point here (theta0 = 1e-8-class coalescent
rate) is the moderately-mixed regime where relative-step FD remains
valid while absolute-step FD does not.

Fixture: the test_lrt_at coalescent class at nr_samples=3 (JSP 56
vertices), 2 epochs, granularity pinned at 2048.
"""
import sys
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

set_log_level("WARNING")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

all_pairs = partial(combinations_with_replacement, r=2)
NR = 3
MU = 1e-4
GRAN = 2048
DTS = [0.5]

indexer = StateIndexer(
    lineages=[Property("descendants", min_value=1, max_value=NR)])


@with_ipv([NR] + [0] * (NR - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t


graph = Graph(coal, indexer=indexer)
jpg = graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=MU,
                             discrete=False)
jsp = jpg.joint_stop_prob_graph()
ipv_full = np.zeros(jpg.vertices_length(), dtype=np.float64)
for edge in jpg.starting_vertex().edges():
    ipv_full[edge.to().index()] = edge.weight()
initial_ipv = ipv_full[jsp._ipv_target_indices]
print(f"JSP vertices: {jsp.vertices_length()}; n_ipv: {len(initial_ipv)}")


def loss(flat_theta):
    et = flat_theta.reshape(2, 2)
    probs = jsp.daisy_chain_joint_probs(
        epoch_thetas=et, epoch_dts=DTS, initial_ipv=initial_ipv,
        granularity=GRAN, final_read='sojourn')
    # a smooth scalar readout over the joint probs
    return jnp.sum(probs * jnp.arange(1.0, probs.shape[0] + 1.0))


def primal_np(flat_theta):
    return float(loss(jnp.asarray(flat_theta)))


def richardson_grad(flat_theta, rel=1e-4):
    th = np.asarray(flat_theta, float)
    g = np.zeros_like(th)
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-12)
        def cd(hh):
            tp = th.copy(); tp[j] += hh
            tm = th.copy(); tm[j] -= hh
            return (primal_np(tp) - primal_np(tm)) / (2 * hh)
        d1 = cd(h)
        d2 = cd(h / 2)
        g[j] = (4 * d2 - d1) / 3.0     # Richardson: O(h^4)
    return g


for tag, th0 in (("benign", 1e-4), ("mixed-scale", 1e-8)):
    flat = np.array([th0, MU, th0 * 2.0, MU])
    g_ref = richardson_grad(flat)
    denom = max(np.max(np.abs(g_ref)), 1e-300)
    print(f"\n[{tag}] theta0={th0:g}")
    print(f"  reference grad: {g_ref.tolist()}")
    try:
        g_ship = np.asarray(jax.grad(loss)(jnp.asarray(flat)))
    except Exception as exc:
        # The shipped backward is absolute-step FD (eps=1e-7,
        # ffi_wrappers.py:1296 / the svgd model's eps_local): its probe
        # theta-eps goes NEGATIVE when a slot is below eps, and the
        # FFI's rate validation raises. At mixed scale the shipped
        # gradient does not merely lose accuracy -- it CRASHES.
        print(f"  shipped-FD grad: RAISED {type(exc).__name__}: "
              f"{str(exc).splitlines()[-1][:120]}")
        continue
    err = np.abs(g_ship - g_ref) / denom
    # slots: [e0_th, e0_mu, e1_th, e1_mu]; mu slots are fixed-in-practice
    print(f"  shipped-FD grad: {g_ship.tolist()}")
    print(f"  per-slot rel err: e0_th={err[0]:.2e} e0_mu={err[1]:.2e} "
          f"e1_th={err[2]:.2e} e1_mu={err[3]:.2e}")
    print(f"  INTERMEDIATE-epoch (e0) worst: {max(err[0], err[1]):.2e}; "
          f"FINAL-epoch (e1) worst: {max(err[2], err[3]):.2e}")

print("\nA2 VALUE TEST COMPLETE (interpretation in the findings)")
