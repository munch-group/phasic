"""Batch 0 de-risk checks for epoch-normalization-plan.md (v2).

Independent closed-form references for the normalised epoch (daisy-chain)
likelihood. Run inside the worktree env:

    pixi run python experiments/dr_epochnorm_checks.py

Sections (printed):
  A  n=2 one epoch, reward_limit / tot_reward_limit, exposure alphas
  B  n=2 two epochs (c1 != c2), granularity convergence
  C  n=3 one epoch, two features: closed form vs discrete table, and
     per-feature vs total cap separation
  D  gradients: exact_final_grad True vs False vs closed-form FD
  E  deterministic expected-log-likelihood gate (truth recovery;
     raw and shared-Z variants must miss)
  F  test.py model (n=4, coal 5, mu 3, reward_limit 3): tied 2-epoch MLE
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import gammainc, comb
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'pytest'))
from test_notebook_model_recovery import coalescent_indexed  # noqa: E402


# ---------------------------------------------------------------- closed forms
def p_one_epoch_n2(k, c, mu, alpha=1.0):
    m = 2 * alpha * mu
    q = m / (c + m)
    return (1 - q) * q ** k


def p_two_epoch_n2(k, c1, mu1, c2, mu2, t1, alpha=1.0):
    m1, m2 = 2 * alpha * mu1, 2 * alpha * mu2
    q1, q2 = m1 / (c1 + m1), m2 / (c2 + m2)
    a = (1 - q1) * q1 ** k * gammainc(k + 1, (c1 + m1) * t1)
    b = np.exp(-c1 * t1) * sum(poisson.pmf(k - j, m1 * t1) * (1 - q2) * q2 ** j
                               for j in range(k + 1))
    return a + b


def p_one_epoch_n3(k1, k2, c, mu, alpha=1.0):
    a = alpha * mu / (c + alpha * mu)
    q = 2 * alpha * mu / (c + 2 * alpha * mu)
    tot = 0.0
    for n in range(k1 + 1):
        s = k1 - n + k2
        tot += (1 - a) * a ** n * (1 - q) * q ** s * comb(s, k2) * 0.5 ** s
    return tot


def cap_ok(row, reward_limit, tot_limit):
    if reward_limit is not None and any(x > reward_limit for x in row):
        return False
    return sum(row) <= tot_limit


# ---------------------------------------------------------------- helpers
def epoch_model_out(jc, obs, epoch_starts, theta, **kw):
    em = jc.epoch_model(obs, epoch_starts, **kw)
    return np.asarray(em.model(jnp.asarray(theta, dtype=float))[0]), em


def builder_out(jc, obs_rows, epoch_starts, theta, **kw):
    mapped = jc._map_joint_observations_to_indices(obs_rows, seed=0)
    m, *_ = jc._daisy_chain_svgd_model(
        observed_indices=np.asarray(mapped), epoch_starts=epoch_starts,
        t_eval=200.0, **kw)
    return m, mapped


def section(title):
    print('\n' + '=' * 72 + '\n' + title + '\n' + '=' * 72, flush=True)


# ================================================================== A
section('A  n=2 one epoch vs closed form (normalised), with exposure')
g2, idx2 = coalescent_indexed(2)
L = 3
c, mu = 1.3, 0.7
for kw in [dict(reward_limit=L), dict(tot_reward_limit=L)]:
    jc = g2.joint_prob_graph(idx2, discrete=False, **kw)
    obs = [[k, 0] for k in range(L + 1)]
    out, _ = epoch_model_out(jc, obs, [0.0], [c, mu])
    ref = np.array([p_one_epoch_n2(k, c, mu) for k in range(L + 1)])
    ref = ref / ref.sum()
    print(f'  {kw}: no exposure, max rel err {np.max(np.abs(out / ref - 1)):.2e}, '
          f'sum {out.sum():.15f}')
    # exposure: one observation per (alpha, k)
    alphas = [0.5, 1.0, 3.0]
    rows, al = [], []
    for a in alphas:
        for k in range(L + 1):
            rows.append([k, 0]); al.append(a)
    m, _ = builder_out(jc, rows, [0.0], None, exposure_arr=np.asarray(al),
                       exposure_param_index=1)
    out = np.asarray(m(jnp.asarray([c, mu]))[0])
    ref = []
    for a in alphas:
        r = np.array([p_one_epoch_n2(k, c, mu, a) for k in range(L + 1)])
        ref.extend(r / r.sum())
    ref = np.asarray(ref)
    print(f'  {kw}: exposure {alphas}, max rel err {np.max(np.abs(out / ref - 1)):.2e}')

# ================================================================== B
section('B  n=2 two epochs vs closed form, granularity convergence')
jc = g2.joint_prob_graph(idx2, reward_limit=L, discrete=False)
obs = [[k, 0] for k in range(L + 1)]
c1, mu1, c2, mu2, t1 = 2.0, 0.7, 7.0, 1.5, 0.5
for alpha in [1.0, 3.0]:
    ref = np.array([p_two_epoch_n2(k, c1, mu1, c2, mu2, t1, alpha) for k in range(L + 1)])
    ref = ref / ref.sum()
    for gran in [0, 1000, 10000, 100000]:
        if alpha == 1.0:
            out, _ = epoch_model_out(jc, obs, [0.0, t1], [c1, mu1, c2, mu2],
                                     daisy_chain_granularity=gran)
        else:
            m, _ = builder_out(jc, obs, [0.0, t1], None,
                               exposure_arr=np.full(len(obs), alpha),
                               exposure_param_index=1, granularity=gran)
            out = np.asarray(m(jnp.asarray([c1, mu1, c2, mu2]))[0])
        print(f'  alpha={alpha} granularity={gran:>6}: max rel err '
              f'{np.max(np.abs(out / ref - 1)):.2e}', flush=True)

# ================================================================== C
section('C  n=3 closed form vs discrete table; cap separation')
g3, idx3 = coalescent_indexed(3)
c, mu = 1.1, 0.6
jd = g3.joint_prob_graph(idx3, reward_limit=6, discrete=True)
jd.update_weights([c, mu])
T = jd.joint_prob_table()
cols = list(T.columns[:-1])
errs = []
for r in T.itertuples(index=False):
    k1, k2 = int(r[0]), int(r[1])
    if any(int(x) != 0 for x in r[2:len(cols)]):
        continue
    errs.append(abs(r[-1] / p_one_epoch_n3(k1, k2, c, mu) - 1))
print(f'  closed form vs discrete table (reward_limit 6, raw): max rel err {max(errs):.2e} '
      f'over {len(errs)} rows; columns {cols}')
for kw, rl, tl in [(dict(reward_limit=2), 2, np.inf), (dict(tot_reward_limit=2), None, 2),
                   (dict(reward_limit=2, tot_reward_limit=3), 2, 3)]:
    jc3 = g3.joint_prob_graph(idx3, discrete=False, **kw)
    jd3 = g3.joint_prob_graph(idx3, discrete=True, **kw)
    jd3.update_weights([c, mu])
    rows = [tuple(int(x) for x in r) for r in jd3.joint_prob_table().iloc[:, :-1].to_numpy()]
    out, _ = epoch_model_out(jc3, [list(r) for r in rows], [0.0], [c, mu])
    ref = np.array([p_one_epoch_n3(r[0], r[1], c, mu) for r in rows])
    allowed = [(k1, k2) for k1 in range(12) for k2 in range(12)
               if cap_ok((k1, k2), rl, tl)]
    Z = sum(p_one_epoch_n3(k1, k2, c, mu) for k1, k2 in allowed)
    print(f'  {kw}: {len(rows)} outcomes (closed-form allowed set {len(allowed)}); '
          f'max rel err {np.max(np.abs(out / (ref / Z) - 1)):.2e}')

# ================================================================== D
section('D  gradients: exact_final_grad vs FD vs closed-form FD')
jc = g2.joint_prob_graph(idx2, reward_limit=L, discrete=False)
obs_rows = [[k, 0] for k in range(L + 1)] * 3
counts = np.array([5, 3, 2, 1] * 3, dtype=float)
al = np.repeat([0.5, 1.0, 3.0], L + 1)


def cf_loglik(th, alpha_arr, epochs):
    out = []
    for i, (row, a) in enumerate(zip(obs_rows, alpha_arr)):
        if epochs == 1:
            ps = np.array([p_one_epoch_n2(k, th[0], th[1], a) for k in range(L + 1)])
        else:
            ps = np.array([p_two_epoch_n2(k, th[0], th[1], th[2], th[3], 0.5, a)
                           for k in range(L + 1)])
        out.append(np.log(ps[row[0]] / ps.sum()))
    return float(np.sum(counts * np.asarray(out)))


for epochs, es, th in [(1, [0.0], [1.3, 0.7]), (2, [0.0, 0.5], [2.0, 0.7, 7.0, 1.5])]:
    for use_exp in [False, True]:
        kw = dict(exposure_arr=al, exposure_param_index=1) if use_exp else {}
        alpha_arr = al if use_exp else np.ones(len(obs_rows))
        grads = {}
        for efg in [False, True]:
            m, _ = builder_out(jc, obs_rows, es, None, exact_final_grad=efg,
                               granularity=100000, **kw)
            f = lambda t: jnp.sum(jnp.asarray(counts) * jnp.log(m(t)[0]))
            grads[efg] = np.asarray(jax.grad(f)(jnp.asarray(th, dtype=float)))
            P = jnp.asarray(np.asarray(th) * np.exp(np.random.default_rng(0).normal(0, .1, (4, len(th)))))
            vg = np.asarray(jax.vmap(jax.grad(f))(P))
            vg_loop = np.stack([np.asarray(jax.grad(f)(p)) for p in P])
            grads[(efg, 'vmap')] = np.max(np.abs(vg - vg_loop)) / np.max(np.abs(vg_loop))
        h = 1e-5
        cf = np.array([(cf_loglik(np.asarray(th) + h * e, alpha_arr, epochs)
                        - cf_loglik(np.asarray(th) - h * e, alpha_arr, epochs)) / (2 * h)
                       for e in np.eye(len(th))])
        sc = np.max(np.abs(cf))
        print(f'  epochs={epochs} exposure={use_exp}: '
              f'FD-vs-cf {np.max(np.abs(grads[False] - cf)) / sc:.1e}  '
              f'exact-vs-cf {np.max(np.abs(grads[True] - cf)) / sc:.1e}  '
              f'exact-vs-FD {np.max(np.abs(grads[True] - grads[False])) / sc:.1e}  '
              f'vmap-vs-loop FD {grads[(False, "vmap")]:.1e} exact {grads[(True, "vmap")]:.1e}',
              flush=True)

# ================================================================== E
section('E  deterministic expected-log-likelihood gate (mu fixed, fit c)')
c_true, mu_fix = 1.3, 0.7
alphas = np.array([0.3, 1.0, 4.0])
jc = g2.joint_prob_graph(idx2, reward_limit=L, discrete=False)
rows, al, w = [], [], []
for a in alphas:
    p = np.array([p_one_epoch_n2(k, c_true, mu_fix, a) for k in range(L + 1)])
    p = p / p.sum()
    for k in range(L + 1):
        rows.append([k, 0]); al.append(a); w.append(p[k])
w = np.asarray(w)
m, _ = builder_out(jc, rows, [0.0], None, exposure_arr=np.asarray(al),
                   exposure_param_index=1)
impl = lambda cc: -float(np.sum(w * np.log(np.asarray(m(jnp.asarray([cc, mu_fix]))[0]))))
r = minimize_scalar(impl, bounds=(0.1, 10), method='bounded', options=dict(xatol=1e-10))
print(f'  implementation: argmax c = {r.x:.10f} (truth {c_true}), rel err {abs(r.x / c_true - 1):.1e}')


def variant(cc, kind):
    tot = 0.0
    Zs = {a: sum(p_one_epoch_n2(k, cc, mu_fix, a) for k in range(L + 1)) for a in alphas}
    Zshared = np.mean(list(Zs.values()))
    for row, a, wi in zip(rows, al, w):
        p = p_one_epoch_n2(row[0], cc, mu_fix, a)
        d = {'raw': 1.0, 'sharedZ': Zshared, 'Z_alpha1': Zs[1.0]}[kind]
        tot += wi * np.log(p / d)
    return -tot


for kind in ['raw', 'sharedZ', 'Z_alpha1']:
    rv = minimize_scalar(lambda cc: variant(cc, kind), bounds=(0.1, 10), method='bounded',
                         options=dict(xatol=1e-10))
    print(f'  variant {kind:9s}: argmax c = {rv.x:.6f}, rel miss {abs(rv.x / c_true - 1):.1e}')

# ================================================================== F
section('F  test.py model: tied 2-epoch normalised MLE (truth 5)')
from phasic import Graph  # noqa: E402
g4, idx4 = coalescent_indexed(4)
jd4 = g4.joint_prob_graph(idx4, reward_limit=3, discrete=True)
jd4.update_weights([5.0, 3.0])
T4 = jd4.joint_prob_table()
p4 = T4['prob'].to_numpy(); p4 = p4 / p4.sum()
rows4 = [list(r) for r in T4.iloc[:, :-1].to_numpy()]
jc4 = g4.joint_prob_graph(idx4, reward_limit=3, discrete=False)
em = jc4.epoch_model(rows4, [0.0, 1.0])
f4 = lambda cc: -float(np.sum(p4 * np.log(np.asarray(em.model(jnp.asarray([cc, 3.0, cc, 3.0]))[0]))))
r4 = minimize_scalar(f4, bounds=(1, 20), method='bounded', options=dict(xatol=1e-8))
print(f'  expected-loglik argmax c = {r4.x:.6f} (truth 5), rel err {abs(r4.x / 5 - 1):.1e}')
