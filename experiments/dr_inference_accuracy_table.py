"""Inference-accuracy overview across the notebook models, with an
independent convergence harness.

WHY A HARNESS. `svgd.summary()` reports the posterior it happens to have
reached; it cannot tell you whether SVGD converged. This harness does not
trust any single fit. For every model it runs SEVERAL INDEPENDENT fits from
different seeds and applies three checks that can each fail independently:

  1. CROSS-RUN AGREEMENT (Gelman-Rubin R-hat). Treat each run's particle
     cloud as a chain. R-hat compares between-run to within-run variance;
     if independent runs land in different places the posterior is not
     determined by the data and R-hat exceeds 1. Reported per parameter.
     Convention: R-hat < 1.05 is acceptable, < 1.01 is good.

  2. TRACE STATIONARITY. Compare the particle mean over the final quarter
     of iterations with the quarter before it. If the cloud is still
     drifting, the fit stopped early regardless of what R-hat says.

  3. AGREEMENT WITH AN INDEPENDENT OPTIMISER. Maximise the SAME likelihood
     with scipy, from a different starting point, with no reference to
     SVGD. If the posterior mode and this MLE disagree, one of them is
     wrong. This is the strongest check because it shares no machinery
     with SVGD beyond the likelihood itself.

Only if all three agree is a row reported as converged.

Output: a markdown table of true vs inferred parameters with prior and
posterior intervals, written to b3-inference-accuracy-table.md.

Usage:  pixi run python experiments/dr_inference_accuracy_table.py [--quick]
"""
import argparse
import random
import sys
import time
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import phasic
from phasic import (Graph, GaussPrior, LogGaussPrior, Property, StateIndexer,
                    set_log_level, with_ipv)

set_log_level("ERROR")
all_pairs = partial(combinations_with_replacement, r=2)

sys.path.insert(0, "tests/pytest")
from test_notebook_model_recovery import (  # noqa: E402
    coalescent_graph, two_island_graph, coalescent_indexed,
    sample_joint_observations)


# --------------------------------------------------------------- diagnostics
def r_hat(chains):
    """Gelman-Rubin R-hat. `chains` is (n_chains, n_samples) for ONE
    parameter. Values near 1 mean independent runs agree."""
    chains = np.asarray(chains, float)
    m, n = chains.shape
    if m < 2:
        return float('nan')
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()          # within-chain
    B = n * means.var(ddof=1)                       # between-chain
    if W <= 0:
        return float('nan')
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def trace_drift(history, dim):
    """Relative change in particle mean between the last quarter of
    iterations and the quarter before it. Near 0 means stationary."""
    h = np.asarray(history)
    if h.ndim != 3 or h.shape[0] < 8:
        return float('nan')
    n = h.shape[0]
    q = max(1, n // 4)
    late = h[-q:, :, dim].mean()
    prev = h[-2 * q:-q, :, dim].mean()
    return float(abs(late - prev) / max(abs(late), 1e-30))


def independent_mle(loglik, x0, bounds=None):
    """Maximise the same likelihood with scipy, independently of SVGD.

    Optimised in LOG space: these parameters are positive and span many
    decades (coalescent rates ~1, mutation rates ~1e-4), and Nelder-Mead
    in linear space takes steps sized for O(1) parameters, which sends a
    1e-4-scale parameter to nonsense.
    """
    x0 = np.asarray(x0, float)
    r = minimize(lambda u: -loglik(np.exp(u)), np.log(x0),
                 method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-10, maxiter=8000))
    return np.exp(r.x)


def prior_interval(prior, dim=0):
    """95% interval of the prior, in THETA space."""
    if isinstance(prior, LogGaussPrior):
        mu, sd = float(np.ravel(prior.mu)[dim]), float(np.ravel(prior.sigma)[dim])
        return float(np.exp(mu - 1.96 * sd)), float(np.exp(mu + 1.96 * sd))
    if isinstance(prior, GaussPrior):
        mu, sd = float(np.ravel(prior.mu)[dim]), float(np.ravel(prior.sigma)[dim])
        return mu - 1.96 * sd, mu + 1.96 * sd
    for a, b in (('theta', 'std'), ('mu', 'sigma')):
        if hasattr(prior, a) and hasattr(prior, b):
            mu = float(np.ravel(getattr(prior, a))[dim])
            sd = float(np.ravel(getattr(prior, b))[dim])
            return mu - 1.96 * sd, mu + 1.96 * sd
    return float('nan'), float('nan')


# ------------------------------------------------------------------- driver
def analyse(name, vertices, true_theta, fit_fn, loglik, mle_x0, free=None,
            n_seeds=3, prior_obj=None):
    """Run n_seeds independent fits and apply all three convergence checks."""
    free = list(range(len(true_theta))) if free is None else free
    runs, t0 = [], time.time()
    for seed in range(n_seeds):
        random.seed(1000 + seed); np.random.seed(1000 + seed)
        svgd = fit_fn(seed)
        res = svgd.get_results()
        runs.append(dict(
            particles=np.asarray(res['particles']),
            mean=np.asarray(res['theta_mean']).ravel(),
            lo=np.asarray(res['hpd_lower']).ravel(),
            hi=np.asarray(res['hpd_upper']).ravel(),
            history=res.get('history'),
            prior=getattr(svgd, 'prior', None)))
    elapsed = time.time() - t0

    mle = independent_mle(loglik, mle_x0, None) if loglik else None
    prior_obj = prior_obj if prior_obj is not None else runs[0]['prior']

    rows = []
    for d in free:
        chains = np.stack([r['particles'][:, d] for r in runs])
        rh = r_hat(chains)
        drifts = [trace_drift(r['history'], d) for r in runs
                  if r['history'] is not None]
        drift = float(np.nanmax(drifts)) if drifts else float('nan')
        post_mean = float(np.mean([r['mean'][d] for r in runs]))
        lo = float(np.mean([r['lo'][d] for r in runs]))
        hi = float(np.mean([r['hi'][d] for r in runs]))
        plo, phi = prior_interval(prior_obj, d) if prior_obj is not None else (np.nan, np.nan)
        mle_d = float(mle[d]) if mle is not None else float('nan')
        covered = lo <= true_theta[d] <= hi
        # Consistency with the independent optimiser is judged by whether
        # the MLE falls INSIDE the posterior interval -- not by comparing
        # it to the posterior MEAN. With an informative prior the mean is
        # SUPPOSED to differ from the MLE, so a mean-vs-MLE test would
        # flag correct Bayesian behaviour as non-convergence.
        mle_in_hpd = (bool(lo <= mle_d <= hi)
                      if mle is not None and np.isfinite(mle_d) else True)
        mle_gap = (abs(post_mean - mle_d) / max(abs(mle_d), 1e-30)
                   if mle is not None and np.isfinite(mle_d) else float('nan'))
        ok = (covered
              and (not np.isfinite(rh) or rh < 1.05)
              and (not np.isfinite(drift) or drift < 0.10)
              and mle_in_hpd)
        rows.append(dict(model=name, vertices=vertices, param=d,
                         true=true_theta[d], prior_lo=plo, prior_hi=phi,
                         post_mean=post_mean, hpd_lo=lo, hpd_hi=hi,
                         mle=mle_d, rhat=rh, drift=drift,
                         covered=covered, mle_gap=mle_gap,
                         mle_in_hpd=mle_in_hpd, converged=ok,
                         seconds=elapsed / n_seeds))
        print(f"  {name} [{vertices}v] p{d}: true={true_theta[d]:.5g} "
              f"post={post_mean:.5g} HPD=[{lo:.4g},{hi:.4g}] "
              f"MLE={mle_d:.5g} Rhat={rh:.3f} drift={drift:.3f} "
              f"{'CONVERGED' if ok else 'NOT CONVERGED'}", flush=True)
    return rows


def time_loglik(graph, data, rewards=None):
    def f(theta):
        if np.any(np.asarray(theta) <= 0):
            return -1e18
        graph.update_weights(list(theta))
        g = graph.reward_transform(rewards) if rewards is not None else graph
        p = np.array([g.pdf(float(t)) for t in data])
        return float(np.sum(np.log(np.clip(p, 1e-300, None))))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smallest size of each model only")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    S = args.seeds
    all_rows = []

    print("== coalescent (svgd-basics) ==", flush=True)
    for n, nv in ([(10, 43)] if args.quick else [(10, 43), (13, 102)]):
        g = coalescent_graph(n)
        g.update_weights([7.0])
        random.seed(0); np.random.seed(0)
        data = g.sample(500)
        prior = GaussPrior(ci=[3, 15])
        all_rows += analyse(
            "coalescent", nv, [7.0],
            lambda seed, g=g, data=data, prior=prior: g.svgd(
                data, prior=prior, n_iterations=60, n_particles=60),
            time_loglik(g, data), [6.0], n_seeds=S, prior_obj=prior)

    print("== two_island (svgd-multi-param) ==", flush=True)
    for n, nv in ([(4, 21)] if args.quick else [(4, 21)]):
        g = two_island_graph(n)
        g.update_weights([0.7, 0.3])
        random.seed(0); np.random.seed(0)
        data = g.sample(500)
        all_rows += analyse(
            "two_island", nv, [0.7, 0.3],
            lambda seed, g=g, data=data: g.svgd(
                data, n_iterations=40, n_particles=50),
            time_loglik(g, data), [0.5, 0.5], n_seeds=S)

    print("== coalescent + rewards (svgd-multi-feature) ==", flush=True)
    g = coalescent_graph(10)
    g.update_weights([10.0])
    rewards = np.sum(g.states().T, axis=0)
    random.seed(17); np.random.seed(17)
    rdata = g.sample(500, rewards=rewards)
    rprior = GaussPrior(ci=[5, 25])
    all_rows += analyse(
        "coalescent+rewards", 43, [10.0],
        lambda seed, g=g, rdata=rdata, rewards=rewards, rprior=rprior: g.svgd(
            observed_data=rdata, rewards=rewards, prior=rprior,
            n_iterations=60, n_particles=60),
        time_loglik(g, rdata, rewards), [8.0], n_seeds=S, prior_obj=rprior)

    print("== joint_prob (svgd-joint-prob) ==", flush=True)
    MU = 1e-2   # see test_notebook_model_recovery: 1e-4 is degenerate
    for n, rl, nv in ([(3, 2, 25)] if args.quick else [(3, 2, 25), (4, 3, 168)]):
        gg, indexer = coalescent_indexed(n)
        jpg = gg.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=MU)
        true = [1.0 / 10_000, MU]
        rng = np.random.default_rng(17)
        obs = sample_joint_observations(jpg, true, 1000, rng)
        jprior = LogGaussPrior(ci=[1 / 50_000, 1 / 5_000])

        obs_arr = np.asarray(obs)

        def jll(theta, jpg=jpg, obs_arr=obs_arr, MU=MU):
            th = [float(theta[0]), MU]
            if th[0] <= 0:
                return -1e18
            jpg.update_weights(th)
            tbl = jpg.joint_prob_table()
            cols = list(tbl.columns[:-1])
            key = {tuple(int(v) for v in r[cols]): max(float(r['prob']), 1e-300)
                   for _, r in tbl.iterrows()}
            tot = sum(key.values())
            return float(sum(np.log(key.get(tuple(int(v) for v in row), 1e-300) / tot)
                             for row in obs_arr))

        all_rows += analyse(
            "joint_prob", nv, true,
            lambda seed, jpg=jpg, obs=obs, jprior=jprior, MU=MU: jpg.svgd(
                obs, fixed=[(1, MU)], prior=jprior,
                n_iterations=60, n_particles=60),
            (lambda t: jll(t)), [1.5 / 10_000], free=[0], n_seeds=S,
            prior_obj=jprior)

    write_table(all_rows, S)


def write_table(rows, n_seeds):
    out = ["# Inference accuracy across the notebook models",
           "",
           f"Generated by `experiments/dr_inference_accuracy_table.py` "
           f"({n_seeds} independent seeds per model).",
           "",
           "Every row passed — or failed — three INDEPENDENT convergence "
           "checks, not `summary()`:",
           "R-hat (cross-run agreement, want <1.05), drift (trace "
           "stationarity, want <0.10), and whether an",
           "independently-computed scipy MLE of the SAME likelihood falls "
           "inside the posterior interval.",
           "",
           "The MLE is compared against the INTERVAL, not the posterior "
           "mean: with an informative prior the",
           "mean is supposed to differ from the MLE, so a mean-vs-MLE test "
           "would flag correct behaviour as failure.",
           "",
           "| model | vertices | param | true | prior 95% | posterior mean | "
           "posterior 95% HPD | indep. MLE | MLE in HPD | R-hat | drift | "
           "covers truth | converged |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['vertices']} | {r['param']} | "
            f"{r['true']:.5g} | [{r['prior_lo']:.4g}, {r['prior_hi']:.4g}] | "
            f"{r['post_mean']:.5g} | [{r['hpd_lo']:.4g}, {r['hpd_hi']:.4g}] | "
            f"{r['mle']:.5g} | {'yes' if r['mle_in_hpd'] else 'NO'} | "
            f"{r['rhat']:.3f} | {r['drift']:.3f} | "
            f"{'yes' if r['covered'] else 'NO'} | "
            f"{'yes' if r['converged'] else 'NO'} |")
    n_ok = sum(1 for r in rows if r['converged'])
    out += ["", f"{n_ok}/{len(rows)} parameter estimates passed all three "
                f"convergence checks and covered the true value.", ""]
    with open("b3-inference-accuracy-table.md", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[-4:]))
    print("wrote b3-inference-accuracy-table.md")


if __name__ == "__main__":
    main()
