import random
import numpy as np
import pandas as pd

def seed_all(n=42):
    """
    Seed BOTH generators.

    `Graph.sample()` does NOT use numpy: the pybind layer seeds the C
    sampler from Python's stdlib `random` module (phasic_pybind.cpp
    set_c_seed), so `np.random.seed` alone leaves sampling
    non-deterministic and makes these tests flaky. Verified: two
    `sample()` calls under the same `random.seed` are identical, under
    the same `np.random.seed` alone they are not.
    """
    random.seed(n)
    np.random.seed(n)


def sample_joint_observations(jpg, theta, n, rng):
    """svgd-joint-prob's sampler: draw joint outcomes from the model's own
    joint-probability table at the true parameters."""
    jpg.update_weights(list(theta))
    table = jpg.joint_prob_table()
    p = table['prob'].to_numpy()
    p = p / p.sum()
    picks = rng.choice(table.index.values, n, p=p)
    return table.loc[picks, table.columns[:-1]].to_numpy().tolist()


def log_likelihood(graph, theta, data, rewards=None):
    """
    Log-likelihood of `data` under `graph` at `theta`.

    With `rewards`, the graph is re-weighted FIRST and reward-transformed
    after: `reward_transform` returns a plain (non-parameterized) graph, so
    it cannot itself take an update_weights call.
    """
    graph.update_weights(list(theta))
    g = graph.reward_transform(rewards) if rewards is not None else graph
    p = np.array([g.pdf(float(t)) for t in data])
    return float(np.sum(np.log(np.clip(p, 1e-300, None))))


def assert_informative(obs, min_distinct=0.2):
    """Guard against a vacuous recovery test: if the sampled observations
    are (nearly) all the same outcome, the data say nothing about theta and
    any 'recovery' is just the prior."""
    distinct = np.unique(obs, axis=0).size
    assert distinct / len(obs) >= min_distinct, (
        f"observations are degenerate: distinct outcome(s) make up only "
        f"{distinct / len(obs)*100:.2f}% of the data set"
    )


def assert_map_recovers(svgd, true_theta, tolerance=0.02):
    """
    Every free parameter's true value is only "tolerance" percent off its MAP estimate.
    """
    map_est = np.asarray(svgd.map_estimate_from_particles()[0]).ravel()
    for i in range(len(true_theta)):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue    
        deviation = abs(true_theta[i] - map_est[i]) / max(abs(true_theta[i]), 1e-30) 
        assert deviation < tolerance, (
            f"parameter {i} true value {map_est[i]:g} is more than {tolerance*100:.3}% "
            f"from the MAP estimate ({deviation*100:.3}% )"
        )

def assert_mean_recovers(svgd, true_theta, tolerance=0.02):
    """
    Every free parameter's true value is only "tolerance" percent off its MAP estimate.
    """
    res = svgd.get_results()
    mean = np.asarray(res['theta_mean']).ravel()
    for i in range(len(true_theta)):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue    
        deviation = abs(mean[i] - true_theta[i]) / max(abs(true_theta[i]), 1e-30)
        assert deviation < tolerance, (
            f"parameter {i} true value {mean[i]:g} is more than {tolerance*100:.3}% "
            f"from the estimated mean ({deviation*100:.3}% )"
        )


def assert_truth_is_local_max(graph, true_theta, data, factors=(0.5, 0.9, 0.95, 1.05, 1.1, 2.0),
                              rewards=None):
    """
    The parameters the data were sampled from must beat every
    perturbation of them under the library's own likelihood.
    """
    base = log_likelihood(graph, true_theta, data, rewards)
    assert np.isfinite(base), "log-likelihood at the true parameters is not finite"
    for i in range(len(true_theta)):
        for f in factors:
            th = list(true_theta)
            th[i] = th[i] * f
            alt = log_likelihood(graph, th, data, rewards)
            assert alt < base, (
                f"parameter {i} scaled by {f}: log-likelihood {alt:.3f} is "
                f"NOT below the value at the true parameters {base:.3f} — "
                f"the data-generating parameters are not a local maximum, "
                f"so inference cannot be expected to find them")


def assert_in_hpd(svgd, true_theta):
    """
    Every free parameter's true value inside its 95% HPD, with
    the interval not uselessly wide.
    """
    res = svgd.get_results()
    lo = np.asarray(res['hpd_lower']).ravel()
    hi = np.asarray(res['hpd_upper']).ravel()
    mean = np.asarray(res['theta_mean']).ravel()
    for i in range(len(true_theta)):        
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        assert lo[i] <= true_theta[i] <= hi[i], (
            f"parameter {i} true value {true_theta[i]:g} OUTSIDE the 95% HPD "
            f"[{lo[i]:g}, {hi[i]:g}] (posterior mean {mean[i]:g}) — "
            f"inference did not recover the value it was sampled from")


def assert_hpd_informative(svgd, true_theta, max_rel_width=0.2):
    """
    HPD interval is not uselessly wide relative to estimate.
    """
    res = svgd.get_results()
    lo = np.asarray(res['hpd_lower']).ravel()
    hi = np.asarray(res['hpd_upper']).ravel()
    mean = np.asarray(res['theta_mean']).ravel()
    for i in range(len(true_theta)):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        width = (hi[i] - lo[i]) / max(abs(mean[i]), 1e-30)
        assert width < max_rel_width, (
            f"width of parameter {i} HPD is {width:.4g}, which is more than {max_rel_width*100:.1f} percent"
            f" of the posterior mean ({mean[i]:.4g}), so wide that containing the truth is not informative"
        )

def assert_in_hdr(svgd, true_theta, hdr_prob=0.95):
    """
    True theta is in HDR region HPD interval is not uselessly wide relative to estimate.
    """
    particles, min_like = svgd.estimate_hdr(hdr_prob)
    lows = np.min(particles, axis=0)
    highs = np.max(particles, axis=0)
    for true, low, high in zip(true_theta, lows, highs):
        assert true > low
        assert true < high



def assert_mean_std_converged(svgd, window=10, tail=10, rtol=0.05):
    h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
    n_part, n_iter, n_param = h.shape
    for i in range(n_param):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        d = h[:, :, i]
        for name, stat in dict(mean= np.mean(d, axis=1), std=np.std(d, axis=1)).items():
            rolling_std = pd.Series(stat).rolling(window=window, center=False, min_periods=window).std().to_numpy()
            last = rolling_std[-tail:]
            rel_diffs = np.diff(last, n=1) / np.mean(last)
            max_rel_diff = rel_diffs.max()
            assert max_rel_diff < rtol, (
                f"Among the last {tail} rolling window {name} (size={window}), a window {name} differs "
                f"too much from the mean of {name} over all {tail} windows: {max_rel_diff} for  < {rtol}. "
                f"Suggests particle {name} has not converged."
            )

def assert_map_in_hdr(svgd):
    map_est = svgd.map_estimate_from_particles()[0]
    for p in [0.95, 2/3, 1/3]: # to trigger the largest one first 
        particles = svgd.estimate_hdr(p)[0]
        for i in range(particles.shape[1]):
            if svgd.fixed_mask and svgd.fixed_mask[i]:
                continue            
            print(p, i, map_est[i], min(particles[:, i]), max(particles[:, i]))
            assert map_est[i] > particles[:, i].min(), f"MAP for parameter with index {i} is below {p*100:.2f}% HDR"
            assert map_est[i] < particles[:, i].max(), f"MAP for parameter with index {i} is above {p*100:.2f}% HDR"


def assert_smooth_trajectories(svgd, tol=50):
    """
    Individual particle trajectories may "spike" if the step is too large 
    compared to the gradient. Such spikes inflate the HDP and generally flag 
    an unreliable/failed fit. Diagnosed as large differences between running 
    differences in each estimate for each particle.
    """
    h = np.asarray(svgd.get_results()['history'])
    n_iter, n_part, n_param = h.shape
    for i in range(n_param):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue        
        h = np.asarray(svgd.get_results()['history'])[:, :, i]
        diffs = np.diff(h, n=1, axis=0)[:-1, :]
        rel_diff_of_diffs = np.absolute(np.diff(diffs, n=1, axis=0) / diffs[:-1, :])
        max_across_part = np.amax(rel_diff_of_diffs, axis=1)
        max_rel_diff_of_diffs = np.max(max_across_part)
        idx = np.argwhere(max_across_part == max_rel_diff_of_diffs).item()
        assert max_rel_diff_of_diffs < tol, (
            f"parameter {i} particle trajectories has spikes produces by"
            f"a step too large for the gradient. Reduce step size, or"
            f"handle how it behaves when the fit plateaus."
            f"max rel_diff_of_diffs is {max_rel_diff_of_diffs:.2f} at iteration {idx+1}"
        )


def assert_monotone_convergence(svgd, true_theta, rtol=1e-2):
    h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
    n_iter, n_part, n_param = h.shape
    for i in range(n_param):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        means = np.mean(h[:, :, i], axis=1)
        rel_diffs = np.diff(np.absolute(means - true_theta[i]), n=1)
        msg = (
            f"The mean estimate of param with index {i} does not monotinically move towards the its true value. "
            f"If so, the running diff of the absolute difference between particle mean and true theta should be "
            f"negative. "
        )
        max_rel_diff = np.max(rel_diffs)
        prop_iter_with_pos_rel_diff = np.sum(rel_diffs > 0) / n_iter
        assert max_rel_diff <= rtol, msg + f"Max rel diff is: {max_rel_diff:.4g}"
        assert prop_iter_with_pos_rel_diff <= rtol, msg + f"Proportion of positive diffs is {prop_iter_with_pos_rel_diff:.4g}"

        
def test_svgd_model(graph, true_theta, data):
    assert_truth_is_local_max(graph, true_theta, data)


def test_svgd_data(data):
    assert_informative(data)


def test_svgd_fit(svgd, true_theta):
    assert_mean_std_converged(svgd)      
    assert_map_in_hdr(svgd)
    assert_smooth_trajectories(svgd)
    assert_monotone_convergence(svgd, true_theta)
    assert_map_recovers(svgd, true_theta)
    assert_hpd_informative(svgd, true_theta)
    assert_in_hdr(svgd, true_theta)
    assert_in_hdr(svgd, true_theta)


def all(graph, true_theta, data, svgd):
    test_svgd_model(graph, true_theta, data)
    test_svgd_data(data)
    test_svgd_fit(svgd, true_theta)
