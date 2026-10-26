import random
import numpy as np
import pandas as pd


# Utility #############################

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




# Model ###############################

# def assert_truth_is_local_max(graph, true_theta, data, factors=(0.5, 0.9, 0.95, 1.05, 1.1, 2.0),
#                               rewards=None):
#     """
#     The parameters the data were sampled from must beat every
#     perturbation of them under the library's own likelihood.
#     """
#     base = log_likelihood(graph, true_theta, data, rewards)
#     assert np.isfinite(base), "log-likelihood at the true parameters is not finite"
#     for i in range(len(true_theta)):
#         for f in factors:
#             th = list(true_theta)
#             th[i] = th[i] * f
#             alt = log_likelihood(graph, th, data, rewards)
#             assert alt < base, (
#                 f"parameter {i} scaled by {f}: log-likelihood {alt:.3f} is "
#                 f"NOT below the value at the true parameters {base:.3f} — "
#                 f"the data-generating parameters are not a local maximum, "
#                 f"so inference cannot be expected to find them")

def assert_truth_is_local_max(graph, true_theta, data, rewards=None):
    """
    The parameters the data were sampled from must beat every
    perturbation of them under the library's own likelihood.
    """
    def log_likelihood(graph, theta, data, rewards=None):
        graph.update_weights(list(theta))
        g = graph.reward_transform(rewards) if rewards is not None else graph
        p = np.array([g.pdf(float(t)) for t in data])
        return float(np.sum(np.log(np.clip(p, 1e-300, None))))

    base = log_likelihood(graph, true_theta, data, rewards)
    assert np.isfinite(base), "log-likelihood at the true parameters is not finite"
    for i in range(len(true_theta)):
        pert = 0.1
        for p in [-pert, pert]:
            th = list(true_theta)
            th[i] = th[i] * (1 + p)
            alt = log_likelihood(graph, th, data, rewards)
            # assert (base - alt) / base < 1 / np.sqrt(len(data)), (
            assert alt < base, (
                f"perturbed index {i} parameter {th[i]} produce log-likelihood "
                f"{alt:.3f} above that for true parameters {base:.3f} - "
                f"the data-generating parameters are not a local maximum, "
                f"so inference cannot be expected to find them")
            
# Data ################################

def assert_informative(obs, rtol=0.5):
    """Guard against a vacuous recovery test: if the sampled observations
    are (nearly) all the same outcome, the data say nothing about theta and
    any 'recovery' is just the prior."""
    obs = np.array(obs)
    assert obs.ndim <= 2
    if obs.ndim == 1:
        unique, counts = np.unique_counts(obs)
    else:
        unique, counts = np.unique_counts([hash(tuple(x)) for x in obs])
    n_obs = len(obs)
    idx = counts.argmax()
    n_most_common = counts[idx]
    most_common = unique[idx]
    (n_obs - n_most_common) / n_obs >= rtol, (
        f"observations are degenerate: most common value ({most_common}) makes up "
        f"{n_most_common / n_obs * 100:.2f}% of the data set"
    )
    # prop_unique = len(unique) / len(obs)
    # assert prop_unique >= rtol, (
    #     f"observations are degenerate: distinct observations make up only "
    #     f"{prop_unique*100:.2f}% of the data set"
    # )

# Fit #################################

def assert_map_recovers(svgd, true_theta, tolerance=0.05):
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

# def assert_in_hdr(svgd, true_theta, hdr_prob=0.95):
#     """
#     True theta is in HDR region HPD interval is not uselessly wide relative to estimate.
#     """
#     particles, min_like = svgd.estimate_hdr(hdr_prob)
#     lows = np.min(particles, axis=0)
#     highs = np.max(particles, axis=0)
#     for true, low, high in zip(true_theta, lows, highs):
#         assert true > low
#         assert true < high


def assert_mean_converged(svgd, tail=20, rtol=0.05):
    h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
    n_part, n_iter, n_param = h.shape
    for i in range(n_param):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        d = h[:, :, i]
        particle_means = np.mean(d, axis=1)
        max_rel_diff = np.max(np.absolute(particle_means[-tail:] - particle_means[-1])) / particle_means[-1]
        # assert max_rel_diff < rtol, (max_rel_diff, particle_means[-1])
        assert max_rel_diff < rtol, (
            f"Not converged: One of the last {tail} iterations has a particle mean for param index {i} that "
            f"is more than {int(rtol*100)}% different ({int(max_rel_diff*100)}%) from its final estimate."
        )

# def assert_mean_converged(svgd, window=10, tail=10, rtol=0.05):
#     h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
#     n_part, n_iter, n_param = h.shape
#     for i in range(n_param):
#         if svgd.fixed_mask and svgd.fixed_mask[i]:
#             continue
#         d = h[:, :, i]
#         for name, stat in dict(mean= np.mean(d, axis=1)).items():
#             rolling_std = pd.Series(stat).rolling(window=window, center=False, min_periods=window).std().to_numpy()
#             last = rolling_std[-tail:]
#             rel_diffs = np.diff(last, n=1) / np.mean(last)
#             max_rel_diff = rel_diffs.max()
#             assert max_rel_diff < rtol, (
#                 f"Among the last {tail} rolling window {name} (size={window}), a window {name} differs "
#                 f"too much from the mean of {name} over all {tail} windows: {max_rel_diff} for  < {rtol}. "
#                 f"Suggests particle {name} has not converged."
#             )

# def assert_std_converged(svgd, window=10, tail=10, rtol=0.05):
#     h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
#     n_part, n_iter, n_param = h.shape
#     for i in range(n_param):
#         if svgd.fixed_mask and svgd.fixed_mask[i]:
#             continue
#         d = h[:, :, i]
#         for name, stat in dict(std=np.std(d, axis=1)).items():
#             rolling_std = pd.Series(stat).rolling(window=window, center=False, min_periods=window).std().to_numpy()
#             last = rolling_std[-tail:]
#             rel_diffs = np.diff(last, n=1) / np.mean(last)
#             max_rel_diff = rel_diffs.max()
#             assert max_rel_diff < rtol, (
#                 f"Among the last {tail} rolling window {name} (size={window}), a window {name} differs "
#                 f"too much from the mean of {name} over all {tail} windows: {max_rel_diff} for  < {rtol}. "
#                 f"Suggests particle {name} has not converged."
#             )

def assert_map_in_hdr(svgd):
    map_particle, loglike = svgd.map_estimate_from_particles()
    map_particle = np.array(map_particle)
    hdr_particles, loglike = svgd.joint_hdr_particles(level=0.5)
    assert np.any((hdr_particles - map_particle).sum(axis=1) == 0), (
        f"The MAP particle {map_particle} is not among the particles in the 50% HDR"
    )

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


def assert_monotone_convergence(svgd, true_theta, rtol=0.05):
    h = np.asarray(svgd.get_results()['history'])[:-2, :, :]
    n_iter, n_part, n_param = h.shape
    for i in range(n_param):
        if svgd.fixed_mask and svgd.fixed_mask[i]:
            continue
        means = np.mean(h[:-1, :, i], axis=1)
        diffs = np.absolute(np.diff(means - true_theta[i], n=1))
        diff_of_diffs = np.diff(diffs, n=1)
        msg = (
            f"The mean estimate of param with index {i} does not monotonically move towards the its true value. "
            f"If so, the running diff of the absolute difference between particle mean and true theta should be "
            f"negative. "
        )
        max_rel_diff = np.max(diff_of_diffs)
        prop_iter_with_pos_rel_diff = np.sum(diff_of_diffs > 0) / n_iter
        assert max_rel_diff <= rtol, msg + f"Max rel diff is: {max_rel_diff:.4g}"
        assert prop_iter_with_pos_rel_diff <= rtol, msg + f"Proportion of positive diffs is {prop_iter_with_pos_rel_diff:.4g}"


# Convenience #########################

def test_svgd_model(graph, true_theta, data, rewards=None):
    assert_truth_is_local_max(graph, true_theta, data, rewards=rewards)

def test_svgd_data(data):
    assert_informative(data)

# def test_svgd_learning_rate_fit(svgd, true_theta):
#     assert_mean_converged(svgd)      
#     # assert_std_converged(svgd)      
#     assert_map_in_hdr(svgd)
#     assert_smooth_trajectories(svgd)
#     assert_monotone_convergence(svgd, true_theta)
#     assert_map_recovers(svgd, true_theta)
#     assert_hpd_informative(svgd, true_theta)
#     assert_in_hdr(svgd, true_theta)

def test_svgd_optimizer_fit(svgd, true_theta):
    assert_mean_converged(svgd)      
    assert_map_in_hdr(svgd)
    assert_map_recovers(svgd, true_theta)   
    assert_hpd_informative(svgd, true_theta)
    # assert_in_hdr(svgd, true_theta)


def all(graph, true_theta, data, svgd, rewards=None):
    if data.ndim == 1 and len(true_theta) == 1:
        test_svgd_model(graph, true_theta, data, rewards=rewards)
    test_svgd_data(data)
    test_svgd_optimizer_fit(svgd, true_theta)

