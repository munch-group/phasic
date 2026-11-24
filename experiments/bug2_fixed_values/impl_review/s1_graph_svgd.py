from common import *
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_indexed
L=3; EPOCHS=[0.0,0.4]
g, idx = coalescent_indexed(2)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EPOCHS[1]) for k in range(L+1)])
rows = [[k, 0] for k in range(L+1) for _ in range(int(round(p[k]*200)))]
KW = dict(n_particles=6, n_iterations=3, learning_rate=0.01, progress=False, seed=1)
def a1():
    s = jc.svgd(rows, fixed=[(1, 0.3)], prior=[gauss(), None], **KW); report(s,[1])
scen("A1 non-epoch joint, fixed (1,0.3)", a1)
def a2():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, [0.3, 0.7])], prior=[gauss(), None], **KW); report(s,[1,3])
scen("A2 epoch, fixed (1,[0.3,0.7])", a2)
def a2b():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.3)], prior=[gauss(), None], **KW); report(s,[1,3])
scen("A2b epoch, fixed (1,0.3)", a2b)
def a3():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.3)], tied=[(0,[0,1])], prior=[gauss(), None], n_particles=40, n_iterations=3, learning_rate=0.01, progress=False, seed=1); report(s,[0,1,2,3])
    print("  fixed_mask", np.asarray(s.fixed_mask), "fixed_theta", np.asarray(s.fixed_theta), "user_fixed", s._user_fixed_indices())
    print("  MAP", s.map_estimate_from_particles()[0])
scen("A3 tied (0,[0,1]) + fixed (1,0.3), 40 particles", a3)
def a3b():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.3)], tied=[(1,[0,1])], prior=[gauss(), None], n_particles=40, n_iterations=3, learning_rate=0.01, progress=False, seed=1); report(s,[0,1,2,3])
    print("  fixed_mask", np.asarray(s.fixed_mask), "fixed_theta", np.asarray(s.fixed_theta), "user_fixed", s._user_fixed_indices())
    s.summary()
scen("A3b tie ON the fixed param (1,[0,1]) + fixed (1,0.3), 40 particles", a3b)
alpha = np.linspace(1.0, 3.0, len(rows))
def a4():
    s = jc.svgd(rows, fixed=[(1, 1e-10)], exposure=alpha, exposure_param_index=1, prior=[gauss(), None], **KW); report(s,[1])
scen("A4 non-epoch exposure, fixed exposure param 1e-10", a4)
def a4b():
    s = jc.svgd(rows, fixed=[(1, 0.3)], exposure=alpha, exposure_param_index=1, prior=[gauss(), None], **KW); report(s,[1])
scen("A4b non-epoch exposure, fixed exposure param 0.3", a4b)
def a4c():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.3)], exposure=alpha, exposure_param_index=1, prior=[gauss(), None], **KW); report(s,[1,3])
scen("A4c epoch exposure, fixed 0.3", a4c)
def a4d():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 1e-10)], exposure=alpha, exposure_param_index=1, prior=[gauss(), None], **KW); report(s,[1,3])
scen("A4d epoch exposure, fixed 1e-10", a4d)
def a5():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.0)], prior=[gauss(), None], **KW); report(s,[1,3])
scen("A5 epoch, fixed 0.0 (mutation rate; degenerate)", a5)
def a5b():
    s = jc.svgd(rows, fixed=[(1, 0.0)], prior=[gauss(), None], **KW); report(s,[1])
scen("A5b non-epoch joint, fixed 0.0", a5b)
def a5c():
    s = chain2().svgd(REAL, fixed=[(1, 0.0)], prior=[gauss(), None], theta_dim=2, **KW); report(s,[1])
    print("  ll", s.log_likelihood(), "ll refine", s.log_likelihood(refine=True))
scen("A5c chain2 Graph.svgd fixed 0.0", a5c)
def a5d():
    s = chain2().svgd(REAL, fixed=[(1, 1e-10)], prior=[gauss(), None], theta_dim=2, **KW); report(s,[1])
scen("A5d chain2 Graph.svgd fixed 1e-10", a5d)
