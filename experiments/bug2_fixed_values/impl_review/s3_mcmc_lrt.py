from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_indexed
import phasic.model_selection as ms
def mk(): return Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2)
def mrep(m, sl):
    r = m.get_results(); P = np.asarray(r['particles'])
    print(f"  seen={col(sl)[:5]} uniq={np.unique(P[:,sl])[:4]} mean={float(np.asarray(r['theta_mean'])[sl])!r} std={float(np.asarray(r['theta_std'])[sl])!r} hpd=({r['hpd_lower'][sl]!r},{r['hpd_upper'][sl]!r}) ncalls={len(SEEN)} acc={np.asarray(m.acceptance_rate)}")
    print("  hpd_intervals(0.5):", m.hpd_intervals(0.5))
    print("  free mean", float(np.asarray(r['theta_mean'])[1-sl]))
MK = dict(n_samples=120, n_chains=3, burn_in=20, verbose=False, progress=False, seed=0)
for v in (1e-10, 0.3, 0.0):
    def c1(v=v):
        m = chain2().mcmc(REAL, prior=[gauss(), None], fixed=[(1, v)], theta_dim=2, **MK); print("  parallel:", m.parallel); mrep(m,1)
        try:
            m.summary()
        except Exception as e: print("  summary RAISED", type(e).__name__, str(e)[:200])
    scen(f"C1 Graph.mcmc (vmap) fixed {v}", c1)
def c2():
    m = MCMC(model=mk(), observed_data=REAL, prior=[gauss(), None], fixed=[(1, 1e-10)], theta_dim=2, parallel='none', **MK); print("  parallel:", m.parallel); m.run(); mrep(m,1)
scen("C2 MCMC sequential (parallel='none')", c2)
def c2b():
    m = MCMC(model=mk(), observed_data=REAL, prior=[gauss(), None], fixed=[(1, 1e-10)], theta_dim=2, n_chains=1, n_samples=300, burn_in=20, adaptive=True, adapt_after=50, verbose=False, progress=False, seed=0); print("  parallel:", m.parallel); m.run(); mrep(m,1)
scen("C2b MCMC 1 chain adaptive", c2b)
def c2c():
    m = MCMC(model=mk(), observed_data=REAL, prior=[gauss(), None], fixed=[(1, 1e-10)], theta_dim=2, adaptive=True, adapt_after=30, **MK); m.run(); mrep(m,1)
scen("C2c MCMC vmap adaptive", c2c)
def c3():
    init = np.full((3,2), 2.0)
    m = MCMC(model=mk(), observed_data=REAL, prior=[gauss(), None], fixed=[(1, 0.25)], theta_dim=2, theta_init=init, **MK); m.run(); mrep(m,1)
    m = MCMC(model=mk(), observed_data=REAL, prior=[gauss(), None], fixed=[0, 1], theta_dim=2, **MK); m.run(); mrep(m,1); print("  mask dtype", m.fixed_mask.dtype)
    m = MCMC(model=mk(), observed_data=REAL, prior=[GaussPrior(ci=[0.5,5]), None], fixed=[(1, -0.0)], theta_dim=2, positive_params=False, **MK); print("  no-transform: param_transform", m.param_transform, "fixed_values", np.asarray(m.fixed_values))
scen("C3 MCMC theta_init / mask / no transform", c3)
def c4():
    for bad in ([(0,1.0),(1,0.3)],):
        try:
            MCMC(model=mk(), observed_data=REAL, prior=[None, None], fixed=bad, theta_dim=2, **MK); print("  all-fixed constructed")
        except Exception as e: print("  all fixed RAISED", type(e).__name__, str(e)[:100])
    try:
        MCMC(model=mk(), observed_data=REAL, prior=[GaussPrior(ci=[0.5,5]), None], fixed=[(1,0.3)], theta_dim=2, positive_params=False, param_transform=lambda p: jnp.cumsum(jnp.exp(p))[::-1], **MK); print("  coupled accepted!")
    except Exception as e: print("  coupled RAISED", type(e).__name__, str(e)[:100])
scen("C4 MCMC validation", c4)

# ---- LRT on real nested pair
L=3; EPOCHS=[0.0,0.4]
g, idx = coalescent_indexed(2)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EPOCHS[1]) for k in range(L+1)])
rows = [[k, 0] for k in range(L+1) for _ in range(int(round(p[k]*200)))]
EKW = dict(n_particles=8, n_iterations=20, learning_rate=0.01, progress=False, seed=1)
def d1():
    em = jc.epoch_model(rows, epoch_starts=EPOCHS)
    full = em.fit(fixed=[(1, 1.0), (3, 1.0)], prior=[gauss(), None]*2, **EKW)
    report(full,[1,3])
    nested = em.fit(fixed=[(1, 1.0), (3, 1.0), (2, 6.0)], prior=[gauss(), None, None, None], **EKW)
    report(nested,[1,2,3])
    print("  LRT:", ms.likelihood_ratio_test(full, nested))
    nested2 = em.fit(fixed=[(1, 1.0), (3, 1.0000001), (2, 6.0)], prior=[gauss(), None, None, None], **EKW)
    try: print("  LRT mismatched:", ms.likelihood_ratio_test(full, nested2))
    except Exception as e: print("  LRT mismatched RAISED:", str(e)[:200])
    try: print("  LRT refine:", ms.likelihood_ratio_test(full, nested, refine=True))
    except Exception as e: print("  LRT refine RAISED:", type(e).__name__, str(e)[:300])
scen("D1 FreeEpochModel LRT full vs nested", d1)
def d2():
    free = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 1.0)], prior=[gauss(), None], **EKW)
    tied = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 1.0)], tied=[(0,[0,1])], prior=[gauss(), None], **EKW)
    for f in (ms.likelihood_ratio_test, ms.likelihood_ratio_test_at):
        try: print("  ", f.__name__, f(free, tied))
        except Exception as e: print("  ", f.__name__, "RAISED:", type(e).__name__, str(e)[:300])
scen("D2 tied vs free LRT", d2)
for v in (1e-10, 0.0, 1e-6):
    def d3(v=v):
        em = jc.epoch_model(rows, epoch_starts=EPOCHS)
        s = em.fit(fixed=[(1, v), (3, v)], prior=[gauss(), None]*2, n_particles=4, n_iterations=2, learning_rate=0.01, progress=False, seed=1); report(s,[1,3])
    scen(f"D3 FreeEpochModel fixed {v} (deviation claim: <1e-7 raises negative-rate)", d3)
def d4():
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 1e-10)], prior=[gauss(), None], n_particles=4, n_iterations=2, learning_rate=0.01, progress=False, seed=1); report(s,[1,3])
    print("  ll", s.log_likelihood())
scen("D4 Graph.svgd epoch fixed 1e-10 no exposure", d4)
