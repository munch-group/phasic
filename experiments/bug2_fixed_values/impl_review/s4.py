import sys
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/impl_review')
from common import *
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_indexed
import phasic.model_selection as ms
L=3; EPOCHS=[0.0,0.4]
g, idx = coalescent_indexed(2)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EPOCHS[1]) for k in range(L+1)])
rows = [[k, 0] for k in range(L+1) for _ in range(int(round(p[k]*200)))]
EKW = dict(n_particles=8, n_iterations=150, learning_rate=0.02, progress=False, seed=1)
def d1():
    em = jc.epoch_model(rows, epoch_starts=EPOCHS)
    full = em.fit(fixed=[(1, 1.0), (3, 1.0)], prior=[gauss(), None]*2, **EKW)
    report(full,[1,3]); SEEN.clear()
    nested = em.fit(fixed=[(1, 1.0), (3, 1.0), (2, 6.0)], prior=[gauss(), None, None, None], **EKW)
    report(nested,[1,2,3])
    print("  ll full", full.log_likelihood(), "nested", nested.log_likelihood(), "means", np.asarray(full.theta_mean), np.asarray(nested.theta_mean))
    for kw in ({}, {'refine': True}):
        try: print("  LRT", kw, ms.likelihood_ratio_test(full, nested, **kw))
        except Exception as e: print("  LRT", kw, "RAISED:", type(e).__name__, str(e)[:300])
    nested2 = em.fit(fixed=[(1, 1.0), (3, 1.0000001), (2, 6.0)], prior=[gauss(), None, None, None], n_particles=4, n_iterations=2, progress=False, seed=1)
    try: print("  LRT mismatched:", ms.likelihood_ratio_test(full, nested2))
    except Exception as e: print("  LRT mismatched RAISED:", str(e)[:200])
scen("D1 FreeEpochModel LRT full vs nested", d1)
for v in (1e-10, 0.0, 1e-8):
    def d3(v=v):
        em = jc.epoch_model(rows, epoch_starts=EPOCHS)
        s = em.fit(fixed=[(1, v), (3, v)], prior=[gauss(), None]*2, n_particles=4, n_iterations=2, learning_rate=0.01, progress=False, seed=1); report(s,[1,3])
    scen(f"D3 FreeEpochModel fixed {v}", d3)
    def d4(v=v):
        s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, v)], prior=[gauss(), None], n_particles=4, n_iterations=2, learning_rate=0.01, progress=False, seed=1); report(s,[1,3])
    scen(f"D4 Graph.svgd epoch fixed {v}", d4)
