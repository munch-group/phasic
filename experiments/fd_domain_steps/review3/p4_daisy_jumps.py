"""P4: does the epoch (daisy) forward have granularity jumps in theta when
an earlier epoch's max rate exceeds 512 (auto granularity)? If so, the full
log rule must NOT be used there (the plan's section 3.3 does not classify
this family). Compare today / capped / full on epoch-1 coalescent slot over
60 nearby theta, against the full rule on a FIXED-granularity model."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, jax, jax.numpy as jnp
from common import *
jax.config.update("jax_enable_x64", True)
src = open(os.path.join(os.path.dirname(__file__), 'p1_daisy.py')).read()
exec(src.split("em = jpg.epoch_model")[0])   # graph, jpg, obs
oi = jpg._map_joint_observations_to_indices(obs)
EP = np.array([0.0, 0.005])
def build(gran):
    m, *_ = jpg._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=EP, user_prior=None,
                                        user_fixed=None, granularity=gran)
    return lambda th: np.asarray(m(jnp.asarray(th))[0], np.float64)
F0 = build(0)
Ffix = build(20000)
rng = np.random.default_rng(0)
for scale in (30.0, 200.0):
    errs = {k: [] for k in ('today', 'cap s=1e-4', 'full s=1e-4', 'full s=1e-3')}
    for k in range(60):
        th = np.array([scale, 1e-3, 1.0, 1e-3]) * np.r_[1 + 1e-3 * rng.standard_normal(), 1, 1, 1]
        cot = 1.0 / F0(th)
        ref = fd_log(Ffix, th, 0, cot, 1e-4)
        errs['today'].append(rel(fd_today(F0, th, 0, cot), ref))
        errs['cap s=1e-4'].append(rel(fd_log(F0, th, 0, cot, 1e-4, cap=1e-7), ref))
        errs['full s=1e-4'].append(rel(fd_log(F0, th, 0, cot, 1e-4), ref))
        errs['full s=1e-3'].append(rel(fd_log(F0, th, 0, cot, 1e-3), ref))
    for k, v in errs.items():
        v = np.array(v)
        print(f"  coal1 scale={scale:g} {k:12s} median {np.median(v):.1e} 95pct {np.percentile(v,95):.1e} max {v.max():.1e} frac>1% {np.mean(v>1e-2):.2f}", flush=True)
