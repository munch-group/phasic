import numpy as np, sys
sys.path.insert(0,"tests/pytest")
from test_notebook_model_recovery import (two_locus_arg, seed_all, SCHEDULE,
                                          MIXED_LOCUS_TRUE, MIXED_LOCUS_PRIORS)
from phasic import set_log_level, ExpStepSize
set_log_level("ERROR")
g=two_locus_arg(4); g.update_weights(MIXED_LOCUS_TRUE)
seed_all(0); data=g.sample(600)
print("PIN4 mixed-scale two-locus: baseline 60it/30p -> recomb 0.070 vs true 0.5", flush=True)
print("(prior centre 0.063, so it barely moved)", flush=True)
for iters,parts,sched in ((60,30,SCHEDULE),(400,60,SCHEDULE),
                          (1200,100,ExpStepSize(first_step=0.08,last_step=0.005,tau=250.0))):
    seed_all(1)
    sv=g.svgd(data, prior=MIXED_LOCUS_PRIORS, learning_rate=sched,
              n_iterations=iters, n_particles=parts)
    r=sv.get_results(); m=np.asarray(r['theta_mean']).ravel()
    lo=np.asarray(r['hpd_lower']).ravel(); hi=np.asarray(r['hpd_upper']).ravel()
    print(f"  {iters:>4}it {parts:>3}p: coal={m[0]:6.3f} (true 2.0) "
          f"recomb={m[1]:.4f} (true 0.5, {abs(m[1]-0.5)/0.5:.0%} off) "
          f"recomb_HPD=[{lo[1]:.4f},{hi[1]:.4f}]", flush=True)
