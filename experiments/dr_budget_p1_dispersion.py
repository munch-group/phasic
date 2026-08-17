import numpy as np, sys
sys.path.insert(0,"tests/pytest")
from test_notebook_model_recovery import (coalescent_graph, seed_all,
                                          OFF_TRUTH_COAL, SCHEDULE, COAL_TRUE)
from phasic import set_log_level, ExpStepSize
set_log_level("ERROR")
g=coalescent_graph(10); g.update_weights(COAL_TRUE)
seed_all(0); data=g.sample(500)
print("PIN1 under-dispersion: does the HPD widen to cover true 7.0?", flush=True)
print("baseline was 150it/60p -> HPD [6.44, 6.61], sd ~0.67x asymptotic", flush=True)
for iters,parts,sched in ((150,60,SCHEDULE),(150,200,SCHEDULE),(150,400,SCHEDULE),
                          (600,60,ExpStepSize(first_step=0.05,last_step=0.005,tau=120.0)),
                          (600,200,ExpStepSize(first_step=0.05,last_step=0.005,tau=120.0))):
    seed_all(1)
    sv=g.svgd(data, prior=OFF_TRUTH_COAL, learning_rate=sched,
              n_iterations=iters, n_particles=parts)
    r=sv.get_results(); m=float(np.asarray(r['theta_mean']).ravel()[0])
    lo=float(np.asarray(r['hpd_lower']).ravel()[0]); hi=float(np.asarray(r['hpd_upper']).ravel()[0])
    sd=float(np.std(np.asarray(r['particles'])[:,0]))
    print(f"  {iters:>4}it {parts:>3}p: mean={m:6.3f} HPD=[{lo:6.3f},{hi:6.3f}] "
          f"sd={sd:.4f} covers7={lo<=7.0<=hi}", flush=True)
