import numpy as np, sys
sys.path.insert(0,"tests/pytest")
from test_notebook_model_recovery import (two_island_graph, seed_all,
                                          OFF_TRUTH_ISLAND, SCHEDULE, ISLAND_TRUE)
from phasic import set_log_level, ExpStepSize
set_log_level("ERROR")
g=two_island_graph(8); g.update_weights(ISLAND_TRUE)
seed_all(0); data=g.sample(300)
print("PIN3 two_island 186v: baseline 40it/25p -> mean p0=1.29 vs true 0.7 (84% high)", flush=True)
print("(the 21-vertex case that PASSES runs 150it/50p)", flush=True)
for iters,parts,sched in ((40,25,SCHEDULE),(150,50,SCHEDULE),
                          (400,50,ExpStepSize(first_step=0.05,last_step=0.005,tau=100.0))):
    seed_all(1)
    sv=g.svgd(data, prior=OFF_TRUTH_ISLAND, learning_rate=sched,
              n_iterations=iters, n_particles=parts)
    r=sv.get_results(); m=np.asarray(r['theta_mean']).ravel()
    lo=np.asarray(r['hpd_lower']).ravel(); hi=np.asarray(r['hpd_upper']).ravel()
    rel=[abs(m[i]-ISLAND_TRUE[i])/ISLAND_TRUE[i] for i in range(2)]
    cov=[bool(lo[i]<=ISLAND_TRUE[i]<=hi[i]) for i in range(2)]
    print(f"  {iters:>4}it {parts:>3}p: mean={np.round(m,4).tolist()} "
          f"off={[f'{x:.0%}' for x in rel]} covers={cov}", flush=True)
