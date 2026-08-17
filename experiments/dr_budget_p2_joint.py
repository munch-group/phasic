import numpy as np, sys
sys.path.insert(0,"tests/pytest")
from test_notebook_model_recovery import (coalescent_indexed, sample_joint_observations,
    seed_all, OFF_TRUTH_JOINT, SCHEDULE, JOINT_TRUE, MUTATION_RATE)
from phasic import set_log_level, ExpStepSize
set_log_level("ERROR")
gg,idx=coalescent_indexed(4)
jpg=gg.joint_prob_graph(idx, reward_limit=3, mutation_rate=MUTATION_RATE)
rng=np.random.default_rng(17); obs=sample_joint_observations(jpg,JOINT_TRUE,1000,rng)
print("PIN2 joint_prob 168v: baseline 150it/60p -> 2.12e-5 vs true 1e-4 (79% low)", flush=True)
for iters,parts,sched in ((150,60,SCHEDULE),(600,60,ExpStepSize(first_step=0.05,last_step=0.005,tau=120.0)),
                          (600,200,ExpStepSize(first_step=0.05,last_step=0.005,tau=120.0)),
                          (1500,100,ExpStepSize(first_step=0.08,last_step=0.005,tau=300.0))):
    seed_all(1)
    sv=jpg.svgd(obs, fixed=[(1,MUTATION_RATE)], prior=OFF_TRUTH_JOINT,
                learning_rate=sched, n_iterations=iters, n_particles=parts)
    r=sv.get_results(); m=float(np.asarray(r['theta_mean']).ravel()[0])
    lo=float(np.asarray(r['hpd_lower']).ravel()[0]); hi=float(np.asarray(r['hpd_upper']).ravel()[0])
    print(f"  {iters:>4}it {parts:>3}p: mean={m:.4e} ({abs(m-1e-4)/1e-4:5.1%} off) "
          f"HPD=[{lo:.3e},{hi:.3e}] covers={lo<=1e-4<=hi}", flush=True)
