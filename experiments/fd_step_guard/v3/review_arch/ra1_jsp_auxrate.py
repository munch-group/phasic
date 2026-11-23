"""RA1: is the epoch (JSP) graph's uniformization rate scale-free?
joint_stop_prob_graph installs constant rate-1.0 trap loops (t<->aux).
Measure max exit rate (excluding the starting vertex) of the JSP graph at
production-like parameter scales, and the implied Lambda = G*t."""
import sys, math, time, ctypes
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
from test_notebook_model_recovery import coalescent_indexed
from unif import max_rate, n_terms

print('sizeof(long double) on this machine:', ctypes.sizeof(ctypes.c_longdouble))
for n in (4, 6):
    g, idx = coalescent_indexed(n)
    jc = g.joint_prob_graph(idx, discrete=False, reward_limit=3)
    jsp = jc.joint_stop_prob_graph()
    print(f'n={n}: joint graph {jc.vertices_length()} v, JSP {jsp.vertices_length()} v, param_length {jsp.param_length()}')
    for coal, mu, T in ((1.0, 1.0, 1.0), (1e-4, 1e-2, 1e4), (1e-4, 1e-4, 1e4), (1e-6, 1e-8, 1e6)):
        theta = [coal, mu][:jsp.param_length()]
        jsp.update_weights(theta); jc.update_weights(theta)
        # rates split: aux-loop vertices vs model vertices
        aux = set(int(v) for v in jsp._t_aux_map.values()); tv = set(int(k) for k in jsp._t_aux_map.keys())
        r_model = 0.0; r_trap = 0.0
        for i in range(1, jsp.vertices_length()):
            r = sum(e.weight() for e in jsp.vertex_at(i).edges())
            if i in aux or i in tv: r_trap = max(r_trap, r)
            else: r_model = max(r_model, r)
        mr = max(r_model, r_trap)
        lam_model = r_model * T; lam_jsp = mr * T
        print(f'   theta=({coal:g},{mu:g}) T={T:g}: model max rate {r_model:.3g}, trap-loop rate {r_trap:.3g} '
              f'-> steps intrinsic {n_terms(lam_model)} vs JSP-with-trap {n_terms(lam_jsp)} '
              f'(x{n_terms(lam_jsp)/n_terms(lam_model):.1f}); shipped Euler {int(T*max(1000,int(2*max(512,mr))))}')
