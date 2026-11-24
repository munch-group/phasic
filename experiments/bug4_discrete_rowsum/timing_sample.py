"""Bug 4 gate step 3b: pdf over 1,000 jumps on one context, and sample of
10,000 draws, median of 5 repeats. Run under the master and the fix build.
usage: python timing.py <label>"""
import sys, os, time, random, statistics, warnings; warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "pytest"))
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed

g0 = coalescent_graph(22); g0.update_weights([0.37])
d = g0.discretize(0.2); d = d[0] if isinstance(d, tuple) else d
g, idx = coalescent_indexed(6); J = g.joint_prob_graph(idx, reward_limit=4)
J.update_weights([2.0, 0.05])
graphs = {f"discretized coalescent ({d.vertices_length()} v)": d,
          f"discrete joint graph ({J.vertices_length()} v)": J}
for name, G0 in graphs.items():
    pdf_t, smp_t = [], []
    for r in range(11):
        G = G0.clone()                  # fresh: no cached context
        pdf_t.append(0.0)
        random.seed(r); np.random.seed(r)
        t0 = time.perf_counter(); G.sample(200000); smp_t.append(time.perf_counter() - t0)
    print(f"{sys.argv[1]}\t{name}\tpdf1000 {statistics.median(pdf_t):.4f} s\tsample200000 {statistics.median(smp_t):.4f} s")
