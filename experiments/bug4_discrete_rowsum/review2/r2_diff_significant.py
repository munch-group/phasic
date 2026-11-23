"""Review 2: size of the change each treatment makes to pmf values that matter
(pmf > 1e-14), real joint_prob_graph n=5 theta=[2,0.05]; reuses r1's emulator."""
import sys, warnings, io, contextlib
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest'); sys.path.insert(0, '/Users/kmt/phasic-bug4/experiments/bug4_discrete_rowsum/review2')
with contextlib.redirect_stdout(io.StringIO()):
    import r1_step_variants as r1
n, edges = r1.n, r1.edges
ref = r1.emulate(n, edges, 'A', 300)[0]
big = ref > 1e-14
for var in ('B1', 'B2', 'C'):
    pm = r1.emulate(n, edges, var, 300)[0]
    print(f'{var}: values > 1e-14: {big.sum()}; max rel diff {np.max(np.abs(pm[big]-ref[big])/ref[big]):.2e}; bit-identical values {int((pm[big]==ref[big]).sum())}/{big.sum()}; max abs diff all {np.max(np.abs(pm-ref)):.2e}')
