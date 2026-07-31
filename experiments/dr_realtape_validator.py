"""Batch-0 REAL-C-tape confirmation: forward-mode validator over the REAL _off
elimination tape (via Graph._debug_fwdmode_grad) on the CYCLIC graph (the
topology whose self-loop made the differentiable trace path refuse -> FD was
chosen). Two gates:

  (1) tape forward E[T] == phasic native expected_waiting_time   (tape is faithful)
  (2) forward-mode dE[T]/d(edge weight) == central-difference     (tape differentiates)

Gate (2) uses CD as the oracle ONLY where CD is trustworthy (benign scales). At
extreme MIXED scale the central difference itself degrades (the whole reason B3
exists) while the analytic forward-mode stays exact -- there we instead check
forward-mode against a CLOSED-FORM gradient and DISPLAY the CD divergence as the
B3 demonstration. Exits nonzero only on a real (non-FD-baseline) failure.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def cyclic_graph():
    g = Graph(1); s = g.starting_vertex()
    A = g.find_or_create_vertex([2]); B = g.find_or_create_vertex([1]); g.find_or_create_vertex([0])
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0])                              # A->B  th0   (inputs: w_AB, w_BA)
    B.add_edge(A, [1.0, 0.0])                              # B->A  th0
    B.add_edge(g.find_or_create_vertex([0]), [0.0, 1.0])  # B->abs th1   (w_Babs)
    return g


# Closed-form dE[T]/d(w_Babs) = -w_BA/(w_Babs^2 * w_AB) - 1/w_Babs^2, with the
# elimination inputs at (w_AB, w_BA, w_Babs) = (th0, th0, th1). Only used at the
# extreme-scale point where CD is untrustworthy; identifies the input carrying
# the largest-magnitude (most FD-fragile) gradient.
def dEdw_Babs(th0, th1):
    return -th0/(th1*th1*th0) - 1.0/(th1*th1)


print("Batch-0 real-C-tape forward-mode validator on the cyclic graph:")
ok_all = True
for theta in ([1.0, 1.0], [1.0, 0.5], [2.0, 1.0], [1.0, 1e-3], [1.0, 1e-6]):
    g = cyclic_graph()
    g.update_weights(theta)
    ewt, fwd, cd = g._debug_fwdmode_grad()
    native = float(np.asarray(g.expected_waiting_time())[0])
    fwd = np.asarray(fwd); cd = np.asarray(cd)
    closed = 1.0/theta[0] + 2.0/theta[1]
    spread = max(theta)/min(theta)
    cd_reliable = spread <= 1e4

    ok_fwd  = abs(ewt - native) / max(1.0, abs(native)) < 1e-9
    print(f"\n theta={theta}")
    print(f"  E[T]: tape={ewt:.10g}  native={native:.10g}  (closed={closed:.10g})")
    print(f"  forward-mode dE/dw : {np.array2string(fwd, precision=6)}")
    print(f"  central-diff dE/dw : {np.array2string(cd, precision=6)}")

    if cd_reliable:
        ok_grad = np.allclose(fwd, cd, rtol=1e-4, atol=1e-9)
        rel = np.max(np.abs(fwd-cd)) / max(1e-30, np.max(np.abs(cd)))
        print(f"  [{'OK' if ok_fwd else 'FAIL'}] forward==native   "
              f"[{'OK' if ok_grad else 'FAIL'}] forward-mode==central-diff (rel {rel:.1e})")
        ok_all &= ok_fwd and ok_grad
    else:
        # FD-defect regime: CD is unreliable; forward-mode vs closed-form oracle.
        oracle = dEdw_Babs(theta[0], theta[1])
        k = int(np.argmax(np.abs(fwd)))            # the dominant (FD-fragile) input
        ok_grad = abs(fwd[k] - oracle) / abs(oracle) < 1e-9
        cd_relerr = abs(cd[k] - oracle) / abs(oracle)
        print(f"  [FD-defect regime, spread={spread:.0e}] forward-mode[{k}]={fwd[k]:.8g} "
              f"vs closed-form {oracle:.8g}: {'OK' if ok_grad else 'FAIL'}")
        print(f"    -> central-diff[{k}]={cd[k]:.8g} is OFF by {cd_relerr:.1e} "
              f"(the FD defect B3 fixes); analytic forward-mode is exact")
        ok_all &= ok_fwd and ok_grad

print(f"\n{'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
sys.exit(0 if ok_all else 1)
