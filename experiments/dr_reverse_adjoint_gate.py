"""Batch-1 gate: the REVERSE-mode theta-adjoint (Graph._debug_reverse_grad) over
the real _off two-tier tape, on CYCLIC parameterized fixtures across a regime
grid. Gated against:
  - the Batch-0 FORWARD-MODE oracle (_debug_fwdmode_grad) -- component-wise equal
    is the dot-product identity <Jv,u>==<v,J'u> evaluated at every basis vector,
    which pins the shipped transpose (stage-1 + stage-2 REPLACE/kill + ordering);
  - a scale-matched central difference at benign scales;
  - native E[T]; and the closed-form dominant gradient at extreme mixed scale
    (2-cycle), where CD degrades.
Exits nonzero on any real failure.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def two_cycle():
    # s -> A <-> B -> abs   (B has a self-competing exit; elimination needs INV/OM/DIV)
    g = Graph(1); s = g.starting_vertex()
    A = g.find_or_create_vertex([2]); B = g.find_or_create_vertex([1]); g.find_or_create_vertex([0])
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0])                              # th0
    B.add_edge(A, [1.0, 0.0])                              # th0
    B.add_edge(g.find_or_create_vertex([0]), [0.0, 1.0])  # th1
    return g


def three_cycle():
    # s -> A -> B -> C, with C -> A back-edge (3-cycle A->B->C->A) and C -> abs.
    g = Graph(1); s = g.starting_vertex()
    A = g.find_or_create_vertex([3]); B = g.find_or_create_vertex([2])
    C = g.find_or_create_vertex([1]); g.find_or_create_vertex([0])
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0])                              # th0
    B.add_edge(C, [0.0, 1.0])                              # th1
    C.add_edge(A, [1.0, 0.0])                              # th0  (back-edge -> cycle)
    C.add_edge(g.find_or_create_vertex([0]), [0.0, 1.0])  # th1
    return g


def dEdw_Babs(th0, th1):   # 2-cycle only, dominant input
    return -th0/(th1*th1*th0) - 1.0/(th1*th1)


def check(name, build, thetas, extreme_oracle=None):
    global ok_all
    print(f"\n=== fixture: {name} ===")
    for theta in thetas:
        g = build(); g.update_weights(theta)
        ewt_r, rev = g._debug_reverse_grad()
        ewt_f, fwd, cd = g._debug_fwdmode_grad()
        rev = np.asarray(rev); fwd = np.asarray(fwd); cd = np.asarray(cd)
        native = float(np.asarray(g.expected_waiting_time())[0])
        spread = max(theta)/min(theta); cd_reliable = spread <= 1e4

        ok_ewt = abs(ewt_r - native)/max(1.0, abs(native)) < 1e-9
        # two INDEPENDENT AD paths -> agree to the conditioning floor (~1e-10 rel
        # at extreme scale), not bit-for-bit; a real transpose bug is O(1) rel.
        ok_match = np.allclose(rev, fwd, rtol=1e-7, atol=1e-9)
        print(f" theta={theta}: E[T]={ewt_r:.8g} (native {native:.8g}); "
              f"rev={np.array2string(rev, precision=5)}")
        line = f"   [{'OK' if ok_ewt else 'FAIL'}] rev E[T]==native  " \
               f"[{'OK' if ok_match else 'FAIL'}] rev==fwd-mode (|d|max {np.max(np.abs(rev-fwd)):.1e})"
        if cd_reliable:
            ok_cd = np.allclose(rev, cd, rtol=1e-4, atol=1e-9)
            line += f"  [{'OK' if ok_cd else 'FAIL'}] rev==central-diff"
            ok_all &= ok_ewt and ok_match and ok_cd
        elif extreme_oracle is not None:
            oracle = extreme_oracle(theta[0], theta[1]); k = int(np.argmax(np.abs(rev)))
            ok_or = abs(rev[k]-oracle)/abs(oracle) < 1e-9
            line += f"  [FD-defect] rev[{k}]={rev[k]:.6g} vs closed-form {oracle:.6g}: " \
                    f"{'OK' if ok_or else 'FAIL'} (CD off {abs(cd[k]-oracle)/abs(oracle):.1e})"
            ok_all &= ok_ewt and ok_match and ok_or
        else:
            ok_all &= ok_ewt and ok_match
        print(line)


ok_all = True
print("Batch-1 reverse theta-adjoint vs forward-mode oracle on cyclic fixtures:")
check("2-cycle (s->A<->B->abs)", two_cycle,
      [[1.0, 1.0], [1.0, 0.5], [2.0, 1.0], [1.0, 1e-3], [1.0, 1e-6]],
      extreme_oracle=dEdw_Babs)
check("3-cycle (s->A->B->C->A, C->abs)", three_cycle,
      [[1.0, 1.0], [1.5, 0.7], [3.0, 0.2], [0.5, 2.0]])

print(f"\n{'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
sys.exit(0 if ok_all else 1)
