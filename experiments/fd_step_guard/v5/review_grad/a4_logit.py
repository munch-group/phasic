"""A4 (pure numpy): logit central step for a (0,1)-declared parameter, s = 1e-4.
F(theta) = theta^a (1-theta)^b ; phi-space target = theta(1-theta) F'(theta)/F = a(1-theta) - b theta.
Two probe formulas: (N) sigmoid(logit(theta) +- s) built from log/exp;  (S) theta e^{+-s} / (1 - theta + theta e^{+-s}).
Also: what the formulas return for theta outside (0,1) (exposure-scaled slot, softplus-transformed single BetaPrior)."""
import numpy as np
np.seterr(all='ignore')
s = 1e-4
def probes_N(t):
    x = np.log(t) - np.log1p(-t); return 1/(1+np.exp(-(x+s))), 1/(1+np.exp(-(x-s)))
def probes_S(t):
    return t*np.exp(s)/(1-t+t*np.exp(s)), t*np.exp(-s)/(1-t+t*np.exp(-s))
def F(t, a, b): return a*np.log(t) + b*np.log1p(-t)   # log F, evaluated accurately
for a, b in ((3, 2), (1, 0), (0, 1), (50, 50)):
    print(f'## F = theta^{a} (1-theta)^{b}')
    for t in (1e-9, 1e-3, 0.5, 1-1e-3, 1-1e-6, 1-1e-9):
        ex_phi = a*(1-t) - b*t
        row = []
        for nm, pr in (('N', probes_N), ('S', probes_S)):
            tp, tm = pr(t)
            # FD of F (not log F), divided by the actual probe difference, then chain to phi and to d log F
            d = (np.exp(F(tp, a, b) - F(t, a, b)) - np.exp(F(tm, a, b) - F(t, a, b))) / (tp - tm)
            phi = d * t * (1 - t)
            row.append(f'{nm}: probes inside (0,1): {0 < tm < tp < 1}; phi-space abs err {abs(phi-ex_phi):.1e} (value {ex_phi:+.3g})')
        print(f'  theta = 1-{1-t:.0e}' if t > 0.5 else f'  theta = {t:.0e}', '|', ' | '.join(row))
print('## theta outside (0,1) handed to a (0,1)-declared slot')
for t in (1.0, 1.0000001, 5.0, 50.0, 0.0):
    print(f'  theta={t}: N -> {probes_N(t)}   S -> {probes_S(t)}')
