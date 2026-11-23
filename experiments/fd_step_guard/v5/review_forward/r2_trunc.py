"""R2: is the plan's truncation rule defined/accurate for VECTOR read-outs (advance, stop_probability,
accumulated_visiting_time)?  Erlang-N chain, rate 1 (plan's own example: Erlang-100, t=5, density 1.1e-89)."""
import numpy as np, math
from scipy.stats import poisson, gamma

def erlangP(N, rate=1.0, f=1.02):
    G = f * rate; P = np.zeros((N + 1, N + 1))   # states 0..N-1 transient, N absorbing
    for i in range(N): P[i, i] = 1 - rate / G; P[i, i + 1] = rate / G
    return P, G

def advance(s, P, G, dt, rule, eps=1e-16, cap=200000):
    """returns new transient state, absorbed mass, occupancy, steps. rule: 'abs' | 'scalar' | 'comp'"""
    N = len(s); lam = G * dt
    x = s.copy(); new = np.zeros(N); occ = np.zeros(N); absorbed = 0.0; tot = s.sum(); k = 0
    while True:
        w = poisson.pmf(k, lam); T = poisson.sf(k, lam)      # T = P(N > k) = remaining Poisson mass
        new += w * x; occ += T * x
        y = np.zeros(N); y[1:] += x[:-1] * P[0, 1]; y += x * P[0, 0]; m = x[-1] * P[0, 1]
        supp_stable = np.array_equal(y > 0, x > 0)
        x = y; absorbed += m * T; k += 1
        if rule == 'abs':                     # v3 prototype / classic: absolute Poisson mass
            if T <= eps: break
        elif rule == 'scalar':                # plan 3.1 literally: relative to the accumulated (absorbed) sum, not before first absorbed mass
            if absorbed > 0 and T * tot <= eps * absorbed: break
            if T == 0.0 and k > lam: break    # (without this line the rule never ends when nothing is absorbed)
        elif rule == 'comp':                  # proposed: support closed AND tail bound <= eps x smallest non-zero component
            nz = new[new > 0]
            if supp_stable and T * tot <= eps * nz.min(): break
            if T == 0.0 and k > lam: break
        if k > cap: break
    return new, absorbed, occ / G, k

for N, t in ((100, 5.0), (100, 0.5), (30, 40.0)):
    P, G = erlangP(N); true_pdf = gamma.pdf(t, N); true_sf = gamma.sf(t, N)
    print(f'## Erlang-{N} rate 1, t={t}: true pdf {true_pdf:.3e}, true cdf {gamma.cdf(t, N):.3e}')
    for rule in ('abs', 'scalar', 'comp'):
        for grid in (1, 50):
            s = np.zeros(N); s[0] = 1.0; cdf = 0.0; steps = 0; occ = np.zeros(N)
            for _ in range(grid):
                s, a, o, k = advance(s, P, G, t / grid, rule); cdf += a; occ += o; steps += k
            pdf = s[-1] * 1.0
            # exact transient state: Poisson(t) probabilities; exact occupancy of last transient state = cdf of Erlang-N ... integral
            ex_state = poisson.pmf(np.arange(N), t)
            nzm = ex_state > 0
            comp_err = np.max(np.abs(s[nzm] - ex_state[nzm]) / ex_state[nzm])
            ex_occ = poisson.sf(np.arange(N), t)       # int_0^t Pois(i; u) du = P(Pois(t) > i)
            occ_err = np.max(np.abs(occ - ex_occ) / ex_occ)
            print(f'  rule={rule:6s} advances={grid:3d}: pdf rel err {abs(pdf-true_pdf)/true_pdf:.1e}  cdf rel err {abs(cdf-gamma.cdf(t,N))/gamma.cdf(t,N):.1e}'
                  f'  worst state-component rel err {comp_err:.1e}  worst occupancy rel err {occ_err:.1e}  steps {steps}')

# plan rule on a read-out with NO absorption (stop_probability on a graph whose absorbing vertex is unreachable, e.g. JSP graph outcome traps)
print('## no reachable absorbing vertex (2-cycle), plan rule "never before the first non-zero absorbed mass":')
lam = 3.0; k = 0
while True:
    T = poisson.sf(k, lam); k += 1
    if T == 0.0: break
print(f'  absorbed mass stays 0 forever -> rule never fires; the only exit is Poisson-tail underflow at k={k} for Lambda={lam} (or the work ceiling)')
for lam in (0.006, 0.1, 5, 1e3, 1e5):
    k8 = int(poisson.isf(1e-16, lam)); k300 = int(poisson.isf(1e-300, lam))
    print(f'  Lambda={lam:g}: steps to Poisson tail 1e-16: {k8+1}; to 1e-300 (rule relative to a 1e-284 component): {k300+1}')
