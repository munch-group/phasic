"""R1: plan 3.1a advance(dt) semantics vs scipy expm with piecewise-constant generators.
Prototype acts on a replica of the C context state (non-absorbing probability vector,
scalar absorbed mass, per-vertex accumulated time); weights are read from the real graph
after each real update_weights()."""
import numpy as np, math
from scipy.linalg import expm
from scipy.stats import poisson
from phasic import Graph

def build():
    g = Graph(1); s = g.starting_vertex()
    V = {i: g.find_or_create_vertex([i]) for i in range(1, 6)}
    s.add_edge(V[1], 0.6); s.add_edge(V[2], 0.2); s.add_edge(V[4], 0.1)   # defective (0.9) + atom 0.1
    V[1].add_edge(V[2], [2.0, 0.0]); V[2].add_edge(V[1], [0.0, 1.5]); V[2].add_edge(V[3], [0.7, 0.0])
    V[3].add_edge(V[1], [0.0, 0.3]); V[3].add_edge(V[4], [1.1, 0.0]); V[1].add_edge(V[5], [0.0, 0.2])
    V[3].add_edge(V[5], [0.05, 0.0])
    return g

def gen(g):
    vs = list(g.vertices()); n = len(vs); Q = np.zeros((n, n))
    for v in vs:
        if v.index() == 0: continue
        for e in v.edges():
            Q[v.index(), e.to().index()] += e.weight(); Q[v.index(), v.index()] -= e.weight()
    ipv = np.zeros(n)
    for e in vs[0].edges(): ipv[e.to().index()] += e.weight()
    ab = np.array([len(v.edges()) == 0 for v in vs])
    return Q, ipv, ab

class Ctx:  # replica of ptd_probability_distribution_context after the init step
    def __init__(self, ipv, ab):
        self.ab = ab; self.state = np.where(ab, 0.0, ipv); self.cdf = ipv[ab].sum()
        self.avt = np.zeros(len(ipv)); self.time = 0.0; self.pdf = None; self.steps = 0
    def advance(self, Q, dt, eps=1e-17):
        ab = self.ab; rates = -np.diag(Q); R = rates.max()
        a = Q[:, ab].sum(1)
        if R == 0.0 or dt == 0.0:           # plan's (1/G) sum ... is 0/0 here; needs a special case
            self.avt += self.state * dt; self.time += dt; self.pdf = self.state @ a; return
        G = 1.02 * R; lam = G * dt
        P = np.eye(len(Q)) + Q / G
        n = int(lam + 12 * math.sqrt(lam + 1) + 40)
        k = np.arange(n + 1); w = poisson.pmf(k, lam); sf = poisson.sf(k, lam)
        v = self.state.copy(); new = np.zeros_like(v); acc = np.zeros_like(v); absorbed = 0.0
        for j in range(n + 1):
            new += w[j] * v; acc += sf[j] * v
            v = v @ P; m = v[ab].sum(); v[ab] = 0.0          # C step zeroes absorbing entries -> pmf
            absorbed += m * sf[j]                             # P(N >= j+1) = sf(j)
            self.steps += 1
        self.state = new; self.cdf += absorbed; self.avt += acc / G; self.time += dt
        self.pdf = self.state @ a                              # plan: sum_v state(v) x rate into absorbing

def exact_piecewise(ipv, Qs_dts):
    n = len(ipv); p = ipv.copy(); occ = np.zeros(n)
    for Q, dt in Qs_dts:
        M = np.zeros((2 * n, 2 * n)); M[:n, :n] = Q; M[:n, n:] = np.eye(n)
        E = expm(M * dt); occ += p @ E[:n, n:]; p = p @ E[:n, :n]
    return p, occ

g = build(); g.update_weights([1.0, 1.0]); Q0, ipv, ab = gen(g)
sched = [([1.0, 1.0], 0.3), ([5000.0, 1.0], 0.01), ([1e-4, 2.0], 3.0), ([1.0, 1.0], 0.0),
         ([1e-9, 1e-9], 1.0), ([3.0, 1e-9], 0.5), ([2e5, 7.0], 1e-3), ([1.0, 1.0], 2.0)]
c = Ctx(ipv, ab); hist = []
tr = ~ab; tr[0] = False
print('theta, dt | state rel err | cdf abs err | pdf rel err | avt rel err | steps')
for th, dt in sched:
    g.update_weights(th); Q, _, _ = gen(g); hist.append((Q, dt)); c.advance(Q, dt)
    p, occ = exact_piecewise(ipv, hist)
    a = Q[:, ab].sum(1); pdf_ex = (p * ~ab) @ a
    rel = lambda x, y: np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300))
    print(f'{th} dt={dt:g} | {rel(c.state[tr], p[tr]):.1e} | {abs(c.cdf - p[ab].sum()):.1e} | {abs(c.pdf - pdf_ex)/max(pdf_ex,1e-300):.1e} | {rel(c.avt[tr], occ[tr]):.1e} | {c.steps}  time={c.time:g}')
print('per-absorbing-vertex mass (exact):', p[ab], ' -> context keeps only the scalar sum', c.cdf)

# all rates exactly zero: is that reachable through update_weights?
try:
    g.update_weights([0.0, 0.0]); Qz, _, _ = gen(g); print('update_weights([0,0]) accepted; max rate', (-np.diag(Qz)).max())
    s0 = c.state.copy(); c.advance(Qz, 1.5)
    print(' advance(1.5) at zero rates: state unchanged', np.allclose(c.state, s0), 'avt increment', (c.state * 1.5)[tr], 'pdf', c.pdf,
          ' ; plan formula (1/G) sum ... with G = 1.02 x 0 ->', 'division by zero')
except Exception as e:
    print('update_weights([0,0]) raises:', str(e)[:120])

# pdf right after an update_weights (no advance in between): left vs right limit
g.update_weights([1.0, 1.0]); Qa, _, _ = gen(g); g.update_weights([10.0, 1.0]); Qb, _, _ = gen(g)
print('density just before / just after a change point (same state):', c.state @ Qa[:, ab].sum(1), c.state @ Qb[:, ab].sum(1))

# reward-transformed graph
g = build(); g.update_weights([1.0, 1.0])
rw = np.array([0, 2.0, 0.0, 0.5, 0, 0])
gr = g.reward_transform(rw); Qr, ipvr, abr = gen(gr)
print('reward-transformed: vertices', gr.vertices_length(), 'ipv', ipvr, 'absorbing', abr, 'ipv sum', ipvr.sum())
cr = Ctx(ipvr, abr); 
for dt in (0.2, 0.5, 1.3): cr.advance(Qr, dt)
p = ipvr @ expm(Qr * 2.0); a = Qr[:, abr].sum(1)
print(' advance x3 to t=2: cdf err', abs(cr.cdf - p[abr].sum()), 'pdf rel err', abs(cr.pdf - (p * ~abr) @ a) / ((p * ~abr) @ a),
      '| shipped pdf(2) rel err', abs(gr.pdf(2.0) - (p * ~abr) @ a) / ((p * ~abr) @ a), ' atom at 0 (cdf(0)) =', ipvr[abr].sum())
