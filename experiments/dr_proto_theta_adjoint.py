"""θ-adjoint / cyclic-trace prototype, gated against the DR-A analytic oracle.

phasic's existing trace is JUMP-CHAIN (edge probabilities) and refuses cycles;
its own comment says a full fix needs "a true Gaussian elimination that drops
vertex i". So this prototypes THAT: expected absorption time E[T]=1^T(-S)^-1 a
via RATE-MATRIX Gaussian elimination recorded as a linear op TRACE. Two exact
routes, both cyclic-correct by construction:

  Tier-1  the elimination op-sequence in plain JAX -> jax.grad (reverse-mode
          over the trace).
  Tier-3  the SAME elimination recorded on a scalar tape -> a hand-written
          reverse-mode theta-adjoint (the C-adjoint template: walk the tape in
          reverse, chain leaf adjoints through the edge coefficients).

Gate: both vs the DR-A analytic oracle (jnp.linalg.solve grad == closed form)
and vs phasic's native forward, across a regime grid, on cyclic graphs; and vs
the production FD gradient to show the improvement.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph
import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Graph spec: n transient states 0..n-1, absorbing = -1.
# edges: list of (i, j, coeffs) ; rate(i->j) = coeffs . theta.
# --------------------------------------------------------------------------- #
# Fixture 1: DR-A 2-state cycle.  start->A ; A<->B (th0) ; B->abs (th1).
FIX_CYCLE2 = dict(
    n=2, theta_dim=2, alpha=[1.0, 0.0],
    edges=[(0, 1, [1.0, 0.0]),        # A->B  th0
           (1, 0, [1.0, 0.0]),        # B->A  th0   (cycle)
           (1, -1, [0.0, 1.0])],      # B->abs th1
    closed=lambda th: 1.0 / th[0] + 2.0 / th[1],
    closed_grad=lambda th: np.array([-1.0 / th[0] ** 2, -2.0 / th[1] ** 2]),
)
# Fixture 2: 3-state with a 3-cycle A->B->C->A plus exits, mixed structure.
FIX_CYCLE3 = dict(
    n=3, theta_dim=3, alpha=[1.0, 0.0, 0.0],
    edges=[(0, 1, [1.0, 0.0, 0.0]),   # A->B th0
           (1, 2, [1.0, 0.0, 0.0]),   # B->C th0
           (2, 0, [1.0, 0.0, 0.0]),   # C->A th0  (3-cycle)
           (0, -1, [0.0, 1.0, 0.0]),  # A->abs th1
           (2, -1, [0.0, 0.0, 1.0])], # C->abs th2
    closed=None,
)


# --------------------------------------------------------------------------- #
# (-S)(theta) as a dense matrix (prototype; production would be sparse).
# --------------------------------------------------------------------------- #
def neg_S(theta, fix):
    n = fix['n']; M = jnp.zeros((n, n))
    for (i, j, c) in fix['edges']:
        rate = jnp.dot(jnp.asarray(c, float), theta)
        M = M.at[i, i].add(rate)
        if j >= 0:
            M = M.at[i, j].add(-rate)
    return M


# --------------------------------------------------------------------------- #
# ORACLE: analytic gradient via jnp.linalg.solve (DR-A), exact VJP.
# --------------------------------------------------------------------------- #
def ET_solve(theta, fix):
    tau = jnp.linalg.solve(neg_S(theta, fix), jnp.ones(fix['n']))
    return jnp.dot(jnp.asarray(fix['alpha']), tau)


# --------------------------------------------------------------------------- #
# TIER 1: rate-matrix Gaussian elimination in plain JAX = the differentiable
# elimination TRACE. Fixed pivot order (no pivoting; (-S) is an M-matrix).
# --------------------------------------------------------------------------- #
def ET_trace(theta, fix):
    n = fix['n']
    A = neg_S(theta, fix); b = jnp.ones(n)
    for k in range(n):                      # forward elimination
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A = A.at[i].add(-f * A[k])
            b = b.at[i].add(-f * b[k])
    tau = [None] * n                        # back substitution
    for i in range(n - 1, -1, -1):
        acc = b[i]
        for j in range(i + 1, n):
            acc = acc - A[i, j] * tau[j]
        tau[i] = acc / A[i, i]
    return sum(jnp.asarray(fix['alpha'])[i] * tau[i] for i in range(n))


# --------------------------------------------------------------------------- #
# TIER 3: the SAME elimination on a scalar tape -> hand-written reverse-mode
# theta-adjoint (the C-adjoint template).
# --------------------------------------------------------------------------- #
class Tape:
    def __init__(self, theta_dim):
        self.op = []          # (kind, a, b)
        self.dtheta = []      # for leaves: coeff vector; else None
        self.theta_dim = theta_dim

    def leaf(self, coeffs):   # a theta-linear leaf: value = coeffs . theta
        self.op.append(('leaf', None, None)); self.dtheta.append(np.asarray(coeffs, float))
        return Node(self, len(self.op) - 1)

    def const(self, v):
        self.op.append(('const', v, None)); self.dtheta.append(None)
        return Node(self, len(self.op) - 1)

    def _bin(self, kind, a, b):
        self.op.append((kind, a, b)); self.dtheta.append(None)
        return Node(self, len(self.op) - 1)

    def forward(self, theta):
        val = np.empty(len(self.op))
        for s, (kind, a, b) in enumerate(self.op):
            if kind == 'leaf':   val[s] = self.dtheta[s] @ theta
            elif kind == 'const': val[s] = a
            elif kind == 'add':  val[s] = val[a] + val[b]
            elif kind == 'sub':  val[s] = val[a] - val[b]
            elif kind == 'mul':  val[s] = val[a] * val[b]
            elif kind == 'div':  val[s] = val[a] / val[b]
        return val

    def grad(self, theta, out_slot):
        val = self.forward(theta)
        adj = np.zeros(len(self.op)); adj[out_slot] = 1.0
        for s in range(len(self.op) - 1, -1, -1):     # reverse sweep
            kind, a, b = self.op[s]; g = adj[s]
            if g == 0.0 or kind in ('leaf', 'const'): continue
            if kind == 'add':   adj[a] += g;               adj[b] += g
            elif kind == 'sub': adj[a] += g;               adj[b] -= g
            elif kind == 'mul': adj[a] += g * val[b];      adj[b] += g * val[a]
            elif kind == 'div': adj[a] += g / val[b];      adj[b] += g * (-val[a] / val[b] ** 2)
        dtheta = np.zeros(self.theta_dim)               # chain leaves -> theta
        for s, (kind, _a, _b) in enumerate(self.op):
            if kind == 'leaf' and adj[s] != 0.0:
                dtheta += adj[s] * self.dtheta[s]
        return float(val[out_slot]), dtheta


class Node:
    def __init__(self, tape, slot): self.t = tape; self.s = slot
    def _c(self, o): return o if isinstance(o, Node) else self.t.const(float(o))
    def __add__(s, o): return s.t._bin('add', s.s, s._c(o).s)
    def __sub__(s, o): return s.t._bin('sub', s.s, s._c(o).s)
    def __mul__(s, o): return s.t._bin('mul', s.s, s._c(o).s)
    def __truediv__(s, o): return s.t._bin('div', s.s, s._c(o).s)
    __radd__ = __add__
    def __rsub__(s, o): return s._c(o).__sub__(s)
    __rmul__ = __mul__


def build_tape_and_grad(theta, fix):
    """Record the SAME Gaussian elimination on the tape, run the manual adjoint."""
    tp = Tape(fix['theta_dim']); n = fix['n']
    A = [[tp.const(0.0) for _ in range(n)] for _ in range(n)]
    for (i, j, c) in fix['edges']:
        r = tp.leaf(c)
        A[i][i] = A[i][i] + r
        if j >= 0:
            A[i][j] = A[i][j] - r
    b = [tp.const(1.0) for _ in range(n)]
    for k in range(n):
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(n):
                A[i][j] = A[i][j] - f * A[k][j]
            b[i] = b[i] - f * b[k]
    tau = [None] * n
    for i in range(n - 1, -1, -1):
        acc = b[i]
        for j in range(i + 1, n):
            acc = acc - A[i][j] * tau[j]
        tau[i] = acc / A[i][i]
    out = tau[0]
    for i in range(1, n):
        if fix['alpha'][i] != 0.0:
            out = out + tp.const(fix['alpha'][i]) * tau[i]
    # alpha[0]=1 assumed; general: weight tau[0]
    return tp.grad(np.asarray(theta, float), out.s)


def relerr(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def run(fix, name, thetas):
    print("=" * 88); print(f"FIXTURE: {name}  (n={fix['n']}, theta_dim={fix['theta_dim']})"); print("=" * 88)
    pg = Graph  # native forward via phasic, built to match
    # phasic native graph
    g = Graph(1); s = g.starting_vertex()
    verts = [g.find_or_create_vertex([fix['n'] - i]) for i in range(fix['n'])] + [g.find_or_create_vertex([0])]
    s.add_edge(verts[0], 1.0)
    for (i, j, c) in fix['edges']:
        verts[i].add_edge(verts[j if j >= 0 else fix['n']], list(c))
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=fix['theta_dim'])
    ET_native = lambda th: float(model(jnp.asarray(th), jnp.array([1.0]))[1][0])
    grad_fd = jax.grad(lambda th: model(th, jnp.array([1.0]))[1][0])
    grad_solve = jax.grad(lambda th: ET_solve(th, fix))
    grad_tr = jax.grad(lambda th: ET_trace(th, fix))

    for th in thetas:
        thj = jnp.asarray(th)
        f_solve = float(ET_solve(thj, fix)); f_tr = float(ET_trace(thj, fix)); f_nat = ET_native(th)
        print(f"\n  theta={th}")
        cl = fix['closed'](th) if fix['closed'] else None
        print(f"    forward  solve={f_solve:.8g}  trace={f_tr:.8g}  native={f_nat:.8g}" + (f"  closed={cl:.8g}" if cl else ""))
        g_oracle = np.asarray(grad_solve(thj))
        g_trace = np.asarray(grad_tr(thj))
        _, g_adj = build_tape_and_grad(th, fix)
        try:
            g_fd = np.asarray(grad_fd(thj)); fd_ok = True
        except Exception as e:
            g_fd = None; fd_err = f"{type(e).__name__}"; fd_ok = False
        print(f"    {'grad':>6} {'ORACLE(solve)':>15} {'TIER1(trace)':>14} {'t1.relerr':>10} "
              f"{'TIER3(adjoint)':>15} {'t3.relerr':>10} {'FD':>13} {'fd.relerr':>10}")
        for i in range(fix['theta_dim']):
            fd_s = f"{g_fd[i]:.5g}" if fd_ok else fd_err
            fd_re = f"{relerr(g_fd[i], g_oracle[i]):.1e}" if fd_ok else "-"
            print(f"    {'dth'+str(i):>6} {g_oracle[i]:>15.6g} {g_trace[i]:>14.6g} "
                  f"{relerr(g_trace[i], g_oracle[i]):>10.1e} {g_adj[i]:>15.6g} "
                  f"{relerr(g_adj[i], g_oracle[i]):>10.1e} {fd_s:>13} {fd_re:>10}")
        cg = fix.get('closed_grad')
        if cg is not None:
            true = cg(th)
            print(f"    vs TRUE (closed-form gradient): true={list(np.round(true,4))}")
            print(f"      relerr-vs-true   oracle={relerr(g_oracle,true)}  trace={relerr(g_trace,true)}  "
                  f"adjoint={relerr(g_adj,true)}" + (f"  FD={relerr(g_fd,true)}" if fd_ok else "  FD=CRASH"))


run(FIX_CYCLE2, "2-state cycle (DR-A)", [[1.0, 1.0], [1.0, 1e-4], [1.0, 1e-8]])
run(FIX_CYCLE3, "3-state 3-cycle",      [[1.0, 1.0, 1.0], [1.0, 1e-4, 1.0], [1.0, 1e-8, 1.0]])
