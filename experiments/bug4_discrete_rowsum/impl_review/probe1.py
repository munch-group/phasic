import numpy as np, phasic, sys
from phasic import Graph
print("phasic from", phasic.__file__, flush=True)

def tryit(label, f):
    try:
        r = f()
        print(f"{label}: OK -> {r}", flush=True)
    except Exception as e:
        print(f"{label}: RAISE {type(e).__name__}: {str(e)[:160]!r}", flush=True)

# A. zero-reward vertex with row sum 1.5 in a discrete reward transform
def gA(w=0.75):
    g = Graph(1); s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2]); a = g.find_or_create_vertex([0])
    s.add_edge(v1, 1.0); v1.add_edge(v2, w); v1.add_edge(a, w); v2.add_edge(a, 0.5)
    g.is_discrete = True
    return g
g = gA()
tryit("A direct pdf(2)", lambda: g.pdf(2))
tryit("A rt[0,0,1,0] pdf(1..4)", lambda: [g.reward_transform_discrete(np.array([0,0,1,0])).pdf(k) for k in range(1,5)])
tryit("A rt[0,0,1,0] row sums", lambda: [sum(e.weight() for e in v.edges()) for v in g.reward_transform_discrete(np.array([0,0,1,0])).vertices()])
# A2. negative weight at zero-reward vertex
g2 = gA(); 
for e in g2.vertex_at(1).edges():
    if e.to().index()==3: e.update_weight(-0.25)
tryit("A2 neg weight direct pdf", lambda: g2.pdf(2))
tryit("A2 neg rt[0,0,1,0] pdf", lambda: [g2.reward_transform_discrete(np.array([0,0,1,0])).pdf(k) for k in range(1,5)])

# B. starting vertex sums 1.5, reward transform with all ones
def gB(w1, w2):
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); z = g.find_or_create_vertex([0])
    s.add_edge(a, w1); s.add_edge(b, w2); a.add_edge(z, 0.5); b.add_edge(z, 0.25)
    g.is_discrete = True
    return g
gb = gB(1.0, 0.5)
tryit("B direct pdf", lambda: gb.pdf(1))
tryit("B rt ones pdf/cdf", lambda: (gb.reward_transform_discrete(np.array([1,1,1,1])).pdf(1), gb.reward_transform_discrete(np.array([1,1,1,1])).cdf(100)))
tryit("B rt [0,1,1,0] cdf100", lambda: gb.reward_transform_discrete(np.array([0,1,1,0])).cdf(100))
tryit("B rt [0,2,1,0] cdf100", lambda: gb.reward_transform_discrete(np.array([0,2,1,0])).cdf(100))
tryit("B expectation", lambda: gb.expectation())

# C. step invariant stress: random rows summing to exactly 1 / within ulp; count negatives
rng = np.random.default_rng(1)
neg_pmf = neg_stop = 0; maxcdf = 0
for trial in range(200):
    n = 12
    g = Graph(1); s = g.starting_vertex()
    vs = [g.find_or_create_vertex([i+1]) for i in range(n)]
    z = g.find_or_create_vertex([0])
    s.add_edge(vs[0], 1.0)
    for i, v in enumerate(vs):
        targets = [j for j in range(n) if j != i]
        k = rng.integers(1, 6)
        tg = rng.choice(targets, size=k, replace=False)
        w = rng.random(k + 1); w = w / w.sum()  # row sums ~1 incl absorb edge
        for t, ww in zip(tg, w[:-1]): v.add_edge(vs[t], float(ww))
        v.add_edge(z, float(w[-1]))
    g.is_discrete = True
    rs = max(sum(e.weight() for e in v.edges()) for v in vs)
    if rs > 1 + 1e-9: continue
    ctx = g.distribution_context()
    for _ in range(150):
        ctx.step()
        if ctx.pmf() < 0: neg_pmf += 1
        sp = np.asarray(ctx.stop_probability(), float)
        neg_stop += int((sp < 0).sum())
        maxcdf = max(maxcdf, ctx.cdf())
print(f"C random rows: neg_pmf={neg_pmf} neg_stop={neg_stop} maxcdf-1={maxcdf-1:.3e}", flush=True)

# D. FD gradient near 0 for a native discrete parameterized graph
import jax, jax.numpy as jnp
def pgeom():
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0]); g.is_discrete = True
    return g
m = Graph.pmf_from_graph(pgeom(), discrete=True)
k = jnp.array([1.0, 2.0, 3.0])
loss = lambda t: jnp.sum(jnp.log(m(t, k)))
tryit("D grad at 5e-8", lambda: np.asarray(jax.grad(loss)(jnp.array([5e-8]))))
tryit("D grad at 0.5", lambda: np.asarray(jax.grad(loss)(jnp.array([0.5]))))

# E. overflow: finite huge weights summing to inf
g = Graph(1); s = g.starting_vertex(); v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0]); b = g.find_or_create_vertex([2])
s.add_edge(v, 1.0); v.add_edge(a, 1e308); v.add_edge(b, 1e308); g.is_discrete = True
tryit("E huge weights pdf", lambda: g.pdf(1))
