"""§4.5 theta == 0: the one-sided step evaluates the forward AT 0. Which
forwards accept an exact-zero slot when the slot is an exit BESIDE another
(so 0 is a valid model, not a trap)? Fixture: s -> a; a -> b (t0), a -> c (t1);
b -> c (1). Also today's gradient at 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TIMES = jnp.array([0.5, 1.0, 2.0])
JUMPS = jnp.array([2.0, 3.0, 5.0])


def beside(mode):
    g = Graph(1)
    s = g.starting_vertex(); a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    if mode == "log":
        a.add_edge(b, [1.0, 1.0]); a.add_edge(c, [1.0, 1.0]); b.add_edge(c, [1.0, 1.0])
        g.weight_mode = "log"
    else:
        a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); b.add_edge(c, [1.0, 0.0])
        if mode == "formula":
            g.weight_formula = "c0*t0 + c1*t1"
        elif mode == "callback":
            g.weight_callback = lambda t, c: c[0] * t[0] + c[1] * t[1]
    return g


rows = []
for mode in ("linear", "log", "formula", "callback"):
    for disc in (False, True):
        g = beside(mode)
        x = JUMPS if disc else TIMES
        base = [0.3, 0.0] if disc else [1.0, 0.0]
        sites = {
            "pfg": lambda g=g, d=disc, x=x: (lambda m: (lambda th: m(th, x)))(Graph.pmf_from_graph(g, discrete=d, theta_dim=2)),
            "pam": lambda g=g, d=disc, x=x: (lambda m: (lambda th: m(th, x)))(Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=d, theta_dim=2)),
            "pam_rw": lambda g=g, d=disc, x=x: (lambda m: (lambda th: m(th, x, rewards=jnp.asarray([0.0, 1.0, 2.0, 0.0]))))(Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=d, theta_dim=2)),
        }
        if mode == "linear" and not disc and os.environ.get("PHASIC_SOURCE_DIR"):
            sites["mfg"] = lambda g=g: (lambda m: m)(Graph.moments_from_graph(g, nr_moments=2))
        for name, mk in sites.items():
            try:
                f = mk()
            except Exception as e:  # noqa: BLE001
                print(f"{mode:9s} disc={disc!s:5s} {name:7s} BUILD {tag(e)}"); continue
            L = lambda th, f=f: sum(jnp.nansum(z) for z in jax.tree_util.tree_leaves(f(th)))
            print(f"{mode:9s} disc={disc!s:5s} {name:7s} fwd@0={outcome(f, jnp.asarray(base))}  "
                  f"fwd@1e-7={outcome(f, jnp.asarray([base[0], 1e-7]))}  "
                  f"todays grad@0={outcome(jax.grad(L), jnp.asarray(base))}", flush=True)
if os.environ.get("PHASIC_SOURCE_DIR"):
    cpp = os.path.join(HERE, "m_cpp_beside.cpp")
    with open(cpp, "w") as fh:
        fh.write('#include "user_model.h"\nphasic::Graph build_model(const double* theta, int n_params) {\n'
                 '    phasic::Graph g(1); auto s = g.starting_vertex();\n'
                 '    auto a = g.find_or_create_vertex({3}); auto b = g.find_or_create_vertex({2});\n'
                 '    auto c = g.find_or_create_vertex({1});\n'
                 '    s.add_edge(a, 1.0); a.add_edge(b, theta[0]); a.add_edge(c, theta[1]); b.add_edge(c, theta[0]);\n'
                 '    return g;\n}\n')
    m = Graph.pmf_from_cpp(cpp)
    f = lambda th: m(th, TIMES)
    print(f"cpp       disc=False pmf_from_cpp fwd@0={outcome(f, jnp.asarray([1.0, 0.0]))}  "
          f"todays grad@0={outcome(jax.grad(lambda t: jnp.sum(f(t))), jnp.asarray([1.0, 0.0]))}")
