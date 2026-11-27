"""Shared definitions of the Plan A bit-identity cells (fd-domain-steps-plan.md
§6 Batch 1, §4.7).

Each cell is a model built through one finite-difference site, a theta at which
today's FD rule works, and a scalar loss. ``grad_in_mode`` evaluates its
gradient eagerly, under ``jit`` and under ``vmap(grad)``. The golden values were
captured on master ``ff049679`` (before any Plan A source change) by
``experiments/fd_domain_steps/batch1/capture_golden.py`` into
``tests/pytest/data/fd_domain_steps_golden.json``; ``test_fd_domain_steps.py``
recomputes them on the current code and requires BIT equality in the same mode
on the capture platform. Every cell is UNDECLARED (no ``param_domains``) and
outside the §2 band, so Plan A must leave it bit for bit unchanged.
"""
from __future__ import annotations

import os
from functools import lru_cache, partial
from itertools import combinations_with_replacement

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GOLDEN_PATH = os.path.join(HERE, "data", "fd_domain_steps_golden.json")

TIMES = [0.5, 1.0, 2.0]
JUMPS = [2.0, 3.0, 5.0]

CPP_SRC = """#include "user_model.h"
phasic::Graph build_model(const double* theta, int n_params) {
    phasic::Graph g(1); auto s = g.starting_vertex();
    auto a = g.find_or_create_vertex({3}); auto b = g.find_or_create_vertex({2});
    auto c = g.find_or_create_vertex({1});
    s.add_edge(a, 1.0); a.add_edge(b, theta[0]); a.add_edge(c, theta[1]);
    b.add_edge(c, theta[0]);
    return g;
}
"""


def _ensure_source_dir():
    # moments_from_graph / pmf_from_cpp compile against the source tree
    # (OI-058); point them at this checkout when not set.
    os.environ.setdefault("PHASIC_SOURCE_DIR", REPO)


def beside(mode="linear"):
    """s -> a; a -> b (t0), a -> c (t1); b -> c (t0). Slot 1 is an exit beside
    another, so every slot value >= 0 is a valid model."""
    from phasic import Graph
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    if mode == "log":
        a.add_edge(b, [1.0, 1.0]); a.add_edge(c, [2.0, 1.0]); b.add_edge(c, [1.0, 1.0])
        g.weight_mode = "log"
        return g
    a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); b.add_edge(c, [1.0, 0.0])
    if mode == "formula":
        g.weight_formula = "c0*t0 + c1*t1"
    elif mode == "callback":
        g.weight_callback = lambda t, c: c[0] * t[0] + c[1] * t[1]
    return g


REWARDS = [0.0, 1.0, 2.0, 0.0]


def joint_graphs():
    """(continuous joint graph, observed indices) for a 4-sample coalescent."""
    jpg, oi, _ = _joint()
    return jpg, oi


def joint_rows():
    """The observed outcome rows behind ``joint_graphs()``'s indices."""
    return [list(r) for r in _joint()[2]]


@lru_cache(maxsize=None)
def _joint():
    from phasic import Graph, with_ipv, StateIndexer, Property
    nr = 4
    idx = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
    pairs = partial(combinations_with_replacement, r=2)

    @with_ipv([nr] + [0] * (nr - 1))
    def coal(state, indexer=None):
        t = []
        for i, j in pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy(); new[i] -= 1; new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return t

    g = Graph(coal, indexer=idx)
    jd = g.joint_prob_graph(idx, reward_limit=3, event_rate_coefficient=1e-2)
    jd.update_weights([1.0, 1e-2])
    tab = jd.joint_prob_table()
    obs = tab.iloc[:8][tab.columns[:-1]].to_numpy().tolist()
    jpg = g.joint_prob_graph(idx, reward_limit=3, event_rate_coefficient=1e-2, discrete=False)
    oi = jpg._map_joint_observations_to_indices(obs)
    return jpg, tuple(int(x) for x in oi), tuple(tuple(int(v) for v in r) for r in obs)


def build(name):
    """Return f(theta) -> pytree for cell ``name`` (fresh model each call)."""
    import jax.numpy as jnp
    from phasic import Graph
    times = jnp.asarray(TIMES); jumps = jnp.asarray(JUMPS)
    rw = jnp.asarray(REWARDS)
    if name == "pfg_wasdph":
        m = Graph.pmf_from_graph(beside().discretize(1.0), discrete=True, theta_dim=3)
        return lambda th: m(th, jnp.asarray([1.0, 2.0, 4.0]))
    if name.startswith("pfg_"):
        mode, disc = {"pfg_linear": ("linear", False), "pfg_log": ("log", False),
                      "pfg_formula": ("formula", False), "pfg_callback": ("callback", False),
                      "pfg_dph": ("linear", True)}[name]
        m = Graph.pmf_from_graph(beside(mode), discrete=disc, theta_dim=2)
        return lambda th: m(th, jumps if disc else times)
    if name.startswith("pam_"):
        mode = "formula" if "formula" in name else ("log" if "log" in name else "linear")
        ffi = "ffi" in name
        m = Graph.pmf_and_moments_from_graph(beside(mode), nr_moments=2, theta_dim=2, use_ffi=ffi)
        if "rw" in name:
            return lambda th: m(th, times, rewards=rw)
        return lambda th: m(th, times)
    if name == "cdf_zero":
        m = Graph.pmf_and_moments_from_graph(beside(), nr_moments=2, theta_dim=2)
        return lambda th: m._cdf_zero_fn(th, rw)
    if name == "multivar":
        m = Graph.pmf_and_moments_from_graph_multivariate(beside(), nr_moments=2, theta_dim=2)
        return lambda th: m(th, times, rewards=jnp.stack([rw, 2 * rw]))
    if name == "mfg":
        _ensure_source_dir()
        m = Graph.moments_from_graph(beside(), nr_moments=2)
        return m
    if name == "cpp":
        _ensure_source_dir()
        import tempfile
        d = tempfile.mkdtemp(prefix="fdgold_")
        p = os.path.join(d, "fd_golden_model.cpp")
        with open(p, "w") as fh:
            fh.write(CPP_SRC)
        m = Graph.pmf_from_cpp(p)
        return lambda th: m(th, times)
    if name == "rvp_linear":
        g = beside()
        return lambda th: g.reward_visit_probability(np.asarray(REWARDS), theta=th)
    if name == "jix":
        jpg, oi = joint_graphs()
        m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2)
        return lambda th: m(th, jnp.asarray(oi))
    if name.startswith("dc_"):
        jpg, oi = joint_graphs()
        kw = {"dc_noexp": {}, "dc_efg": {"exact_final_grad": True},
              "dc_exp": {"exposure_arr": np.linspace(0.5, 2.0, len(oi)),
                         "exposure_param_index": 1}}[name]
        m, _, _, _ = jpg._daisy_chain_svgd_model(
            observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
            user_prior=None, user_fixed=None, **kw)
        return lambda th: m(th, None)
    if name == "dcjp":
        jpg, _ = joint_graphs()
        jsp = jpg.joint_stop_prob_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        return lambda th: jsp.daisy_chain_joint_probs(
            epoch_thetas=jnp.asarray(th).reshape(2, 2), epoch_dts=[0.5],
            initial_ipv=np.full(n_ipv, 1.0 / n_ipv))
    raise KeyError(name)


# (cell, theta) pairs: ordinary values, plus slots just ABOVE the §2 band
# (2e-7) on refusing models, which must stay bit-identical too.
CELLS = [
    ("pfg_linear", [0.7, 1.3]),
    ("pfg_log", [0.7, 1.3]),
    ("pfg_formula", [0.7, 1.3]),
    ("pfg_callback", [0.7, 1.3]),
    ("pfg_dph", [0.3, 0.4]),
    ("pfg_dph", [0.3, 2e-7]),
    ("pfg_wasdph", [0.7, 1.3, 0.9]),
    ("pam_linear", [0.7, 1.3]),
    ("pam_linear_ffi", [0.7, 1.3]),
    ("pam_linear", [0.7, 5e-8]),        # non-refusing linear: today's clamp, unchanged
    ("pam_formula", [0.7, 1.3]),
    ("pam_log", [0.7, 1.3]),
    ("pam_linear_rw", [0.7, 1.3]),
    ("pam_linear_rw", [0.7, 2e-7]),
    ("cdf_zero", [0.7, 1.3]),
    ("multivar", [0.7, 1.3]),
    ("mfg", [0.7, 1.3]),
    ("cpp", [0.7, 1.3]),
    ("rvp_linear", [0.7, 1.3]),
    ("jix", [1.0, 0.5]),
    ("dc_noexp", [1.0, 0.5, 1.5, 0.3]),
    ("dc_noexp", [1.0, 2e-7, 1.5, 3e-7]),
    ("dc_exp", [1.0, 0.5, 1.5, 0.3]),
    ("dc_efg", [1.0, 0.5, 1.5, 0.3]),
    ("dcjp", [1.0, 0.5, 1.5, 0.3]),
]
MODES = ("eager", "jit", "vmap")
# pmf_from_cpp cannot be vmapped on master (its pure_callback returns one row
# for a batch of particles: "Expected (2, 3), Actual (1, 3)"; OI-123).
NO_VMAP = {"cpp"}


# Reward-transform cells are NOT reproducible run to run on master (OI-022 /
# OI-015: address-ordered reward transform). Two fresh-process captures
# differed by up to 9.6e-8 relative in the FD gradient (forward last-bit noise
# times 1/(2e-7)), so these compare at rtol 1e-6 instead of bitwise. A wrong
# rule selection outside the band would change the gradient by orders of
# magnitude more (the step would go from 0.5 relative to 3e-5).
NON_REPRODUCIBLE = {"pam_linear_rw", "multivar", "cdf_zero"}
NON_REPRODUCIBLE_RTOL = 1e-6


def modes_for(name):
    return tuple(m for m in MODES if not (m == "vmap" and name in NO_VMAP))


def loss_of(f):
    import jax
    import jax.numpy as jnp

    def L(th):
        leaves = jax.tree_util.tree_leaves(f(th))
        return sum(jnp.nansum(x) * (1.0 if k == 0 else 1e-3) for k, x in enumerate(leaves))
    return L


def grad_in_mode(name, theta, mode):
    """Gradient of the cell's loss as a float64 numpy array (vmap: 2 rows)."""
    import jax
    import jax.numpy as jnp
    L = loss_of(build(name))
    th = jnp.asarray(theta, dtype=jnp.float64)
    if mode == "eager":
        return np.asarray(jax.grad(L)(th), dtype=np.float64)
    if mode == "jit":
        return np.asarray(jax.jit(jax.grad(L))(th), dtype=np.float64)
    if mode == "vmap":
        P = jnp.stack([th, th * 1.1])
        return np.asarray(jax.vmap(jax.grad(L))(P), dtype=np.float64)
    raise KeyError(mode)


def key(name, theta, mode):
    return f"{name}|{','.join(repr(float(x)) for x in theta)}|{mode}"


def to_hex(a):
    return [float(x).hex() for x in np.ravel(a)]


def platform_tag():
    import platform
    import sys
    return f"{sys.platform}-{platform.machine()}"
