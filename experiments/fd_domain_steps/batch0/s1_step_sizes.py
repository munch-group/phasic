"""Plan A Batch 0: step size per forward family (plan §6 Batch 0, §4.1-4.3).

usage: python s1_step_sizes.py <model>     (one model per process; see MODELS)

For each model, theta point and slot, the partial derivative of the
log-likelihood-type scalar L(theta) = sum(cot * F(theta)) (cot = 1/F at the
point, i.e. d log-likelihood; cot = 1 for moment outputs) is computed by:

  today   theta +/- 1e-7 absolute, divided by 2e-7 (master's rule)
  full    theta * exp(+/- s), divided by the actual probe difference
  cap     theta * exp(+/- u), u = min(log1p(1e-7/theta), s)
  ufull / ucap   (probability slots of discrete models only) the 'unit'
          rule: logit step s, capped at min(s, 1e-7/(theta(1-theta)))/(1+s)

for s on the grid S, against a reference: the EXACT gradient where phasic has
one (moments Jacobian, joint-index sojourn adjoint, epoch final-epoch block),
otherwise a Richardson extrapolation of log-coordinate central differences
(s = 1e-2, 5e-3, 2.5e-3; spread reported). Rows go to s1_<model>.jsonl; the
summary (s2_summarise.py) applies the acceptance criterion.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from functools import partial
from itertools import combinations_with_replacement

S = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "pytest"))


# ------------------------------------------------------------------ models
def _nmr():
    import importlib.util
    p = os.path.join(HERE, "..", "..", "..", "tests", "pytest",
                     "test_notebook_model_recovery.py")
    spec = importlib.util.spec_from_file_location("_nmr", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def times_for(scale, n=6):
    # natural time units: rate * t ~ 0.05 .. 3
    return jnp.asarray(np.geomspace(0.05, 3.0, n) / scale)


def m_euler(builder, n, P, pts):
    """Continuous Euler pdf, pmf_from_graph (FFI)."""
    def make():
        g = builder(n)
        m = Graph.pmf_from_graph(g, theta_dim=P)
        out = []
        for regime, th in pts:
            t = times_for(float(np.median(th)))
            out.append((regime, th, (lambda x, t=t: m(x, t)), None, "loglik"))
        return out
    return make


def m_reward_euler(n, pts):
    """Reward-transformed Euler pdf: pmf_and_moments_from_graph with 1-D
    rewards (refusing by §4.4 because of the rewards)."""
    def make():
        mod = _nmr()
        g = mod.coalescent_graph(n)
        rw = jnp.asarray(np.sum(g.states().T, axis=0), dtype=jnp.float64)
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1,
                                             exact_moment_grad=False)
        out = []
        for regime, th in pts:
            t = times_for(float(np.median(th))) * 3
            out.append((regime, th, (lambda x, t=t: m(x, t, rewards=rw)[0]), None, "loglik"))
        return out
    return make


def m_moments(builder, n, P, pts):
    """Elimination: the moments output of pmf_and_moments_from_graph; exact
    reference = the shipped exact moments Jacobian."""
    def make():
        g = builder(n)
        m_fd = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=P,
                                                exact_moment_grad=False)
        m_ex = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=P,
                                                exact_moment_grad=True)
        t1 = jnp.asarray([1.0])
        out = []
        for regime, th in pts:
            F = (lambda x: m_fd(x, t1)[1])
            y = np.asarray(F(jnp.asarray(th)))
            cot = jnp.asarray(1.0 / y)
            ex = (lambda x, cot=cot: jax.grad(lambda z: jnp.sum(cot * m_ex(z, t1)[1]))(x))
            out.append((regime, th, F, ex, "loglik"))
        return out
    return make


def _jpg():
    mod = _nmr()
    g, idx = mod.coalescent_indexed(4)
    return g, idx


def m_jix(pts):
    """Elimination: joint-index sojourn probabilities; exact reference =
    exact_grad=True (the sojourn adjoint)."""
    def make():
        g, idx = _jpg()
        jpg = g.joint_prob_graph(idx, reward_limit=3, mutation_rate=1e-2, discrete=False)
        jd = g.joint_prob_graph(idx, reward_limit=3, mutation_rate=1e-2)
        jd.update_weights([1e-4, 1e-2])
        tab = jd.joint_prob_table()
        obs = tab.iloc[:10][tab.columns[:-1]].to_numpy().tolist()
        oi = jnp.asarray(jpg._map_joint_observations_to_indices(obs))
        m_fd = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2)
        m_ex = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2, exact_grad=True)
        out = []
        for regime, th in pts:
            F = (lambda x: m_fd(x, oi)[0])
            cot = jnp.asarray(1.0 / np.asarray(F(jnp.asarray(th))))
            ex = (lambda x, cot=cot: jax.grad(lambda z: jnp.sum(cot * m_ex(z, oi)[0]))(x))
            out.append((regime, th, F, ex, "loglik"))
        return out
    return make


def m_epoch(pts, granularity=0):
    """Multi-epoch daisy chain (2 epochs, 2 slots each); exact reference for
    the FINAL epoch's slots from exact_final_grad=True; Richardson for the
    earlier epoch."""
    def make():
        g, idx = _jpg()
        jpg = g.joint_prob_graph(idx, reward_limit=3, mutation_rate=1e-2, discrete=False)
        jd = g.joint_prob_graph(idx, reward_limit=3, mutation_rate=1e-2)
        jd.update_weights([1e-4, 1e-2])
        tab = jd.joint_prob_table()
        obs = tab.iloc[:10][tab.columns[:-1]].to_numpy().tolist()
        oi = jpg._map_joint_observations_to_indices(obs)
        out = []
        for regime, th in pts:
            dt = 0.5 / min(th)
            kw = dict(observed_indices=oi, epoch_starts=np.array([0.0, dt]),
                      user_prior=None, user_fixed=None, granularity=granularity)
            m, _, _, _ = jpg._daisy_chain_svgd_model(**kw)
            F = (lambda x, m=m: m(x, None)[0])
            ex = None
            if granularity == 0:
                me, _, _, _ = jpg._daisy_chain_svgd_model(exact_final_grad=True, **kw)
                cot = jnp.asarray(1.0 / np.asarray(F(jnp.asarray(th))))
                ex = (lambda x, me=me, cot=cot: jax.grad(
                    lambda z: jnp.sum(cot * me(z, None)[0]))(x))
            out.append((regime, th, F, ex, "loglik"))
        return out
    return make


def m_dph(n, pts):
    """Native discrete (DPH): the two-island model read as per-generation
    probabilities (coalescence C(k,2)*t0, migration k*t1)."""
    def make():
        mod = _nmr()
        g = mod.two_island_graph(n)
        m = Graph.pmf_from_graph(g, discrete=True, theta_dim=2)
        out = []
        for regime, th in pts:
            jumps = jnp.asarray(np.unique(np.round(np.geomspace(0.3, 6.0, 6) / max(th))))
            out.append((regime, th, (lambda x, j=jumps: m(x, j)), None, "loglik"))
        return out
    return make


def m_wasdph(n, pts):
    """Discretized (was_dph): coalescent.discretize(1.0); slot 1 is the
    discretize (mutation-rate) slot."""
    def make():
        mod = _nmr()
        g = mod.coalescent_graph(n).discretize(1.0)
        m = Graph.pmf_from_graph(g, discrete=True, theta_dim=2)
        # jumps where the pmf has mass (odd counts are structurally zero)
        jumps = jnp.asarray([8.0, 10.0, 12.0, 16.0, 20.0, 26.0])
        out = []
        for regime, th in pts:
            out.append((regime, th, (lambda x: m(x, jumps)), None, "loglik"))
        return out
    return make


def _nmr_b(name):
    return lambda n: getattr(_nmr(), name)(n)


ORD1, PG1 = [("ordinary", [7.0]), ("ordinary", [1.0])], [("popgen", [1e-4]), ("popgen", [1e-5])]
ORD2 = [("ordinary", [0.7, 0.3]), ("ordinary", [1.0, 2.0])]
PG2 = [("popgen", [1e-4, 3e-5]), ("popgen", [1e-5, 1e-4])]
ORD3 = [("ordinary", [1.0, 0.2, 0.05])]
PG3 = [("popgen", [1e-4, 2e-5, 5e-5])]
MODELS = {
    "euler_coal":     ("euler", m_euler(_nmr_b("coalescent_graph"), 5, 1, ORD1 + PG1)),
    "euler_island":   ("euler", m_euler(_nmr_b("two_island_graph"), 3, 2, ORD2 + PG2)),
    "euler_split":    ("euler", m_euler(_nmr_b("island_split_populations"), 1, 3, ORD3 + PG3)),
    "euler_reward":   ("euler", m_reward_euler(5, ORD1 + PG1)),
    "elim_moments_coal":   ("elimination", m_moments(_nmr_b("coalescent_graph"), 10, 1, ORD1 + PG1)),
    "elim_moments_island": ("elimination", m_moments(_nmr_b("two_island_graph"), 4, 2, ORD2 + PG2)),
    "elim_moments_split":  ("elimination", m_moments(_nmr_b("island_split_populations"), 2, 3, ORD3 + PG3)),
    "elim_jix":       ("elimination", m_jix([("ordinary", [1.0, 0.5]), ("ordinary", [0.3, 2.0]),
                                             ("popgen", [1e-4, 1e-2]), ("popgen", [1e-5, 1e-4])])),
    "epoch":          ("epoch", m_epoch([("ordinary", [1.0, 0.5, 1.5, 0.3]),
                                         ("ordinary", [0.3, 2.0, 0.6, 1.0]),
                                         ("popgen", [1e-4, 1e-4, 1e-4, 1e-4]),
                                         ("popgen", [1e-5, 1e-4, 3e-5, 1e-4])])),
    "epoch_gran":     ("epoch_granularity", m_epoch([("ordinary", [1.0, 0.5, 1.5, 0.3]),
                                                     ("ordinary", [0.3, 2.0, 0.6, 1.0])],
                                                    granularity=200)),
    "dph_island":     ("native_discrete", m_dph(3, [("ordinary", [0.05, 0.02]), ("ordinary", [0.02, 0.1]),
                                                     ("popgen", [1e-4, 3e-5]), ("popgen", [1e-5, 1e-4])])),
    "wasdph_coal":    ("discretized", m_wasdph(5, [("ordinary", [1.0, 0.5]), ("ordinary", [0.3, 2.0]),
                                                   ("popgen", [1e-4, 1e-4]), ("popgen", [1e-5, 3e-5])])),
}


# ------------------------------------------------------------------ rules
def L_of(F, cot):
    def L(th):
        return float(np.nansum(np.asarray(cot) * np.asarray(F(jnp.asarray(th)), dtype=np.float64)))
    return L


def d_today(L, th, i):
    tp = list(th); tp[i] += 1e-7
    tm = list(th); tm[i] -= 1e-7
    return (L(tp) - L(tm)) / 2e-7


def d_pos(L, th, i, s, cap):
    t = th[i]
    u = min(np.log1p(1e-7 / t), s) if cap else s
    tp = list(th); tp[i] = t * np.exp(u)
    tm = list(th); tm[i] = t * np.exp(-u)
    d = tp[i] - tm[i]
    return (L(tp) - L(tm)) / d if d != 0 else 0.0


def d_unit(L, th, i, s, cap):
    t = th[i]
    z = np.log(t / (1 - t))
    h = min(s, 1e-7 / (t * (1 - t))) / (1 + s) if cap else s
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    tp = list(th); tp[i] = sig(z + h)
    tm = list(th); tm[i] = sig(z - h)
    d = tp[i] - tm[i]
    return (L(tp) - L(tm)) / d if d != 0 else 0.0


def richardson(L, th, i, s=1e-2):
    def d(ss):
        t = th[i]
        tp = list(th); tp[i] = t * np.exp(ss)
        tm = list(th); tm[i] = t * np.exp(-ss)
        return (L(tp) - L(tm)) / (2 * ss) / t
    a, b, c = d(s), d(s / 2), d(s / 4)
    r1 = (4 * b - a) / 3; r2 = (4 * c - b) / 3
    r = (16 * r2 - r1) / 15
    return r, abs(r2 - r1) / max(abs(r), 1e-300)


def main(name):
    family, make = MODELS[name]
    out_path = os.path.join(HERE, f"s1_{name}.jsonl")
    fh = open(out_path, "w")
    t0 = time.time()
    for regime, th, F, ex, kind in make():
        y = np.asarray(F(jnp.asarray(th)), dtype=np.float64)
        # Entries below 1e-12 of the largest are structural zeros carrying
        # rounding noise (a5_wasdph_scan.log); 1/y would amplify that noise.
        keep = np.isfinite(y) & (np.abs(y) > 1e-12 * np.nanmax(np.abs(y)))
        cot = np.where(keep, 1.0 / np.where(keep, y, 1.0), 0.0)
        L = L_of(F, cot)
        exact = np.asarray(ex(jnp.asarray(th))) if ex is not None else None
        for i in range(len(th)):
            rr, spread = richardson(L, th, i)
            row = dict(model=name, family=family, regime=regime, theta=th, slot=i,
                       richardson=rr, spread=spread)
            ref = rr
            if exact is not None:
                row["exact"] = float(exact[i])
                row["exact_vs_rich"] = rel(rr, float(exact[i]))
                # epoch: only the final epoch's slots are exact (the rest FD)
                if family != "epoch" or i >= len(th) // 2:
                    ref = float(exact[i])
                    row["ref"] = "exact"
            row.setdefault("ref", "richardson")
            row["refval"] = ref
            row["today"] = rel(d_today(L, th, i), ref)
            for s in S:
                row[f"full_{s:g}"] = rel(d_pos(L, th, i, s, False), ref)
                row[f"cap_{s:g}"] = rel(d_pos(L, th, i, s, True), ref)
                if family == "native_discrete":
                    row[f"ufull_{s:g}"] = rel(d_unit(L, th, i, s, False), ref)
                    row[f"ucap_{s:g}"] = rel(d_unit(L, th, i, s, True), ref)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(name, regime, th, "slot", i, f"ref={ref:.6e} ({row['ref']}) spread={spread:.1e}",
                  "today=%.1e" % row["today"],
                  " ".join(f"{s:g}:{row[f'full_{s:g}']:.1e}/{row[f'cap_{s:g}']:.1e}" for s in S),
                  f"[{time.time()-t0:.0f}s]", flush=True)
    fh.close()


if __name__ == "__main__":
    main(sys.argv[1])
