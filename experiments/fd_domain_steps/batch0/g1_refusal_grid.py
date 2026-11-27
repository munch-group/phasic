"""Plan A Batch 0, refusal grid part 1: the single-graph FD sites.

Axes (plan §6 Batch 0): builder {graph-based, pmf_from_cpp} x rewards
{none, 1-D, 2-D} x weight mode {linear, log, formula (lin, sqrt, division),
callback (lin, sqrt)} x discreteness {continuous, native DPH, was_dph} x
slot value {1, 5e-8, 0, 1-5e-8}.

For every (site, cell) three things are recorded on master:
  P  the FORWARD at the primal theta (slot = v), v in VALS;
  N  the FORWARD at a negative probe (slot = p), p in PROBES  -> the model
     property of §4.4 ("does this forward refuse a negative probe?");
  G  today's jax.grad at slot = v (today's FD loop, incl. the 1e-9 clamps).
The refactor plan's phase-1 matrix uses PROBES; Plan A's grid uses VALS.

Prediction checked (§4.4): a model refuses iff multi-epoch, discrete
(native or was_dph), weight mode in {log, formula, callback}, built by
pmf_from_cpp, or called with rewards.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403

VALS = [(1.0, "1"), (5e-8, "5e-8"), (0.0, "0"), (1.0 - 5e-8, "1-5e-8")]
PROBES = [(-1e-12, "-1e-12"), (-1e-9, "-1e-9"), (-5e-8, "-5e-8"), (-1e-7, "-1e-7")]
TIMES = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
JUMPS = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)

# Coefficient patterns: 'sep' puts the test slot alone on its edge (a pure
# linear exit), 'mix' gives every edge both slots (needed by log mode, whose
# product runs over ALL coefficients).
SEP = ([1.0, 0.0], [0.0, 1.0])
MIX = ([1.0, 1.0], [2.0, 1.0])
MODES = {
    "linear":        (SEP, None),
    "log":           (MIX, "log"),
    "formula_lin":   (SEP, "c0*t0 + c1*t1"),
    "formula_div":   (MIX, "c0*t0/(c1*t1)"),
    "formula_sqrt":  (MIX, "c0*t0*sqrt(c1*t1)"),
    "callback_lin":  (SEP, lambda t, c: c[0] * t[0] + c[1] * t[1]),
    "callback_sqrt": (MIX, lambda t, c: c[0] * t[0] * jnp.sqrt(c[1] * t[1])),
}


def build(mode, kind):
    (cab, cbc), wm = MODES[mode]
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, list(cab))
    b.add_edge(c, list(cbc))
    if wm == "log":
        g.weight_mode = "log"
    elif isinstance(wm, str):
        g.weight_formula = wm
    elif callable(wm):
        g.weight_callback = wm
    if kind == "wasdph":
        g = g.discretize(1.0)
    return g


def base_theta(kind, P):
    th = [0.5 if kind != "cont" else 1.0] * P
    if kind == "wasdph":
        th[-1] = 1.0  # discretize rate slot
    return th


def rewards_for(g):
    nv = g.vertices_length()
    r = np.zeros(nv)
    r[1 % nv] = 1.0
    r[2 % nv] = 2.0
    return jnp.asarray(r)


def sites_for(mode, kind, g, P):
    """{site_name: forward(theta) -> pytree}, built lazily; a build failure is
    recorded as that site's outcome."""
    disc = kind != "cont"
    rw = rewards_for(g)
    rw2 = jnp.stack([rw, 2 * rw])
    S = {}

    def pfg():
        m = Graph.pmf_from_graph(g, discrete=disc, theta_dim=P)
        return lambda th: m(th, JUMPS if disc else TIMES)
    S["pfg"] = pfg

    def pam(ffi, with_rw):
        def mk():
            m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=disc,
                                                 theta_dim=P, use_ffi=ffi)
            kw = {"rewards": rw} if with_rw else {}
            return lambda th: m(th, JUMPS if disc else TIMES, **kw)
        return mk
    S["pam_pybind"] = pam(False, False)
    S["pam_ffi"] = pam(True, False)
    S["pam_pybind_rw1d"] = pam(False, True)

    def cz():
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=disc, theta_dim=P)
        f = getattr(m, "_cdf_zero_fn", None)
        if f is None:
            raise RuntimeError("no _cdf_zero_fn")
        return lambda th: f(th, rw)
    S["cdf_zero"] = cz

    def mv():
        m = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, discrete=disc,
                                                          theta_dim=P)
        return lambda th: m(th, JUMPS if disc else TIMES, rewards=rw2)
    S["multivar_rw2d"] = mv

    def mfg():
        m = Graph.moments_from_graph(g, nr_moments=2)
        return lambda th: m(th)
    if not disc:
        S["moments_from_graph"] = mfg

    def rvp():
        return lambda th: g.reward_visit_probability(rw, theta=th)
    S["rvp"] = rvp
    return S


def scalar_loss(f):
    def L(th):
        leaves = jax.tree_util.tree_leaves(f(th))
        return sum(jnp.nansum(x) * (1.0 if i == 0 else 1e-3) for i, x in enumerate(leaves))
    return L


def run_cell(f, base, slot):
    rec = {}
    for v, lab in VALS:
        th = list(base); th[slot] = v
        rec["P" + lab] = outcome(f, jnp.asarray(th))
        rec["G" + lab] = outcome(jax.grad(scalar_loss(f)), jnp.asarray(th))
    for p, lab in PROBES:
        th = list(base); th[slot] = p
        rec["N" + lab] = outcome(f, jnp.asarray(th))
    return rec


def predicted(mode, kind, site, cpp=False):
    # §4.4: multi-epoch (not on these sites), discrete (native or was_dph),
    # weight mode log/formula/callback, pmf_from_cpp, or called with rewards.
    rw = site in ("pam_pybind_rw1d", "cdf_zero", "multivar_rw2d")
    return bool(cpp or kind != "cont" or mode != "linear" or rw)


def emit(rec, t0=None):
    with open(os.path.join(HERE, "g1_refusal_grid.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), "" if t0 is None else f"[{time.time()-t0:.1f}s]", flush=True)


def main(kind, mode):
    """One (kind, mode) per process: compilation caches grow across models."""
    if mode == "cpp":
        disc = kind == "dph"
        try:
            m = Graph.pmf_from_cpp(os.path.join(HERE, "m_cpp.cpp"), discrete=disc)
            f = lambda th: m(th, JUMPS if disc else TIMES)
            built = "ok"
        except Exception as e:  # noqa: BLE001
            f, built = None, tag(e)
        rec = dict(kind=kind, mode="cpp", site="pmf_from_cpp", slot=1, build=built)
        if f is not None:
            rec.update(run_cell(f, [0.5 if disc else 1.0] * 2, 1))
        rec["pred_refuse"] = True
        emit(rec)
        return
    try:
        g = build(mode, kind)
    except Exception as e:  # noqa: BLE001
        emit(dict(kind=kind, mode=mode, site="(build graph)", slot=-1, build=tag(e)))
        return
    P = 3 if kind == "wasdph" else 2
    base = base_theta(kind, P)
    slots = [1, 2] if kind == "wasdph" else [1]
    only = os.environ.get("SITES")
    for site, mk in sites_for(mode, kind, g, P).items():
        if only and site not in only.split(","):
            continue
        t0 = time.time()
        try:
            f = mk(); built = "ok"
        except Exception as e:  # noqa: BLE001
            f, built = None, tag(e)
        for slot in slots:
            rec = dict(kind=kind, mode=mode, site=site, slot=slot, build=built)
            if f is not None:
                rec.update(run_cell(f, base, slot))
            rec["pred_refuse"] = predicted(mode, kind, site)
            emit(rec, t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
