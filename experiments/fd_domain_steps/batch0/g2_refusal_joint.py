"""Plan A Batch 0, refusal grid part 2: joint-probability and multi-epoch sites.

These builders require weight_mode='linear' (joint_prob_graph refuses the
others), so the axes are: joint graph {continuous, discrete (was_dph)} x site
x slot value. Same P / N / G record as part 1.

Sites:
  jix_fd        pmf_from_graph_joint_index, FD loop (__init__.py:9888)
  jix_hostfwd   the forward the host fallback (:9552) probes: update_weights
                + expected_sojourn_time on the joint graph
  dc_noexp      _daisy_chain_svgd_model, no exposure (:5306)
  dc_exp        _daisy_chain_svgd_model, exposure (:5647)
  dc_efg        _daisy_chain_svgd_model, exact_final_grad=True (:5344)
  dc_gran       _daisy_chain_svgd_model, explicit granularity=200
  dcjp          JSP.daisy_chain_joint_probs (:12018)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from functools import partial
from itertools import combinations_with_replacement
from phasic import with_ipv, StateIndexer, Property

VALS = [(1.0, "1"), (5e-8, "5e-8"), (0.0, "0"), (1.0 - 5e-8, "1-5e-8")]
PROBES = [(-1e-12, "-1e-12"), (-1e-9, "-1e-9"), (-5e-8, "-5e-8"), (-1e-7, "-1e-7")]

nr = 4
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
all_pairs = partial(combinations_with_replacement, r=2)


@with_ipv([nr] + [0] * (nr - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy(); new[i] -= 1; new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t


graph = Graph(coal, indexer=indexer)
MU = 1e-2


def jpg_of(disc):
    return graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=MU, discrete=disc)


def obs_of(jpg):
    jd =graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=MU)
    jd.update_weights([1.0, MU])
    tab = jd.joint_prob_table()
    obs = tab.iloc[:8][tab.columns[:-1]].to_numpy().tolist()
    return obs, jpg._map_joint_observations_to_indices(obs)


def scalar_loss(f):
    def L(th):
        leaves = jax.tree_util.tree_leaves(f(th))
        return jnp.nansum(leaves[0])
    return L


def run_cell(f, base, slot, grad=True):
    rec = {}
    for v, lab in VALS:
        th = list(base); th[slot] = v
        rec["P" + lab] = outcome(f, jnp.asarray(th))
        if grad:
            rec["G" + lab] = outcome(jax.grad(scalar_loss(f)), jnp.asarray(th))
    for p, lab in PROBES:
        th = list(base); th[slot] = p
        rec["N" + lab] = outcome(f, jnp.asarray(th))
    return rec


def main(disc):
    rows = []
    for disc in (disc,):
        kind = "jpg_disc" if disc else "jpg_cont"
        jpg = jpg_of(disc)
        obs, oi = obs_of(jpg)
        P = jpg.param_length()
        base1 = [1.0, 0.5]
        S = {}

        def jix():
            m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=P)
            return lambda th: m(th, jnp.asarray(oi))
        S["jix_fd"] = (jix, base1, [1], True)

        def hostfwd():
            idx = list(range(jpg.vertices_length()))
            def f(th):
                jpg.update_weights([float(x) for x in np.asarray(th)])
                return np.asarray(jpg.expected_sojourn_time(idx))
            return f
        S["jix_hostfwd"] = (hostfwd, base1, [1], False)

        if not disc:
            base2 = [1.0, 0.5, 1.0, 0.5]
            for name, kw in (("dc_noexp", {}),
                             ("dc_exp", dict(exposure_arr=np.linspace(0.5, 2.0, len(oi)),
                                             exposure_param_index=1)),
                             ("dc_efg", dict(exact_final_grad=True)),
                             ("dc_gran", dict(granularity=200))):
                def dc(kw=kw):
                    m, td, _, _ = jpg._daisy_chain_svgd_model(
                        observed_indices=oi, epoch_starts=np.array([0.0, 0.5]),
                        user_prior=None, user_fixed=None, **kw)
                    return lambda th: m(th, None)
                S[name] = (dc, base2, [1, 3], True)

            def dcjp():
                jsp = jpg.joint_stop_prob_graph()
                n_ipv = jsp.starting_vertex().edges_length()
                def f(th):
                    return jsp.daisy_chain_joint_probs(
                        epoch_thetas=jnp.asarray(th).reshape(2, P), epoch_dts=[0.5],
                        initial_ipv=np.full(n_ipv, 1.0 / n_ipv))
                return f
            S["dcjp"] = (dcjp, base2, [1, 3], True)

        for site, (mk, base, slots, grad) in S.items():
            t0 = time.time()
            try:
                f = mk(); built = "ok"
            except Exception as e:  # noqa: BLE001
                f, built = None, tag(e)
            for slot in slots:
                rec = dict(kind=kind, mode="linear", site=site, slot=slot, build=built)
                if f is not None:
                    rec.update(run_cell(f, base, slot, grad=grad))
                # §4.4: multi-epoch or discrete refuses; single-epoch continuous
                # elimination does not.
                rec["pred_refuse"] = bool(disc or site.startswith("dc"))
                with open(os.path.join(HERE, "g2_refusal_joint.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), f"[{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] == "disc")
