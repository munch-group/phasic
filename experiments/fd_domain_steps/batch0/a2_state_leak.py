"""Is the pybind pmf_and_moments model stateful across calls?
(1) valid theta only: repeated rewards / no-rewards calls must agree bitwise;
(2) after ONE refused call (negative probe with rewards), do later calls at a
    valid theta change?  Fresh model per experiment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from g1_refusal_grid import build, rewards_for, TIMES

g = build("linear", "cont")
rw = rewards_for(g)
th_ok = jnp.asarray([1.0, 0.5])


def call(m, th, with_rw):
    try:
        p, mo = m(th, TIMES, rewards=rw) if with_rw else m(th, TIMES)
        return np.concatenate([np.asarray(p), np.asarray(mo)])
    except Exception as e:  # noqa: BLE001
        return "RAISE " + str(e).replace("\n", " ")[-120:]


for ffi in (False, True):
    print(f"=== use_ffi={ffi}")
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, use_ffi=ffi)
    seq = [call(m, th_ok, False), call(m, th_ok, True), call(m, th_ok, True), call(m, th_ok, False),
           call(m, th_ok, True)]
    for i, v in enumerate(seq):
        print("  (1) valid call", i, v if isinstance(v, str) else np.array2string(v, precision=12))
    nr = [v for i, v in enumerate(seq) if i in (0, 3)]
    wr = [v for i, v in enumerate(seq) if i in (1, 2, 4)]
    print("  (1) no-rewards calls identical:", all(np.array_equal(nr[0], x) for x in nr),
          "| rewards calls identical:", all(np.array_equal(wr[0], x) for x in wr))
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, use_ffi=ffi)
    ref_r = call(m, th_ok, True); ref_n = call(m, th_ok, False)
    bad = call(m, jnp.asarray([1.0, -5e-8]), True)
    print("  (2) refused call:", bad if isinstance(bad, str) else "NOT refused " + str(bad))
    for k in range(4):
        r = call(m, th_ok, True); n = call(m, th_ok, False)
        print(f"  (2) after, round {k}: rewards same as before: "
              f"{(not isinstance(r, str)) and np.array_equal(r, ref_r)}  {'' if not isinstance(r, str) else r}"
              f" | no-rewards same: {(not isinstance(n, str)) and np.array_equal(n, ref_n)} {'' if not isinstance(n, str) else n}")
    if not isinstance(r, str) and not np.array_equal(r, ref_r):
        print("     before:", np.array2string(ref_r, precision=10), "\n     after: ", np.array2string(r, precision=10))
