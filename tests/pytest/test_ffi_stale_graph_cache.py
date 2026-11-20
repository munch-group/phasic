"""FFI per-thread graph caches must not serve a destroyed builder's graph (T1).

The batched sojourn / daisy-chain FFI handlers keep, per OpenMP worker
thread, a Graph built from each GraphBuilder, keyed by the builder's
ADDRESS. Builders live in the calling thread's builder_cache and are
destroyed when that thread exits, while OpenMP workers are long-lived,
so their entries outlive the builder. A later builder allocated at the
same address used to hit the old entry and run a different model on the
previous model's graph: "Invalid vertex index" errors from the C layer
and NaN results, with no exception. Measured on master (macOS): about
10-50 wrong results per 3000 short-lived threads alternating two models
(13-50 depending on OMP_NUM_THREADS), 0 with the fix (entries store the
builder's instance id and are rebuilt on mismatch). With one OpenMP
thread the hazard cannot occur, so the subprocess pins
OMP_NUM_THREADS=4. Detection on Linux has not been measured. See
joint-prob-followups-plan.md, ticket T1.

The workload runs in a subprocess so its many short-lived threads (and
their FFI builders) do not interact with the rest of the session.
"""
import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent('''
    import sys, threading
    import numpy as np, jax, jax.numpy as jnp
    sys.path.insert(0, sys.argv[1])
    from phasic import Graph
    from test_notebook_model_recovery import coalescent_indexed

    def build(n):
        g, idx = coalescent_indexed(n)
        j = g.joint_prob_graph(idx, discrete=False, reward_limit=2)
        term = jnp.array(sorted({v.index() for v in j.vertices() if v.edges()
                                 and any(len(e.to().edges()) == 0 for e in v.edges())}),
                         dtype=jnp.int32)
        m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
        return jax.jit(jax.vmap(lambda t: m(t, term)[0]))

    TH = jnp.array([[1.0 + 0.1 * k, 0.5 + 0.05 * k] for k in range(8)])
    f = {"A": build(3), "B": build(4)}          # different structures
    ref = {k: np.asarray(fn(TH)) for k, fn in f.items()}
    bad = 0
    for i in range(int(sys.argv[2])):
        k = "AB"[i % 2]
        out = {}
        t = threading.Thread(target=lambda: out.__setitem__("r", np.asarray(f[k](TH))))
        t.start(); t.join()
        if not np.allclose(out["r"], ref[k], rtol=1e-12):
            bad += 1
    print("BAD", bad, flush=True)
''')


def test_new_builder_at_reused_address_gets_its_own_graph(tmp_path):
    script = tmp_path / "stale_cache.py"
    script.write_text(_SCRIPT)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run([sys.executable, str(script), tests_dir, "3000"],
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "OMP_NUM_THREADS": "4"})
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BAD")]
    assert lines, proc.stdout[-2000:]
    n_bad = int(lines[-1].split()[1])
    assert n_bad == 0, (
        f"{n_bad} of 3000 calls returned another model's results "
        f"(stale per-thread graph); stderr tail:\n{proc.stderr[-1500:]}")
