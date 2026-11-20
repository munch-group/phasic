"""The native (FFI) model caches are capped and can be cleared (ticket T11).

Every distinct model evaluated through the FFI path used to keep its
builder and one graph per worker thread until the process exited
(measured ~290 MB per 45,704-vertex model). Now at most
``model_memory_cache_size`` models stay cached process-wide (least
recently used dropped first) and ``phasic.clear_model_memory()`` frees
them. Each test runs in a subprocess: the capacity is process-wide.
"""
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

PRELUDE = textwrap.dedent(f"""
    import sys, gc
    sys.path.insert(0, {HERE!r})
    import numpy as np, jax, jax.numpy as jnp
    import phasic
    from phasic import Graph
    from test_notebook_model_recovery import coalescent_indexed

    def model(k, n=4):
        # distinct structure JSON per k (mutation coefficient differs)
        g, idx = coalescent_indexed(n)
        jc = g.joint_prob_graph(idx, reward_limit=2, discrete=False,
                                mutation_rate=1.0 + 0.1 * k)
        return Graph.pmf_from_graph_joint_index(jc, theta_dim=2), jc.vertices_length()

    P = jnp.exp(jnp.linspace(-0.5, 0.5, 32)).reshape(16, 2)   # 16 particles: OpenMP fan-out

    def evaluate(m, nv):
        vi = jnp.arange(min(20, nv), dtype=jnp.int32)
        return np.asarray(jax.vmap(lambda t: m(t, vi)[0])(P))
""")


def _run(body, env_extra=None):
    env = dict(os.environ, OMP_NUM_THREADS="4", JAX_PLATFORMS="cpu")
    env.pop("PHASIC_MODEL_MEMORY_CACHE_SIZE", None)
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
                       capture_output=True, text=True, env=env, timeout=600)
    return r


def test_cap_bounds_models_and_rebuilt_models_give_same_values():
    r = _run("""
        phasic.configure(model_memory_cache_size=2)
        models = [model(k) for k in range(4)]
        first = [evaluate(*mk) for mk in models]
        info = phasic.model_memory_info()
        assert info['capacity'] == 2 and info['models'] == 2, info
        # Every model was evicted at least once by now; evaluating them
        # again rebuilds from the structure JSON and must agree exactly.
        second = [evaluate(*mk) for mk in models]
        for a, b in zip(first, second):
            assert np.array_equal(a, b), (a, b)
            assert np.all(np.isfinite(a))
        info = phasic.model_memory_info()
        assert info['models'] == 2, info
        print("OK", info)
    """)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_per_thread_graphs_of_evicted_models_are_released():
    # Without the sweep, every OpenMP worker kept a graph per model ever seen.
    r = _run("""
        phasic.configure(model_memory_cache_size=1)
        for k in range(6):
            evaluate(*model(k))
        gc.collect()
        info = phasic.model_memory_info()
        # one model cached, at most one graph per thread (4 workers + caller)
        assert info['models'] == 1 and info['cached_graphs'] <= 5, info
        print("OK", info)
    """)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_clear_model_memory():
    r = _run("""
        a = model(0); b = model(1)
        va = evaluate(*a); evaluate(*b)
        assert phasic.model_memory_info()['models'] == 2
        info = phasic.clear_model_memory()
        assert info['models'] == 0, info
        assert np.array_equal(evaluate(*a), va)   # rebuilt after the clear
        info = phasic.model_memory_info()
        assert info['models'] == 1, info
        print("OK", info)
    """)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_configure_validates_and_context_restores():
    r = _run("""
        from phasic.exceptions import PTDConfigError
        for bad in (0, -1, True, 2.5, "3"):
            try:
                phasic.configure(model_memory_cache_size=bad)
            except PTDConfigError:
                pass
            else:
                raise AssertionError(f"accepted {bad!r}")
        phasic.configure(model_memory_cache_size=None)
        evaluate(*model(0))
        assert phasic.model_memory_info()['capacity'] == 4   # default
        with phasic.configure(model_memory_cache_size=3):
            assert phasic.model_memory_info()['capacity'] == 3
        assert phasic.model_memory_info()['capacity'] == 4
        print("OK")
    """)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_start_up_env_value():
    r = _run("""
        evaluate(*model(0))
        assert phasic.model_memory_info()['capacity'] == 7
        print("OK")
    """, {"PHASIC_MODEL_MEMORY_CACHE_SIZE": "7"})
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_bad_env_value_raises():
    r = _run("""
        print("UNREACHED")
    """, {"PHASIC_MODEL_MEMORY_CACHE_SIZE": "lots"})
    assert r.returncode != 0 and "PHASIC_MODEL_MEMORY_CACHE_SIZE" in r.stderr, r.stdout + r.stderr
