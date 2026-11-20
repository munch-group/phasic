"""Thread exit after FFI + direct GraphBuilder use must not crash.

Regression for a use-after-destroy between two C++ thread_local caches:

* ``builder_cache`` (graph_builder_ffi.cpp) holds the GraphBuilder
  instances the FFI handlers create, one cache per OS thread;
* the per-thread persistent-graph map (graph_builder.cpp) is filled by
  the Python-facing ``GraphBuilder.compute_*`` methods, and
  ``GraphBuilder::~GraphBuilder`` erases its entry from it.

thread_local objects are destroyed in reverse construction order at
thread exit. When a thread first runs an FFI handler (cache created) and
then a direct ``compute_pmf`` (map created), the map is destroyed first;
the cache's builders are destroyed afterwards and their destructor
erased from the dead map. libc++ leaves the dead map's element count
intact but nulls its bucket array, so if the map was non-empty (a
Python-held builder still alive) the lookup dereferenced near null and
the process segfaulted. In the test suite this showed up as a segfault
at interpreter exit, when XLA joins its worker threads.

The sequence runs in a subprocess (a crash would take pytest down) on a
short-lived Python thread; on the unfixed build it exits with SIGSEGV
every time ON macOS/libc++ (3/3 in development and in review).

Scope and assumptions:
* Linux/libstdc++: the unfixed code is still undefined behaviour there,
  but libstdc++'s ~_Hashtable clears the element count first, so the
  dead-map erase most likely returns without crashing; this test is a
  reliable guard on macOS only.
* It relies on JAX running the CPU FFI call on the calling Python thread
  (verified with jax 0.7.2 / PjRt CPU), which is what builds builder_cache
  before the persistent-graph map on that thread. If a future JAX moves
  the call to a pool thread, the test still passes but stops
  reproducing the original ordering.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent('''
    import json, sys, threading
    import numpy as np
    import jax.numpy as jnp
    from phasic import Graph, phasic_pybind as cpp
    from phasic.config import get_config
    from phasic.ffi_wrappers import _make_json_serializable

    if not get_config()._use_ffi:
        sys.exit(77)  # FFI unavailable: the sequence cannot be exercised

    # Standard coalescent, n=3, one parameter (coalescence rate).
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [3.0])
    v2.add_edge(v1, [1.0])

    model = Graph.pmf_from_graph(g)          # FFI-backed
    sj = json.dumps(_make_json_serializable(g.serialize()))
    keep = []                                # outlives the thread

    def work():
        p_ffi = np.asarray(model(jnp.array([1.0]), jnp.array([0.5, 1.0])))
        b = cpp.parameterized.GraphBuilder(sj)
        keep.append(b)
        p_direct = np.asarray(b.compute_pmf(np.array([1.0]), np.array([0.5, 1.0]), False, 0))
        assert np.allclose(p_ffi, p_direct), (p_ffi, p_direct)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    print("THREAD_EXIT_OK", flush=True)
''')


def test_thread_exit_after_ffi_then_direct_builder_does_not_crash(tmp_path):
    script = tmp_path / "tl_teardown.py"
    script.write_text(_SCRIPT)
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=300, env=dict(os.environ))
    if proc.returncode == 77:
        pytest.skip("phasic FFI is not available in this build")
    assert proc.returncode == 0, (
        f"subprocess exited with {proc.returncode} (negative = signal; "
        f"-11 = SIGSEGV)\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-4000:]}")
    assert "THREAD_EXIT_OK" in proc.stdout
