"""A new vertex is not an aux vertex, whatever was in the memory malloc
returned. ptd_vertex_create_state did not set is_aux, so on Linux (CI) fresh
vertices could read as aux and joint_prob_graph() refused plain graphs.

The check runs in a subprocess with the allocator filling new memory with
non-zero bytes (MallocScribble on macOS, MALLOC_PERTURB_ on glibc), so it
fails without the fix on both platforms.
"""
import os
import subprocess
import sys
import textwrap

_CHILD = textwrap.dedent("""
    from phasic import Graph, with_ipv

    @with_ipv([4, 0, 0, 0])
    def coal(state):
        out = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                out.append((new, [state[i] * (state[j] - same) / (1 + same)]))
        return out

    g = Graph(coal)
    assert not any(v.is_aux() for v in g.vertices()), "plain vertex reads as aux"
    d = g.discretize(0.1)
    n_aux = sum(v.is_aux() for v in d.vertices())
    assert n_aux == len(d.vertices()) - len(g.vertices()), n_aux
    print("OK")
""")


def test_new_vertices_are_not_aux_with_dirty_malloc():
    env = dict(os.environ, MallocScribble="1", MALLOC_PERTURB_="170")
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr[-2000:]
