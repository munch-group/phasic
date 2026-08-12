"""
Batch D Tier-1 / D.1: `Graph.moments_from_graph` under `jax.vmap`.

Regression tests for the vmap crash (atlas finding; master plan §6-D.1): the
ctypes callback `_compute_moments_pure` had no ndim handling, so under
`vmap` (`vmap_method='expand_dims'`) a 2-D theta batch hit
`lib.compute_moments` with `len(theta_np)` == batch size — an
execution-confirmed RuntimeError. The fix loops rows and returns
(B, nr_moments); ndim > 2 (nested vmap) raises a clear error.
"""

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")


# moments_from_graph JIT-compiles C++ from the source tree (same constraint
# and skip-guard as test_weight_mode_probe_and_guards.py).
_pkg_dir = phasic._get_package_dir()
_HAS_SOURCES = (_pkg_dir / "src" / "cpp" / "phasiccpp.cpp").exists()
requires_sources = pytest.mark.skipif(
    not _HAS_SOURCES,
    reason=(
        f"phasic C/C++ sources not on disk (package root: {_pkg_dir}); "
        "JIT compilation requires an editable install or PHASIC_SOURCE_DIR."
    ),
)

# s -> v3 -[2,3]-> v2 -[1,2]-> v1(absorbing); linear weights.
# theta=[1,2]: rates (8, 5); T ~ Hypoexp(8, 5):
#   E[T]   = 1/8 + 1/5 = 0.325
#   E[T^2] = 2*(1/64 + 1/40 + 1/25) = 0.16125
THETA = np.array([1.0, 2.0])
E_T = 1 / 8 + 1 / 5
E_T2 = 2 * (1 / 64 + 1 / 40 + 1 / 25)


def _param_graph():
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [2.0, 3.0])
    v2.add_edge(v1, [1.0, 2.0])
    return g


@pytest.fixture(scope="module")
def moments_fn():
    return Graph.moments_from_graph(_param_graph(), nr_moments=2)


@requires_sources
class TestMomentsFromGraphVmap:

    def test_non_vmap_values_unchanged(self, moments_fn):
        m = np.asarray(moments_fn(jnp.asarray(THETA)))
        assert m.shape == (2,)
        assert m[0] == pytest.approx(E_T, rel=1e-12)
        assert m[1] == pytest.approx(E_T2, rel=1e-12)

    @pytest.mark.parametrize("batch", [2, 5])
    def test_vmap_forward_matches_per_row(self, moments_fn, batch):
        rng = np.random.default_rng(0)
        thetas = jnp.asarray(rng.uniform(0.5, 3.0, size=(batch, 2)))
        got = np.asarray(jax.vmap(moments_fn)(thetas))
        want = np.stack([np.asarray(moments_fn(t)) for t in thetas])
        assert got.shape == (batch, 2)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert np.all(np.isfinite(got))

    def test_vmap_grad_matches_per_row(self, moments_fn):
        # Exercises the FD backward under vmap (master-plan risk 11): every
        # probe routes through the same fixed callback, so per-row and
        # batched results must agree to fp determinism.
        f = lambda th: moments_fn(th)[0]
        rng = np.random.default_rng(1)
        thetas = jnp.asarray(rng.uniform(0.5, 3.0, size=(3, 2)))
        got = np.asarray(jax.vmap(jax.grad(f))(thetas))
        want = np.stack([np.asarray(jax.grad(f)(t)) for t in thetas])
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_nested_vmap_raises_clearly(self, moments_fn):
        thetas = jnp.asarray(np.full((2, 3, 2), 1.5))
        with pytest.raises(Exception, match="only 1-D theta"):
            np.asarray(jax.vmap(jax.vmap(moments_fn))(thetas))
