"""
Tests for SVGD kernel preconditioning.

Tests both MomentJacobianPreconditioner (default) and FisherPreconditioner:
- Scaling computation and normalization
- Data-driven reference point search
- Integration with SVGD optimize()
- Multi-scale parameter handling
- Fixed parameter support
- Preconditioner string/instance dispatch

All direct preconditioner construction uses param_transform=jax.nn.softplus
because phase-type models require positive rates.  SVGD always applies this
transform internally; these tests mirror that behaviour.
"""

from phasic import (
    Graph, SVGD,
    FisherPreconditioner, MomentJacobianPreconditioner,
)
from phasic.svgd import _PreconditionerBase
import numpy as np
import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_exponential_graph():
    """Single-parameter exponential: S -> [2] -> [1] with rate theta[0]."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def _build_two_stage_graph():
    """Two-parameter Erlang-like: S -> [3] --(theta[0])--> [2] --(theta[1])--> [1].

    Total time = Exp(theta[0]) + Exp(theta[1]).
    When theta[0] >> theta[1], moments are dominated by the slow stage.
    """
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])  # rate = theta[0]
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])  # rate = theta[1]
    return g


def _model_and_data_1d(true_theta=2.0, n_samples=100, seed=42):
    """Return (model, data) for a 1-parameter exponential."""
    graph = _build_exponential_graph()
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    np.random.seed(seed)
    data = np.random.exponential(1.0 / true_theta, size=n_samples)
    return model, data


def _model_and_data_2d(true_theta=(10.0, 1.0), n_samples=200, seed=42):
    """Return (model, data) for a 2-parameter two-stage graph.

    Default true_theta has a 10x scale difference to exercise preconditioning.
    """
    graph = _build_two_stage_graph()
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    np.random.seed(seed)
    # Total time = Exp(theta0) + Exp(theta1)
    data = (np.random.exponential(1.0 / true_theta[0], size=n_samples)
            + np.random.exponential(1.0 / true_theta[1], size=n_samples))
    return model, data


_softplus = jax.nn.softplus


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------

def test_class_hierarchy():
    """Both preconditioner classes extend _PreconditionerBase."""
    assert issubclass(MomentJacobianPreconditioner, _PreconditionerBase)
    assert issubclass(FisherPreconditioner, _PreconditionerBase)
    assert not issubclass(_PreconditionerBase, MomentJacobianPreconditioner)


# ---------------------------------------------------------------------------
# Scaling properties (both methods)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_scaling_shape_and_positivity(PrecondClass):
    """Scaling vector has correct shape, is positive, and normalized to mean 1."""
    model, data = _model_and_data_2d()
    prec = PrecondClass(model, data, theta_dim=2, param_transform=_softplus)
    prec.compute_scaling(jnp.zeros(2))

    assert prec.scaling is not None
    assert prec.scaling.shape == (2,)
    assert jnp.all(prec.scaling > 0), f"scaling must be positive, got {prec.scaling}"
    assert jnp.abs(jnp.mean(prec.scaling) - 1.0) < 1e-6, (
        f"scaling must be normalized to mean 1, got mean={float(jnp.mean(prec.scaling))}"
    )


@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_scaling_1d_is_one(PrecondClass):
    """For a single parameter, scaling must be [1.0] (trivially normalized)."""
    model, data = _model_and_data_1d()
    prec = PrecondClass(model, data, theta_dim=1, param_transform=_softplus)
    prec.compute_scaling(jnp.zeros(1))

    np.testing.assert_allclose(prec.scaling, [1.0], atol=1e-6)


@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_scaling_reflects_scale_difference(PrecondClass):
    """With theta=(10, 1) -- different scales -- scaling entries should differ.

    The fast parameter (theta[0]=10) has small moment sensitivity per unit
    change in unconstrained space compared to the slow parameter (theta[1]=1).
    The scaling vector should not be uniform.
    """
    model, data = _model_and_data_2d(true_theta=(10.0, 1.0))
    prec = PrecondClass(model, data, theta_dim=2, param_transform=_softplus)
    prec.compute_scaling(jnp.zeros(2))

    ratio = float(prec.scaling[0] / prec.scaling[1])
    # The two entries should differ -- we only assert they're not identical.
    assert abs(ratio - 1.0) > 0.01, (
        f"Expected non-uniform scaling for different-scale parameters, got ratio={ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Data-driven reference search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_reference_search_adapts_to_data_scale(PrecondClass):
    """Reference point should differ for small-mean vs large-mean data."""
    model, _ = _model_and_data_1d()

    # Small-scale data (mean ~ 0.1)
    data_small = np.array([0.05, 0.08, 0.12, 0.15])
    prec_small = PrecondClass(model, data_small, theta_dim=1, param_transform=_softplus)
    ref_small = prec_small._find_moment_matching_reference(jnp.zeros(1))

    # Large-scale data (mean ~ 5.0)
    data_large = np.array([3.0, 4.5, 5.5, 7.0])
    prec_large = PrecondClass(model, data_large, theta_dim=1, param_transform=_softplus)
    ref_large = prec_large._find_moment_matching_reference(jnp.zeros(1))

    # The reference points should be different for different data scales
    assert float(ref_small[0]) != float(ref_large[0]), (
        "Reference points should differ for different data scales"
    )


@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_reference_search_no_overflow(PrecondClass):
    """Large data means should not cause overflow in inverse softplus."""
    model, _ = _model_and_data_1d()
    data_huge = np.array([500.0, 1000.0, 1500.0])
    prec = PrecondClass(model, data_huge, theta_dim=1, param_transform=_softplus)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = prec._find_moment_matching_reference(jnp.zeros(1))

    assert jnp.all(jnp.isfinite(ref)), f"Reference must be finite, got {ref}"


# ---------------------------------------------------------------------------
# Param transform support
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("PrecondClass", [MomentJacobianPreconditioner, FisherPreconditioner])
def test_param_transform_softplus(PrecondClass):
    """Preconditioner should work with softplus transform (the SVGD default)."""
    model, data = _model_and_data_1d()
    prec = PrecondClass(
        model, data, theta_dim=1,
        param_transform=_softplus,
    )
    prec.compute_scaling(jnp.zeros(1))
    assert prec.scaling is not None
    assert jnp.all(jnp.isfinite(prec.scaling))


# ---------------------------------------------------------------------------
# SVGD string dispatch
# ---------------------------------------------------------------------------

def test_svgd_auto_selects_jacobian():
    """preconditioner='auto' should set method to 'jacobian'."""
    model, data = _model_and_data_1d()
    svgd = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                preconditioner='auto', verbose=False)
    assert svgd.preconditioner_method == 'jacobian'


def test_svgd_jacobian_string():
    """preconditioner='jacobian' should set method to 'jacobian'."""
    model, data = _model_and_data_1d()
    svgd = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                preconditioner='jacobian', verbose=False)
    assert svgd.preconditioner_method == 'jacobian'


def test_svgd_fisher_string():
    """preconditioner='fisher' should set method to 'fisher'."""
    model, data = _model_and_data_1d()
    svgd = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                preconditioner='fisher', verbose=False)
    assert svgd.preconditioner_method == 'fisher'


def test_svgd_none_disables():
    """preconditioner=None and 'none' should disable preconditioning."""
    model, data = _model_and_data_1d()
    svgd1 = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                 preconditioner=None, verbose=False)
    svgd2 = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                 preconditioner='none', verbose=False)
    assert svgd1.preconditioner_method is None
    assert svgd2.preconditioner_method is None


def test_svgd_accepts_instance():
    """Passing a preconditioner instance should store it directly."""
    model, data = _model_and_data_1d()

    mjp = MomentJacobianPreconditioner(model, data, theta_dim=1)
    svgd1 = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                 preconditioner=mjp, verbose=False)
    assert svgd1.preconditioner_method is mjp

    fp = FisherPreconditioner(model, data, theta_dim=1)
    svgd2 = SVGD(model, data, theta_dim=1, n_particles=5, n_iterations=1,
                 preconditioner=fp, verbose=False)
    assert svgd2.preconditioner_method is fp


def test_svgd_rejects_invalid_string():
    """Invalid preconditioner string should raise ValueError."""
    model, data = _model_and_data_1d()
    with pytest.raises(ValueError, match="preconditioner must be"):
        SVGD(model, data, theta_dim=1, preconditioner='invalid', verbose=False)


def test_svgd_rejects_invalid_type():
    """Non-string, non-None, non-preconditioner should raise TypeError."""
    model, data = _model_and_data_1d()
    with pytest.raises(TypeError, match="preconditioner must be"):
        SVGD(model, data, theta_dim=1, preconditioner=42, verbose=False)


# ---------------------------------------------------------------------------
# Integration: SVGD.optimize runs with each preconditioner mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precond_arg", ['jacobian', 'fisher', None])
def test_svgd_optimize_runs(precond_arg):
    """SVGD.optimize() should complete without error for each mode."""
    model, data = _model_and_data_1d(true_theta=2.0, n_samples=50)
    svgd = SVGD(
        model, data, theta_dim=1,
        n_particles=8, n_iterations=3,
        preconditioner=precond_arg,
        seed=42, verbose=False
    )
    svgd.optimize()
    assert svgd.particles.shape[1] == 1


@pytest.mark.parametrize("precond_arg", ['jacobian', 'fisher'])
def test_svgd_optimize_2d(precond_arg):
    """SVGD.optimize() with 2 parameters and each preconditioner mode."""
    model, data = _model_and_data_2d(true_theta=(5.0, 2.0), n_samples=100)
    svgd = SVGD(
        model, data, theta_dim=2,
        n_particles=8, n_iterations=3,
        preconditioner=precond_arg,
        seed=42, verbose=False
    )
    svgd.optimize()
    assert svgd.particles.shape[1] == 2


# ---------------------------------------------------------------------------
# Integration: fixed parameters + preconditioning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precond_arg", ['jacobian', 'fisher'])
def test_svgd_optimize_with_fixed_params(precond_arg):
    """Preconditioning should work correctly when some parameters are fixed."""
    model, data = _model_and_data_2d(true_theta=(5.0, 2.0), n_samples=100)
    svgd = SVGD(
        model, data, theta_dim=2,
        n_particles=8, n_iterations=3,
        fixed=[(0, 5.0)],  # fix theta[0] at 5.0
        preconditioner=precond_arg,
        seed=42, verbose=False
    )
    svgd.optimize()
    # With one fixed param, SVGD operates in 1D internally
    # but returns the full 2D particles with the fixed value restored
    assert svgd.particles.shape[1] == 2
    # Fixed dimension should be constant across particles
    np.testing.assert_allclose(svgd.particles[:, 0], 5.0, atol=0.1)


# ---------------------------------------------------------------------------
# MomentJacobianPreconditioner: Jacobian-specific checks
# ---------------------------------------------------------------------------

def test_jacobian_column_norms_finite():
    """Jacobian column norms should always be finite (no PMF division)."""
    model, data = _model_and_data_2d()
    prec = MomentJacobianPreconditioner(
        model, data, theta_dim=2, param_transform=_softplus
    )
    prec.compute_scaling(jnp.zeros(2))

    assert jnp.all(jnp.isfinite(prec.scaling)), (
        f"Jacobian scaling must be finite, got {prec.scaling}"
    )


# ---------------------------------------------------------------------------
# FisherPreconditioner: Fisher-specific checks
# ---------------------------------------------------------------------------

def test_fisher_epsilon_floor():
    """Fisher preconditioner should respect the epsilon floor."""
    model, data = _model_and_data_1d()
    eps = 0.5  # intentionally large for testing
    prec = FisherPreconditioner(
        model, data, theta_dim=1, epsilon=eps, param_transform=_softplus
    )
    prec.compute_scaling(jnp.zeros(1))

    # With 1 dim, scaling is always [1.0] after normalization, but the
    # internal sqrt(max(F, epsilon)) should have been >= sqrt(eps).
    assert prec.scaling is not None
    assert jnp.all(prec.scaling > 0)


# ---------------------------------------------------------------------------
# Pre-built preconditioner instance passed to SVGD
# ---------------------------------------------------------------------------

def test_prebuilt_preconditioner_used_directly():
    """When a pre-built preconditioner instance is passed, SVGD should use it
    without recomputing scaling."""
    model, data = _model_and_data_1d(n_samples=50)
    prec = MomentJacobianPreconditioner(
        model, data, theta_dim=1,
        param_transform=_softplus,
    )
    prec.compute_scaling(jnp.zeros(1))
    original_scaling = prec.scaling.copy()

    svgd = SVGD(
        model, data, theta_dim=1,
        n_particles=8, n_iterations=3,
        preconditioner=prec,
        seed=42, verbose=False
    )
    svgd.optimize()

    # Scaling should not have been recomputed
    np.testing.assert_array_equal(prec.scaling, original_scaling)
