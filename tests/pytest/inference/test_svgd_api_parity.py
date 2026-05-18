"""Interface parity between :meth:`Graph.svgd` and :class:`SVGD`.

``Graph.svgd`` is the recommended entry point — it constructs the
model, runs the combinational validator, and (for partial-reward
coverage) attaches the zero-inflated likelihood term before
delegating to :class:`SVGD`. Direct ``SVGD(...)`` construction is
also a public API; this module asserts that the two surfaces:

- accept identical defaults on every shared parameter,
- run the same combinational validator (with rules that can fire
  without a graph reference), and
- produce the same posterior to numerical tolerance when called with
  the same kwargs and fixed seed.

The shared-defaults check is structural (reads the function
signatures via :mod:`inspect`); the parity check is end-to-end.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from phasic import Graph, SVGD
from phasic.exceptions import SvgdConfigError


# Parameters that exist on both surfaces. Defaults must agree.
_SHARED_PARAMS = (
    'prior', 'n_particles', 'n_iterations', 'learning_rate',
    'bandwidth', 'theta_init', 'theta_dim', 'seed', 'verbose',
    'progress', 'jit', 'parallel', 'n_devices', 'precompile',
    'compilation_config', 'positive_params', 'param_transform',
    'regularization', 'nr_moments', 'rewards', 'fixed', 'optimizer',
    'preconditioner', 'exposure', 'exposure_param_index',
)


def test_shared_defaults_agree():
    """Defaults on every shared parameter must be identical."""
    g_sig = inspect.signature(Graph.svgd)
    s_sig = inspect.signature(SVGD.__init__)

    mismatches = []
    for name in _SHARED_PARAMS:
        if name not in g_sig.parameters or name not in s_sig.parameters:
            mismatches.append(f"{name}: missing on one surface")
            continue
        g_default = g_sig.parameters[name].default
        s_default = s_sig.parameters[name].default
        if g_default != s_default:
            mismatches.append(
                f"{name}: Graph.svgd={g_default!r} vs SVGD={s_default!r}"
            )
    assert not mismatches, "Default divergence:\n  " + "\n  ".join(mismatches)


def test_validator_runs_on_direct_svgd():
    """Direct ``SVGD(...)`` must run the combinational validator and
    raise :class:`SvgdConfigError` on a known-bad combo. Here:
    ``exposure_param_index`` out of range for ``theta_dim=1``.
    """
    # A dummy model that satisfies the signature contract.
    def dummy_model(theta, times, rewards=None):
        pmf = jnp.ones_like(times)
        moments = jnp.ones(2)
        return pmf, moments

    with pytest.raises(SvgdConfigError, match=r"exposure_param_index"):
        SVGD(
            model=dummy_model,
            observed_data=jnp.array([1.0, 2.0, 3.0]),
            theta_dim=1,
            exposure=jnp.array([1.0, 1.0, 1.0]),
            exposure_param_index=5,  # out of [0, 1)
            n_iterations=1,
            verbose=False,
            progress=False,
        )


def test_graph_svgd_and_direct_svgd_agree():
    """Equal kwargs through both surfaces produce the same posterior.

    Builds a 2-state exponential phase-type model with one rate
    parameter ``θ``, runs SVGD via both ``Graph.svgd`` and direct
    ``SVGD(model=...)`` with identical settings and a fixed seed,
    then asserts the posterior means and stds agree to a tight
    tolerance. Tolerance is loose enough to absorb JAX cache effects
    but tight enough to catch real divergence.
    """
    # 2-state exponential: S -> v2 (init prob 1) -> v1 (absorb), rate theta.
    def build_graph():
        g = Graph(1)
        v_start = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        v_start.add_edge(v2, 1.0)
        v2.add_edge(v1, [1.0])  # parameterised: weight = theta[0]
        return g

    np.random.seed(0)
    g_data = build_graph()
    g_data.update_weights([3.0])
    data = np.array(g_data.sample(80))

    # Pass a concrete prior so both paths share the same one
    # (Graph.svgd defaults to DataPrior, SVGD defaults to standard
    # normal — these would diverge silently otherwise).
    def shared_prior(phi):
        return -0.5 * jnp.sum((phi / 10.0) ** 2)

    kw = dict(
        observed_data=data,
        prior=shared_prior,
        n_particles=20,
        n_iterations=150,
        learning_rate=0.05,
        bandwidth='median_per_dim',
        positive_params=True,
        theta_dim=1,
        seed=42,
        verbose=False,
        progress=False,
        parallel='vmap',
    )

    # Path A: Graph.svgd
    g_a = build_graph()
    res_a = g_a.svgd(**kw)
    mean_a = float(res_a.theta_mean[0])
    std_a = float(res_a.theta_std[0])

    # Path B: SVGD direct
    g_b = build_graph()
    model_b = Graph.pmf_and_moments_from_graph(g_b, nr_moments=2, theta_dim=1)
    svgd_b = SVGD(model=model_b, **kw)
    svgd_b.optimize()
    mean_b = float(svgd_b.theta_mean[0])
    std_b = float(svgd_b.theta_std[0])

    # Tolerance: relative 5%. JAX cache and prior-construction differ
    # marginally between the two paths (Graph.svgd auto-builds a
    # DataPrior; direct SVGD uses the standard-normal default unless
    # prior= is passed), but the kwargs above pin everything else.
    assert abs(mean_a - mean_b) / max(abs(mean_a), 1e-6) < 0.05, (
        f"mean mismatch: Graph.svgd={mean_a:.4f} vs SVGD direct={mean_b:.4f}"
    )
    assert abs(std_a - std_b) / max(abs(std_a), 1e-6) < 0.10, (
        f"std mismatch: Graph.svgd={std_a:.4f} vs SVGD direct={std_b:.4f}"
    )
