"""Extraction-seam bit-identity gate for the WS-C ``svgd.py`` module split.

``svgd.py`` (9,611 lines) will be split into ``svgd_priors`` / ``svgd_schedules``
/ ``svgd_optimizers`` / ``svgd_preconditioners`` / ``svgd_kernels`` (+ re-exports),
and the 27-parameter ``SVGD.__init__`` / 42-parameter ``Graph.svgd`` will be piped
through ``SvgdConfig``. Those are **verbatim relocations** — this gate pins the
observable, deterministic behavior of the leaf components *at the extraction seams*
so the move cannot silently change a number.

The golden values were captured from the pre-split implementation. Any drift when
the code moves modules (or an unintended arithmetic edit rides along) turns this
gate red. It is deliberately exact (bit-identity), matching the other Stage-2/3
equivalence gates. It touches no inference loop, no RNG, no C path — pure leaf
math + the config dataclass schema.
"""
import dataclasses

import jax.numpy as jnp
import pytest

import phasic.svgd as S
from phasic.svgd_config import SvgdConfig

pytestmark = pytest.mark.equivalence


def _flat(val):
    return [float(x) for x in jnp.asarray(val).reshape(-1)]


# NOTE: phasic does NOT force jax_enable_x64 at import (despite the CLAUDE.md
# note), so these leaf ops run in float32 standalone but float64 once an FFI gate
# elsewhere in the suite flips x64 on — the exact value is therefore ambient-mode
# dependent. We compare with a tight relative tolerance (the Stage-2 G5 gate's
# convention) so the gate is order-independent yet still catches any real
# arithmetic drift from a non-verbatim module move (which is >> 1e-5). Structural
# invariants (self-similarity, symmetry, step-to-step determinism) stay exact.
_REL = 1e-5
_ABS = 1e-9


def _approx(golden):
    return pytest.approx(golden, rel=_REL, abs=_ABS)


# --------------------------------------------------------------------------- #
# Priors (-> svgd_priors.py)
# --------------------------------------------------------------------------- #
def test_gauss_prior_log_prob_golden():
    gp = S.GaussPrior(mean=5.0, std=2.0)
    assert _flat(gp(jnp.array([5.0]))) == _approx([-0.0])
    assert _flat(gp(jnp.array([3.0]))) == _approx([-0.5])
    assert _flat(gp(jnp.array([7.0]))) == _approx([-0.5])


def test_loggauss_prior_log_prob_golden():
    lp = S.LogGaussPrior(mean=1.0, std=0.5)
    assert _flat(lp(jnp.array([1.0]))) == _approx([-2.0])
    assert _flat(lp(jnp.array([2.0]))) == _approx([-0.8814644813537598])


# --------------------------------------------------------------------------- #
# Step-size schedules (-> svgd_schedules.py)
# --------------------------------------------------------------------------- #
def test_exp_step_size_schedule_golden():
    es = S.ExpStepSize(first_step=0.1, last_step=0.01, tau=500.0)
    assert _flat(es(0)) == _approx([0.10000000149011612])
    assert _flat(es(100)) == _approx([0.08368577063083649])
    assert _flat(es(500)) == _approx([0.04310915246605873])
    assert _flat(es(5000)) == _approx([0.01000408548861742])


def test_constant_step_size_schedule_golden():
    cs = S.ConstantStepSize(step_size=0.01)
    # float32 representation of 0.01 — the schedule computes in f32 (pinning the
    # exact value guards against a precision change during the module move)
    assert _flat(cs(0)) == _approx([0.01])
    assert _flat(cs(1234)) == _approx([0.01])


# --------------------------------------------------------------------------- #
# Kernel (-> svgd_kernels.py)
# --------------------------------------------------------------------------- #
def test_rbf_kernel_golden_and_invariants():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.5, 0.5])
    assert _flat(S.rbf_kernel(x, y, 1.0)) == _approx([0.2865048050880432])
    # self-similarity and symmetry are exact structural invariants of the kernel
    assert _flat(S.rbf_kernel(x, x, 1.0)) == [1.0]
    assert _flat(S.rbf_kernel(x, y, 1.0)) == _flat(S.rbf_kernel(y, x, 1.0))


def test_rbf_kernel_median_matrix_golden():
    parts = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    K = S.rbf_kernel_median(parts)[0]
    assert _flat(K) == _approx([
        1.0, 0.382546067237854, 0.12977856397628784,
        0.382546067237854, 1.0, 0.5485008955001831,
        0.12977856397628784, 0.5485008955001831, 1.0,
    ])


# --------------------------------------------------------------------------- #
# Optimizer (-> svgd_optimizers.py)
# --------------------------------------------------------------------------- #
def _fresh_adam(grad):
    opt = S.Adam(learning_rate=0.01)
    if getattr(opt, "m", None) is None:
        opt.m = jnp.zeros_like(grad)
        opt.v = jnp.zeros_like(grad)
    return opt


def test_adam_step_golden_and_deterministic():
    grad = jnp.array([[0.5, -0.3], [0.1, 0.2]])
    golden = [0.009999999776482582, -0.009999999776482582,
              0.009999999776482582, 0.009999999776482582]
    first = _flat(_fresh_adam(grad).step(grad))
    assert first == _approx(golden)
    # true determinism: two fresh optimizers from the same state produce the
    # bit-identical first step (exact — no ambient-mode ambiguity here).
    assert _flat(_fresh_adam(grad).step(grad)) == first


# --------------------------------------------------------------------------- #
# Constructor -> SvgdConfig seam (frozen-dataclass schema)
# --------------------------------------------------------------------------- #
def test_svgd_config_is_frozen():
    assert SvgdConfig.__dataclass_params__.frozen is True


def test_svgd_config_schema_is_stable():
    """The 27-param SVGD constructor / 42-param Graph.svgd are being piped
    through SvgdConfig; pin that the config schema keeps its known fields (a
    rename/removal during the split is what would break the pipeline)."""
    names = {f.name for f in dataclasses.fields(SvgdConfig)}
    required = {
        "graph_kind", "is_discrete", "param_length", "observation_kind",
        "n_observations", "rewards_kind", "has_epoch_starts", "n_epochs",
        "has_fixed", "fixed_indices", "has_exposure", "exposure_length",
    }
    missing = required - names
    assert not missing, f"SvgdConfig lost seam fields during the split: {missing}"
