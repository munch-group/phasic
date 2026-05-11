"""Tests for the svgd_strategy field on PhasicConfig.

The old parallel_config / disable_parallel context managers (and
the global ParallelConfig state they mutated) were removed in
the config refactor. SVGD's default parallelism strategy is
now read from phasic.get_config().svgd_strategy, set via
phasic.configure(svgd_strategy=...) — including as a context
manager (`with configure(svgd_strategy='none'):`).
"""

from __future__ import annotations

import pytest

import phasic
from phasic.config import _phasic_assigned_env, reset_config
from phasic.exceptions import PTDConfigError


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    _phasic_assigned_env.clear()
    yield
    reset_config()
    _phasic_assigned_env.clear()


def test_default_is_auto():
    assert phasic.get_config().svgd_strategy == 'auto'


def test_configure_persistent():
    phasic.configure(svgd_strategy='vmap')
    assert phasic.get_config().svgd_strategy == 'vmap'


def test_context_manager_rolls_back():
    with phasic.configure(svgd_strategy='none'):
        assert phasic.get_config().svgd_strategy == 'none'
    assert phasic.get_config().svgd_strategy == 'auto'


def test_context_manager_yields_cfg():
    with phasic.configure(svgd_strategy='pmap') as cfg:
        assert cfg.svgd_strategy == 'pmap'


def test_invalid_strategy_raises():
    with pytest.raises((PTDConfigError, ValueError, TypeError)):
        phasic.configure(svgd_strategy='not-a-real-mode')


def test_no_env_var_backing():
    """svgd_strategy is a runtime-only field; setting it must
    not write any env var (the C side does not consume it)."""
    import os
    phasic.configure(svgd_strategy='vmap')
    for v in ('PHASIC_SVGD_STRATEGY', 'PHASIC_PARALLEL', 'PARALLEL_STRATEGY'):
        assert os.environ.get(v) is None


def test_old_names_are_gone():
    """The removed parallel_config / disable_parallel / globals
    must not be importable from phasic."""
    for name in ('parallel_config', 'disable_parallel',
                 'get_parallel_config', 'set_parallel_config'):
        assert not hasattr(phasic, name), (
            f"{name} should have been removed in the config refactor"
        )
