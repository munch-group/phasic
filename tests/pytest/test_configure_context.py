"""Tests for `configure()` as a context manager.

`configure(...)` may be used in two ways:
  - As a regular call: settings are applied and persist.
  - As `with configure(...): ...`: settings are applied on
    entry and rolled back on exit (both the dataclass fields
    and the phasic-tracked env vars).
"""

from __future__ import annotations

import os

import pytest

import phasic
from phasic.config import _phasic_assigned_env, reset_config
from phasic.exceptions import PTDConfigError


_PHASIC_ENV_VARS = (
    'OMP_NUM_THREADS', 'PHASIC_HIERAR_ELIMINATION',
    'PHASIC_REWARD_COMPUTE_CACHE', 'PHASIC_DISABLE_GRAPH_CACHE',
    'PHASIC_CACHE_DIR',
    'PHASIC_MIN_SCC_SIZE_TO_CACHE',
    'PHASIC_MAX_PARALLEL_SCCS', 'PHASIC_FORCE_MPFR',
    'PHASIC_MPFR_BITS', 'PHASIC_CONDITION_THRESHOLD',
    'PHASIC_DISABLE_CONDITION_WARNINGS',
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_config()
    _phasic_assigned_env.clear()
    for v in _PHASIC_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    yield
    reset_config()
    _phasic_assigned_env.clear()
    # See test_config_scc_controls.py for rationale: phasic.configure
    # bypasses monkeypatch by writing os.environ directly, so we
    # scrub directly on cleanup.
    import os
    for v in _PHASIC_ENV_VARS:
        os.environ.pop(v, None)


def test_context_manager_rolls_back_field():
    """Inside the `with`, the field is set; outside, restored."""
    assert phasic.get_config().parallel_elimination is False
    with phasic.configure(parallel_elimination=True):
        assert phasic.get_config().parallel_elimination is True
    assert phasic.get_config().parallel_elimination is False


def test_context_manager_rolls_back_env_var():
    """Env var is set inside the block, restored on exit."""
    assert os.environ.get('PHASIC_HIERAR_ELIMINATION') is None
    with phasic.configure(parallel_elimination=True):
        assert os.environ['PHASIC_HIERAR_ELIMINATION'] == '1'
    assert os.environ.get('PHASIC_HIERAR_ELIMINATION') is None


def test_context_manager_restores_pre_existing_value():
    """If a field had a non-default value before the context,
    that value is restored — not the dataclass default."""
    phasic.configure(parallel_elimination_min_subgraph=8)
    assert phasic.get_config().parallel_elimination_min_subgraph == 8

    with phasic.configure(parallel_elimination_min_subgraph=100):
        assert phasic.get_config().parallel_elimination_min_subgraph == 100

    assert phasic.get_config().parallel_elimination_min_subgraph == 8
    assert os.environ['PHASIC_MIN_SCC_SIZE_TO_CACHE'] == '8'


def test_context_manager_yields_config():
    """`with configure(...) as cfg:` yields the live PhasicConfig."""
    with phasic.configure(cpu_threads=4) as cfg:
        assert isinstance(cfg, phasic.PhasicConfig)
        assert cfg.cpu_threads == 4


def test_context_manager_nested():
    """Nested `with` blocks roll back in LIFO order."""
    with phasic.configure(parallel_elimination=True):
        assert phasic.get_config().parallel_elimination is True
        with phasic.configure(cpu_threads=4):
            assert phasic.get_config().parallel_elimination is True
            assert phasic.get_config().cpu_threads == 4
        # Inner rolled back; outer still in effect.
        assert phasic.get_config().parallel_elimination is True
        assert phasic.get_config().cpu_threads != 4
    # Outer rolled back too.
    assert phasic.get_config().parallel_elimination is False


def test_context_manager_rolls_back_on_exception():
    """If the body raises, settings are still rolled back."""
    with pytest.raises(RuntimeError, match='kaboom'):
        with phasic.configure(parallel_elimination=True):
            assert phasic.get_config().parallel_elimination is True
            raise RuntimeError('kaboom')
    # Rolled back despite the exception.
    assert phasic.get_config().parallel_elimination is False


def test_persistent_configure_still_works():
    """configure() without `with` keeps settings applied (the
    return value is just discarded)."""
    assert phasic.get_config().parallel_elimination is False
    phasic.configure(parallel_elimination=True)
    assert phasic.get_config().parallel_elimination is True
    assert os.environ['PHASIC_HIERAR_ELIMINATION'] == '1'


def test_context_manager_restores_multiple_fields():
    """A single `with` block can roll back multiple fields and env vars."""
    with phasic.configure(parallel_elimination=True,
                          cpu_threads=4,
                          cache_dir='/tmp/x',
                          parallel_elimination_max_concurrent=2):
        assert phasic.get_config().parallel_elimination is True
        assert phasic.get_config().cpu_threads == 4
        assert phasic.get_config().cache_dir == '/tmp/x'
        assert phasic.get_config().parallel_elimination_max_concurrent == 2
        assert os.environ['PHASIC_CACHE_DIR'] == '/tmp/x'
        assert os.environ['PHASIC_MAX_PARALLEL_SCCS'] == '2'

    cfg = phasic.get_config()
    assert cfg.parallel_elimination is False
    assert cfg.cache_dir is None
    assert cfg.parallel_elimination_max_concurrent is None
    assert os.environ.get('PHASIC_CACHE_DIR') is None
    assert os.environ.get('PHASIC_MAX_PARALLEL_SCCS') is None


def test_context_manager_conflict_still_raises():
    """Conflict checks fire at __init__ time of the context too:
    invalid kwargs raise before the `with` body runs."""
    with pytest.raises(PTDConfigError):
        with phasic.configure(parallel_elimination=True, cpu_threads=1):
            pass


def test_context_manager_validate_raises_outside_with():
    """Even when not used as a context, configure() raises on
    invalid kwargs — the snapshot/rollback mechanism does not
    suppress validation."""
    with pytest.raises(PTDConfigError):
        phasic.configure(parallel_elimination=True, cpu_threads=1)
