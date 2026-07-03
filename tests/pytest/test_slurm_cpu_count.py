"""Regression test for the SLURM_JOB_CPUS_PER_NODE parsing bug.

Before the fix, `compute_missing_traces_parallel` did
    n_workers = os.environ.get('SLURM_JOB_CPUS_PER_NODE', n_workers)
    n_workers = max(1, min(n_workers, len(work_units)))
and SLURM always sets that variable to a *string*, so `min(str, int)`
raised `TypeError: '<' not supported between instances of 'int' and 'str'`,
aborting the whole hierarchical trace computation on the library's primary
(SLURM) deployment target.
"""
import os
import pytest

from phasic.hierarchical_trace_cache import _slurm_cpu_count


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    yield


def test_unset_returns_default(monkeypatch):
    assert _slurm_cpu_count(7) == 7


def test_plain_integer_string(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "16")
    assert _slurm_cpu_count(4) == 16


def test_heterogeneous_allocation_form(monkeypatch):
    # e.g. "4(x2)" for 2 nodes with 4 CPUs each — parse the leading count.
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "4(x2)")
    assert _slurm_cpu_count(1) == 4


def test_comma_list_form(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "16,8")
    assert _slurm_cpu_count(1) == 16


def test_unparseable_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "unexpected")
    assert _slurm_cpu_count(3) == 3


def test_result_is_always_int_usable_in_min(monkeypatch):
    """The core bug: the result must be an int so min(result, N) works."""
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "16")
    n = _slurm_cpu_count(4)
    # This is exactly the operation that used to raise TypeError.
    assert max(1, min(n, 10)) == 10
