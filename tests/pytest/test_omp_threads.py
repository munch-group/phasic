"""Tests for OpenMP thread count auto-detection and the
omp_num_threads configure() knob.

Auto-detection runs at import time and writes OMP_NUM_THREADS
if it's unset. Priority:
  1. SLURM_CPUS_PER_TASK
  2. SLURM_CPUS_ON_NODE
  3. os.sched_getaffinity(0) (Linux cgroups)
  4. os.cpu_count()

Most tests run the detection in a fresh subprocess so they
can control the env var without interference from the parent
test process (which has already imported phasic).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def run_in_subprocess(env_overrides: dict[str, str | None],
                      script: str) -> str:
    """Run `script` in a clean Python subprocess with env tweaks.
    Returns the LAST non-empty line of stdout. (Phasic emits a
    ``<IPython.core.display.HTML object>`` banner at import time
    that we strip off.)"""
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    lines = [line for line in result.stdout.splitlines()
             if line.strip() and "<IPython.core.display.HTML" not in line]
    assert lines, f"no usable stdout:\n{result.stdout}"
    return lines[-1].strip()


def test_auto_detect_sets_omp_when_unset():
    """With OMP_NUM_THREADS unset, importing phasic sets it to
    a positive integer (typically os.cpu_count())."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None,
         "SLURM_CPUS_PER_TASK": None,
         "SLURM_CPUS_ON_NODE": None},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    n = int(out)
    assert n >= 1


def test_auto_detect_respects_existing_omp():
    """A pre-set OMP_NUM_THREADS is preserved verbatim."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": "2"},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "2"


def test_auto_detect_uses_slurm_cpus_per_task():
    """SLURM_CPUS_PER_TASK takes priority over machine CPU count."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None,
         "SLURM_CPUS_PER_TASK": "4",
         "SLURM_CPUS_ON_NODE": None},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "4"


def test_auto_detect_falls_back_to_slurm_cpus_on_node():
    """SLURM_CPUS_ON_NODE is used when SLURM_CPUS_PER_TASK is unset."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None,
         "SLURM_CPUS_PER_TASK": None,
         "SLURM_CPUS_ON_NODE": "6"},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "6"


def test_auto_detect_slurm_cpus_per_task_beats_node():
    """SLURM_CPUS_PER_TASK takes priority over SLURM_CPUS_ON_NODE."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None,
         "SLURM_CPUS_PER_TASK": "4",
         "SLURM_CPUS_ON_NODE": "16"},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "4"


def test_auto_detect_malformed_slurm_var_falls_through():
    """A non-numeric SLURM env var is ignored and detection
    falls through to the next source."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None,
         "SLURM_CPUS_PER_TASK": "not_a_number",
         "SLURM_CPUS_ON_NODE": "5"},
        "import phasic, os; print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "5"


# ---------------------------------------------------------------
# configure(omp_num_threads=...)
# ---------------------------------------------------------------


def test_configure_writes_omp_env_var():
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None},
        "import phasic, os; "
        "phasic.configure(omp_num_threads=7); "
        "print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "7"


def test_configure_overrides_existing_omp():
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": "2"},
        "import phasic, os; "
        "phasic.configure(omp_num_threads=11); "
        "print(os.environ['OMP_NUM_THREADS'])",
    )
    assert out == "11"


def test_configure_rejects_zero():
    """configure(omp_num_threads=0) raises."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None},
        "import phasic\n"
        "try:\n"
        "    phasic.configure(omp_num_threads=0)\n"
        "    print('NO_ERROR')\n"
        "except phasic.PTDConfigError as e:\n"
        "    print('CAUGHT:', e)",
    )
    assert "CAUGHT" in out
    assert "omp_num_threads" in out


def test_configure_rejects_negative():
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": None},
        "import phasic\n"
        "try:\n"
        "    phasic.configure(omp_num_threads=-1)\n"
        "    print('NO_ERROR')\n"
        "except phasic.PTDConfigError as e:\n"
        "    print('CAUGHT:', e)",
    )
    assert "CAUGHT" in out


def test_get_config_reads_omp_from_env():
    """get_config() reflects OMP_NUM_THREADS from the environment."""
    out = run_in_subprocess(
        {"OMP_NUM_THREADS": "5"},
        "import phasic; print(phasic.get_config().omp_num_threads)",
    )
    assert out == "5"
