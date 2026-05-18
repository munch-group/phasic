"""SLURM-WP-5: sbatch job-script generator + work-unit submission.

Verifies:
  - generate_sbatch_script produces a well-formed array job
    script with the right SBATCH directives.
  - write_work_units_for_level emits one JSON per SCC in a
    level-set.
  - submit_sbatch handles missing sbatch gracefully.
  - run_workers_locally (the no-SLURM fallback) produces the
    same cache state as a real SLURM submission would.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import phasic.cache as cache
from phasic import distributed_scc
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _force_cache_all_sccs(monkeypatch):
    """Override the size threshold so every SCC produces a
    cache entry; sbatch / local-fallback tests count files."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


def test_generate_sbatch_script_has_required_directives(tmp_path):
    work_units = [str(tmp_path / "wu_0.json"),
                  str(tmp_path / "wu_1.json")]
    script = distributed_scc.generate_sbatch_script(
            work_units=work_units,
            cache_dir=str(tmp_path / "cache"),
            log_dir=str(tmp_path / "logs"))
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=" in script
    assert "#SBATCH --array=0-1" in script
    assert "#SBATCH --time=" in script
    assert "#SBATCH --mem=" in script
    assert "#SBATCH --cpus-per-task=" in script
    assert "PHASIC_CACHE_DIR=" in script
    assert "phasic.scc_worker" in script


def test_generate_sbatch_array_size_matches_work_units(tmp_path):
    work_units = [str(tmp_path / f"wu_{i}.json") for i in range(7)]
    script = distributed_scc.generate_sbatch_script(
            work_units=work_units,
            cache_dir=str(tmp_path / "cache"),
            log_dir=str(tmp_path / "logs"))
    m = re.search(r"#SBATCH --array=0-(\d+)", script)
    assert m is not None
    assert int(m.group(1)) == 6


def test_generate_sbatch_max_concurrent(tmp_path):
    work_units = [str(tmp_path / f"wu_{i}.json") for i in range(5)]
    script = distributed_scc.generate_sbatch_script(
            work_units=work_units,
            cache_dir=str(tmp_path / "cache"),
            log_dir=str(tmp_path / "logs"),
            max_concurrent=3)
    assert "#SBATCH --array=0-4%3" in script


def test_generate_sbatch_extra_directives(tmp_path):
    work_units = [str(tmp_path / "wu_0.json")]
    extra = "#SBATCH --partition=short\n#SBATCH --account=phasic"
    script = distributed_scc.generate_sbatch_script(
            work_units=work_units,
            cache_dir=str(tmp_path / "cache"),
            log_dir=str(tmp_path / "logs"),
            extra_directives=extra)
    assert "--partition=short" in script
    assert "--account=phasic" in script


def test_generate_sbatch_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        distributed_scc.generate_sbatch_script(
                work_units=[],
                cache_dir=str(tmp_path / "cache"),
                log_dir=str(tmp_path / "logs"))


def test_generate_sbatch_handles_paths_with_spaces(tmp_path):
    """Paths with spaces (rare but possible) must be quoted."""
    wu = str(tmp_path / "with space" / "wu.json")
    script = distributed_scc.generate_sbatch_script(
            work_units=[wu],
            cache_dir=str(tmp_path / "cache"),
            log_dir=str(tmp_path / "logs"))
    # The bash array literal must contain the path within quotes
    # so word-splitting doesn't break it.
    assert f'"{wu}"' in script


def test_write_work_units_for_level(tmp_path):
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    work_dir = tmp_path / "wus"
    work_dir.mkdir()

    paths = distributed_scc.write_work_units_for_level(
            scc_decomp, scc_indices=[0, 1, 2],
            work_dir=str(work_dir))

    assert len(paths) == 3
    for p in paths:
        assert os.path.exists(p)


def test_submit_sbatch_missing_executable_raises(tmp_path):
    """If sbatch is not on PATH, submit_sbatch raises with a
    clear message rather than failing silently."""
    script = tmp_path / "fake.sh"
    script.write_text("#!/bin/bash\necho hi")

    with pytest.raises(RuntimeError, match="not found"):
        distributed_scc.submit_sbatch(
                str(script), sbatch_exe="this-binary-does-not-exist")


def test_run_workers_locally_sequential_pipeline(tmp_path, monkeypatch):
    """Local fallback: serial subprocess execution produces the
    same cache state as a SLURM array would."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    n = len(scc_decomp)

    work_dir = tmp_path / "wus"
    work_dir.mkdir()
    paths = distributed_scc.write_work_units_for_level(
            scc_decomp, list(range(n)), str(work_dir))

    rcs = distributed_scc.run_workers_locally(paths, max_workers=1)
    assert all(rc == 0 for rc in rcs), f"non-zero exit codes: {rcs}"

    files = list((cache_dir / "parameterized_reward_compute").glob("scc_*.bin"))
    # n files since toy_b has n SCCs.
    assert len(files) == n


def test_run_workers_locally_parallel_pipeline(tmp_path, monkeypatch):
    """Local fallback with max_workers>1 still produces the
    correct cache state — parallelism is safe because
    write-then-rename in the cache write path is atomic."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    n = len(scc_decomp)

    work_dir = tmp_path / "wus"
    work_dir.mkdir()
    paths = distributed_scc.write_work_units_for_level(
            scc_decomp, list(range(n)), str(work_dir))

    rcs = distributed_scc.run_workers_locally(paths, max_workers=4)
    assert all(rc == 0 for rc in rcs)

    # Orchestrator: every SCC should hit cache.
    cache.reset_scc_compose_stats()
    g_run = build_toy_b()
    g_run.update_weights(THETA)
    result = g_run.expected_waiting_time()
    stats = cache.scc_compose_stats()
    assert stats["cache_hits"] == n
    assert stats["cache_misses"] == 0
    assert result[0] == pytest.approx(1.88, rel=1e-12)
