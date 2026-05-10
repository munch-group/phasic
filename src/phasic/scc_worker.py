"""SLURM-WP-3: standalone scc_worker CLI.

Run as ``python -m phasic.scc_worker <work_unit_path>``.

The worker:
  1. Loads the work-unit JSON written by the orchestrator
     (:func:`phasic.distributed_scc.write_work_unit`).
  2. Reconstructs the synthetic SCC graph
     (:func:`phasic.distributed_scc.deserialize_scc_synth`).
  3. Calls
     :func:`phasic.distributed_scc.cache_synth_prc` which
     populates the on-disk PRC cache.
  4. Exits with status 0 on success, non-zero on failure.

The cache location is controlled by ``PHASIC_CACHE_DIR``
(SLURM-WP-1) — set this to a shared-filesystem path so the
worker's writes are visible to the orchestrator and other
workers.

Each worker is a single SLURM array task. Failures are
isolated; the orchestrator retries by re-checking the cache
after the level-set's array job completes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S")


def run_worker(work_unit_path: str) -> int:
    """Execute a single work unit.

    Loads the work unit from disk, deserialises the synthetic
    SCC graph, and populates the on-disk PRC cache. Returns
    0 on success, non-zero on failure.

    Parameters
    ----------
    work_unit_path : str
        Path to the work-unit JSON file.

    Returns
    -------
    int
        Exit status.
    """
    from phasic import distributed_scc

    log = logging.getLogger("phasic.scc_worker")

    log.info("Loading work unit: %s", work_unit_path)
    try:
        data = distributed_scc.read_work_unit(work_unit_path)
    except Exception as e:
        log.error("Failed to load work unit %s: %s", work_unit_path, e)
        return 1

    scc_index = data.get("scc_index", "?")
    log.info("Reconstructing synth for SCC index %s", scc_index)
    try:
        synth = distributed_scc.deserialize_scc_synth(data)
    except Exception as e:
        log.error("Failed to deserialise synth (SCC %s): %s",
                  scc_index, e)
        return 2

    log.info("Computing PRC and writing to cache "
             "(synth has %d vertices)", synth.vertices_length())
    try:
        distributed_scc.cache_synth_prc(synth)
    except Exception as e:
        log.error("Failed to cache PRC (SCC %s): %s", scc_index, e)
        return 3

    log.info("OK: SCC %s done.", scc_index)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``python -m phasic.scc_worker``."""
    parser = argparse.ArgumentParser(
            description="Phasic SLURM SCC worker — populate the "
                        "on-disk PRC cache for one synthetic SCC "
                        "graph from a work-unit JSON file.")
    parser.add_argument(
            "work_unit",
            help="Path to the work-unit JSON file.")
    parser.add_argument(
            "-v", "--verbose", action="store_true",
            help="Enable debug logging.")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    return run_worker(args.work_unit)


if __name__ == "__main__":
    sys.exit(main())
