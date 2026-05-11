"""``phasic`` console-script entry point.

Implements ``phasic publish-compute``, a one-shot helper for
contributing a C-elimination ``.bin`` artifact to the
``munch-group/phasic-traces`` registry. The flow:

1. Import the user's Python script (path argument). Look for a
   top-level callable named ``build_graph``.
2. Call ``build_graph()`` → must return a ``phasic.Graph``.
3. Populate the symbolic elimination cache by calling
   ``graph.expectation()`` (or whatever the user's script does
   inside ``build_graph``).
4. Compute the graph hash and save the parent ``.bin`` to a staging
   dir; also bundle any matching ``scc_*.bin`` files from the local
   cache.
5. Build a registry entry and either:
   - ``--dry-run``: print the entry JSON to stdout. Skip clone/push.
   - default: clone ``munch-group/phasic-traces``, copy artifacts
     into ``artifacts/``, splice the entry into ``registry.json``,
     ``git add``/``commit``/``push`` to a feature branch, and
     ``gh pr create``.

Usage
-----

::

    phasic publish-compute scripts/coalescent.py \\
        --id coal_n5 \\
        --description "Kingman coalescent for n=5"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .compute_repository import (
    ComputeRegistry,
    _graph_hash_hex,
    _param_compute_cache_dir,
)
from .exceptions import PTDBackendError
from .logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _import_user_script(script_path: Path):
    """Import *script_path* as a one-off module and return the module."""
    if not script_path.is_file():
        raise PTDBackendError(f"script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(
        "_phasic_publish_script", script_path)
    if spec is None or spec.loader is None:
        raise PTDBackendError(
            f"cannot import {script_path} (importlib couldn't make a spec)")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _build_graph_from_script(script_path: Path):
    """Run *script_path*'s ``build_graph()`` and return the Graph."""
    module = _import_user_script(script_path)
    builder = getattr(module, "build_graph", None)
    if builder is None or not callable(builder):
        raise PTDBackendError(
            f"{script_path}: no top-level callable named 'build_graph' "
            f"(must return a phasic.Graph)")
    graph = builder()
    from . import Graph
    if not isinstance(graph, Graph):
        raise PTDBackendError(
            f"{script_path}:build_graph() returned {type(graph).__name__}, "
            f"expected phasic.Graph")
    return graph


def _populate_prc_cache(graph) -> None:
    """Force the symbolic elimination cache to be populated.

    The C-side ``_save_param_compute_graph`` refuses to save if
    ``parameterized_reward_compute_graph`` is NULL. The cheapest way
    to populate it is to call ``expectation()``.
    """
    # ``expectation`` may not be defined for every graph (e.g. DPH).
    # Try a few entry points, in order of cheapness.
    for fn_name in ("expectation", "moments", "pdf"):
        fn = getattr(graph, fn_name, None)
        if fn is None:
            continue
        try:
            if fn_name == "pdf":
                fn(1.0)
            elif fn_name == "moments":
                fn(1)
            else:
                fn()
            return
        except Exception as e:  # pragma: no cover — model-dependent
            logger.debug("%s() failed: %s", fn_name, e)
            continue
    raise PTDBackendError(
        "could not populate the C-side compute cache: expectation(), "
        "moments(), and pdf() all failed on this graph")


def _save_artifacts(graph, staging_dir: Path) -> tuple[Path, list[Path]]:
    """Save the parent ``.bin`` and bundle any per-SCC files.

    Returns ``(parent_path, scc_paths)`` — the parent file is always
    present; the SCC list is empty unless the hierarchical compose
    path has already populated them in the local cache.
    """
    hash_hex = _graph_hash_hex(graph)
    parent = staging_dir / f"{hash_hex}.bin"

    if not getattr(graph, "_has_param_compute_graph_cache", lambda: False)():
        _populate_prc_cache(graph)

    graph._save_param_compute_graph(str(parent))

    scc_paths: list[Path] = []
    local_cache = _param_compute_cache_dir()
    if local_cache.exists():
        for scc_file in local_cache.glob("scc_*.bin"):
            target = staging_dir / scc_file.name
            shutil.copy2(scc_file, target)
            scc_paths.append(target)

    return parent, scc_paths


def _git_config_value(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _default_author() -> str:
    name = _git_config_value("user.name")
    email = _git_config_value("user.email")
    if name and email:
        return f"{name} <{email}>"
    if name:
        return name
    if email:
        return email
    return "unknown"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    """Run *cmd*, capturing output. Raises on non-zero exit."""
    logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    if check and proc.returncode != 0:
        raise PTDBackendError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}")
    return proc.stdout


# ============================================================================
# Publish-compute command
# ============================================================================


def cmd_publish_compute(args: argparse.Namespace) -> int:
    script_path = Path(args.script).resolve()
    graph = _build_graph_from_script(script_path)

    metadata: dict[str, Any] = {
        "description": args.description or "",
        "author": args.author or _default_author(),
        "license": args.license,
        "vertices": graph.vertices_length(),
        "param_length": graph.param_length(),
    }
    if args.domain:
        metadata["domain"] = args.domain
    if args.model_type:
        metadata["model_type"] = args.model_type
    if args.tags:
        metadata["tags"] = list(args.tags)

    # 1) save artifacts to a staging dir
    with tempfile.TemporaryDirectory(prefix="phasic_publish_") as tmp:
        staging = Path(tmp)
        parent_path, scc_paths = _save_artifacts(graph, staging)

        # 2) build the registry entry
        registry = ComputeRegistry(
            registry_repo=args.registry_repo, auto_update=False)
        entry = registry.build_publish_entry(
            graph,
            compute_id=args.id,
            metadata=metadata,
            artifact_paths={"parent": parent_path, "scc": scc_paths},
        )

        if args.dry_run:
            print(json.dumps({args.id: entry}, indent=2))
            print(f"\n# staging artifacts:", file=sys.stderr)
            print(f"#   parent: {parent_path}", file=sys.stderr)
            for p in scc_paths:
                print(f"#   scc:    {p}", file=sys.stderr)
            return 0

        # 3) clone the registry repo & open a PR
        return _publish_via_pr(
            registry_repo=args.registry_repo,
            compute_id=args.id,
            entry=entry,
            parent_path=parent_path,
            scc_paths=scc_paths,
        )


def _publish_via_pr(
    registry_repo: str,
    compute_id: str,
    entry: dict[str, Any],
    parent_path: Path,
    scc_paths: list[Path],
) -> int:
    """Clone the registry repo, splice in the entry, and open a PR."""
    if not shutil.which("gh"):
        raise PTDBackendError(
            "`gh` (GitHub CLI) not found on PATH. Install from "
            "https://cli.github.com/ or run with --dry-run.")

    branch = f"phasic-publish/{compute_id}"
    with tempfile.TemporaryDirectory(prefix="phasic_pr_") as clone_tmp:
        clone_dir = Path(clone_tmp) / "registry"
        _run(["gh", "repo", "clone", registry_repo, str(clone_dir)])
        registry_path = clone_dir / "registry.json"
        if not registry_path.is_file():
            # Bootstrap a fresh v2 registry on the first publish.
            data: dict[str, Any] = {
                "version": "2.0", "format": "ptd_pcg", "computes": {}}
        else:
            data = json.loads(registry_path.read_text())
            if "computes" not in data:
                data = {"version": "2.0", "format": "ptd_pcg",
                        "computes": {}, **{k: v for k, v in data.items()
                                           if k not in ("traces",)}}
        if compute_id in data["computes"]:
            raise PTDBackendError(
                f"compute id '{compute_id}' already exists in "
                f"{registry_repo}/registry.json")
        data["computes"][compute_id] = entry

        # Copy artifacts into artifacts/.
        artifacts_dir = clone_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        shutil.copy2(parent_path, artifacts_dir / parent_path.name)
        for p in scc_paths:
            shutil.copy2(p, artifacts_dir / p.name)

        # Rewrite registry.json with stable formatting.
        registry_path.write_text(json.dumps(data, indent=2) + "\n")

        _run(["git", "checkout", "-b", branch], cwd=clone_dir)
        _run(["git", "add", "registry.json", "artifacts/"], cwd=clone_dir)
        _run(["git", "commit", "-m",
              f"Add compute artifact {compute_id}\n\n"
              f"graph_hash: {entry['graph_hash']}\n"
              f"format_revision: {entry['format_revision']}"],
             cwd=clone_dir)
        _run(["git", "push", "-u", "origin", branch], cwd=clone_dir)

        out = _run([
            "gh", "pr", "create",
            "--title", f"Add compute artifact: {compute_id}",
            "--body", (
                f"Adds the C-elimination compute artifact for "
                f"`{compute_id}`.\n\n"
                f"- graph_hash: `{entry['graph_hash']}`\n"
                f"- format_revision: {entry['format_revision']}\n"
                f"- vertices: "
                f"{entry['metadata'].get('vertices', 'unspecified')}"),
        ], cwd=clone_dir)
        print(out)
    return 0


# ============================================================================
# Argument parsing
# ============================================================================


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phasic",
        description="phasic command-line utilities")
    sub = p.add_subparsers(dest="command", required=True)

    pub = sub.add_parser(
        "publish-compute",
        help="Publish a C-elimination compute artifact to the "
             "phasic-traces registry.")
    pub.add_argument("script", type=str,
                     help="Path to a Python script defining build_graph() "
                          "(must return a phasic.Graph).")
    pub.add_argument("--id", required=True,
                     help="Unique human-readable id for this artifact "
                          "(e.g. 'coal_n5_theta1').")
    pub.add_argument("--description", default="",
                     help="Free-form description of the model.")
    pub.add_argument("--domain", default=None,
                     help="Filtering key (e.g. 'population-genetics').")
    pub.add_argument("--model-type", default=None, dest="model_type",
                     help="Filtering key (e.g. 'coalescent').")
    pub.add_argument("--tags", nargs="+", default=None,
                     help="One or more tags to attach to the metadata.")
    pub.add_argument("--license", default="MIT",
                     help="SPDX license identifier (default: MIT).")
    pub.add_argument("--author", default=None,
                     help="Override the author string (default: from git).")
    pub.add_argument("--registry-repo", default="munch-group/phasic-traces",
                     dest="registry_repo",
                     help="GitHub owner/name of the registry repo.")
    pub.add_argument("--dry-run", action="store_true",
                     help="Build artifacts and print the registry entry; "
                          "do not clone or push.")
    pub.set_defaults(func=cmd_publish_compute)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PTDBackendError as e:
        print(f"phasic: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
