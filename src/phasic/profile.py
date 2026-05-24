"""Graph profiler — recommend build/eval settings before committing to an
expensive first evaluation.

``profile_graph(graph)`` analyses a parameterised :class:`~phasic.Graph` and
recommends, with reasons:

- **parallel_elimination** (= hierarchical elimination) — from the cheap SCC
  structure (``scc_decomposition`` + level sets). ON only when the condensation
  has independent SCCs to run concurrently and isn't dominated by one giant SCC.
- **dyn_ordering** — *measured*: the largest SCC's synthetic graph is eliminated
  twice (default vs min-out-degree ordering) and the wall-times compared. dyn's
  fill benefit is model-dependent and not reliably predictable statically, so we
  probe rather than guess.
- **eval path** — ``max_rate`` → forward-PDF auto-granularity (``2·max_rate``).
  A high ``max_rate`` makes the forward-PDF likelihood slow per evaluation; the
  joint/sojourn path (:meth:`Graph.pmf_from_graph_joint_index`) is
  ``max_rate``-independent.

All analysis is read-only with respect to graph topology; the structural tier is
O(V+E). The dyn probe costs two eliminations of the *largest* SCC only (skipped
by default when that SCC is large — see ``probe_dyn``).

Example
-------
>>> import phasic
>>> prof = phasic.profile_graph(graph)
>>> print(prof)
>>> prof.recommendations["parallel_elimination"]
(True, 'RECOMMEND ON (23 independent SCCs at the widest level, ...)')
>>> prof.apply_snippet
'phasic.configure(parallel_elimination=True)'
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Recommendation thresholds (single source of truth).
_PARALLEL_MIN_FRAC = 0.25       # >=25% of work must sit at width>=2 levels
_PARALLEL_MAX_DOMINANCE = 0.5   # one SCC >50% of vertices => Amdahl-bound
_DYN_MIN_REDUCTION = 0.15       # >=15% time cut on largest SCC => recommend dyn
_DYN_AUTO_MAX_SCC = 4000        # 'auto': skip probe if largest SCC bigger
_DYN_NOISE_FLOOR_S = 1e-2       # eliminations <10 ms are noise-dominated
_PATH_GRANULARITY_WARN = 1e5    # auto-granularity above this => forward is slow


@dataclass
class GraphProfile:
    """Result of :func:`profile_graph`. ``recommendations`` maps each lever to a
    ``(decision, reason)`` tuple; ``apply_snippet`` is a ready-to-paste config."""

    n_vertices: int
    n_edges: int
    param_length: int
    max_rate: float
    auto_granularity: float
    n_sccs: int
    largest_scc: int
    largest_scc_frac: float
    n_levels: int
    max_level_width: int
    parallelizable_frac: float
    speedup_ceiling: float
    cpu_count: int
    dyn_probe: dict[str, Any] | None = None
    recommendations: dict[str, tuple] = field(default_factory=dict)

    @property
    def apply_snippet(self) -> str:
        flags = []
        if self.recommendations.get("parallel_elimination", (False,))[0]:
            flags.append("parallel_elimination=True")
        if self.recommendations.get("dyn_ordering", (False,))[0]:
            # dyn_ordering is a per-graph property, not a configure() field — see report()
            flags.append("# graph.dyn_ordering = True")
        if not flags:
            return "# defaults are fine for this graph"
        cfg = [f for f in flags if not f.startswith("#")]
        line = ("phasic.configure(" + ", ".join(cfg) + ")") if cfg else ""
        extra = [f for f in flags if f.startswith("#")]
        return "; ".join(([line] if line else []) + extra)

    def report(self) -> str:
        L = [f"phasic graph profile — {self.n_vertices} vertices, "
             f"{self.n_edges} edges, param_length={self.param_length}",
             f"  SCC structure : {self.n_sccs} SCCs, largest {self.largest_scc} "
             f"({self.largest_scc_frac:.0%}), {self.n_levels} levels, "
             f"widest level {self.max_level_width}"]
        for key, label in (("parallel_elimination", "parallel_elim"),
                           ("dyn_ordering", "dyn_ordering "),
                           ("path", "eval path    ")):
            rec = self.recommendations.get(key)
            if rec is not None:
                L.append(f"  {label} : {rec[1]}")
        L.append(f"  -> {self.apply_snippet}")
        return "\n".join(L)

    def __repr__(self) -> str:
        return self.report()


def _scc_structure(graph) -> dict[str, Any]:
    """Cheap structural metrics (Tarjan O(V+E))."""
    from .distributed_scc import compute_scc_levels
    sd = graph.scc_decomposition()
    n = len(sd)
    sizes = [sd.scc_at(i).size() for i in range(n)]
    total = sum(sizes) or 1
    levels = compute_scc_levels(sd)              # sink-first level sets
    widths = [len(lv) for lv in levels]
    max_width = max(widths) if widths else 0
    # Work at levels with >=2 independent SCCs = the parallelisable part.
    parallel_vertices = sum(sizes[i] for lv in levels if len(lv) >= 2 for i in lv)
    # Critical path with infinite cores, cost proxy ∝ size: levels run
    # sequentially; within a level the slowest SCC gates the level.
    critical = sum(max((sizes[i] for i in lv), default=0) for lv in levels) or 1
    return {
        "scc_graph": sd, "sizes": sizes, "levels": levels,
        "n_sccs": n, "total": total, "largest": max(sizes) if sizes else 0,
        "max_width": max_width,
        "parallelizable_frac": parallel_vertices / total,
        "speedup_ceiling_inf": total / critical,
    }


def _max_rate(graph, theta) -> float:
    """Max out-rate over vertices at reference theta (drives auto-granularity)."""
    pl = graph.param_length()
    if pl > 0:
        th = np.ones(pl) if theta is None else np.asarray(theta, dtype=float)
        if th.shape != (pl,):
            raise ValueError(
                f"theta must have shape ({pl},) to match param_length; got {th.shape}")
        graph.update_weights(th)
    return max((v.rate() for v in graph.vertices()), default=0.0)


def _probe_dyn(graph, struct) -> dict[str, Any] | None:
    """Time the largest SCC's synth elimination under default vs dyn ordering.
    Caching disabled so both orderings actually run. Returns a dict describing
    the outcome (possibly ``{'skipped': ...}`` or ``{'negligible': True}``)."""
    from .phasic_pybind import _synth_get_or_compute_prc
    sd, sizes = struct["scc_graph"], struct["sizes"]
    if not sizes:
        return {"skipped": "no SCCs"}
    largest_i = int(np.argmax(sizes))
    if sizes[largest_i] < 2:
        return {"skipped": "no non-trivial SCC to probe"}
    if sizes[largest_i] > _DYN_AUTO_MAX_SCC:
        return {"skipped": f"largest SCC {sizes[largest_i]} > {_DYN_AUTO_MAX_SCC} "
                           f"(would cost ~2 full builds); pass probe_dyn=True to force"}

    prev = os.environ.get("PHASIC_DISABLE_CACHE")
    os.environ["PHASIC_DISABLE_CACHE"] = "1"
    try:
        def _time(dyn: bool) -> float:
            synth, _meta = sd.scc_at(largest_i).as_synthetic_graph()
            synth.set_dyn_ordering(dyn)
            t = time.perf_counter()
            _synth_get_or_compute_prc(synth)
            return time.perf_counter() - t
        d0 = min(_time(False) for _ in range(3))
        d1 = min(_time(True) for _ in range(3))
    finally:
        if prev is None:
            os.environ.pop("PHASIC_DISABLE_CACHE", None)
        else:
            os.environ["PHASIC_DISABLE_CACHE"] = prev

    pct = (d0 - d1) / d0 if d0 > 0 else 0.0
    out = {"scc_size": sizes[largest_i], "default_s": d0, "dyn_s": d1, "pct": pct}
    if max(d0, d1) < _DYN_NOISE_FLOOR_S:
        out["negligible"] = True
    return out


def profile_graph(graph, theta=None, probe_dyn: bool | str = "auto") -> GraphProfile:
    """Profile ``graph`` and recommend ``parallel_elimination``, ``dyn_ordering``,
    and the evaluation path. See the module docstring.

    Parameters
    ----------
    graph : Graph
        A parameterised graph (pre-elimination).
    theta : array-like, optional
        Reference parameter vector for the ``max_rate`` computation. Defaults to
        all-ones. Must match ``graph.param_length()``.
    probe_dyn : bool or 'auto', default 'auto'
        Whether to run the dyn-ordering probe. ``'auto'`` runs it unless the
        largest SCC exceeds an internal size cap (where two probe eliminations
        would approach the cost of two full builds). ``True`` forces it; ``False``
        skips it.

    Returns
    -------
    GraphProfile
    """
    cpu = os.cpu_count() or 1
    st = _scc_structure(graph)
    mr = _max_rate(graph, theta)
    auto_g = 2.0 * mr
    largest_frac = st["largest"] / st["total"]
    speedup = min(st["speedup_ceiling_inf"], cpu)

    prof = GraphProfile(
        n_vertices=graph.vertices_length(), n_edges=graph.edges_length(),
        param_length=graph.param_length(), max_rate=mr, auto_granularity=auto_g,
        n_sccs=st["n_sccs"], largest_scc=st["largest"], largest_scc_frac=largest_frac,
        n_levels=len(st["levels"]), max_level_width=st["max_width"],
        parallelizable_frac=st["parallelizable_frac"], speedup_ceiling=speedup,
        cpu_count=cpu,
    )

    # --- parallel_elimination (structural) ---
    if cpu <= 1:
        prof.recommendations["parallel_elimination"] = (
            False, "leave OFF (single core available)")
    elif st["max_width"] < 2:
        prof.recommendations["parallel_elimination"] = (
            False, "leave OFF (condensation is a chain/one SCC — nothing "
                   "independent to run in parallel)")
    elif st["parallelizable_frac"] < _PARALLEL_MIN_FRAC:
        prof.recommendations["parallel_elimination"] = (
            False, f"leave OFF (only {st['parallelizable_frac']:.0%} of work is at "
                   f"width>=2 levels)")
    elif largest_frac > _PARALLEL_MAX_DOMINANCE:
        prof.recommendations["parallel_elimination"] = (
            False, f"marginal (one SCC is {largest_frac:.0%} of vertices — "
                   f"Amdahl-bound, ceiling ~{speedup:.1f}x)")
    else:
        prof.recommendations["parallel_elimination"] = (
            True, f"RECOMMEND ON ({st['max_width']} independent SCCs at the widest "
                  f"level, {cpu} cores; ceiling ~{speedup:.1f}x; "
                  f"parallelizable_frac {st['parallelizable_frac']:.0%})")

    # --- dyn_ordering (measured probe) ---
    if probe_dyn is True or probe_dyn == "auto":
        dp = _probe_dyn(graph, st)
        prof.dyn_probe = dp
        if dp is None or "skipped" in dp:
            prof.recommendations["dyn_ordering"] = (
                False, f"not probed ({dp.get('skipped') if dp else 'n/a'})")
        elif dp.get("negligible"):
            prof.recommendations["dyn_ordering"] = (
                False, f"leave OFF (largest SCC only {dp['scc_size']} vertices — "
                       f"elimination <10 ms, ordering is immaterial)")
        elif dp["pct"] >= _DYN_MIN_REDUCTION:
            prof.recommendations["dyn_ordering"] = (
                True, f"RECOMMEND ON (probe on largest SCC: {dp['default_s']:.3f}s vs "
                      f"{dp['dyn_s']:.3f}s, -{dp['pct']:.0%})")
        else:
            prof.recommendations["dyn_ordering"] = (
                False, f"leave OFF (probe: {dp['default_s']:.3f}s vs {dp['dyn_s']:.3f}s, "
                       f"{dp['pct']:+.0%} — default ordering already near-optimal)")
    else:
        prof.recommendations["dyn_ordering"] = (False, "not probed (probe_dyn=False)")

    # --- eval path (max_rate -> granularity) ---
    if auto_g >= _PATH_GRANULARITY_WARN:
        prof.recommendations["path"] = (
            "sojourn", f"forward-PDF will be SLOW per eval (max_rate {mr:.3g} -> "
                       f"granularity {auto_g:.3g}); IF your likelihood is over a "
                       f"joint/discrete index rather than continuous times, use "
                       f"Graph.pmf_from_graph_joint_index (max_rate-independent)")
    else:
        prof.recommendations["path"] = (
            "forward", f"forward-PDF OK (max_rate {mr:.3g} -> granularity {auto_g:.3g})")

    return prof
