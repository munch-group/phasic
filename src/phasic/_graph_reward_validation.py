"""Reward validation / vertex-index helpers for :class:`~phasic.Graph`.

Extracted verbatim from ``Graph`` (Stage-3 WS-C). Pure relocation; bodies
unchanged. Assigned onto ``Graph`` as class attributes in ``__init__.py`` so
they stay direct members (docs/introspection).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def absorbing_state_rewards(self) -> np.ndarray:
    """
    Get a reward vector that is 1 for states with edges to absorbing states.

    This reward vector is used for computing the Laplace transform value
    from a Laplace-transformed graph.

    Returns
    -------
    np.ndarray
        Reward vector of length n_vertices. Element i is 1.0 if vertex i
        has an edge to an absorbing state, 0.0 otherwise.

    Examples
    --------
    >>> graph = Graph(coalescent_callback)
    >>> rewards = graph.absorbing_state_rewards()
    >>> laplace_graph = graph.laplace_transform(0.5)
    >>> L_0_5 = laplace_graph.expectation(rewards=rewards)

    See Also
    --------
    laplace_transform : Create a Laplace-transformed graph
    """
    n_vertices = self.vertices_length()
    rewards = np.zeros(n_vertices)
    for i in range(n_vertices):
        vertex = self.vertex_at(i)
        # Check if any edge goes to an absorbing state (no outgoing edges)
        for edge in vertex.edges():
            to_vertex = edge.to()
            if to_vertex.edges_length() == 0:
                rewards[i] = 1.0
                break
    return rewards

def _starting_vertex_indices(self) -> list[int]:
    """Indices of vertices with positive initial probability mass.

    Reads the synthetic start vertex's outgoing edges; each edge's
    ``weight()`` is the IPV per-target (cf. line 4944 in this file).
    """
    sv = self.starting_vertex()
    return [int(e.to().index()) for e in sv.edges() if e.weight() > 0.0]

def _absorbing_vertex_indices(self) -> list[int]:
    """Indices of vertices with no outgoing edges (absorbing)."""
    n = self.vertices_length()
    return [i for i in range(n) if self.vertex_at(i).edges_length() == 0]

def _validate_reward_coverage(
    self, rewards_1d: np.ndarray, *, context: str,
) -> None:
    """Raise ValueError if some trajectories accumulate zero reward.

    Such trajectories contribute a point mass at :math:`r = 0` to
    the reward-transformed distribution: a legitimate (atom +
    continuous) mixture, not an error. Callers decide the response
    via ``_validate_rewards(..., coverage_mode=...)``; this helper
    raises so the orchestrator can collect and re-emit the
    per-feature diagnostics.

    Assumes ``rewards_1d`` is already shape-validated (1D, length
    n_vertices). Called from ``_validate_rewards``.

    Algorithm: BFS in the subgraph that excludes all rewarded
    vertices. If any absorbing vertex is reachable from any
    starting vertex in the reduced subgraph, a trajectory exists
    that skips every reward — and an atom at :math:`r = 0` will
    appear in the reward-transformed distribution. O(V + E).
    """
    from collections import deque
    rewards_arr = np.asarray(rewards_1d, dtype=np.float64)

    rewarded_raw = set(int(v) for v in np.where(rewards_arr > 0.0)[0])
    starts = set(int(v) for v in self._starting_vertex_indices())
    absorbing = set(int(v) for v in self._absorbing_vertex_indices())

    # Absorbing vertices have zero sojourn time, so a reward placed
    # on an absorbing vertex contributes nothing to the accumulated
    # reward. Treat absorbing rewards as null for coverage
    # purposes — otherwise rewarding only the absorbing vertex
    # would topologically pass the BFS (every trajectory visits
    # absorbing) while every accumulated reward is actually zero.
    rewarded = rewarded_raw - absorbing

    # BFS over the subgraph excluding rewarded vertices.
    parent: dict[int, int | None] = {v: None for v in starts}
    queue: deque[int] = deque(starts - rewarded)
    visited: set[int] = set(starts - rewarded)
    bad_absorbing: int | None = None
    while queue:
        v = queue.popleft()
        if v in absorbing:
            bad_absorbing = v
            break
        for edge in self.vertex_at(v).edges():
            to_idx = int(edge.to().index())
            if to_idx in rewarded or to_idx in visited:
                continue
            visited.add(to_idx)
            parent[to_idx] = v
            queue.append(to_idx)

    if bad_absorbing is None:
        return  # every trajectory hits at least one rewarded vertex.

    # Witness path reconstruction (parent chain back to a start).
    path = [bad_absorbing]
    cur = bad_absorbing
    while parent.get(cur) is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    path_str = " -> ".join(str(v) for v in path)

    def _trunc(s: set[int], k: int = 10) -> str:
        ss = sorted(s)
        return f"{ss[:k]}{'...' if len(ss) > k else ''}"

    # If all of the user's rewarded vertices were absorbing, the
    # effective rewarded set is empty and *every* trajectory
    # accumulates zero. Surface this as a specific hint so the
    # user knows their reward vector did nothing useful.
    absorbing_only_hint = ""
    absorbing_rewards = rewarded_raw & absorbing
    if absorbing_rewards and not rewarded:
        absorbing_only_hint = (
            "\n  Note: the supplied rewards land only on absorbing\n"
            f"  vertices {sorted(absorbing_rewards)!r}. Absorbing\n"
            "  vertices have zero sojourn time, so they contribute\n"
            "  nothing to the accumulated reward — equivalent to\n"
            "  no rewards at all. Move the reward weight onto\n"
            "  non-absorbing vertices."
        )

    raise ValueError(
        f"{context}: some trajectories accumulate zero reward.\n"
        "\n"
        "The reward-transformed distribution therefore has a point\n"
        "mass at r = 0 with weight 1 - P(visit a rewarded vertex).\n"
        "The continuous density returned by `pdf(t)` integrates to\n"
        "the complement; the atomic mass equals `cdf(0)` of the\n"
        "reward-transformed graph. This is a legitimate mixture\n"
        "distribution, not an error — `Graph.svgd` automatically\n"
        "combines both into a zero-inflated likelihood.\n"
        "\n"
        f"  Rewarded vertices (count: {len(rewarded)}): {_trunc(rewarded)}\n"
        f"  Starting vertices: {_trunc(starts)}\n"
        f"  Absorbing vertices: {_trunc(absorbing)}\n"
        f"  Witness path producing the atom at r = 0:\n"
        f"    {path_str}"
        f"{absorbing_only_hint}\n"
        "\n"
        "If you specifically want a strictly continuous reward-\n"
        "transformed distribution, modify the reward decomposition\n"
        "so every trajectory visits at least one rewarded vertex:\n"
        "  (1) Include more states in the decomposition.\n"
        "  (2) Use `graph.states().T` (full transpose, no slice).\n"
        "  (3) Merge this reward vector with another that covers\n"
        "      the missing trajectories.\n"
        "\n"
        "To skip this check entirely (e.g. you are scripting a\n"
        "one-shot read on a known sub-stochastic distribution),\n"
        "pass `validate_rewards=False` at the public entry point."
    )

def _validate_rewards(
    self,
    rewards,
    *,
    allow_2d: bool = True,
    check_coverage: bool = True,
    coverage_mode: str = "warn",
    context: str = "rewards",
) -> np.ndarray:
    """Unified validator for reward arrays (shape + coverage).

    Shape errors always raise (wrong-length rewards is a bug, not a
    modelling choice). Coverage failures — reward vectors that
    some trajectories skip entirely, producing a point mass at
    :math:`r = 0` — are *not* errors: they describe a legitimate
    mixture distribution (atom + continuous part). The default
    coverage_mode is ``"warn"``; the SVGD path overrides this to
    ``"report"`` so it can automatically wire the zero-inflated
    likelihood without bothering the user.

    Parameters
    ----------
    rewards : array-like
        1D ``(n_vertices,)`` or 2D ``(n_features, n_vertices)``.
    allow_2d : bool
        If False, reject 2D arrays (e.g. when called from
        ``Graph.expectation``, ``Graph.moments``, or any 1D-only
        entry point).
    check_coverage : bool
        If True, run the coverage check (BFS for an unrewarded
        trajectory). If False, skip the check entirely; callers
        opt out via ``validate_rewards=False`` at their public
        entry points.
    coverage_mode : {"warn", "report", "raise"}, default "warn"
        How to handle coverage failures:

        - ``"warn"`` (default): emit one ``UserWarning`` per
          offending feature; no raise. Used by ``Graph.sample``,
          ``Graph.reward_transform``, and any other ad-hoc path
          where the atom at zero is correct and the user just
          needs a heads-up.
        - ``"report"``: never raise, never warn; caller inspects
          the returned array via ``_partial_coverage_features``
          to decide what to do. Used by ``Graph.svgd`` to switch
          to a zero-inflated likelihood automatically.
        - ``"raise"``: raise ``ValueError`` with the witness-path
          diagnostic. Available for explicit input-rejection
          workflows; no in-tree caller uses this today.
    context : str
        Used to label error / warning messages (e.g. "rewards",
        "feature 2 rewards").

    Returns
    -------
    np.ndarray
        The same rewards as a float64 np.ndarray (caller can use
        it directly without re-converting).

    Raises
    ------
    ValueError
        On shape mismatch (always) or coverage failure when
        ``check_coverage=True`` and ``coverage_mode="raise"``.
    """
    arr = np.asarray(rewards, dtype=np.float64)
    n_v = self.vertices_length()

    # ---- Shape check (O(1), always runs) -----------------------
    if arr.ndim == 1:
        if arr.shape[0] != n_v:
            raise ValueError(
                f"{context}: 1D rewards must have shape "
                f"(n_vertices={n_v},), got shape {arr.shape}."
            )
    elif arr.ndim == 2:
        if not allow_2d:
            raise ValueError(
                f"{context}: 2D rewards not accepted here (this "
                f"entry point requires 1D, shape "
                f"(n_vertices={n_v},))."
            )
        if arr.shape[1] != n_v:
            # Friendly transpose suggestion if the user clearly has
            # it backwards: (n_vertices, n_features) instead of
            # (n_features, n_vertices).
            transpose_hint = ""
            if arr.shape[0] == n_v:
                transpose_hint = (
                    "\n  Hint: you appear to have a "
                    f"(n_vertices={n_v}, n_features={arr.shape[1]}) "
                    "array. Transpose it: rewards = rewards.T."
                )
            raise ValueError(
                f"{context}: 2D rewards must have shape "
                f"(n_features, n_vertices={n_v}), got shape "
                f"{arr.shape}.{transpose_hint}"
            )
    else:
        raise ValueError(
            f"{context}: must be 1D (n_vertices={n_v},) or "
            f"2D (n_features, n_vertices={n_v}); "
            f"got shape {arr.shape}."
        )

    # No separate absorbing-vertex check: rewards on absorbing
    # vertices contribute nothing (zero sojourn) but are not an
    # error. If the user supplies a reward vector with weight only
    # on the absorbing vertex, the coverage check below catches
    # the resulting partial coverage and the caller chooses how
    # to respond (warn / report / raise).

    # ---- Coverage check (O(V+E) per feature, gated) ------------
    if not check_coverage:
        return arr

    if coverage_mode not in ("raise", "report", "warn"):
        raise ValueError(
            f"coverage_mode must be one of "
            f"'raise' | 'report' | 'warn'; got {coverage_mode!r}."
        )

    if arr.ndim == 1:
        offenders: list[int] = []
        messages: list[str] = []
        try:
            self._validate_reward_coverage(arr, context=context)
        except ValueError as exc:
            offenders.append(0)
            messages.append(exc.args[0])
    else:
        offenders = []
        messages = []
        for j in range(arr.shape[0]):
            try:
                self._validate_reward_coverage(
                    arr[j], context=f"{context}[feature {j}]"
                )
            except ValueError as exc:
                offenders.append(j)
                messages.append(exc.args[0])

    if not offenders:
        return arr

    if coverage_mode == "raise":
        if arr.ndim == 1:
            raise ValueError(messages[0])
        joined = "\n\n---\n\n".join(messages)
        raise ValueError(
            f"{context}: invalid rewards for features "
            f"{offenders} (of {arr.shape[0]}).\n\n{joined}"
        )
    if coverage_mode == "warn":
        import warnings
        for msg in messages:
            first_line = msg.splitlines()[0]
            warnings.warn(
                f"{first_line} The reward-transformed distribution "
                "has a point mass at r = 0 with weight "
                "1 - P(visit a rewarded vertex); the continuous "
                "density returned by `pdf(t)` covers only r > 0. "
                "Read the atomic mass with `cdf(0)` on the "
                "transformed graph. `Graph.svgd` combines both "
                "into a zero-inflated likelihood automatically.",
                UserWarning,
                stacklevel=3,
            )
    # mode == "report": silent; caller inspects via
    # _partial_coverage_features to act on offenders.

    return arr
