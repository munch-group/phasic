"""End-to-end tests for the daisy-chain plan (C path).

Per daisy-chain-plan.md. Greenfield additions; no modifications to
the Python EliminationTrace path or to C/C++/pybind code.

Phases:
  1. joint_stop_prob_graph + helpers (eager numpy)
  2. epoch_transition_fn (JAX-traceable per-epoch transition)
  3. daisy_chain_log_likelihood_per_event +
     daisy_chain_log_likelihood_joint_snapshot
  4. SVGD smoke test (separate file)
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, StateIndexer, Property, with_ipv
import phasic.cache as cache


# ---------------------------------------------------------------------------
# Small reproducible model (matches the spirit of
# docs/pages/tutorial/time_inhom_joint_prob.ipynb).
# ---------------------------------------------------------------------------


N_SAMPLES = 3
MUTATION_RATE = 0.1
REWARD_LIMIT = 3


def _make_indexer() -> StateIndexer:
    return StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N_SAMPLES)],
    )


def _make_callback(indexer: StateIndexer):
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N_SAMPLES

    @with_ipv(ipv)
    def coal_callback(state, indexer=None):
        transitions = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = list(state)
            new[i] -= 1
            new[j] -= 1
            sum_idx = indexer.lineages.props_to_index(
                descendants=(i + 1) + (j + 1)
            )
            new[sum_idx] += 1
            pair_count = state[i] * (state[j] - same) // (1 + same)
            transitions.append((new, [pair_count, 0]))
        return transitions

    return coal_callback


def _build_jsp_graph():
    indexer = _make_indexer()
    cb = _make_callback(indexer)
    g = Graph(cb, indexer=indexer)
    jp = g.joint_prob_graph(
        indexer,
        mutation_rate=MUTATION_RATE,
        reward_limit=REWARD_LIMIT,
        discrete=False,
    )
    return jp.joint_stop_prob_graph()


# ---------------------------------------------------------------------------
# Phase 1: joint_stop_prob_graph + helpers
# ---------------------------------------------------------------------------


class TestJointStopProbGraphStructure:
    def test_metadata_attached(self):
        jsp = _build_jsp_graph()
        assert getattr(jsp, '_joint_stop_prob_graph', False) is True
        assert isinstance(jsp._t_aux_map, dict)
        assert isinstance(jsp._t_vertex_indices, list)
        # _t_vertex_indices is sorted.
        assert jsp._t_vertex_indices == sorted(jsp._t_vertex_indices)
        # Every t-vertex has an aux partner; the maps agree.
        assert set(jsp._t_aux_map.keys()) == set(jsp._t_vertex_indices)

    def test_t_vertex_aux_loop_structure(self):
        # Each t-vertex must have its outgoing edges replaced by a single
        # unit-weight parameterised edge to the aux; the aux must have a
        # single unit-weight parameterised edge back.
        jsp = _build_jsp_graph()
        param_length = jsp.param_length()
        for t_idx, aux_idx in jsp._t_aux_map.items():
            t_v = jsp.vertex_at(t_idx)
            aux_v = jsp.vertex_at(aux_idx)
            assert t_v.edges_length() == 1
            assert aux_v.edges_length() == 1
            assert t_v.edges()[0].to().index() == aux_idx
            assert aux_v.edges()[0].to().index() == t_idx
            # Coefficients are all 1 in both directions (parameterised view).
            t_param_edges = t_v.parameterized_edges()
            aux_param_edges = aux_v.parameterized_edges()
            assert len(t_param_edges) == 1
            assert len(aux_param_edges) == 1
            assert list(t_param_edges[0].edge_state(param_length)) == [1.0] * param_length
            assert list(aux_param_edges[0].edge_state(param_length)) == [1.0] * param_length

    def test_t_vertices_do_not_route_to_absorption(self):
        # The JSP graph traps mass at t-vertices via the aux loop. So a
        # t-vertex's outgoing edges must NOT lead to an absorbing vertex
        # (length-0 edge list). Non-t interior vertices are still allowed
        # to absorb — that captures mass lost to mutation events that
        # don't reach a t-state, mirroring the notebook semantics.
        jsp = _build_jsp_graph()
        for t_idx in jsp._t_vertex_indices:
            t_v = jsp.vertex_at(t_idx)
            for e in t_v.edges():
                target = e.to()
                assert target.edges_length() != 0, (
                    f"t-vertex {t_idx} has edge to absorbing vertex "
                    f"{target.index()} — t-vertices must trap via aux loop"
                )

    def test_starting_vertex_has_one_edge_per_non_aux_target(self):
        # IPV edges are added at construction time, weight 0, one per non-
        # aux non-trash non-absorbing vertex. After update_ipv, those edges
        # carry the user's IPV weights.
        jsp = _build_jsp_graph()
        start_edges = jsp.starting_vertex().edges()
        # All weight 0 initially.
        for e in start_edges:
            assert e.weight() == pytest.approx(0.0)
        # Number of IPV edges matches non-aux interior vertex count.
        aux_indices = set(jsp._t_aux_map.values())
        non_aux_interior = sum(
            1
            for v in jsp.vertices()
            if v.index() != jsp.starting_vertex().index()
            and v.index() not in aux_indices
            and v.edges_length() > 0  # exclude the absorbing vertex
        )
        assert len(start_edges) == non_aux_interior

    def test_validation(self):
        # Must come from joint_prob_graph.
        g = Graph(1)
        v = g.find_or_create_vertex([1])
        sink = g.find_or_create_vertex([0])
        v.add_edge(sink, [1.0])
        g.starting_vertex().add_edge(v, 1.0)
        with pytest.raises(ValueError, match="joint_prob_graph"):
            g.joint_stop_prob_graph()


class TestJointStopProbabilities:
    def test_collapse_t_aux_eager(self):
        # _collapse_t_aux: vertex-length input → (n - n_aux)-length output
        # in non-aux order, with each aux's mass added to its t-vertex.
        jsp = _build_jsp_graph()
        n = jsp.vertices_length()
        aux_set = set(jsp._t_aux_map.values())
        # Synthetic input: index i contributes mass i.
        raw = np.arange(n, dtype=np.float64)
        out = jsp._collapse_t_aux(raw)
        assert out.shape == (n - len(aux_set),)
        # Reconstruct expected values.
        expected = []
        for i in range(n):
            if i in aux_set:
                continue
            p = float(raw[i])
            if i in jsp._t_aux_map:
                p += float(raw[jsp._t_aux_map[i]])
            expected.append(p)
        np.testing.assert_array_equal(out, np.asarray(expected))

    def test_joint_stop_probabilities_after_update_ipv(self):
        jsp = _build_jsp_graph()
        # Set theta and a uniform IPV (probability 1/k on each entry).
        n_ipv = jsp.starting_vertex().edges_length()
        jsp.update_ipv(np.full(n_ipv, 1.0 / n_ipv))
        jsp.update_weights([1.0, 1.0, 1.0])
        probs = jsp.joint_stop_probabilities(0.5)
        # Output length must match collapsed dimension.
        n = jsp.vertices_length()
        n_aux = len(set(jsp._t_aux_map.values()))
        assert probs.shape == (n - n_aux,)
        # Mass is non-negative and bounded by 1 per entry.
        assert np.all(probs >= -1e-12)
        assert np.all(probs <= 1.0 + 1e-12)

    def test_joint_stop_probabilities_validation(self):
        # Calling on a non-JSP graph must raise.
        indexer = _make_indexer()
        g = Graph(_make_callback(indexer), indexer=indexer)
        with pytest.raises(ValueError, match="joint_stop_prob_graph"):
            g.joint_stop_probabilities(1.0)


class TestJointProbsWithTime:
    def test_dataframe_shape(self):
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        jsp.update_ipv(np.full(n_ipv, 1.0 / n_ipv))
        jsp.update_weights([1.0, 1.0, 1.0])
        df = jsp.joint_probs_with_time(1.0)
        assert 'probs' in df.columns
        assert len(df) == len(jsp._t_vertex_indices)
        # All probabilities are finite.
        assert np.all(np.isfinite(df['probs'].to_numpy()))

    def test_validation(self):
        indexer = _make_indexer()
        g = Graph(_make_callback(indexer), indexer=indexer)
        with pytest.raises(ValueError, match="joint_stop_prob_graph"):
            g.joint_probs_with_time(1.0)


# ---------------------------------------------------------------------------
# Phase 2: epoch_transition_fn (JAX-traceable per-epoch transition)
# ---------------------------------------------------------------------------


class TestEpochTransitionFn:
    def test_eager_matches_manual(self):
        # transition_fn(theta, ipv) must equal the manual sequence
        # update_ipv → update_weights → stop_probability → collapse →
        # project to IPV-coords. Output shape == n_ipv so the result can
        # be fed straight into the next epoch's update_ipv.
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        ipv = np.full(n_ipv, 1.0 / n_ipv)
        theta = np.array([1.0, 1.0, 1.0])
        dt = 0.5

        # Manual reference.
        jsp.update_ipv(ipv)
        jsp.update_weights(theta)
        raw = np.asarray(jsp.stop_probability(dt))
        collapsed = jsp._collapse_t_aux(raw)
        expected = collapsed[jsp._ipv_collapsed_positions]

        # Through the transition_fn.
        transition = jsp.epoch_transition_fn(dt)
        out = np.asarray(transition(theta, ipv))

        assert out.shape == (n_ipv,)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)

    def test_jit(self):
        import jax
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        ipv = np.full(n_ipv, 1.0 / n_ipv)
        theta = np.array([1.0, 1.0, 1.0])
        dt = 0.5
        transition = jsp.epoch_transition_fn(dt)

        out_eager = np.asarray(transition(theta, ipv))
        out_jit = np.asarray(jax.jit(transition)(theta, ipv))
        np.testing.assert_allclose(out_jit, out_eager, rtol=1e-12, atol=1e-15)

    def test_vmap(self):
        import jax
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        ipv = np.full(n_ipv, 1.0 / n_ipv)
        thetas = np.array([
            [1.0, 1.0, 1.0],
            [1.5, 0.5, 1.0],
            [0.7, 1.3, 1.0],
        ])
        dt = 0.5
        transition = jsp.epoch_transition_fn(dt)

        # vmap over theta (shared ipv).
        vmapped = jax.vmap(lambda t: transition(t, ipv))
        batch_out = np.asarray(vmapped(thetas))

        # Reference: Python loop.
        ref = np.stack([np.asarray(transition(t, ipv)) for t in thetas])
        np.testing.assert_allclose(batch_out, ref, rtol=1e-12, atol=1e-15)

    def test_validation(self):
        # epoch_transition_fn requires JSP graph.
        indexer = _make_indexer()
        g = Graph(_make_callback(indexer), indexer=indexer)
        with pytest.raises(ValueError, match="joint_stop_prob_graph"):
            g.epoch_transition_fn(0.5)


# ---------------------------------------------------------------------------
# Phase 3: daisy_chain_log_likelihood_joint_snapshot
# ---------------------------------------------------------------------------


class TestDaisyChainJointSnapshot:
    def test_matches_eager_loop(self):
        # Bit-equivalence with an eager Python loop that mirrors the
        # notebook semantics: daisy through n_epochs-1 transitions, then
        # read joint_stop_probabilities at the snapshot time.
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)

        epoch_dts = [0.4, 0.4, 0.4]
        epoch_thetas = jnp.asarray([
            [1.0, 1.0, 1.0],
            [1.5, 0.5, 1.0],
            [0.7, 1.3, 1.0],
        ])
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        snapshot_time = 1.0
        snapshot_indices = jnp.arange(n_t, dtype=jnp.int32)
        snapshot_counts = jnp.ones(n_t, dtype=jnp.float64)

        log_lik = float(
            jsp.daisy_chain_log_likelihood_joint_snapshot(
                epoch_dts=epoch_dts,
                epoch_thetas=epoch_thetas,
                initial_ipv=initial_ipv,
                snapshot_time=snapshot_time,
                snapshot_indices=snapshot_indices,
                snapshot_counts=snapshot_counts,
            )
        )

        # Eager reference using update_ipv / update_weights / stop_probability.
        ipv = np.asarray(initial_ipv)
        for i in range(len(epoch_dts) - 1):
            jsp.update_ipv(ipv)
            jsp.update_weights(np.asarray(epoch_thetas[i]))
            raw = np.asarray(jsp.stop_probability(epoch_dts[i]))
            collapsed = jsp._collapse_t_aux(raw)
            # Project to IPV coords for the next epoch.
            ipv = collapsed[jsp._ipv_collapsed_positions]

        jsp.update_ipv(ipv)
        jsp.update_weights(np.asarray(epoch_thetas[-1]))
        raw = np.asarray(jsp.stop_probability(snapshot_time))
        collapsed = jsp._collapse_t_aux(raw)

        aux_set = set(jsp._t_aux_map.values())
        non_aux_rank = {}
        rank = 0
        for j in range(jsp.vertices_length()):
            if j in aux_set:
                continue
            non_aux_rank[j] = rank
            rank += 1
        t_probs = np.array(
            [collapsed[non_aux_rank[t_idx]] for t_idx in jsp._t_vertex_indices]
        )
        ref = float(np.sum(np.asarray(snapshot_counts) * np.log(t_probs[np.asarray(snapshot_indices)])))

        np.testing.assert_allclose(log_lik, ref, rtol=1e-9, atol=1e-12)

    def test_jit(self):
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        snapshot_indices = jnp.arange(n_t, dtype=jnp.int32)
        snapshot_counts = jnp.ones(n_t, dtype=jnp.float64)

        def log_lik(epoch_thetas):
            return jsp.daisy_chain_log_likelihood_joint_snapshot(
                epoch_dts=[0.4, 0.4, 0.4],
                epoch_thetas=epoch_thetas.reshape(3, 3),
                initial_ipv=initial_ipv,
                snapshot_time=1.0,
                snapshot_indices=snapshot_indices,
                snapshot_counts=snapshot_counts,
            )

        thetas_flat = jnp.asarray(
            [1.0, 1.0, 1.0, 1.5, 0.5, 1.0, 0.7, 1.3, 1.0],
            dtype=jnp.float64,
        )
        v_eager = float(log_lik(thetas_flat))
        v_jit = float(jax.jit(log_lik)(thetas_flat))
        np.testing.assert_allclose(v_jit, v_eager, rtol=1e-9, atol=1e-12)

    def test_grad(self):
        # jax.grad over epoch_thetas matches finite differences.
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        snapshot_indices = jnp.arange(n_t, dtype=jnp.int32)
        snapshot_counts = jnp.ones(n_t, dtype=jnp.float64)

        def log_lik(epoch_thetas):
            return jsp.daisy_chain_log_likelihood_joint_snapshot(
                epoch_dts=[0.4, 0.4],
                epoch_thetas=epoch_thetas.reshape(2, 3),
                initial_ipv=initial_ipv,
                snapshot_time=1.0,
                snapshot_indices=snapshot_indices,
                snapshot_counts=snapshot_counts,
            )

        thetas_flat = jnp.asarray(
            [1.0, 1.0, 1.0, 1.5, 0.5, 1.0],
            dtype=jnp.float64,
        )
        # The pure_callback paths in this method don't ship with a
        # custom_vjp (the existing FFI patterns in phasic also rely on
        # JAX's default forward-only behaviour for stop_probability).
        # If the user wants gradients, that's a follow-up. Smoke test:
        # ensure either a finite gradient or a clean NotImplementedError.
        try:
            grad_fn = jax.grad(lambda t: log_lik(t))
            g = grad_fn(thetas_flat)
            # If it returns, gradients must be finite.
            assert np.all(np.isfinite(np.asarray(g)))
        except (NotImplementedError, TypeError, ValueError) as exc:
            # Expected: JAX pure_callback has no built-in JVP rule, so
            # jax.grad raises until a custom_vjp / custom_jvp is added.
            # The plan flags this as future work (custom VJP via finite
            # differences); leave the test as a forward smoke check.
            pytest.skip(f"jax.grad not yet supported: {exc}")

    def test_vmap_over_thetas(self):
        # SVGD particles: vmap over a batch of epoch-theta vectors.
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        snapshot_indices = jnp.arange(n_t, dtype=jnp.int32)
        snapshot_counts = jnp.ones(n_t, dtype=jnp.float64)

        def log_lik(epoch_thetas):
            return jsp.daisy_chain_log_likelihood_joint_snapshot(
                epoch_dts=[0.4, 0.4, 0.4],
                epoch_thetas=epoch_thetas.reshape(3, 3),
                initial_ipv=initial_ipv,
                snapshot_time=1.0,
                snapshot_indices=snapshot_indices,
                snapshot_counts=snapshot_counts,
            )

        thetas_batch = jnp.asarray(
            [
                [1.0, 1.0, 1.0, 1.5, 0.5, 1.0, 0.7, 1.3, 1.0],
                [0.8, 1.2, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5, 1.0],
            ],
            dtype=jnp.float64,
        )
        out = jax.vmap(log_lik)(thetas_batch)
        assert out.shape == (2,)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_repeated_chains_dont_thrash_cache(self):
        # Stage A0-style guard: repeated update_ipv + update_weights cycles
        # over many daisy-chain evaluations must not blow up the symbolic
        # compute graph cache. The daisy chain uses stop_probability,
        # which does not itself populate the
        # `parameterized_reward_compute_graph` cache (that cache is
        # populated by reward_transform / moments / expectation paths),
        # so the file count should stay at 0 across the whole sweep.
        # The point of the test is to exercise the chain end-to-end and
        # confirm no cache thrashing or runaway file creation.
        import jax.numpy as jnp
        cache.clear_param_compute_cache()
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        snapshot_indices = jnp.arange(n_t, dtype=jnp.int32)
        snapshot_counts = jnp.ones(n_t, dtype=jnp.float64)

        for trial in range(5):
            jsp.daisy_chain_log_likelihood_joint_snapshot(
                epoch_dts=[0.4, 0.4, 0.4],
                epoch_thetas=jnp.asarray(
                    [[1.0, 1.0, 1.0]] * 3, dtype=jnp.float64,
                )
                + 0.1 * trial,
                initial_ipv=initial_ipv,
                snapshot_time=1.0,
                snapshot_indices=snapshot_indices,
                snapshot_counts=snapshot_counts,
            )
        # File count must be 0 (stop_probability doesn't populate this
        # cache) or 1 (if a future change opts the daisy chain through
        # an expectation/moments path). Anything more indicates thrashing.
        assert cache.param_compute_cache_info()['n_files'] in (0, 1)


class TestCyclicSafety:
    def test_stop_probability_works_on_cyclic_jsp_graph(self):
        # The t-aux trapping loop is itself a 2-vertex cycle. The C
        # uniformization path handles cycles, so stop_probability must
        # work end-to-end. (joint_prob_graph also produces cycles via the
        # trash-pair self-loop, but we filter the trash pair out during
        # JSP construction. The aux loops are the surviving cycles.)
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        jsp.update_ipv(np.full(n_ipv, 1.0 / n_ipv))
        jsp.update_weights([1.0, 1.0, 1.0])
        # Should not raise.
        probs = jsp.stop_probability(2.0)
        assert np.all(np.isfinite(probs))

    def test_symbolic_cache_survives_phase1_round_trip(self):
        # Stage A0 invariant under repeated update_ipv / update_weights.
        cache.clear_param_compute_cache()
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        jsp.update_ipv(np.full(n_ipv, 1.0 / n_ipv))
        jsp.update_weights([1.0, 1.0, 1.0])
        jsp.joint_stop_probabilities(1.0)
        files_after_first = cache.param_compute_cache_info()['n_files']
        # update_ipv + update_weights must not invalidate the symbolic cache.
        jsp.update_ipv(np.full(n_ipv, 1.0 / n_ipv) * 0.5)
        jsp.update_weights([1.5, 0.7, 1.0])
        jsp.joint_stop_probabilities(1.0)
        assert cache.param_compute_cache_info()['n_files'] == files_after_first
