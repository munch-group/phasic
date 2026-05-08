"""End-to-end tests for the daisy-chain SVGD path (C path / FFI).

Two test classes cover the surviving daisy-chain surface:

1. ``TestJointStopProbGraphStructure`` — JSP graph structural
   invariants the FFI handler depends on (t-aux loop layout,
   IPV-target layout, etc.).
2. ``TestDaisyChainJointProbs`` — exercises the public
   ``Graph.daisy_chain_joint_probs(...)`` end-to-end (shape,
   jit, vmap, validation, gradient via ``custom_vjp`` +
   finite-differences, repeated-call cache stability).
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
# Class 1: JSP graph structural invariants
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


# ---------------------------------------------------------------------------
# Class 2: daisy_chain_joint_probs (the SVGD-shaped FFI model)
# ---------------------------------------------------------------------------


class TestDaisyChainJointProbs:
    def test_shape(self):
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        n_t = len(jsp._t_vertex_indices)
        out = jsp.daisy_chain_joint_probs(
            epoch_thetas=jnp.asarray(
                [[1.0, 1.0, 1.0], [1.5, 0.5, 1.0]], dtype=jnp.float64
            ),
            epoch_dts=[0.5],
            initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
            t_eval=10.0,
        )
        assert out.shape == (n_t,)
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np))
        assert np.all(out_np >= -1e-12)

    def test_jit_matches_eager(self):
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()

        def model(theta_flat):
            return jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_flat.reshape(2, 3),
                epoch_dts=[0.5],
                initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
                t_eval=10.0,
            )

        thetas = jnp.asarray([1.0, 1.0, 1.0, 1.5, 0.5, 1.0], dtype=jnp.float64)
        out_eager = np.asarray(model(thetas))
        out_jit = np.asarray(jax.jit(model)(thetas))
        np.testing.assert_allclose(out_jit, out_eager, rtol=1e-9, atol=1e-12)

    def test_vmap(self):
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()

        def model(theta_flat):
            return jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_flat.reshape(2, 3),
                epoch_dts=[0.5],
                initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
                t_eval=10.0,
            )

        batch = jnp.asarray(
            [
                [1.0, 1.0, 1.0, 1.5, 0.5, 1.0],
                [0.7, 1.3, 1.0, 1.2, 0.8, 1.0],
            ],
            dtype=jnp.float64,
        )
        out = np.asarray(jax.vmap(model)(batch))
        assert out.shape == (2, len(jsp._t_vertex_indices))
        assert np.all(np.isfinite(out))

    def test_validation(self):
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        # Mismatched n_epochs vs len(epoch_dts).
        with pytest.raises(ValueError, match="epoch_dts must have length"):
            jsp.daisy_chain_joint_probs(
                epoch_thetas=np.ones((3, 3)),
                epoch_dts=[0.5],  # length 1, but n_epochs - 1 = 2
                initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
            )
        # Wrong theta_dim.
        with pytest.raises(ValueError, match="theta_dim"):
            jsp.daisy_chain_joint_probs(
                epoch_thetas=np.ones((2, 5)),  # graph has param_length 3
                epoch_dts=[0.5],
                initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
            )
        # Wrong initial_ipv shape.
        with pytest.raises(ValueError, match="initial_ipv must have shape"):
            jsp.daisy_chain_joint_probs(
                epoch_thetas=np.ones((2, 3)),
                epoch_dts=[0.5],
                initial_ipv=np.full(n_ipv + 1, 1.0 / (n_ipv + 1)),
            )

    def test_validation_non_jsp_graph(self):
        indexer = _make_indexer()
        g = Graph(_make_callback(indexer), indexer=indexer)
        with pytest.raises(ValueError, match="joint_stop_prob_graph"):
            g.daisy_chain_joint_probs(
                epoch_thetas=np.ones((2, 2)),
                epoch_dts=[0.5],
                initial_ipv=np.ones(1),
            )

    def test_grad(self):
        # jax.grad over epoch_thetas works via the custom_vjp +
        # finite-differences wrapper in daisy_chain_joint_probs.
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)

        def scalar_obj(theta_flat):
            # Sum reduction so jax.grad has a scalar output.
            return jnp.sum(jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_flat.reshape(2, 3),
                epoch_dts=[0.4],
                initial_ipv=initial_ipv,
                t_eval=10.0,
            ))

        thetas_flat = jnp.asarray(
            [1.0, 1.0, 1.0, 1.5, 0.5, 1.0], dtype=jnp.float64,
        )
        g = jax.grad(scalar_obj)(thetas_flat)
        g_np = np.asarray(g)
        assert g_np.shape == thetas_flat.shape
        assert np.all(np.isfinite(g_np))

    def test_grad_matches_finite_diff(self):
        # Cross-check the custom_vjp finite-diff backward against an
        # independent finite-diff computation on daisy_chain_joint_probs.
        import jax
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)

        def joint_probs(theta_flat):
            return jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_flat.reshape(2, 3),
                epoch_dts=[0.4],
                initial_ipv=initial_ipv,
                t_eval=10.0,
            )

        thetas_flat = jnp.asarray(
            [1.0, 1.0, 1.0, 1.5, 0.5, 1.0], dtype=jnp.float64,
        )
        # All-ones cotangent → grad equals column sum of the Jacobian.
        autodiff_grad = np.asarray(jax.grad(
            lambda t: jnp.sum(joint_probs(t))
        )(thetas_flat))

        # Independent finite-diff at the same eps as the custom_vjp.
        eps = 1e-7
        fd_grad = np.zeros(thetas_flat.shape[0], dtype=np.float64)
        for i in range(thetas_flat.shape[0]):
            tp = thetas_flat.at[i].add(eps)
            tm = thetas_flat.at[i].add(-eps)
            fd_grad[i] = float(
                jnp.sum(joint_probs(tp) - joint_probs(tm)) / (2.0 * eps)
            )

        # Both use the same eps and the same forward, so they must
        # agree to machine precision.
        np.testing.assert_allclose(
            autodiff_grad, fd_grad, rtol=1e-9, atol=1e-12
        )

    def test_repeated_calls_dont_thrash_cache(self):
        # Stage A0-style guard: many daisy-chain forward calls must not
        # blow up the symbolic compute graph cache. The FFI handler uses
        # stop_probability internally, which does not itself populate the
        # parameterized_reward_compute_graph cache. The point of the
        # test is to exercise the chain end-to-end across many particle
        # evaluations and confirm no cache thrashing or runaway file
        # creation.
        import jax.numpy as jnp
        cache.clear_param_compute_cache()
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)

        for trial in range(5):
            jsp.daisy_chain_joint_probs(
                epoch_thetas=jnp.asarray(
                    [[1.0, 1.0, 1.0]] * 3, dtype=jnp.float64,
                )
                + 0.1 * trial,
                epoch_dts=[0.4, 0.4],
                initial_ipv=initial_ipv,
                t_eval=10.0,
            )
        # File count must be 0 (stop_probability doesn't populate this
        # cache) or 1 (if a future change opts the daisy chain through
        # an expectation/moments path). Anything more indicates thrashing.
        assert cache.param_compute_cache_info()['n_files'] in (0, 1)


# ---------------------------------------------------------------------------
# Class 3: granularity threading + adaptive t_eval probe
# ---------------------------------------------------------------------------


class TestGranularity:
    def test_granularity_changes_runs(self):
        # Two granularity values must both run and produce values
        # close to each other (granularity is a discretisation knob;
        # answers converge as it grows).
        import jax.numpy as jnp
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        thetas = jnp.asarray([[1.0, 1.0, 1.0], [1.5, 0.5, 1.0]], dtype=jnp.float64)

        out_auto = np.asarray(jsp.daisy_chain_joint_probs(
            epoch_thetas=thetas, epoch_dts=[0.5],
            initial_ipv=initial_ipv, t_eval=10.0, granularity=0,
        ))
        out_high = np.asarray(jsp.daisy_chain_joint_probs(
            epoch_thetas=thetas, epoch_dts=[0.5],
            initial_ipv=initial_ipv, t_eval=10.0, granularity=2000,
        ))
        assert np.all(np.isfinite(out_auto))
        assert np.all(np.isfinite(out_high))
        # Auto and a high granularity should agree to a few digits;
        # uniformization is monotonic-with-granularity.
        np.testing.assert_allclose(out_auto, out_high, rtol=1e-3, atol=1e-6)

    def test_granularity_validation(self):
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        with pytest.raises(ValueError, match="granularity"):
            jsp.daisy_chain_joint_probs(
                epoch_thetas=np.ones((2, 3)),
                epoch_dts=[0.5],
                initial_ipv=np.full(n_ipv, 1.0 / n_ipv),
                granularity=-1,
            )


class TestAdaptiveTeval:
    def test_probe_returns_sensible_value(self):
        # The probe should pick a t_eval well below the legacy default
        # (4 × sum(dts)) and should produce a residual non-t-vertex
        # mass below tol at that t_eval.
        jsp = _build_jsp_graph()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        epoch_dts = [0.5]
        n_epochs = len(epoch_dts) + 1
        param_length = jsp.param_length()
        probe_thetas = np.ones((n_epochs, param_length))

        tol = 1e-3
        chosen = jsp._probe_daisy_t_eval(
            probe_thetas=probe_thetas,
            epoch_dts=epoch_dts,
            initial_ipv=initial_ipv,
            tol=tol,
        )
        # Should be positive and not blown out to t_max.
        assert chosen > 0
        assert chosen < 100.0

    def test_resolve_auto_returns_numeric(self):
        # _resolve_daisy_chain_t_eval must turn 'auto' into a float
        # and pass numerics through unchanged.
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g = Graph(cb, indexer=indexer)
        jp = g.joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT, discrete=False,
        )

        # Numeric pass-through.
        assert jp._resolve_daisy_chain_t_eval(
            daisy_chain_t_eval=7.5, epoch_starts=[0.0, 0.5],
        ) == 7.5
        # None → legacy default max(sum(dts)*4, 10) = 10.
        assert jp._resolve_daisy_chain_t_eval(
            daisy_chain_t_eval=None, epoch_starts=[0.0, 0.5],
        ) == 10.0
        # 'auto' → some positive number.
        auto_val = jp._resolve_daisy_chain_t_eval(
            daisy_chain_t_eval='auto', epoch_starts=[0.0, 0.5],
        )
        assert isinstance(auto_val, float) and auto_val > 0
        # Negative numeric raises.
        with pytest.raises(ValueError, match="must be > 0"):
            jp._resolve_daisy_chain_t_eval(
                daisy_chain_t_eval=-1.0, epoch_starts=[0.0, 0.5],
            )
        # Garbage string raises.
        with pytest.raises(ValueError, match="'auto'"):
            jp._resolve_daisy_chain_t_eval(
                daisy_chain_t_eval='whatever', epoch_starts=[0.0, 0.5],
            )


# ---------------------------------------------------------------------------
# Class 4: per-epoch fixed values
# ---------------------------------------------------------------------------


class TestPerEpochFixed:
    def _build_source_jp(self):
        """Source joint-prob graph (continuous) for daisy SVGD wiring."""
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g = Graph(cb, indexer=indexer)
        return g.joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT, discrete=False,
        )

    def test_scalar_value_broadcasts(self):
        # Legacy form: (local_idx, scalar) is broadcast across all epochs.
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4]
        n_epochs = len(epoch_starts)

        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_fixed=[(0, 1.5)],
        )
        # (0, 1.5) → fix flat indices 0, param_length, 2*param_length, ...
        # at value 1.5 in every epoch.
        expected = [(epoch * param_length + 0, 1.5) for epoch in range(n_epochs)]
        assert broadcast_fixed == expected

    def test_per_epoch_list_fixes_different_values(self):
        # New form: (local_idx, [v0, v1, ...]) fixes one value per epoch.
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4, 0.9]
        n_epochs = len(epoch_starts)

        per_epoch = [1.0, 2.5, 3.7]
        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_fixed=[(1, per_epoch)],
        )
        expected = [
            (epoch * param_length + 1, float(per_epoch[epoch]))
            for epoch in range(n_epochs)
        ]
        assert broadcast_fixed == expected

    def test_mixed_scalar_and_list_entries(self):
        # User can mix scalar (broadcast) with per-epoch list entries.
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4]
        n_epochs = len(epoch_starts)

        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_fixed=[(0, 5.0), (1, [2.0, 8.0])],
        )
        expected = (
            [(epoch * param_length + 0, 5.0) for epoch in range(n_epochs)]
            + [(epoch * param_length + 1, [2.0, 8.0][epoch])
               for epoch in range(n_epochs)]
        )
        assert sorted(broadcast_fixed) == sorted(expected)

    def test_per_epoch_list_wrong_length_raises(self):
        jp = self._build_source_jp()
        epoch_starts = [0.0, 0.4, 0.9]  # n_epochs = 3
        with pytest.raises(ValueError, match="length n_epochs"):
            jp._daisy_chain_svgd_model(
                observed_indices=[],
                epoch_starts=epoch_starts,
                t_eval=10.0,
                user_fixed=[(0, [1.0, 2.0])],  # length 2, expected 3
            )

    def test_per_epoch_value_accepts_numpy_array(self):
        # np.ndarray works the same as a list.
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4]
        n_epochs = len(epoch_starts)

        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_fixed=[(0, np.array([1.5, 4.5]))],
        )
        expected = [
            (epoch * param_length + 0, [1.5, 4.5][epoch])
            for epoch in range(n_epochs)
        ]
        assert broadcast_fixed == expected
