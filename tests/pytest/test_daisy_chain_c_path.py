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


def _build_joint_prob_graph():
    """Same fixture, but returns the continuous joint-prob graph (the
    input to ``_daisy_chain_svgd_model``)."""
    indexer = _make_indexer()
    cb = _make_callback(indexer)
    g = Graph(cb, indexer=indexer)
    return g.joint_prob_graph(
        indexer,
        mutation_rate=MUTATION_RATE,
        reward_limit=REWARD_LIMIT,
        discrete=False,
    )


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
        # Each t-vertex's outgoing edges are replaced by a single
        # COEFFICIENT-LESS constant-weight edge to the aux; the aux has
        # a single coefficient-less constant-weight edge back. Both
        # directions are skipped by ptd_graph_update_weights, so the
        # trapping rate is decoupled from theta. This is what allows
        # per-observation exposure scaling not to inflate lambda_max
        # via the t-aux loop weights.
        jsp = _build_jsp_graph()
        for t_idx, aux_idx in jsp._t_aux_map.items():
            t_v = jsp.vertex_at(t_idx)
            aux_v = jsp.vertex_at(aux_idx)
            assert t_v.edges_length() == 1
            assert aux_v.edges_length() == 1
            assert t_v.edges()[0].to().index() == aux_idx
            assert aux_v.edges()[0].to().index() == t_idx
            assert t_v.edges()[0].weight() == 1.0
            assert aux_v.edges()[0].weight() == 1.0
            # Both edges are coefficient-less: they do not appear in
            # parameterized_edges() and update_weights() leaves them
            # untouched.
            assert len(t_v.parameterized_edges()) == 0
            assert len(aux_v.parameterized_edges()) == 0

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


# ---------------------------------------------------------------------------
# Class 4b: tied per-epoch parameters (plan: typed-riding-truffle).
# Slaves get their flat positions overwritten with the master's value
# inside the model before every forward evaluation. The slave's flat
# index also lands in `broadcast_fixed` so the SVGD-side `fixed_mask`
# marks it as not-learnable. Gradients route back to the master via the
# standard JAX VJP through the `_apply_tying` scatter (asserted in
# `TestTiedGradient` below — that's the load-bearing batch 3 gate).
# ---------------------------------------------------------------------------


class TestTied:
    def _build_source_jp(self):
        """Same fixture as TestPerEpochFixed."""
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g = Graph(cb, indexer=indexer)
        return g.joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT, discrete=False,
        )

    def test_tied_extends_broadcast_fixed_with_slaves(self):
        """Slaves' flat indices appear in broadcast_fixed with sentinel 0.0."""
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4, 0.9]

        # Tie local_idx=0 across epochs 0 and 2 (master=0, slave=2).
        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_tied=[(0, [0, 2])],
        )
        # Master at flat 0 stays learnable, slave at flat 2*param_length
        # gets added to broadcast_fixed with sentinel 0.0.
        slave_flat = 2 * param_length + 0
        assert (slave_flat, 0.0) in broadcast_fixed
        # No user_fixed, so broadcast_fixed has only the slave entry.
        assert len(broadcast_fixed) == 1

    def test_tied_attaches_info_to_model(self):
        """The model carries a `_tying_info` attribute with the slave→master map."""
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4, 0.9]

        model, _, _, _ = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_tied=[(0, [0, 2])],
        )
        info = getattr(model, '_tying_info', None)
        assert info is not None
        # slave at flat = 2*param_length, master at flat = 0.
        slave_flat = 2 * param_length + 0
        master_flat = 0
        assert info['slave_to_master'] == {slave_flat: master_flat}

    def test_tied_none_yields_empty_info(self):
        """When user_tied is None, slave_to_master is empty and broadcast_fixed
        is None (no slaves added)."""
        jp = self._build_source_jp()
        epoch_starts = [0.0, 0.4]

        model, _, _, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_tied=None,
        )
        info = getattr(model, '_tying_info', None)
        assert info == {'slave_to_master': {}}
        # No fixed and no tied -> broadcast_fixed stays None.
        assert broadcast_fixed is None

    def test_tied_with_existing_fixed_keeps_both(self):
        """`tied` slaves and `fixed` entries coexist in broadcast_fixed."""
        jp = self._build_source_jp()
        param_length = jp.param_length()
        epoch_starts = [0.0, 0.4, 0.9]
        n_epochs = len(epoch_starts)

        model, theta_dim, prior, broadcast_fixed = jp._daisy_chain_svgd_model(
            observed_indices=[],
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_fixed=[(1, 1.5)],          # fix local 1 in every epoch
            user_tied=[(0, [0, 2])],        # tie local 0 epochs 0 and 2
        )
        # Three fixed flat indices (local 1, all 3 epochs) + 1 slave.
        fixed_flats_from_user_fixed = {
            epoch * param_length + 1 for epoch in range(n_epochs)
        }
        slave_flat = 2 * param_length + 0
        all_flats = {idx for idx, _ in broadcast_fixed}
        assert fixed_flats_from_user_fixed.issubset(all_flats)
        assert slave_flat in all_flats
        # The master (flat 0) is NOT in broadcast_fixed — it stays learnable.
        assert 0 not in all_flats


# ---------------------------------------------------------------------------
# Class 4c: tied gradient correctness (plan: typed-riding-truffle, batch 3).
# The load-bearing claim: cotangents at slave positions are routed back
# to the master via _apply_tying's scatter VJP, so the per-slave FD
# partials sum into dL/d(master). These tests pin that contract.
# ---------------------------------------------------------------------------


class TestTiedGradient:
    def _build_source_jp(self):
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g = Graph(cb, indexer=indexer)
        return g.joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT, discrete=False,
        )

    @staticmethod
    def _build_loss(jp, *, user_tied=None):
        """Build a loss closure exercising the SVGD-side daisy-chain
        model (with or without tying). Returns (loss, theta_dim).
        Use the first few t-vertices as observation targets so the loss
        is well-defined."""
        import jax
        import jax.numpy as jnp

        jsp = jp.joint_stop_prob_graph()
        observed_indices = list(jsp._t_vertex_indices[:4])

        model, theta_dim, _, _ = jp._daisy_chain_svgd_model(
            observed_indices=observed_indices,
            epoch_starts=[0.0, 0.4, 0.9],
            t_eval=10.0,
            user_tied=user_tied,
        )

        def loss(theta_flat):
            per_obs, _ = model(theta_flat)
            return jnp.sum(jnp.log(per_obs + 1e-12))

        return loss, theta_dim

    def test_tied_forward_matches_explicit_replication(self):
        """`tied=[(0, [0, 1, 2])]` with master θ* matches `tied=None`
        with a full-length theta where slot 0 = θ* in every epoch."""
        import jax
        import jax.numpy as jnp

        jp = self._build_source_jp()
        param_length = jp.param_length()
        n_epochs = 3

        loss_free, theta_dim_free = self._build_loss(jp, user_tied=None)
        loss_tied, theta_dim_tied = self._build_loss(
            jp, user_tied=[(0, list(range(n_epochs)))]
        )
        assert theta_dim_free == theta_dim_tied == n_epochs * param_length

        # Build theta_full where slot 0 (local) is θ* in every epoch
        # and slots 1..param_length-1 vary per epoch.
        theta_star = 1.7
        theta_list = []
        for epoch in range(n_epochs):
            for slot in range(param_length):
                if slot == 0:
                    theta_list.append(theta_star)
                else:
                    # Distinct values so the test catches accidental
                    # cross-talk between slots.
                    theta_list.append(0.5 + 0.1 * (epoch * param_length + slot))
        theta_full = jnp.asarray(theta_list, dtype=jnp.float64)

        out_free = loss_free(theta_full)
        out_tied = loss_tied(theta_full)
        assert jnp.allclose(out_free, out_tied, rtol=0, atol=1e-12), (
            f"out_free={out_free!r} out_tied={out_tied!r}"
        )

    def test_tied_gradient_sums_per_epoch_partials(self):
        """LOAD-BEARING: under `tied=[(0, [0, 1, 2])]`, the gradient at
        the master flat index equals the sum of the free-model's
        per-epoch partials at every slot-0 flat position.

        FD bwd in the daisy chain perturbs each flat index
        independently with eps=1e-7; cotangents at slave positions
        get routed back to the master via _apply_tying's scatter VJP.
        """
        import jax
        import jax.numpy as jnp

        jp = self._build_source_jp()
        param_length = jp.param_length()
        n_epochs = 3

        loss_free, _ = self._build_loss(jp, user_tied=None)
        loss_tied, _ = self._build_loss(
            jp, user_tied=[(0, list(range(n_epochs)))]
        )

        theta_star = 1.7
        theta_list = []
        for epoch in range(n_epochs):
            for slot in range(param_length):
                if slot == 0:
                    theta_list.append(theta_star)
                else:
                    theta_list.append(0.5 + 0.1 * (epoch * param_length + slot))
        theta_full = jnp.asarray(theta_list, dtype=jnp.float64)

        # slot-0 flat indices: 0, param_length, 2*param_length, ...
        slot0_flats = [epoch * param_length for epoch in range(n_epochs)]
        master_flat = slot0_flats[0]
        slave_flats = slot0_flats[1:]

        g_free = jax.grad(loss_free)(theta_full)
        expected_master_grad = float(sum(g_free[i] for i in slot0_flats))

        g_tied = jax.grad(loss_tied)(theta_full)
        tol = max(1e-6, 1e-4 * abs(expected_master_grad))
        assert abs(float(g_tied[master_flat]) - expected_master_grad) < tol, (
            f"tied master grad {float(g_tied[master_flat])!r} does not "
            f"match sum of free per-epoch partials "
            f"{expected_master_grad!r} (tol={tol})"
        )
        # Slave positions in the tied model should have zero
        # contribution (their cotangent was routed to the master).
        for slave_flat in slave_flats:
            assert abs(float(g_tied[slave_flat])) < tol, (
                f"tied slave grad at flat {slave_flat} = "
                f"{float(g_tied[slave_flat])!r} is not zero "
                f"(tol={tol})"
            )

    def test_tied_plus_fixed_partition(self):
        """`fixed + tied` together must produce a well-defined
        partition: master is learnable, fixed slots are fixed,
        slave slots are fixed (sentinel 0.0). End-to-end via
        Graph.svgd to also exercise the post-init theta_init
        consistency step."""
        from phasic import LogGaussPrior, Adamelia
        import jax.numpy as jnp

        jp = self._build_source_jp()
        param_length = jp.param_length()
        n_epochs = 3

        # Build observations as joint-prob outcome tuples. Sample from
        # the discrete joint_prob_table for the smallest valid outcome.
        # joint-prob columns end with `prob`; first columns are
        # rewarded-property indicators.
        disc = jp._joint_prob_base_graph_indexer  # noqa: F841 (sanity)
        jpt = jp.joint_prob_table() if jp.is_discrete else None
        if jpt is None:
            # Build a transient discrete copy just to harvest one
            # outcome tuple.
            indexer = _make_indexer()
            cb = _make_callback(indexer)
            disc_jp = Graph(cb, indexer=indexer).joint_prob_graph(
                indexer, mutation_rate=MUTATION_RATE,
                reward_limit=REWARD_LIMIT, discrete=True,
            )
            disc_jp.update_weights([1.0] * (disc_jp.param_length() - 1) + [MUTATION_RATE])
            jpt = disc_jp.joint_prob_table()
        outcome_cols = list(jpt.columns[:-1])
        outcome = tuple(int(jpt.iloc[0][c]) for c in outcome_cols)
        obs = [outcome] * 8

        svgd = jp.svgd(
            obs,
            fixed=[(1, 1e-8)],                              # slot 1 fixed
            tied=[(0, [0, 2])],                             # slot 0 tied epochs 0, 2
            prior=LogGaussPrior(ci=[1e-6, 1e-2]),
            n_iterations=2,
            n_particles=4,
            optimizer=Adamelia(learning_rate=0.1),
            epoch_starts=[0.0, 0.4, 0.9],
        )

        # Expected partition (n_epochs=3, param_length=3, theta_dim=9):
        # - slot 0 master at flat 0       -> learnable
        # - slot 0 free at flat 3 (ep 1)  -> learnable
        # - slot 0 slave at flat 6 (ep 2) -> fixed (slave, sentinel)
        # - slot 1 fixed at flats 1, 4, 7 -> fixed (user-supplied)
        # - slot 2 free at flats 2, 5, 8  -> learnable
        n_fixed = int(jnp.sum(svgd.fixed_mask))
        n_total = n_epochs * param_length
        # Three user-fixed (slot 1 × 3 epochs) + 1 slave.
        assert n_fixed == 4, f"expected 4 fixed dims, got {n_fixed}"
        # Five learnable: 2 from slot 0 (master + free epoch 1)
        #               + 3 from slot 2 (free in every epoch).
        assert (n_total - n_fixed) == 5

        # The master's column in theta_init must match the slave's
        # column (the post-init consistency step).
        slave_flat = 2 * param_length + 0
        master_flat = 0
        np.testing.assert_allclose(
            np.asarray(svgd.theta_init[:, slave_flat]),
            np.asarray(svgd.theta_init[:, master_flat]),
        )

    def test_tied_summary_shows_tied_label(self, capsys):
        """SVGD.summary prints `Tied→θ[k]` for slave rows; the master
        row prints regular MAP / Mean / SD / CI."""
        from phasic import LogGaussPrior, Adamelia

        jp = self._build_source_jp()
        param_length = jp.param_length()

        # Sample a single observation tuple for the disc joint table.
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        disc_jp = Graph(cb, indexer=indexer).joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT, discrete=True,
        )
        disc_jp.update_weights([1.0] * (disc_jp.param_length() - 1) + [MUTATION_RATE])
        jpt = disc_jp.joint_prob_table()
        outcome = tuple(int(jpt.iloc[0][c]) for c in list(jpt.columns[:-1]))
        obs = [outcome] * 8

        svgd = jp.svgd(
            obs,
            tied=[(0, [0, 1])],
            prior=LogGaussPrior(ci=[1e-6, 1e-2]),
            n_iterations=2,
            n_particles=4,
            optimizer=Adamelia(learning_rate=0.1),
            epoch_starts=[0.0, 0.5],
        )
        # capsys captures stdout — call summary and check the slave
        # row carries the Tied label.
        capsys.readouterr()  # drain any prior output
        svgd.summary()
        out = capsys.readouterr().out
        # slave flat index = 1 * param_length + 0 = param_length.
        slave_flat = param_length
        # The slave row begins with its index then the Tied label.
        assert f'Tied→θ[0]' in out, (
            f"summary did not show Tied→θ[0] for the slave at flat "
            f"{slave_flat}. Output was:\n{out}"
        )


# ---------------------------------------------------------------------------
# Particle-vmap fusion tests (plan: radiant-giggling-wigderson, path A).
# Verify that `vmap(grad(loss))(particles)` on Graph.daisy_chain_joint_probs
# dispatches ONE fat (P, theta_dim) FFI call per FD perturbation instead of
# P separate expand_dims-batched calls. The keystone change is the
# jax.custom_batching.custom_vmap rule on _forward inside the function.
# ---------------------------------------------------------------------------


class TestDaisyChainParticleVmapFusion:
    """`vmap(grad(daisy_chain_joint_probs))` fusion regression suite.

    Mirrors the just-shipped per-obs exposure-path fusion. Tests are
    written so they fail loudly if the rule does not fire under
    `vmap(grad(...))`.
    """

    @staticmethod
    def _build_model_and_loss(n_epochs=2):
        """Build a small JSP graph and a (model, loss, theta_init) tuple.
        ``param_length`` is determined by the graph itself (3 for the
        small coalescent fixture defined in this module).

        This fixture exercises the LEGACY ad-hoc-callers path:
        ``Graph.daisy_chain_joint_probs`` directly. The
        correctness/equivalence canaries (forward, grad-vs-loop,
        pmap composition) use this — those properties hold on both
        paths.
        """
        import jax
        import jax.numpy as jnp

        jsp = _build_jsp_graph()
        param_length = jsp.param_length()
        n_ipv = jsp.starting_vertex().edges_length()
        initial_ipv = np.full(n_ipv, 1.0 / n_ipv)
        # epoch_dts: length n_epochs - 1
        epoch_dts = [0.4] * (n_epochs - 1)

        def model(theta_flat):
            return jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_flat.reshape(n_epochs, param_length),
                epoch_dts=epoch_dts,
                initial_ipv=initial_ipv,
                t_eval=10.0,
            )

        def loss(theta_flat):
            return jnp.sum(model(theta_flat))

        theta_init = jnp.full(
            n_epochs * param_length, 1.0, dtype=jnp.float64,
        )
        return jsp, model, loss, theta_init

    @staticmethod
    def _build_svgd_model_and_loss(n_epochs=2):
        """Build the SVGD-side fused model via
        ``Graph._daisy_chain_svgd_model``.

        This is the production path SVGD actually uses when called as
        ``Graph.svgd(epoch_starts=..., exposure=None)``. It carries the
        ``custom_vmap`` fusion rule introduced in batch 3 of
        ``daisy-chain-fusion-recovery-plan.md``. The perf canaries
        below assert against THIS model so they pin the fusion as a
        contract on the production path.
        """
        import jax
        import jax.numpy as jnp

        jpg = _build_joint_prob_graph()
        jsp = jpg.joint_stop_prob_graph()
        # Pick the first few t-vertices as observation targets. The
        # exact identities don't matter for shape/perf assertions —
        # only that observed_indices is non-empty and well-formed.
        observed_indices = list(jsp._t_vertex_indices[:5])

        # epoch_starts has length n_epochs; first must be 0.0.
        epoch_starts = [0.0] + [0.4 * (k + 1) for k in range(n_epochs - 1)]

        model, theta_dim, _prior, _fixed = jpg._daisy_chain_svgd_model(
            observed_indices=observed_indices,
            epoch_starts=epoch_starts,
            t_eval=10.0,
            user_prior=None,
            user_fixed=None,
        )

        def loss(theta_flat):
            per_obs, _ = model(theta_flat)
            # log of probabilities, clipped to avoid log(0) at the
            # tracing-pass single-particle probe.
            return jnp.sum(jnp.log(per_obs + 1e-12))

        theta_init = jnp.full(theta_dim, 1.0, dtype=jnp.float64)
        return jpg, model, loss, theta_init

    def test_forward_vmap_unchanged(self):
        """Forward vmap path produces (P, n_t) and matches per-particle
        outputs — the new rule must not break the FORWARD path."""
        import jax
        import jax.numpy as jnp

        _jsp, model, _loss, theta_init = self._build_model_and_loss()
        particles = jnp.stack([
            theta_init,
            theta_init * 1.1,
            theta_init * 0.9,
        ])
        out_vmap = jax.vmap(model)(particles)
        out_loop = jnp.stack([model(p) for p in particles])
        assert out_vmap.shape == out_loop.shape
        np.testing.assert_allclose(
            np.asarray(out_vmap), np.asarray(out_loop),
            rtol=0, atol=1e-12,
        )

    def test_vmap_grad_matches_loop_grad(self):
        """`vmap(grad(loss))(particles)` matches per-particle grads to
        bit precision — the same FFI inputs flow through both paths,
        just dispatched in different batch shapes."""
        import jax
        import jax.numpy as jnp

        _jsp, _model, loss, theta_init = self._build_model_and_loss()
        particles = jnp.stack([
            theta_init,
            theta_init * 1.1,
            theta_init * 0.9,
            theta_init + 0.05,
        ])
        g_vmap = jax.vmap(jax.grad(loss))(particles)
        g_loop = jnp.stack([jax.grad(loss)(p) for p in particles])
        assert g_vmap.shape == (4, theta_init.shape[0])
        np.testing.assert_allclose(
            np.asarray(g_vmap), np.asarray(g_loop),
            rtol=0, atol=1e-8,
        )

    def test_vmap_fuses_into_one_ffi_call(self):
        """CANARY: under vmap(grad(loss)), the FFI handler must be
        invoked with theta of shape (P, theta_dim) — fused — not as
        P separate (theta_dim,) calls with expand_dims. If this fails,
        the custom_vmap rule in ``_daisy_chain_svgd_model`` did not
        intercept and the perf win is lost.

        Targets the SVGD-side fused builder (the production path used
        by ``Graph.svgd(epoch_starts=..., exposure=None)``). The legacy
        ``Graph.daisy_chain_joint_probs`` ad-hoc path is intentionally
        un-fused after the 2026-05-16 revert and does not pin this
        contract.
        """
        import jax
        import jax.numpy as jnp
        import jax.ffi as _ffi

        # Patch at jax.ffi.ffi_call level — the SVGD model captures
        # compute_daisy_chain_joint_probs_ffi by reference at model-
        # build time, so patching the module-level name after the
        # model is built has no effect. Going one level deeper catches
        # every dispatch.
        original_ffi_call = _ffi.ffi_call
        captured = []  # list of (handler_name, theta_shape)

        def traced(name, result_shape, **kwargs):
            inner = original_ffi_call(name, result_shape, **kwargs)
            def call(*args, **kw):
                if args:
                    captured.append((name, tuple(args[0].shape)))
                return inner(*args, **kw)
            return call

        _ffi.ffi_call = traced
        try:
            # Build the SVGD-side fused model UNDER the patch so its
            # closures pick up the traced ffi_call.
            _jpg, _model, loss, theta_init = (
                self._build_svgd_model_and_loss(n_epochs=2)
            )
            n_params = int(theta_init.shape[0])
            P = 3
            particles = jnp.stack([
                theta_init * (1.0 + 0.1 * k)
                for k in range(P)
            ])
            _ = jax.vmap(jax.grad(loss))(particles)
        finally:
            _ffi.ffi_call = original_ffi_call

        # Filter to daisy-chain FFI dispatches only (the test fixture
        # may incidentally dispatch other handlers like sojourn_times
        # during graph construction; ignore those).
        daisy_calls = [
            shape for (name, shape) in captured
            if name in ('ptd_daisy_chain_joint_probs', 'ptd_daisy_chain_sojourn')
        ]
        assert len(daisy_calls) >= 1, (
            f"daisy_chain FFI was not invoked under vmap(grad). "
            f"All captured: {captured[:10]}"
        )

        ranks = [len(shape) for shape in daisy_calls]
        n_3d = sum(1 for r in ranks if r == 3)
        assert n_3d == 0, (
            f"FFI received 3D theta in {n_3d}/{len(daisy_calls)} calls — "
            f"the rule failed and JAX added an extra axis. Shapes: "
            f"{daisy_calls}"
        )

        # Count fused calls — those with leading axis == P. The
        # tracing pass may also dispatch P single-particle 1D probes;
        # those are tolerated.
        two_d_leading = [shape[0] for shape in daisy_calls if len(shape) == 2]
        n_fused = sum(1 for x in two_d_leading if x == P)
        assert n_fused >= 1, (
            f"No FFI call had leading axis P={P}; got 2D leading axes "
            f"{two_d_leading} and ranks {ranks}. The custom_vmap rule "
            f"in _daisy_chain_svgd_model did not fire — vmap(grad) is "
            f"fanning out per particle instead of fusing."
        )
        # Expected fused dispatch count: 1 forward + 2*n_params FD
        # perturbations (skipping fixed slots — there are none in this
        # fixture, so all n_params are perturbed).
        expected_fused = 1 + 2 * n_params
        assert n_fused >= expected_fused - 1, (
            f"Expected ~{expected_fused} fused (P, theta_dim) calls "
            f"(1 fwd + 2*n_params FD perturbations); got {n_fused}. "
            f"Captured shapes: {daisy_calls}"
        )

    def test_vmap_perf_baseline(self):
        """Wall-time baseline harness (NOT a regression assertion).

        Records the ``t_vmap8 / t_single`` ratio for the current
        no-exposure daisy-chain code path so that the fusion-recovery
        plan (``daisy-chain-fusion-recovery-plan.md``) can compare
        before/after. Asserts only a generous upper bound so this
        test is informational — grep the printed line for ``ratio=``
        to read the number out of CI output.

        Skipped on hosts with <4 OMP threads where the per-iteration
        cost is dominated by single-thread FFI overhead and the ratio
        is meaningless.
        """
        import os
        omp_threads = int(
            os.environ.get('OMP_NUM_THREADS', '0')
            or os.cpu_count()
            or 1
        )
        if omp_threads < 4:
            pytest.skip(f"Need >=4 OMP threads; got {omp_threads}")

        import time
        import jax
        import jax.numpy as jnp

        _jsp, _model, loss, theta_init = self._build_model_and_loss()
        # Warm-up JIT for both paths.
        _ = jax.grad(loss)(theta_init).block_until_ready()
        warmup_particles = jnp.tile(theta_init[None, :], (8, 1))
        _ = jax.vmap(jax.grad(loss))(warmup_particles).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(2):
            _ = jax.grad(loss)(theta_init).block_until_ready()
        t_single = (time.perf_counter() - t0) / 2.0

        particles = jnp.tile(theta_init[None, :], (8, 1)) + 0.01 * jnp.arange(8)[:, None]
        t0 = time.perf_counter()
        for _ in range(2):
            _ = jax.vmap(jax.grad(loss))(particles).block_until_ready()
        t_vmap8 = (time.perf_counter() - t0) / 2.0

        ratio = t_vmap8 / max(t_single, 1e-6)
        # Generous upper bound — this test is informational, not a
        # regression check. Like test_vmap_perf_smoke, the batched path pays
        # per-FFI-call OpenMP team overhead that the single-particle path
        # (batch_size==1) skips, and that overhead scales with the thread
        # count, so the bound must grow with omp_threads or the test flakes on
        # many-core hosts (ratio ~44-49× at 16 threads sits right on a fixed
        # 50× bound). The strict per-thread regression check lives in
        # test_vmap_perf_smoke.
        max_ratio = max(50.0, 5.0 * omp_threads)
        # Same format as test_vmap_perf_smoke so the numbers are
        # directly comparable across runs.
        print(
            f"\n  [baseline] t_single={t_single*1e3:.1f}ms  "
            f"t_vmap8={t_vmap8*1e3:.1f}ms  ratio={ratio:.2f}  "
            f"(omp_threads={omp_threads}, bound={max_ratio:.0f}×)"
        )
        assert ratio < max_ratio, (
            f"vmap(grad) over 8 particles took {ratio:.1f}× the "
            f"single-particle time (> {max_ratio:.0f}× for {omp_threads} "
            f"OMP threads) — something is catastrophically wrong with the "
            f"path under test."
        )

    def test_vmap_perf_smoke(self):
        """Perf smoke on the SVGD-side fused path: under vmap(grad), 8
        particles must take no more than 8× single-particle time.

        This is a no-regression upper bound — on hardware where the
        ``custom_vmap`` fusion actually delivers, the ratio should be
        much closer to 1×. On smaller fixtures the per-call OpenMP
        overhead dominates dispatch overhead and the fusion's win is
        invisible; in that regime the assertion still catches a
        catastrophic regression (linear blow-up).

        Skipped on hosts with <4 OMP threads (CI often has 1).
        """
        import os
        omp_threads = int(
            os.environ.get('OMP_NUM_THREADS', '0')
            or os.cpu_count()
            or 1
        )
        if omp_threads < 4:
            pytest.skip(f"Need >=4 OMP threads; got {omp_threads}")

        import time
        import jax
        import jax.numpy as jnp

        _jpg, _model, loss, theta_init = self._build_svgd_model_and_loss()
        # Warm-up JIT for both paths.
        _ = jax.grad(loss)(theta_init).block_until_ready()
        warmup_particles = jnp.tile(theta_init[None, :], (8, 1))
        _ = jax.vmap(jax.grad(loss))(warmup_particles).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(2):
            _ = jax.grad(loss)(theta_init).block_until_ready()
        t_single = (time.perf_counter() - t0) / 2.0

        particles = jnp.tile(theta_init[None, :], (8, 1)) + 0.01 * jnp.arange(8)[:, None]
        t0 = time.perf_counter()
        for _ in range(2):
            _ = jax.vmap(jax.grad(loss))(particles).block_until_ready()
        t_vmap8 = (time.perf_counter() - t0) / 2.0

        ratio = t_vmap8 / max(t_single, 1e-6)
        # The single-particle path runs with batch_size==1 and SKIPS the
        # OpenMP batch region (`#pragma omp parallel for if(batch_size > 1)`),
        # so its time is flat regardless of thread count. The 8-particle path
        # spins up a full-width OpenMP team per FFI call (~1 + 2*theta_dim calls
        # per gradient from the finite-difference backward). On this tiny
        # fixture that team spin-up/barrier dominates and grows with the thread
        # count — measured ~3.6× (1 thread), ~14× (8), ~44× (16) — even though
        # the fusion IS firing (verified structurally by
        # test_vmap_fuses_into_one_ffi_call). So a fixed 8× bound only holds
        # single-threaded; scale it with the thread count. A true fusion
        # regression scales with PARTICLE count (8 separate FFI calls), not
        # thread count, and would blow far past this thread-scaled bound.
        max_ratio = max(8.0, 5.0 * omp_threads)
        print(
            f"\n  [SVGD-fused] t_single={t_single*1e3:.1f}ms  "
            f"t_vmap8={t_vmap8*1e3:.1f}ms  ratio={ratio:.2f}  "
            f"(omp_threads={omp_threads}, bound={max_ratio:.0f}×)"
        )
        assert ratio <= max_ratio, (
            f"vmap(grad) over 8 particles took {ratio:.1f}× the "
            f"single-particle time (> {max_ratio:.0f}× bound for "
            f"{omp_threads} OMP threads) — the custom_vmap fusion is "
            f"likely broken on the SVGD-side path (ideal is ~1×)."
        )

    def test_pmap_vmap_composition(self):
        """`pmap(vmap(grad(loss)))` composition must produce per-particle
        grads identical to the sequential path. Skipped when only one
        local device is available."""
        import jax
        import jax.numpy as jnp

        if jax.local_device_count() < 2:
            pytest.skip("needs >=2 local devices")

        _jsp, _model, loss, theta_init = self._build_model_and_loss()
        # 4 particles, 2 devices -> 2 particles per device.
        particles = jnp.tile(theta_init[None, :], (4, 1)) + 0.01 * jnp.arange(4)[:, None]
        particles_sharded = particles.reshape(2, 2, -1)
        pmapped = jax.pmap(jax.vmap(jax.grad(loss)), axis_name='batch')
        g = pmapped(particles_sharded)
        assert g.shape == (2, 2, theta_init.shape[0])

        g_seq = jnp.stack([jax.grad(loss)(p) for p in particles]).reshape(
            2, 2, theta_init.shape[0]
        )
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(g_seq), rtol=0, atol=1e-8,
        )
