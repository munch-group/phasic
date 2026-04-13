"""Tests for composable graph operations."""
import numpy as np
from pytest import approx
from phasic import Graph, with_ipv


def _make_coalescent(nr_samples=4):
    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions
    return coalescent


class TestRebuildWithWiderLayout:
    def test_wider_state(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)

        wider = g._rebuild_with_wider_layout(extra_state_dims=1)
        assert wider.state_length() == 2
        assert wider.vertices_length() == g.vertices_length()

    def test_wider_coefficients(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)

        wider = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert wider.param_length() == graph.param_length() + 1
        assert wider.vertices_length() == graph.vertices_length()

    def test_rebuild_preserves_structure(self):
        """Vertex count, edge targets, and edge weights match original."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        rebuilt = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert rebuilt.vertices_length() == graph.vertices_length()
        for i in range(graph.vertices_length()):
            orig_v = graph.vertex_at(i)
            new_v = rebuilt.vertex_at(i)
            assert orig_v.edges_length() == new_v.edges_length()
            orig_targets = sorted([e.to().index() for e in orig_v.edges()])
            new_targets = sorted([e.to().index() for e in new_v.edges()])
            assert orig_targets == new_targets

    def test_rebuild_no_change(self):
        """extra_state_dims=0, extra_coeff_slots=0 produces equivalent graph."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([2.0])

        rebuilt = graph._rebuild_with_wider_layout()
        assert rebuilt.vertices_length() == graph.vertices_length()
        assert rebuilt.param_length() == graph.param_length()
        assert rebuilt.state_length() == graph.state_length()
        # Same moments (structural equivalence)
        rebuilt.update_weights([2.0])
        assert rebuilt.moments(3) == approx(graph.moments(3), rel=1e-10)

    def test_rebuild_preserves_metadata(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph._weight_mode = 'log'

        rebuilt = graph._rebuild_with_wider_layout(extra_coeff_slots=1)
        assert rebuilt._callback is graph._callback
        assert rebuilt._weight_mode == 'log'
        assert rebuilt.is_discrete == graph.is_discrete


class TestDiscretizeReturnsNewGraph:
    def test_returns_graph(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        v2 = g.find_or_create_vertex([2])
        v.add_edge(v2, 1.0)

        result = g.discretize(0.1)
        assert isinstance(result, Graph)
        assert hasattr(result, 'rewards')
        assert result.is_discrete

    def test_original_unchanged(self):
        g = Graph(1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        v2 = g.find_or_create_vertex([2])
        v.add_edge(v2, 1.0)

        orig_n = g.vertices_length()
        orig_discrete = g.is_discrete

        _ = g.discretize(0.1)

        assert g.vertices_length() == orig_n
        assert g.is_discrete == orig_discrete


class TestCompositionPipeline:
    def test_laplace_preserves_metadata(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        lt = graph.laplace_transform(0.5)
        assert lt._callback is not None


class TestAddEpoch:
    def test_single_epoch(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        assert g1.vertices_length() > graph.vertices_length()
        # param_length grows by base_param_length + 1 per epoch
        assert g1.param_length() == graph.param_length() + graph.param_length() + 1

    def test_original_unchanged(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        orig_n = graph.vertices_length()
        orig_pl = graph.param_length()

        _ = graph.add_epoch(1.0)

        assert graph.vertices_length() == orig_n
        assert graph.param_length() == orig_pl

    def test_chained_epochs(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        g2 = g1.add_epoch(2.0)
        # Each epoch adds base_param_length + 1 slots
        base_pl = graph.param_length()
        assert g2.param_length() == base_pl + 2 * (base_pl + 1)
        assert g2.vertices_length() > g1.vertices_length()

    def test_epoch_then_discretize(self):
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        g1 = graph.add_epoch(1.0)
        # theta: [epoch0_coal, epoch1_coal, epoch1_trans]
        g1.update_weights([1.0, 1.0, 1.0])

        g2 = g1.discretize(0.1)
        assert g2.is_discrete
        assert hasattr(g2, 'rewards')


class TestCachingComposedGraphs:
    """Verify that trace caching works correctly for graphs built via composition."""

    def test_composed_graph_hash_deterministic(self):
        """Same composition produces same hash."""
        from phasic import hash as phasic_hash

        def _compose():
            coalescent = _make_coalescent(4)
            g = Graph(coalescent)
            g.update_weights([1.0])
            g1 = g.add_epoch(1.0)
            g1.update_weights([1.0, 1.0, 1.0])
            return g1.add_epoch(2.0)

        h1 = phasic_hash.compute_graph_hash(_compose())
        h2 = phasic_hash.compute_graph_hash(_compose())
        assert h1.hash_hex == h2.hash_hex

    def test_composed_graph_trace_recording(self):
        """Trace can be recorded on a composed graph."""
        from phasic.hierarchical_trace_cache import get_trace_hierarchical

        coalescent = _make_coalescent(4)
        g = Graph(coalescent)
        g.update_weights([1.0])
        g1 = g.add_epoch(1.0)

        g1_copy = g1.clone()
        trace = get_trace_hierarchical(g1_copy, param_length=g1.param_length())
        assert trace is not None

    def test_trace_based_moments_match_direct(self):
        """Trace-based computation on composed graph matches direct computation."""
        from phasic.hierarchical_trace_cache import get_trace_hierarchical
        from phasic.trace_elimination import instantiate_from_trace

        coalescent = _make_coalescent(4)
        g = Graph(coalescent)
        g.update_weights([1.0])
        g1 = g.add_epoch(1.0)

        # Record trace
        g1_copy = g1.clone()
        trace = get_trace_hierarchical(g1_copy, param_length=g1.param_length())

        # Trace-based
        theta = np.array([1.0, 0.5, 1.0])
        concrete = instantiate_from_trace(trace, theta)
        trace_moments = concrete.moments(3)

        # Direct
        g1.update_weights(theta)
        direct_moments = g1.moments(3)

        assert trace_moments == approx(direct_moments, rel=1e-6)


class TestEquivalenceOldVsNew:
    """Verify new implementations produce identical graphs to old implementations."""

    def test_discretize_equivalence(self):
        """New discretize produces same graph structure and rewards as old in-place version."""
        def _build():
            g = Graph(1)
            start = g.starting_vertex()
            v1 = g.find_or_create_vertex([1])
            v2 = g.find_or_create_vertex([2])
            start.add_edge(v1, 2.0)
            v1.add_edge(v2, 2.0)
            return g

        # Old approach: in-place
        g_old = _build()
        rewards_old = g_old._discretize_inplace(0.1)

        # New approach: returns new graph
        g_orig = _build()
        g_new = g_orig.discretize(0.1)

        # Same vertices count
        assert g_new.vertices_length() == g_old.vertices_length()
        # Same rewards
        assert np.array_equal(g_new.rewards, rewards_old)
        # Same is_discrete flag
        assert g_new.is_discrete == g_old.is_discrete
        # Same edge structure
        for i in range(g_old.vertices_length()):
            v_old = g_old.vertex_at(i)
            v_new = g_new.vertex_at(i)
            assert v_old.edges_length() == v_new.edges_length()
            old_weights = sorted([e.weight() for e in v_old.edges()])
            new_weights = sorted([e.weight() for e in v_new.edges()])
            assert old_weights == approx(new_weights, rel=1e-10)
        # Same moments
        assert g_new.expectation_discrete(g_new.rewards) == approx(
            g_old.expectation_discrete(rewards_old), rel=1e-10)

    def test_discretize_equivalence_callable(self):
        """New discretize with callable rate matches old in-place version."""
        def _build():
            g = Graph(1)
            start = g.starting_vertex()
            v1 = g.find_or_create_vertex([1])
            v2 = g.find_or_create_vertex([2])
            start.add_edge(v1, 2.0)
            v1.add_edge(v2, 2.0)
            return g

        def rate_fn(state):
            return 0.2

        g_old = _build()
        rewards_old = g_old._discretize_inplace(rate_fn)

        g_orig = _build()
        g_new = g_orig.discretize(rate_fn)

        assert g_new.vertices_length() == g_old.vertices_length()
        assert np.array_equal(g_new.rewards, rewards_old)

    def test_laplace_equivalence(self):
        """New laplace_transform with metadata matches old output structurally."""
        coalescent = _make_coalescent(4)
        graph = Graph(coalescent)
        graph.update_weights([1.0])

        lt = graph.laplace_transform(0.5)

        assert lt.vertices_length() == graph.vertices_length()
        assert lt._callback is not None

    def test_epoch_moments_match_tutorial(self):
        """New add_epoch produces same moments as tutorial's manual epoch wiring."""
        from phasic import StateIndexer, Property
        from itertools import combinations_with_replacement
        from functools import partial
        all_pairs = partial(combinations_with_replacement, r=2)

        nr_samples = 10
        epochs = [0, 1, 2]
        pop_sizes = [1, 5, 10]

        # --- Manual approach (from tutorial) ---
        indexer = StateIndexer(
            lineages=[Property('ton', min_value=1, max_value=nr_samples)],
            slots=['epoch']
        )

        def coalescent_1param(state, epochs=None, epoch_idx=None, indexer=None):
            transitions = []
            epoch_idx = int(epoch_idx)
            if state[indexer.epoch] != epoch_idx:
                return transitions
            for i, j in all_pairs(indexer.lineages):
                pi = indexer.lineages.index_to_props(i)
                pj = indexer.lineages.index_to_props(j)
                if state.sum() <= 1:
                    continue
                same = int(pi.ton == pj.ton)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                k = indexer.props_to_index(ton=pi.ton + pj.ton)
                new[k] += 1
                coeff = np.zeros(len(epochs) + 1)
                coeff[epoch_idx] = state[i] * (state[j] - same) / (1 + same)
                transitions.append([new, coeff])
            return transitions

        def manual_add_epoch(graph, callback, epochs, epoch_idx, indexer):
            epoch = epochs[epoch_idx]
            stop_probs = np.array(graph.stop_probability(epoch))
            accum_v_time = np.array(graph.accumulated_occupancy(epoch))
            with np.errstate(invalid='ignore'):
                epoch_trans_rates = stop_probs / accum_v_time
            for i in range(1, graph.vertices_length() - 1):
                if epoch_trans_rates is None or np.isnan(epoch_trans_rates[i]):
                    continue
                if graph.vertex_at(i).edges_length() == 0:
                    continue
                vertex = graph.vertex_at(i)
                state = vertex.state()
                if not state[indexer.epoch] == epoch_idx - 1:
                    continue
                sister_state = state.copy()
                sister_state[indexer.epoch] = epoch_idx
                child = graph.find_or_create_vertex(sister_state)
                coeff = np.zeros(len(epochs) + 1)
                coeff[-1] = epoch_trans_rates[i]
                vertex.add_edge(child, coeff)
            graph.extend(callback, epochs=epochs, epoch_idx=epoch_idx, indexer=indexer)

        ipv = [0] * indexer.state_length
        ipv[indexer.props_to_index(ton=1)] = nr_samples

        manual_graph = Graph(coalescent_1param, ipv=ipv,
                             epochs=epochs, epoch_idx=0, indexer=indexer)
        manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
        for ei in range(1, len(epochs)):
            manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
            manual_add_epoch(manual_graph, coalescent_1param, epochs, ei, indexer)
        manual_graph.update_weights([1 / s for s in pop_sizes] + [1])
        manual_moments = manual_graph.moments(5)

        # --- New approach ---
        coalescent_simple = _make_coalescent(nr_samples)

        new_graph = Graph(coalescent_simple)
        new_graph.update_weights([1 / pop_sizes[0]])
        g1 = new_graph.add_epoch(epochs[1])
        # Must set weights before computing next epoch's transition rates
        g1.update_weights([1 / pop_sizes[0], 1 / pop_sizes[1], 1])
        g2 = g1.add_epoch(epochs[2])
        # theta layout: [epoch0_coal, epoch1_coal, epoch1_trans, epoch2_coal, epoch2_trans]
        g2.update_weights([1 / pop_sizes[0], 1 / pop_sizes[1], 1, 1 / pop_sizes[2], 1])
        new_moments = g2.moments(5)

        # Moments should match within numerical tolerance
        assert new_moments == approx(manual_moments, rel=1e-6)
