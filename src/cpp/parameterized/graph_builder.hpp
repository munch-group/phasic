#ifndef PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_HPP
#define PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <memory>
#include "../phasiccpp.h"

namespace py = pybind11;

namespace phasic {
namespace parameterized {

/**
 * @brief GraphBuilder: Efficient parameterized graph construction and computation
 *
 * This class separates graph structure (topology) from parameters (theta values).
 * The structure is parsed once from JSON, then graphs can be rapidly built with
 * different theta values for efficient batch processing.
 *
 * Thread-safety: Each GraphBuilder instance is NOT thread-safe. Create separate
 * instances for concurrent access, or use external synchronization.
 *
 * GIL management: All public methods that return numpy arrays should be called
 * with py::call_guard<py::gil_scoped_release>() to release GIL during C++ computation.
 */
class GraphBuilder {
public:
    /**
     * @brief Construct GraphBuilder from JSON-serialized graph structure
     *
     * @param structure_json JSON string from Graph.serialize()
     *        Expected format:
     *        {
     *          "states": [[s00, s01, ...], [s10, s11, ...], ...],
     *          "edges": [[from, to, weight], ...],
     *          "start_edges": [[to, weight], ...],
     *          "param_edges": [[from, to, coeff1, coeff2, ...], ...],
     *          "start_param_edges": [[to, coeff1, coeff2, ...], ...],
     *          "param_length": int,
     *          "state_length": int,
     *          "n_vertices": int
     *        }
     *
     * @throws std::invalid_argument if JSON is malformed or required fields missing
     */
    explicit GraphBuilder(const std::string& structure_json);

    /**
     * @brief Destructor: evicts this builder's entry from the
     *        thread-local persistent-graph cache.
     *
     * The cache is keyed by ``GraphBuilder*`` address. Without this
     * cleanup, a builder destroyed in one scope could leave a stale
     * entry behind, and a *different* builder allocated at the same
     * address would silently "hit" that entry — returning a graph
     * built from a previous theta with a different model structure.
     *
     * Note: only evicts the entry on the *current* thread. Builders
     * destroyed on threads other than the one that populated the
     * cache (uncommon) leave a stale entry; this is acceptable
     * because the next call from that other thread will overwrite
     * the entry anyway (every cache miss writes a fresh entry).
     */
    ~GraphBuilder();

    /**
     * @brief Build graph with specific parameter values
     *
     * @param theta Pointer to parameter array
     * @param theta_len Length of theta array (must match param_length)
     * @return Graph instance with edges weighted by theta
     *
     * @throws std::invalid_argument if theta_len doesn't match param_length
     *
     * Note: This is the core low-level method. Higher-level methods call this
     * internally and handle numpy array conversions.
     */
    Graph build(const double* theta, size_t theta_len);

    /**
     * @brief Compute distribution moments: E[T^k] for k=1,2,...,nr_moments
     *
     * @param theta Parameter array (numpy array)
     * @param nr_moments Number of moments to compute
     * @return Numpy array of shape (nr_moments,) with [E[T], E[T^2], ..., E[T^nr_moments]]
     *
     * Uses Graph::expected_waiting_time() iteratively to compute higher moments.
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>() to enable
     * parallel execution across multiple threads/processes.
     */
    py::array_t<double> compute_moments(
        py::array_t<double> theta,
        int nr_moments
    );

    /**
     * @brief Compute probability mass function (PMF) or probability density function (PDF)
     *
     * @param theta Parameter array (numpy array)
     * @param times Time points or jump counts to evaluate (numpy array)
     * @param discrete If true, compute DPH (discrete), else PDF (continuous)
     * @param granularity Discretization granularity for PDF computation
     * @return Numpy array of PMF/PDF values, shape matches times
     *
     * Continuous (discrete=false): Computes PDF using Graph::pdf(time, granularity)
     * Discrete (discrete=true): Computes DPH PMF using Graph::dph_pmf(jump_count)
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>()
     */
    py::array_t<double> compute_pmf(
        py::array_t<double> theta,
        py::array_t<double> times,
        bool discrete = false,
        int granularity = 100
    );

    /**
     * @brief Compute both PMF and moments efficiently in single pass
     *
     * @param theta Parameter array (numpy array)
     * @param times Time points or jump counts to evaluate (numpy array)
     * @param nr_moments Number of moments to compute
     * @param discrete If true, use DPH mode, else PDF mode
     * @return Pair of (pmf_array, moments_array)
     *
     * This is more efficient than calling compute_pmf() and compute_moments()
     * separately because the graph is built only once.
     *
     * Used by: SVGD with moment-based regularization
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>()
     */
    std::pair<py::array_t<double>, py::array_t<double>>
    compute_pmf_and_moments(
        py::array_t<double> theta,
        py::array_t<double> times,
        int nr_moments,
        bool discrete = false,
        int granularity = 100,
        py::object rewards = py::none()
    );

    /**
     * @brief Compute PMF/PDF, moments, and the reward-transformed
     *        atomic-mass-at-zero in a single pass.
     *
     * Identical to compute_pmf_and_moments() but additionally returns
     * the per-feature mass that the reward-transformed distribution
     * places on the atom at r = 0, equivalent to
     * ``g.reward_transform(rewards).cdf(0)`` (continuous) or
     * ``g.reward_transform(rewards).dph_cdf(0)`` (discrete). When
     * called without rewards (rewards=None), the third output is
     * a single 0.0 (the untransformed distribution has no
     * reward-induced atom).
     *
     * Used by: zero-inflated likelihood term in Graph.svgd, to avoid
     * a redundant ``backward_probabilities`` solve per particle.
     *
     * Shapes:
     * - 1D / no rewards: cdf_zero is a length-1 numpy array.
     * - 2D rewards: cdf_zero is shape (n_features,).
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>()
     */
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    compute_pmf_moments_and_cdf_zero(
        py::array_t<double> theta,
        py::array_t<double> times,
        int nr_moments,
        bool discrete = false,
        int granularity = 100,
        py::object rewards = py::none()
    );

    /**
     * @brief Compute multivariate PMF/PDF for multiple feature dimensions
     *
     * NEW in v0.23.0: Native C++ support for multivariate observations.
     *
     * Supports two modes of operation:
     * 1. Sparse mode (compute_joint=false): Each feature dimension computed
     *    independently with its own reward vector. Zero entries in times array
     *    are treated as "no observation" and produce zero PDF.
     * 2. Joint mode (compute_joint=true): Computes joint PDF across features
     *    [NOT YET IMPLEMENTED - raises error]
     *
     * @param theta Parameter array, shape (n_params,)
     * @param times Time points array, shape (n_times, n_features). Zero = no observation.
     * @param rewards Reward vectors, shape (n_vertices, n_features). Column j defines
     *                reward vector for feature dimension j.
     * @param discrete If true, compute DPH. If false, compute PDF.
     * @param granularity Discretization granularity for PDF computation
     * @param compute_joint If true, compute joint PDF (raises NotImplementedError).
     *                      If false, compute independent PDFs per feature (sparse mode).
     * @return Numpy array of shape (n_times, n_features) with PDF/PMF values.
     *         Zero wherever times[i,j] == 0.0 in sparse mode.
     *
     * Example (sparse mode):
     *   times = [[1.5, 0.0],   // Observe feature 0 only
     *            [0.0, 2.1],   // Observe feature 1 only
     *            [1.2, 1.8]]   // Observe both features
     *   rewards = [[1.0, 0.5], // n_vertices rows, 2 features
     *              [2.0, 1.0]]
     *   Result[0,0] = PDF(t=1.5, rewards[:,0]), Result[0,1] = 0.0
     *   Result[1,0] = 0.0,                      Result[1,1] = PDF(t=2.1, rewards[:,1])
     *   Result[2,0] = PDF(t=1.2, rewards[:,0]), Result[2,1] = PDF(t=1.8, rewards[:,1])
     *
     * Validation with length-1 vectors:
     *   times_1d = [[1.5]] (shape 1,1)
     *   rewards_1d = [[1.0], [2.0]] (shape n_vertices,1)
     *   Result should match compute_pmf() after reward transform.
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>()
     *
     * @throws std::invalid_argument if dimensions mismatch or compute_joint=true
     */
    py::array_t<double> compute_pmf_multivariate(
        py::array_t<double> theta,
        py::array_t<double> times,
        py::array_t<double> rewards,
        bool discrete = false,
        int granularity = 100,
        bool compute_joint = false
    );

    /**
     * @brief Compute converged accumulated visits for specified vertices (joint index mode)
     *
     * For each vertex index, iterates accumulated_visits(jumps) until convergence,
     * returning the limiting accumulated visits (equivalent to expected_sojourn_time).
     *
     * This is used for joint index distributions where observed_data contains vertex
     * indices rather than time values, and likelihood is computed from converged
     * accumulated visits.
     *
     * @param theta Parameter array (numpy array)
     * @param vertex_indices Array of vertex indices to compute (numpy array of int)
     * @param tolerance Convergence tolerance (default 1e-15)
     * @param max_iterations Maximum iterations before giving up (default 10000)
     * @return Numpy array of converged accumulated visits, shape matches vertex_indices
     *
     * @throws std::runtime_error if convergence not achieved within max_iterations
     *
     * GIL Note: Call with py::call_guard<py::gil_scoped_release>()
     */
    py::array_t<double> compute_accumulated_visits_converged(
        py::array_t<double> theta,
        py::array_t<int> vertex_indices,
        double tolerance = 1e-15,
        int max_iterations = 10000
    );

    // Getters for metadata
    int param_length() const { return param_length_; }
    int vertices_length() const { return n_vertices_; }
    int state_length() const { return state_length_; }

    /**
     * @brief Compute moments using iterative expected_waiting_time calls
     *
     * Public for use by FFI handlers. Internal implementation used by
     * compute_moments() and compute_pmf_and_moments()
     */
    std::vector<double> compute_moments_impl(Graph& g, int nr_moments, const std::vector<double>& rewards);

    /** True if this graph is a DPH (serialize 'is_discrete'). Public for FFI handlers. */
    bool is_discrete() const { return is_discrete_; }

    /**
     * Convert a double reward vector to the integer vector the discrete (DPH)
     * reward transform requires; throws on a negative or non-integer value.
     * Public/static for use by both the pybind and FFI reward paths.
     */
    static std::vector<int> rewards_to_int_or_throw(const std::vector<double>& r);

    /**
     * In place, convert continuous raw moments [E[T], E[T^2], ...] to discrete
     * raw moments [E[N], E[N^2], ...] for a DPH. Graph-independent (U=(I-P)^-1
     * commutes with P). Public/static for use by the pybind and FFI moment paths.
     */
    static void continuous_to_discrete_moments(std::vector<double>& m);

    /** Weight computation mode for parameterized edges. */
    enum class WeightMode { LINEAR, LOG, FORMULA };

    /**
     * @brief Get or initialise this thread's persistent graph and refresh its weights.
     *
     * On first call from a given thread for a given GraphBuilder
     * instance, builds a parameterised graph (via build()) and stores
     * it in thread-local storage keyed by ``this``. Subsequent calls
     * from the same thread call ``update_weights(theta)`` on the
     * cached graph and return it.
     *
     * The cached graph holds the C-level
     * ``parameterized_reward_compute_graph`` once it's been built by
     * the first forward call. After Stage A0, that cache survives
     * across ``update_weights`` calls (only structural mutations or
     * graph destruction invalidate it). So the O(n^3) symbolic
     * elimination runs once per (thread, GraphBuilder) instead of
     * once per theta call.
     *
     * Thread-safety: thread_local storage isolates graph instances
     * per OS thread. Safe under the FFI handler's
     * ``#pragma omp parallel for`` and under JAX pmap (each device
     * runs in its own Python interpreter, hence its own GraphBuilder).
     * Concurrent calls within a thread are serialised by the GIL on
     * the pybind11 entry boundary; the GIL is released for the C++
     * portion but each thread has its own slot.
     */
    Graph& get_or_init_persistent_graph(const double* theta, size_t theta_len);

private:
    // Cached structure data (parsed from JSON once)
    int param_length_;      // Number of parameters
    int state_length_;      // Dimension of state vectors
    int n_vertices_;        // Number of vertices (excluding starting vertex)
    WeightMode weight_mode_ = WeightMode::LINEAR;
    bool dyn_ordering_ = false;   // dynamic min-degree elimination ordering
    bool is_discrete_ = false;    // graph is a DPH (from serialize 'is_discrete')

    // weight_mode_ == FORMULA: per-edge weight tape (parsed once from the
    // 'weight_formula_tape' JSON). Evaluated in C via
    // ptd_weight_tape_eval_arrays (compute_weight / IPV edges) and installed
    // on the graph in build() so update_weights() runs it for every theta.
    bool has_tape_ = false;
    std::vector<int> tape_ops_;
    std::vector<double> tape_consts_;
    size_t tape_stack_depth_ = 0;
    size_t tape_n_theta_ = 0;
    size_t tape_n_coeff_ = 0;

    // Vertex states: (n_vertices, state_length)
    std::vector<std::vector<int>> states_;

    // Regular edges: (from_idx, to_idx, weight)
    struct RegularEdge {
        int from_idx;
        int to_idx;
        double weight;
    };
    std::vector<RegularEdge> edges_;
    std::vector<RegularEdge> start_edges_;  // From starting vertex

    // Constant (coefficient-less) edges: (from_idx, to_idx, weight).
    // Created by Vertex::add_aux_vertex_constant. They have
    // coefficients_length == 0 in the C representation, so
    // ptd_graph_update_weights skips them. The build() method
    // reconstructs them by direct ptd_edge struct manipulation,
    // bypassing the EDGE_MODE lock so they coexist with parameterised
    // edges on the same graph (the t-aux trapping loops in the
    // joint_stop_prob_graph).
    std::vector<RegularEdge> constant_edges_;

    // Parameterized edges: (from_idx, to_idx, coefficients...)
    struct ParameterizedEdge {
        int from_idx;
        int to_idx;
        std::vector<double> coefficients;  // Length = param_length
    };
    std::vector<ParameterizedEdge> param_edges_;
    std::vector<ParameterizedEdge> start_param_edges_;  // From starting vertex

    // Helper methods

    /**
     * @brief Parse JSON structure into internal representation
     * @throws std::runtime_error if JSON parsing fails
     */
    void parse_structure(const std::string& json_str);

    /**
     * @brief Compute edge weight from coefficients and theta using current weight_mode_
     */
    double compute_weight(const std::vector<double>& coefficients, const double* theta) const;

    /**
     * @brief Compute factorial: n!
     */
    double factorial(int n);
};

} // namespace parameterized
} // namespace phasic

#endif // PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_HPP
