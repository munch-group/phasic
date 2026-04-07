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

private:
    // Cached structure data (parsed from JSON once)
    int param_length_;      // Number of parameters
    int state_length_;      // Dimension of state vectors
    int n_vertices_;        // Number of vertices (excluding starting vertex)

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
     * @brief Compute factorial: n!
     */
    double factorial(int n);
};

} // namespace parameterized
} // namespace phasic

#endif // PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_HPP
