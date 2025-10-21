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
        int granularity = 100
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
    std::vector<double> compute_moments_impl(Graph& g, int nr_moments);

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
