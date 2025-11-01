#include "graph_builder.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <cmath>

using json = nlohmann::json;

namespace phasic {
namespace parameterized {

GraphBuilder::GraphBuilder(const std::string& structure_json) {
    parse_structure(structure_json);
}

void GraphBuilder::parse_structure(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        // Extract metadata
        param_length_ = j.at("param_length").get<int>();
        state_length_ = j.at("state_length").get<int>();
        n_vertices_ = j.at("n_vertices").get<int>();

        // Parse states
        states_.reserve(n_vertices_);
        auto states_json = j.at("states");
        for (const auto& state_arr : states_json) {
            std::vector<int> state;
            state.reserve(state_length_);
            for (const auto& val : state_arr) {
                state.push_back(val.get<int>());
            }
            states_.push_back(state);
        }

        // Parse regular edges
        auto edges_json = j.at("edges");
        edges_.reserve(edges_json.size());
        for (const auto& edge_arr : edges_json) {
            RegularEdge edge;
            edge.from_idx = edge_arr[0].get<int>();
            edge.to_idx = edge_arr[1].get<int>();
            edge.weight = edge_arr[2].get<double>();
            edges_.push_back(edge);
        }

        // Parse starting vertex edges
        auto start_edges_json = j.at("start_edges");
        start_edges_.reserve(start_edges_json.size());
        for (const auto& edge_arr : start_edges_json) {
            RegularEdge edge;
            edge.from_idx = -1;  // Starting vertex
            edge.to_idx = edge_arr[0].get<int>();
            edge.weight = edge_arr[1].get<double>();
            start_edges_.push_back(edge);
        }

        // Parse parameterized edges (if present)
        // Format: [from_idx, to_idx, x1, x2, ...]
        if (j.contains("param_edges")) {
            auto param_edges_json = j.at("param_edges");
            param_edges_.reserve(param_edges_json.size());
            for (const auto& edge_arr : param_edges_json) {
                ParameterizedEdge edge;
                edge.from_idx = edge_arr[0].get<int>();
                edge.to_idx = edge_arr[1].get<int>();
                // No base_weight
                edge.coefficients.reserve(param_length_);
                for (int i = 2; i < 2 + param_length_; i++) {
                    edge.coefficients.push_back(edge_arr[i].get<double>());
                }
                param_edges_.push_back(edge);
            }
        }

        // Parse starting vertex parameterized edges (if present)
        // Format: [to_idx, x1, x2, ...]
        // NOTE: This should be EMPTY after refactoring (starting edges not parameterized)
        if (j.contains("start_param_edges")) {
            auto start_param_edges_json = j.at("start_param_edges");
            start_param_edges_.reserve(start_param_edges_json.size());
            for (const auto& edge_arr : start_param_edges_json) {
                ParameterizedEdge edge;
                edge.from_idx = -1;  // Starting vertex
                edge.to_idx = edge_arr[0].get<int>();
                // No base_weight
                edge.coefficients.reserve(param_length_);
                for (int i = 1; i < 1 + param_length_; i++) {
                    edge.coefficients.push_back(edge_arr[i].get<double>());
                }
                start_param_edges_.push_back(edge);
            }
        }

    } catch (const json::exception& e) {
        std::ostringstream oss;
        oss << "Failed to parse graph structure JSON: " << e.what();
        throw std::runtime_error(oss.str());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Error parsing graph structure: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

Graph GraphBuilder::build(const double* theta, size_t theta_len) {
    // Validate theta length
    if (static_cast<int>(theta_len) != param_length_) {
        std::ostringstream oss;
        oss << "Theta length mismatch: expected " << param_length_
            << ", got " << theta_len;
        throw std::invalid_argument(oss.str());
    }

    // Create graph with proper state dimension
    Graph g(state_length_);

    // Get starting vertex
    Vertex* start = g.starting_vertex_p();

    // Create all vertices
    std::vector<Vertex*> vertices;
    vertices.reserve(n_vertices_);

    // Check if first vertex is starting vertex (all zeros)
    bool first_is_start = true;
    if (n_vertices_ > 0) {
        for (int i = 0; i < state_length_; i++) {
            if (states_[0][i] != 0) {
                first_is_start = false;
                break;
            }
        }
    }

    for (int i = 0; i < n_vertices_; i++) {
        // Check if this is the starting vertex
        bool is_start = true;
        for (int j = 0; j < state_length_; j++) {
            if (states_[i][j] != 0) {
                is_start = false;
                break;
            }
        }

        if (is_start && i == 0) {
            vertices.push_back(start);
        } else {
            vertices.push_back(g.find_or_create_vertex_p(states_[i]));
        }
    }

    // Add regular edges
    for (const auto& edge : edges_) {
        Vertex* from_v = vertices[edge.from_idx];
        Vertex* to_v = vertices[edge.to_idx];
        from_v->add_edge(*to_v, edge.weight);
    }

    // Add starting vertex edges
    for (const auto& edge : start_edges_) {
        Vertex* to_v = vertices[edge.to_idx];
        start->add_edge(*to_v, edge.weight);
    }

    // Add parameterized edges
    for (const auto& edge : param_edges_) {
        Vertex* from_v = vertices[edge.from_idx];
        Vertex* to_v = vertices[edge.to_idx];

        // Compute weight: dot product only (no base_weight)
        double weight = 0.0;
        for (int i = 0; i < param_length_; i++) {
            weight += edge.coefficients[i] * theta[i];
        }

        from_v->add_edge(*to_v, weight);
    }

    // Add starting vertex parameterized edges
    for (const auto& edge : start_param_edges_) {
        Vertex* to_v = vertices[edge.to_idx];

        // Compute weight: dot product only (no base_weight)
        double weight = 0.0;
        for (int i = 0; i < param_length_; i++) {
            weight += edge.coefficients[i] * theta[i];
        }

        start->add_edge(*to_v, weight);
    }

    return g;
}

double GraphBuilder::factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= static_cast<double>(i);
    }
    return result;
}

std::vector<double> GraphBuilder::compute_moments_impl(Graph& g, int nr_moments, const std::vector<double>& rewards) {
    std::vector<double> result(nr_moments);

    // First moment: E[T] or E[R·T] if rewards provided
    // If rewards is empty → standard moments
    // If rewards provided → reward-transformed moments
    std::vector<double> rewards2 = g.expected_waiting_time(rewards);

    if (rewards2.empty()) {
        throw std::runtime_error("expected_waiting_time returned empty vector");
    }

    result[0] = rewards2[0];

    // Higher moments: E[T^k]
    // This follows the algorithm from _moments in phasic_pybind.cpp
    std::vector<double> rewards3(rewards2.size());

    for (int k = 1; k < nr_moments; k++) {
        // For standard moments (empty rewards), just copy rewards2
        // For custom rewards, multiply by the original rewards
        if (!rewards.empty()) {
            for (size_t i = 0; i < rewards2.size(); i++) {
                rewards3[i] = rewards2[i] * rewards[i];
            }
        } else {
            // Standard moments: copy rewards2 (not square it!)
            rewards3 = rewards2;
        }

        rewards2 = g.expected_waiting_time(rewards3);

        if (rewards2.empty()) {
            throw std::runtime_error("expected_waiting_time returned empty vector for higher moment");
        }

        // E[T^(k+1)] = (k+1)! * result
        result[k] = factorial(k + 1) * rewards2[0];
    }

    return result;
}

py::array_t<double> GraphBuilder::compute_moments(
    py::array_t<double> theta,
    int nr_moments
) {
    // Step 1: Extract data from numpy arrays (requires GIL)
    auto theta_buf = theta.unchecked<1>();
    size_t theta_len = theta_buf.shape(0);

    // Copy theta to C++ vector
    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }

    // Step 2: Release GIL for C++ computation
    std::vector<double> moments;
    {
        py::gil_scoped_release release;

        // Build graph (pure C++)
        Graph g = build(theta_vec.data(), theta_len);

        // Compute moments (pure C++) - empty rewards = standard moments
        std::vector<double> rewards;  // Empty for standard moments
        moments = compute_moments_impl(g, nr_moments, rewards);
    }
    // GIL automatically reacquired here

    // Step 3: Convert to numpy array (requires GIL, which we now have)
    py::array_t<double> result(moments.size());
    auto result_buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < moments.size(); i++) {
        result_buf(i) = moments[i];
    }

    return result;
}

py::array_t<double> GraphBuilder::compute_pmf(
    py::array_t<double> theta,
    py::array_t<double> times,
    bool discrete,
    int granularity
) {
    // Step 1: Extract data from numpy arrays (requires GIL)
    auto theta_buf = theta.unchecked<1>();
    size_t theta_len = theta_buf.shape(0);
    auto times_buf = times.unchecked<1>();
    size_t n_times = times_buf.shape(0);

    // Copy theta and times to C++ vectors (still have GIL)
    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }
    std::vector<double> times_vec(n_times);
    for (size_t i = 0; i < n_times; i++) {
        times_vec[i] = times_buf(i);
    }

    // Step 2: Release GIL for C++ computation
    std::vector<double> result_vec(n_times);
    {
        py::gil_scoped_release release;

        // Build graph (pure C++, no Python objects)
        Graph g = build(theta_vec.data(), theta_len);

        // Compute PMF/PDF (pure C++)
        if (discrete) {
            for (size_t i = 0; i < n_times; i++) {
                int jump_count = static_cast<int>(times_vec[i]);
                result_vec[i] = g.dph_pmf(jump_count);
            }
        } else {
            for (size_t i = 0; i < n_times; i++) {
                result_vec[i] = g.pdf(times_vec[i], granularity);
            }
        }
    }
    // GIL automatically reacquired here

    // Step 3: Create numpy array from C++ vector (requires GIL, which we now have)
    py::array_t<double> result(n_times);
    auto result_buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n_times; i++) {
        result_buf(i) = result_vec[i];
    }

    return result;
}

std::pair<py::array_t<double>, py::array_t<double>>
GraphBuilder::compute_pmf_and_moments(
    py::array_t<double> theta,
    py::array_t<double> times,
    int nr_moments,
    bool discrete,
    int granularity,
    py::object rewards_obj
) {
    // Step 1: Extract data from numpy arrays (requires GIL)
    auto theta_buf = theta.unchecked<1>();
    size_t theta_len = theta_buf.shape(0);
    auto times_buf = times.unchecked<1>();
    size_t n_times = times_buf.shape(0);

    // Copy to C++ vectors
    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }
    std::vector<double> times_vec(n_times);
    for (size_t i = 0; i < n_times; i++) {
        times_vec[i] = times_buf(i);
    }

    // Detect reward dimensionality: None, 1D, or 2D
    bool has_rewards = !rewards_obj.is_none();
    bool is_2d_rewards = false;
    size_t n_features = 1;
    size_t n_vertices = 0;

    std::vector<double> rewards_vec_1d;  // For 1D case
    std::vector<std::vector<double>> rewards_2d;  // For 2D case: [feature][vertex]

    if (has_rewards) {
        auto rewards_array = rewards_obj.cast<py::array_t<double>>();
        auto rewards_info = rewards_array.request();

        if (rewards_info.ndim == 1) {
            // 1D rewards: (n_vertices,)
            n_vertices = rewards_info.shape[0];
            rewards_vec_1d.resize(n_vertices);
            double* rewards_ptr = static_cast<double*>(rewards_info.ptr);
            for (size_t i = 0; i < n_vertices; i++) {
                rewards_vec_1d[i] = rewards_ptr[i];
            }
        } else if (rewards_info.ndim == 2) {
            // 2D rewards: (n_vertices, n_features)
            is_2d_rewards = true;
            n_vertices = rewards_info.shape[0];
            n_features = rewards_info.shape[1];

            // Extract each feature's reward vector
            rewards_2d.resize(n_features, std::vector<double>(n_vertices));
            double* rewards_ptr = static_cast<double*>(rewards_info.ptr);

            for (size_t j = 0; j < n_features; j++) {
                for (size_t i = 0; i < n_vertices; i++) {
                    // Row-major: rewards[i, j] = ptr[i * n_features + j]
                    rewards_2d[j][i] = rewards_ptr[i * n_features + j];
                }
            }
        } else {
            throw std::runtime_error("Rewards must be 1D or 2D array");
        }
    }

    // Step 2: Release GIL for C++ computation
    std::vector<std::vector<double>> pmf_2d;  // [time][feature]
    std::vector<std::vector<double>> moments_2d;  // [feature][moment]

    {
        py::gil_scoped_release release;

        // Build graph ONCE (pure C++)
        Graph g = build(theta_vec.data(), theta_len);

        if (is_2d_rewards) {
            // === MULTIVARIATE CASE: Compute PDF per feature ===
            pmf_2d.resize(n_times, std::vector<double>(n_features));
            moments_2d.resize(n_features, std::vector<double>(nr_moments));

            // Compute PDF/PMF and moments per feature (each with its own reward transformation)
            for (size_t j = 0; j < n_features; j++) {
                // Transform graph for feature j
                Graph g_transformed = g.reward_transform(rewards_2d[j]);

                // Compute PDF/PMF for feature j from transformed graph
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        int jump_count = static_cast<int>(times_vec[t]);
                        pmf_2d[t][j] = g_transformed.dph_pmf(jump_count);
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        pmf_2d[t][j] = g_transformed.pdf(times_vec[t], granularity);
                    }
                }

                // Compute moments for feature j from transformed graph
                // Note: g_transformed already has rewards applied, so pass empty vector
                moments_2d[j] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
            }
        } else {
            // === UNIVARIATE CASE: Single PDF ===
            pmf_2d.resize(n_times, std::vector<double>(1));
            moments_2d.resize(1, std::vector<double>(nr_moments));

            // Check if we need to transform the graph
            if (!rewards_vec_1d.empty()) {
                // Transform graph with rewards
                Graph g_transformed = g.reward_transform(rewards_vec_1d);

                // Compute PDF/PMF from transformed graph
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        int jump_count = static_cast<int>(times_vec[t]);
                        pmf_2d[t][0] = g_transformed.dph_pmf(jump_count);
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        pmf_2d[t][0] = g_transformed.pdf(times_vec[t], granularity);
                    }
                }

                // Compute moments from transformed graph
                // Note: g_transformed already has rewards applied, so pass empty vector
                moments_2d[0] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
            } else {
                // Use original graph (no transformation)
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        int jump_count = static_cast<int>(times_vec[t]);
                        pmf_2d[t][0] = g.dph_pmf(jump_count);
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        pmf_2d[t][0] = g.pdf(times_vec[t], granularity);
                    }
                }

                // Compute moments from original graph (no rewards)
                moments_2d[0] = compute_moments_impl(g, nr_moments, std::vector<double>());
            }
        }
    }
    // GIL automatically reacquired here

    // Step 3: Convert to numpy arrays (requires GIL, which we now have)
    py::array_t<double> pmf_result, moments_result;

    if (is_2d_rewards) {
        // Return 2D arrays
        pmf_result = py::array_t<double>({n_times, n_features});
        auto pmf_buf = pmf_result.mutable_unchecked<2>();
        for (size_t t = 0; t < n_times; t++) {
            for (size_t j = 0; j < n_features; j++) {
                pmf_buf(t, j) = pmf_2d[t][j];
            }
        }

        moments_result = py::array_t<double>({n_features, (size_t)nr_moments});
        auto moments_buf = moments_result.mutable_unchecked<2>();
        for (size_t j = 0; j < n_features; j++) {
            for (int m = 0; m < nr_moments; m++) {
                moments_buf(j, m) = moments_2d[j][m];
            }
        }
    } else {
        // Return 1D arrays (backward compatible)
        pmf_result = py::array_t<double>(n_times);
        auto pmf_buf = pmf_result.mutable_unchecked<1>();
        for (size_t t = 0; t < n_times; t++) {
            pmf_buf(t) = pmf_2d[t][0];
        }

        moments_result = py::array_t<double>(nr_moments);
        auto moments_buf = moments_result.mutable_unchecked<1>();
        for (int m = 0; m < nr_moments; m++) {
            moments_buf(m) = moments_2d[0][m];
        }
    }

    return std::make_pair(pmf_result, moments_result);
}

} // namespace parameterized
} // namespace phasic
