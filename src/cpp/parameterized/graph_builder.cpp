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
        if (j.contains("param_edges")) {
            auto param_edges_json = j.at("param_edges");
            param_edges_.reserve(param_edges_json.size());
            for (const auto& edge_arr : param_edges_json) {
                ParameterizedEdge edge;
                edge.from_idx = edge_arr[0].get<int>();
                edge.to_idx = edge_arr[1].get<int>();
                edge.coefficients.reserve(param_length_);
                for (int i = 2; i < 2 + param_length_; i++) {
                    edge.coefficients.push_back(edge_arr[i].get<double>());
                }
                param_edges_.push_back(edge);
            }
        }

        // Parse starting vertex parameterized edges (if present)
        if (j.contains("start_param_edges")) {
            auto start_param_edges_json = j.at("start_param_edges");
            start_param_edges_.reserve(start_param_edges_json.size());
            for (const auto& edge_arr : start_param_edges_json) {
                ParameterizedEdge edge;
                edge.from_idx = -1;  // Starting vertex
                edge.to_idx = edge_arr[0].get<int>();
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

        // Compute weight: dot product of coefficients and theta
        double weight = 0.0;
        for (int i = 0; i < param_length_; i++) {
            weight += edge.coefficients[i] * theta[i];
        }

        from_v->add_edge(*to_v, weight);
    }

    // Add starting vertex parameterized edges
    for (const auto& edge : start_param_edges_) {
        Vertex* to_v = vertices[edge.to_idx];

        // Compute weight
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

    // Extract optional rewards vector
    std::vector<double> rewards_vec;
    if (!rewards_obj.is_none()) {
        auto rewards_array = rewards_obj.cast<py::array_t<double>>();
        auto rewards_buf = rewards_array.unchecked<1>();
        size_t n_rewards = rewards_buf.shape(0);
        rewards_vec.resize(n_rewards);
        for (size_t i = 0; i < n_rewards; i++) {
            rewards_vec[i] = rewards_buf(i);
        }
    }
    // If rewards_obj is None, rewards_vec remains empty → standard moments

    // Step 2: Release GIL for C++ computation
    std::vector<double> pmf_vec(n_times);
    std::vector<double> moments;
    {
        py::gil_scoped_release release;

        // Build graph ONCE (pure C++)
        Graph g = build(theta_vec.data(), theta_len);

        // Compute PMF/PDF (pure C++)
        if (discrete) {
            for (size_t i = 0; i < n_times; i++) {
                int jump_count = static_cast<int>(times_vec[i]);
                pmf_vec[i] = g.dph_pmf(jump_count);
            }
        } else {
            for (size_t i = 0; i < n_times; i++) {
                pmf_vec[i] = g.pdf(times_vec[i], granularity);
            }
        }

        // Compute moments using same graph (pure C++)
        // Pass rewards_vec (empty for standard moments, filled for reward transformation)
        moments = compute_moments_impl(g, nr_moments, rewards_vec);
    }
    // GIL automatically reacquired here

    // Step 3: Convert to numpy arrays (requires GIL, which we now have)
    py::array_t<double> pmf_result(n_times);
    auto pmf_buf = pmf_result.mutable_unchecked<1>();
    for (size_t i = 0; i < n_times; i++) {
        pmf_buf(i) = pmf_vec[i];
    }

    py::array_t<double> moments_result(moments.size());
    auto moments_buf = moments_result.mutable_unchecked<1>();
    for (size_t i = 0; i < moments.size(); i++) {
        moments_buf(i) = moments[i];
    }

    return std::make_pair(pmf_result, moments_result);
}

} // namespace parameterized
} // namespace phasic
