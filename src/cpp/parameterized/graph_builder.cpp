#include "graph_builder.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <limits>
#include <unordered_map>  // tl_persistent_graphs cache
#include <memory>         // std::unique_ptr

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

        // Parse weight computation mode (default: linear)
        if (j.contains("weight_mode")) {
            std::string mode = j.at("weight_mode").get<std::string>();
            if (mode == "log") {
                weight_mode_ = WeightMode::LOG;
            } else if (mode == "formula") {
                weight_mode_ = WeightMode::FORMULA;
            } else if (mode != "linear") {
                throw std::invalid_argument("Unknown weight_mode: " + mode);
            }
        }

        // Formula mode: parse the compiled per-edge weight tape. Strict
        // pairing (no silent fallback): 'formula' requires the tape, and the
        // tape must not appear under any other mode.
        if (weight_mode_ == WeightMode::FORMULA) {
            if (!j.contains("weight_formula_tape")) {
                throw std::invalid_argument(
                    "weight_mode='formula' requires a 'weight_formula_tape' "
                    "object in the serialized graph");
            }
            const auto& tj = j.at("weight_formula_tape");
            tape_ops_ = tj.at("ops").get<std::vector<int>>();
            tape_consts_ = tj.at("consts").get<std::vector<double>>();
            tape_stack_depth_ = tj.at("stack_depth").get<size_t>();
            tape_n_theta_ = tj.at("n_theta").get<size_t>();
            tape_n_coeff_ = tj.at("n_coeff").get<size_t>();
            has_tape_ = true;
        } else if (j.contains("weight_formula_tape")) {
            throw std::invalid_argument(
                "'weight_formula_tape' present but weight_mode is not "
                "'formula'");
        }

        // Elimination ordering flag (default false). Lets graph.dyn_ordering
        // propagate to this rebuilt graph (applied to the ptd_graph in build()).
        if (j.contains("dyn_ordering")) {
            dyn_ordering_ = j.at("dyn_ordering").get<bool>();
        }

        // Discreteness flag (default false). Lets graph.is_discrete propagate
        // to this rebuilt graph so the reward transform and moments dispatch
        // on it -- without this the parameterized path applies the continuous
        // reward transform and continuous moments to a DPH (N1/N2).
        if (j.contains("is_discrete")) {
            is_discrete_ = j.at("is_discrete").get<bool>();
        }

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

        // Parse coefficient-less constant edges (optional; older
        // serialised graphs do not have this field). These are rebuilt
        // via direct struct manipulation in build() so they bypass the
        // EDGE_MODE lock and coexist with parameterised edges.
        if (j.contains("constant_edges")) {
            auto const_edges_json = j.at("constant_edges");
            constant_edges_.reserve(const_edges_json.size());
            for (const auto& edge_arr : const_edges_json) {
                RegularEdge edge;
                edge.from_idx = edge_arr[0].get<int>();
                edge.to_idx = edge_arr[1].get<int>();
                edge.weight = edge_arr[2].get<double>();
                constant_edges_.push_back(edge);
            }
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

                // Read the FULL coefficient row, which may be LONGER than
                // param_length_ when theta_dim < coefficient length (e.g. a
                // weight_formula referencing per-edge data beyond the optimized
                // parameters). The tape VM bounds-checks c-indices against this
                // length; theta length stays param_length_ (the theta dim).
                edge.coefficients.reserve(edge_arr.size() - 2);
                for (int i = 2; i < static_cast<int>(edge_arr.size()); i++) {
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

                // Read the FULL coefficient row (see param_edges_ note above).
                edge.coefficients.reserve(edge_arr.size() - 1);
                for (int i = 1; i < static_cast<int>(edge_arr.size()); i++) {
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

    // Pin param_length (= theta dimension) up front so edges whose coefficient
    // vectors are LONGER than theta_dim (weight_formula with per-edge data
    // beyond the optimized parameters) are accepted, and the graph's theta
    // length stays param_length_ instead of locking to the first parameterized
    // edge's coefficient length. When param_length_ == coefficient length (the
    // common case) this is equivalent to that implicit lock.
    if (param_length_ > 0) {
        g.set_param_length(static_cast<size_t>(param_length_));
    }

    // Inherit the elimination-ordering flag from the serialized graph so
    // ptd_precompute_reward_compute_graph picks the dynamic min-degree
    // variant for this rebuilt graph (matches PHASIC_DYN_ORDERING / the
    // dyn_ordering property on the originating Graph).
    g.c_graph()->use_dyn_ordering = dyn_ordering_;

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
            // Always create new vertex, even if state is duplicate
            // Serialization preserves vertex identity via indices, not states
            vertices.push_back(g.create_vertex_p(states_[i]));
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

    // Add parameterised edges via the parameterised constructor so the
    // resulting phasic::Graph is locked into PARAMETERIZED edge_mode.
    // That is required for update_weights() to work on the persistent
    // graph cached by get_or_init_persistent_graph(): without it,
    // update_weights() rejects the call with a "constant graph" error.
    //
    // C-side ptd_graph_add_edge initialises edge->weight as
    // sum(coefficients * 1) (default theta = 1) — the ``weight``
    // argument to add_edge_parameterized is ignored. The caller
    // (get_or_init_persistent_graph) follows up with
    // update_weights_parameterized(theta) immediately after build()
    // to install the correct weights for the requested theta.
    for (const auto& edge : param_edges_) {
        Vertex* from_v = vertices[edge.from_idx];
        Vertex* to_v = vertices[edge.to_idx];
        double initial_weight = compute_weight(edge.coefficients, theta);
        from_v->add_edge_parameterized(*to_v, initial_weight, edge.coefficients);
    }

    // Add coefficient-less constant edges via direct ptd_edge struct
    // manipulation, bypassing the EDGE_MODE lock. This mirrors the
    // construction path used by Vertex::add_aux_vertex_constant in
    // phasiccpp.cpp. update_weights skips edges with
    // coefficients_length == 0 (see src/c/phasic.c around L4435), so
    // their weights remain at the constant value below regardless of
    // any later theta updates.
    for (const auto& edge : constant_edges_) {
        Vertex* from_v = vertices[edge.from_idx];
        Vertex* to_v = vertices[edge.to_idx];
        struct ptd_edge *new_edge =
            (struct ptd_edge *)malloc(sizeof(*new_edge));
        if (new_edge == NULL) {
            throw std::runtime_error(
                "GraphBuilder: failed to allocate constant edge"
            );
        }
        new_edge->to = to_v->c_vertex();
        new_edge->weight = edge.weight;
        new_edge->coefficients_length = 0;
        new_edge->coefficients = NULL;
        new_edge->should_free_coefficients = false;

        struct ptd_vertex *src = from_v->c_vertex();
        struct ptd_edge **new_edges_arr = (struct ptd_edge **)realloc(
            src->edges,
            (src->edges_length + 1) * sizeof(struct ptd_edge *)
        );
        if (new_edges_arr == NULL) {
            free(new_edge);
            throw std::runtime_error(
                "GraphBuilder: failed to grow constant-edge array"
            );
        }
        src->edges = new_edges_arr;
        src->edges[src->edges_length] = new_edge;
        src->edges_length++;
    }

    // Starting-vertex (IPV) edges are always treated as constants by
    // ptd_graph_update_weights (the C path skips IPV edges per
    // phasic.c:3245). So leaving these as scalar add_edge is correct
    // and avoids needlessly putting the starting vertex's edges into
    // PARAMETERIZED mode (the mode lock only applies to non-IPV edges
    // anyway, see phasiccpp.cpp:256).
    for (const auto& edge : start_param_edges_) {
        Vertex* to_v = vertices[edge.to_idx];
        start->add_edge(*to_v, compute_weight(edge.coefficients, theta));
    }

    /* C-side ptd_graph_add_edge initialises edge->weight as
     * sum(coefficients * 1) (default theta=1) — the ``initial_weight``
     * argument we passed to add_edge_parameterized above is ignored.
     * Apply the requested theta now so the returned graph has the
     * correct edge weights for the caller. This applies to every
     * caller of build() — get_or_init_persistent_graph (which would
     * call update_weights anyway), but also the direct
     * ``builder->build()`` sites in graph_builder_ffi.cpp that do not
     * cache the graph and thus did not call update_weights. */
    // Install the compiled weight tape on the C graph BEFORE update_weights,
    // so update_weights (here, and on every later theta via the persistent-
    // graph cache) fills each non-IPV edge weight by running the formula in C.
    // Ownership transfers to the graph (freed by ptd_graph_destroy).
    if (has_tape_) {
        struct ptd_weight_tape* tape = ptd_weight_tape_create(
            tape_ops_.data(), tape_ops_.size(),
            tape_consts_.data(), tape_consts_.size(),
            tape_stack_depth_, tape_n_theta_, tape_n_coeff_);
        if (tape == nullptr) {
            throw std::runtime_error(
                "failed to allocate weight_formula tape");
        }
        ptd_graph_set_weight_tape(g.c_graph(), tape);
    }

    if (!param_edges_.empty()) {
        std::vector<double> theta_vec(theta, theta + theta_len);
        bool use_log = (weight_mode_ == WeightMode::LOG);
        g.update_weights_parameterized(theta_vec, use_log);
    }

    return g;
}

namespace {
/* Per-thread cache of persistent graphs, keyed by GraphBuilder
 * identity. Each GraphBuilder may be used from multiple OS threads
 * (e.g. JAX pmap with multi-threaded pure_callback, OpenMP-parallel
 * vmap in graph_builder_ffi.cpp). Each thread needs its own
 * phasic::Graph instance because Graph is not thread-safe under
 * update_weights — it mutates edge->weight in place.
 *
 * thread_local guarantees per-OS-thread storage; the inner map is
 * keyed by the GraphBuilder*'s address so multiple GraphBuilder
 * instances (different model structures) don't share entries. */
thread_local std::unordered_map<const phasic::parameterized::GraphBuilder*,
                                std::unique_ptr<phasic::Graph>>
        tl_persistent_graphs;
}  // namespace

GraphBuilder::~GraphBuilder() {
    /* Evict this builder's entry from the current thread's
     * persistent-graph cache so a new builder allocated at the same
     * address can't accidentally hit a stale entry. ``erase`` on a
     * key that isn't present is a no-op. */
    tl_persistent_graphs.erase(this);
}

phasic::Graph& GraphBuilder::get_or_init_persistent_graph(
        const double* theta, size_t theta_len) {
    /* Graphs with no non-IPV parameterised edges (only IPV
     * parameterised edges, or no parameterised edges at all) cannot be
     * refreshed via ``ptd_graph_update_weights``: the C function
     * rejects UNLOCKED graphs (no non-IPV edges → mode never locked),
     * and IPV edges are treated as constants by update_weights anyway.
     *
     * For these graphs the persistent-graph optimisation also has
     * nothing to amortise — there's no parameterised reward compute
     * graph to cache because there are no parameterised non-IPV
     * edges. So we fall back to fresh-graph-per-call, which gives the
     * pre-Stage-A1 behaviour and is correct for all theta values.
     *
     * The fresh graph is still stored in tl_persistent_graphs (keyed
     * by ``this``) so subsequent calls overwrite the entry rather
     * than leaking. */
    if (param_edges_.empty()) {
        auto g = std::make_unique<phasic::Graph>(build(theta, theta_len));
        tl_persistent_graphs[this] = std::move(g);
        return *tl_persistent_graphs[this];
    }

    auto it = tl_persistent_graphs.find(this);
    if (it == tl_persistent_graphs.end()) {
        // First call from this thread for this builder: build the
        // graph in PARAMETERIZED mode (build() now uses
        // add_edge_parameterized and applies update_weights for the
        // requested theta internally). The first forward call on
        // this graph will populate parameterized_reward_compute_graph;
        // Stage A0 ensures that cache survives subsequent
        // update_weights calls.
        auto g = std::make_unique<phasic::Graph>(build(theta, theta_len));
        auto [inserted_it, ok] = tl_persistent_graphs.emplace(this, std::move(g));
        return *inserted_it->second;
    }
    // Cached: refresh edge weights for the new theta. update_weights
    // bumps weight_version (Stage 1) and clears the *concrete*
    // reward_compute_graph but preserves the symbolic
    // parameterized_reward_compute_graph (Stage A0). The next forward
    // call rebuilds the concrete cache via the cheap O(commands) path
    // rather than the expensive O(n^3) Gaussian elimination.
    std::vector<double> theta_vec(theta, theta + theta_len);
    bool use_log = (weight_mode_ == WeightMode::LOG);
    it->second->update_weights_parameterized(theta_vec, use_log);
    return *it->second;
}

double GraphBuilder::compute_weight(const std::vector<double>& coefficients, const double* theta) const {
    if (weight_mode_ == WeightMode::FORMULA) {
        // Build-time / IPV edge weights via the same C tape VM that
        // update_weights uses. (Non-IPV edge values are recomputed by
        // update_weights right after build(); IPV edges keep this value.)
        double w;
        if (ptd_weight_tape_eval_arrays(
                tape_ops_.data(), tape_ops_.size(),
                tape_consts_.data(), tape_consts_.size(),
                tape_stack_depth_,
                theta, static_cast<size_t>(param_length_),
                coefficients.data(), coefficients.size(), &w) != 0) {
            std::string msg = std::string("weight_formula evaluation failed: ")
                            + reinterpret_cast<const char*>(ptd_err);
            ptd_err[0] = '\0';   // clear so the next C entry sees no stale error
            throw std::invalid_argument(msg);
        }
        if (!std::isfinite(w) || w < 0.0) {
            throw std::invalid_argument(
                "weight_formula produced a non-finite or negative edge weight");
        }
        return w;
    }
    if (weight_mode_ == WeightMode::LOG) {
        // Multiplicative weight: exp(Σ log(c_k * θ_k)) = Π(c_k * θ_k)
        double log_sum = 0.0;
        for (int i = 0; i < param_length_; i++) {
            double product = coefficients[i] * theta[i];
            if (product <= 0.0) {
                throw std::invalid_argument(
                    "log weight mode requires all (coefficient * parameter) products to be positive. "
                    "Got non-positive product at index " + std::to_string(i));
            }
            log_sum += std::log(product);
        }
        return std::exp(log_sum);
    } else {
        // Linear weight: Σ c_k * θ_k
        double weight = 0.0;
        for (int i = 0; i < param_length_; i++) {
            weight += coefficients[i] * theta[i];
        }
        return weight;
    }
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

        // Reuse this thread's persistent graph: O(n^3) elimination is
        // amortised across calls; only edge weights are refreshed for
        // the new theta.
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

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

        // Reuse this thread's persistent graph (Stage A1) so the
        // symbolic elimination cache is amortised across theta calls.
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

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

// Convert a double reward vector to the integer vector the discrete (DPH)
// reward transform requires. DPH rewards count steps, so a non-integer or
// negative value is a user error to surface, not something to silently
// truncate (no silent fallback).
static std::vector<int> rewards_to_int_or_throw(const std::vector<double>& r) {
    std::vector<int> out(r.size());
    for (size_t i = 0; i < r.size(); i++) {
        double v = r[i];
        double rounded = std::round(v);
        if (v < 0.0 || std::abs(v - rounded) > 1e-9) {
            throw std::runtime_error(
                "discrete (DPH) reward transform requires non-negative integer "
                "rewards; got " + std::to_string(v) + " at index " +
                std::to_string(i));
        }
        out[i] = static_cast<int>(rounded);
    }
    return out;
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
            // 2D rewards: (n_features, n_vertices) - each row is one feature's reward vector
            is_2d_rewards = true;
            n_features = rewards_info.shape[0];
            n_vertices = rewards_info.shape[1];

            // Extract each feature's reward vector (each row = one feature)
            rewards_2d.resize(n_features, std::vector<double>(n_vertices));
            double* rewards_ptr = static_cast<double*>(rewards_info.ptr);

            for (size_t j = 0; j < n_features; j++) {
                for (size_t i = 0; i < n_vertices; i++) {
                    // Row-major: rewards[j, i] = ptr[j * n_vertices + i]
                    rewards_2d[j][i] = rewards_ptr[j * n_vertices + i];
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

        // Reuse this thread's persistent graph (Stage A1). Stage 2's
        // clone-first reward_transform ensures the per-feature
        // ``g.reward_transform(rewards_2d[j])`` calls below do NOT
        // mutate the cached source graph, so reuse is correctness-
        // preserving even on the multivariate path.
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

        // A reward vector shorter than the graph reads out of bounds inside
        // ptd_graph_reward_transform / expected_waiting_time (both take a raw
        // pointer and loop over vertices_length). n_vertices above is taken
        // from the reward array's own shape, so validate it against the graph
        // before any reward is consumed. Fail loud rather than read heap.
        if (has_rewards) {
            size_t graph_nv = g.c_graph()->vertices_length;
            if (n_vertices != graph_nv) {
                throw std::runtime_error(
                    "rewards length (" + std::to_string(n_vertices) +
                    ") must equal the number of graph vertices (" +
                    std::to_string(graph_nv) + ")");
            }
        }

        // Discreteness: honour the per-call flag OR the graph's own
        // is_discrete (propagated through serialize). A DPH must use the
        // discrete reward transform (integer rewards), dph_pmf, and the
        // discrete moment correction -- not their continuous counterparts.
        bool is_disc = discrete || is_discrete_;

        if (is_2d_rewards) {
            // === MULTIVARIATE CASE: Compute PDF per feature ===
            pmf_2d.resize(n_times, std::vector<double>(n_features));
            moments_2d.resize(n_features, std::vector<double>(nr_moments));

            // Compute PDF/PMF and moments per feature (each with its own reward transformation)
            for (size_t j = 0; j < n_features; j++) {
                // Transform graph for feature j (discrete transform for a DPH)
                Graph g_transformed = is_disc
                    ? g.reward_transform_discrete(rewards_to_int_or_throw(rewards_2d[j]))
                    : g.reward_transform(rewards_2d[j]);

                // Compute PDF/PMF for feature j from transformed graph
                // Skip NaN times - return NaN in output for those positions
                if (is_disc) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][j] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][j] = g_transformed.dph_pmf(jump_count);
                        }
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][j] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][j] = g_transformed.pdf(times_vec[t], granularity);
                        }
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
                // Transform graph with rewards (discrete transform for a DPH)
                Graph g_transformed = is_disc
                    ? g.reward_transform_discrete(rewards_to_int_or_throw(rewards_vec_1d))
                    : g.reward_transform(rewards_vec_1d);

                // Compute PDF/PMF from transformed graph
                // Skip NaN times - return NaN in output for those positions
                if (is_disc) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][0] = g_transformed.dph_pmf(jump_count);
                        }
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][0] = g_transformed.pdf(times_vec[t], granularity);
                        }
                    }
                }

                // Compute moments from transformed graph
                // Note: g_transformed already has rewards applied, so pass empty vector
                moments_2d[0] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
            } else {
                // Use original graph (no transformation)
                // Skip NaN times - return NaN in output for those positions
                if (is_disc) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][0] = g.dph_pmf(jump_count);
                        }
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][0] = g.pdf(times_vec[t], granularity);
                        }
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

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
GraphBuilder::compute_pmf_moments_and_cdf_zero(
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

    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }
    std::vector<double> times_vec(n_times);
    for (size_t i = 0; i < n_times; i++) {
        times_vec[i] = times_buf(i);
    }

    bool has_rewards = !rewards_obj.is_none();
    bool is_2d_rewards = false;
    size_t n_features = 1;
    size_t n_vertices = 0;

    std::vector<double> rewards_vec_1d;
    std::vector<std::vector<double>> rewards_2d;

    if (has_rewards) {
        auto rewards_array = rewards_obj.cast<py::array_t<double>>();
        auto rewards_info = rewards_array.request();
        if (rewards_info.ndim == 1) {
            n_vertices = rewards_info.shape[0];
            rewards_vec_1d.resize(n_vertices);
            double* rewards_ptr = static_cast<double*>(rewards_info.ptr);
            for (size_t i = 0; i < n_vertices; i++) {
                rewards_vec_1d[i] = rewards_ptr[i];
            }
        } else if (rewards_info.ndim == 2) {
            is_2d_rewards = true;
            n_features = rewards_info.shape[0];
            n_vertices = rewards_info.shape[1];
            rewards_2d.resize(n_features, std::vector<double>(n_vertices));
            double* rewards_ptr = static_cast<double*>(rewards_info.ptr);
            for (size_t j = 0; j < n_features; j++) {
                for (size_t i = 0; i < n_vertices; i++) {
                    rewards_2d[j][i] = rewards_ptr[j * n_vertices + i];
                }
            }
        } else {
            throw std::runtime_error("Rewards must be 1D or 2D array");
        }
    }

    // Step 2: C++ computation (release GIL)
    std::vector<std::vector<double>> pmf_2d;
    std::vector<std::vector<double>> moments_2d;
    std::vector<double> cdf_zero_vec(n_features, 0.0);
    {
        py::gil_scoped_release release;
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

        if (is_2d_rewards) {
            pmf_2d.resize(n_times, std::vector<double>(n_features));
            moments_2d.resize(n_features, std::vector<double>(nr_moments));
            for (size_t j = 0; j < n_features; j++) {
                Graph g_transformed = g.reward_transform(rewards_2d[j]);
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][j] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][j] = g_transformed.dph_pmf(jump_count);
                        }
                    }
                    cdf_zero_vec[j] = g_transformed.dph_cdf(0);
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][j] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][j] = g_transformed.pdf(times_vec[t], granularity);
                        }
                    }
                    cdf_zero_vec[j] = g_transformed.cdf(0.0, granularity);
                }
                moments_2d[j] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
            }
        } else {
            pmf_2d.resize(n_times, std::vector<double>(1));
            moments_2d.resize(1, std::vector<double>(nr_moments));
            if (!rewards_vec_1d.empty()) {
                Graph g_transformed = g.reward_transform(rewards_vec_1d);
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][0] = g_transformed.dph_pmf(jump_count);
                        }
                    }
                    cdf_zero_vec[0] = g_transformed.dph_cdf(0);
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][0] = g_transformed.pdf(times_vec[t], granularity);
                        }
                    }
                    cdf_zero_vec[0] = g_transformed.cdf(0.0, granularity);
                }
                moments_2d[0] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
            } else {
                if (discrete) {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            int jump_count = static_cast<int>(times_vec[t]);
                            pmf_2d[t][0] = g.dph_pmf(jump_count);
                        }
                    }
                } else {
                    for (size_t t = 0; t < n_times; t++) {
                        if (std::isnan(times_vec[t])) {
                            pmf_2d[t][0] = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            pmf_2d[t][0] = g.pdf(times_vec[t], granularity);
                        }
                    }
                }
                moments_2d[0] = compute_moments_impl(g, nr_moments, std::vector<double>());
                // No rewards → no reward-induced atom; leave cdf_zero_vec[0] = 0.0.
            }
        }
    }

    // Step 3: numpy arrays (GIL held)
    py::array_t<double> pmf_result, moments_result, cdf_zero_result;
    if (is_2d_rewards) {
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
        cdf_zero_result = py::array_t<double>(n_features);
        auto cdf_buf = cdf_zero_result.mutable_unchecked<1>();
        for (size_t j = 0; j < n_features; j++) {
            cdf_buf(j) = cdf_zero_vec[j];
        }
    } else {
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
        // Scalar cdf_zero exposed as a 0-D numpy array of size 1
        // so the Python side has a uniform "indexable" shape.
        cdf_zero_result = py::array_t<double>(1);
        auto cdf_buf = cdf_zero_result.mutable_unchecked<1>();
        cdf_buf(0) = cdf_zero_vec[0];
    }
    return std::make_tuple(pmf_result, moments_result, cdf_zero_result);
}

py::array_t<double> GraphBuilder::compute_pmf_multivariate(
    py::array_t<double> theta,
    py::array_t<double> times,
    py::array_t<double> rewards,
    bool discrete,
    int granularity,
    bool compute_joint
) {
    // Validate compute_joint mode
    if (compute_joint) {
        throw std::runtime_error(
            "Joint PDF computation not yet implemented.\n"
            "  Use compute_joint=false for independent feature PDFs (sparse mode).\n"
            "  Joint PDF support planned for Phase 2."
        );
    }

    // Step 1: Extract and validate data from numpy arrays (requires GIL)
    auto theta_buf = theta.unchecked<1>();
    size_t theta_len = theta_buf.shape(0);

    auto times_buf = times.unchecked<2>();
    size_t n_times = times_buf.shape(0);
    size_t n_features = times_buf.shape(1);

    auto rewards_buf = rewards.unchecked<2>();
    size_t n_vertices_r = rewards_buf.shape(0);
    size_t n_features_r = rewards_buf.shape(1);

    // Validate dimensions
    if (n_features_r != n_features) {
        throw std::invalid_argument(
            "Dimension mismatch: times has " + std::to_string(n_features) +
            " features but rewards has " + std::to_string(n_features_r) + " features"
        );
    }

    if ((int)n_vertices_r != n_vertices_) {
        throw std::invalid_argument(
            "Dimension mismatch: rewards has " + std::to_string(n_vertices_r) +
            " vertices but graph has " + std::to_string(n_vertices_) + " vertices"
        );
    }

    // Copy theta to C++ vector
    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }

    // Copy times to C++ 2D vector (row-major: [time][feature])
    std::vector<std::vector<double>> times_2d(n_times, std::vector<double>(n_features));
    for (size_t i = 0; i < n_times; i++) {
        for (size_t j = 0; j < n_features; j++) {
            times_2d[i][j] = times_buf(i, j);
        }
    }

    // Copy rewards to C++ 2D vector (column-major: [feature][vertex])
    std::vector<std::vector<double>> rewards_2d(n_features, std::vector<double>(n_vertices_r));
    for (size_t j = 0; j < n_features; j++) {
        for (size_t v = 0; v < n_vertices_r; v++) {
            rewards_2d[j][v] = rewards_buf(v, j);
        }
    }

    // Step 2: Release GIL for C++ computation
    std::vector<std::vector<double>> pmf_2d(n_times, std::vector<double>(n_features));
    {
        py::gil_scoped_release release;

        // Reuse this thread's persistent graph (Stage A1). Stage 2's
        // clone-first reward_transform makes the per-feature
        // transformations safe even when ``g`` is the cached source.
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

        // === SPARSE MODE: Compute PDF per feature independently ===
        // Each feature gets its own reward transformation and independent PDF computation
        for (size_t j = 0; j < n_features; j++) {
            // Transform graph for feature j
            Graph g_transformed = g.reward_transform(rewards_2d[j]);

            // Compute PDF/PMF for all time points for feature j
            for (size_t i = 0; i < n_times; i++) {
                double time_ij = times_2d[i][j];

                // Skip NaN times - return NaN in output for missing data
                if (std::isnan(time_ij)) {
                    pmf_2d[i][j] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }

                // Sparse mode: zero observation → zero PDF
                if (time_ij == 0.0) {
                    pmf_2d[i][j] = 0.0;
                    continue;
                }

                // Compute PDF at this time point
                if (discrete) {
                    int jump_count = static_cast<int>(time_ij);
                    pmf_2d[i][j] = g_transformed.dph_pmf(jump_count);
                } else {
                    pmf_2d[i][j] = g_transformed.pdf(time_ij, granularity);
                }
            }
        }
    }
    // GIL automatically reacquired here

    // Step 3: Convert to numpy array (requires GIL, which we now have)
    py::array_t<double> result({n_times, n_features});
    auto result_buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n_times; i++) {
        for (size_t j = 0; j < n_features; j++) {
            result_buf(i, j) = pmf_2d[i][j];
        }
    }

    return result;
}

py::array_t<double> GraphBuilder::compute_accumulated_visits_converged(
    py::array_t<double> theta,
    py::array_t<int> vertex_indices,
    double tolerance,
    int max_iterations
) {
    // Step 1: Extract data from numpy arrays (requires GIL)
    auto theta_buf = theta.unchecked<1>();
    size_t theta_len = theta_buf.shape(0);
    auto indices_buf = vertex_indices.unchecked<1>();
    size_t n_indices = indices_buf.shape(0);

    // Copy to C++ vectors
    std::vector<double> theta_vec(theta_len);
    for (size_t i = 0; i < theta_len; i++) {
        theta_vec[i] = theta_buf(i);
    }
    std::vector<int> indices_vec(n_indices);
    for (size_t i = 0; i < n_indices; i++) {
        indices_vec[i] = indices_buf(i);
    }

    // Step 2: Release GIL for C++ computation
    std::vector<double> result_vec(n_indices);
    {
        py::gil_scoped_release release;

        // Reuse this thread's persistent graph (Stage A1).
        Graph& g = get_or_init_persistent_graph(theta_vec.data(), theta_len);

        // For each vertex index, iterate until convergence
        // Use accumulated_visiting_time (continuous) which handles uniformization internally
        for (size_t i = 0; i < n_indices; i++) {
            int vertex_idx = indices_vec[i];

            // Validate vertex index
            if (vertex_idx < 0 || vertex_idx >= n_vertices_) {
                throw std::invalid_argument(
                    "Vertex index " + std::to_string(vertex_idx) +
                    " out of range [0, " + std::to_string(n_vertices_) + ")"
                );
            }

            // Iterate accumulated_visiting_time until convergence
            // Use time increments (granularity=0 means auto-select)
            int time_step = 1;
            std::vector<double> visits = g.accumulated_visiting_time(static_cast<double>(time_step), 0);
            double prev = -1.0;
            double curr = visits[vertex_idx];

            while (std::abs(curr - prev) > tolerance && time_step < max_iterations) {
                prev = curr;
                time_step++;
                visits = g.accumulated_visiting_time(static_cast<double>(time_step), 0);
                curr = visits[vertex_idx];
            }

            if (time_step >= max_iterations) {
                throw std::runtime_error(
                    "Convergence not achieved for vertex " + std::to_string(vertex_idx) +
                    " after " + std::to_string(max_iterations) + " iterations. " +
                    "Current diff: " + std::to_string(std::abs(curr - prev))
                );
            }

            result_vec[i] = curr;
        }
    }
    // GIL automatically reacquired here

    // Step 3: Convert to numpy array (requires GIL, which we now have)
    py::array_t<double> result(n_indices);
    auto result_buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n_indices; i++) {
        result_buf(i) = result_vec[i];
    }

    return result;
}

} // namespace parameterized
} // namespace phasic
