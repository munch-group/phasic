#include "scc_graph.h"
#include "phasiccpp.h"
#include "../c/phasic_hash.h"
#include "../../src/c/phasic_log.h"
#include <stdexcept>
#include <algorithm>

namespace phasic {

// ============================================================================
// SCCGraph Implementation
// ============================================================================

SCCGraph::SCCGraph(struct ptd_scc_graph* scc_graph)
    : scc_graph_(scc_graph), original_graph_wrapper_(nullptr) {
    if (!scc_graph_) {
        throw std::invalid_argument("SCCGraph: null C struct pointer");
    }
}

SCCGraph::~SCCGraph() {
    if (scc_graph_) {
        ptd_scc_graph_destroy(scc_graph_);
        scc_graph_ = nullptr;
    }
}

SCCGraph::SCCGraph(SCCGraph&& other) noexcept
    : scc_graph_(other.scc_graph_),
      original_graph_wrapper_(std::move(other.original_graph_wrapper_)),
      scc_vertices_(std::move(other.scc_vertices_)) {
    other.scc_graph_ = nullptr;
}

SCCGraph& SCCGraph::operator=(SCCGraph&& other) noexcept {
    if (this != &other) {
        if (scc_graph_) {
            ptd_scc_graph_destroy(scc_graph_);
        }
        scc_graph_ = other.scc_graph_;
        original_graph_wrapper_ = std::move(other.original_graph_wrapper_);
        scc_vertices_ = std::move(other.scc_vertices_);
        other.scc_graph_ = nullptr;
    }
    return *this;
}

size_t SCCGraph::n_sccs() const {
    return scc_graph_->vertices_length;
}

const SCCVertex& SCCGraph::scc_at(size_t index) const {
    if (index >= n_sccs()) {
        throw std::out_of_range("SCCGraph::scc_at: index out of range");
    }

    // Lazy initialize cache
    if (scc_vertices_.empty()) {
        scc_vertices_.reserve(n_sccs());
        for (size_t i = 0; i < n_sccs(); ++i) {
            scc_vertices_.push_back(
                std::make_unique<SCCVertex>(scc_graph_->vertices[i], this)
            );
        }
    }

    return *scc_vertices_[index];
}

std::vector<SCCVertex> SCCGraph::sccs_in_topo_order() const {
    std::vector<SCCVertex> result;
    result.reserve(n_sccs());

    // C struct already stores vertices in topological order
    // (guaranteed by Tarjan's algorithm implementation)
    for (size_t i = 0; i < n_sccs(); ++i) {
        result.emplace_back(scc_graph_->vertices[i], this);
    }

    return result;
}

std::vector<size_t> SCCGraph::scc_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(n_sccs());

    for (size_t i = 0; i < n_sccs(); ++i) {
        sizes.push_back(scc_graph_->vertices[i]->internal_vertices_length);
    }

    return sizes;
}

const Graph& SCCGraph::original_graph() const {
    // Lazy initialize wrapper around original graph pointer
    if (!original_graph_wrapper_) {
        // Non-owning reference to original graph
        original_graph_wrapper_ = std::make_unique<Graph>(scc_graph_->graph);
    }
    return *original_graph_wrapper_;
}

std::vector<std::string> SCCGraph::scc_hashes() const {
    std::vector<std::string> hashes;
    hashes.reserve(n_sccs());

    for (size_t i = 0; i < n_sccs(); ++i) {
        hashes.push_back(scc_at(i).hash());
    }

    return hashes;
}


// ============================================================================
// SCCVertex Implementation
// ============================================================================

SCCVertex::SCCVertex(struct ptd_scc_vertex* scc_vertex, const SCCGraph* parent)
    : scc_vertex_(scc_vertex), parent_scc_graph_(parent) {
    if (!scc_vertex_) {
        throw std::invalid_argument("SCCVertex: null C struct pointer");
    }
}

size_t SCCVertex::size() const {
    return scc_vertex_->internal_vertices_length;
}

size_t SCCVertex::index() const {
    return scc_vertex_->index;
}

Graph SCCVertex::as_graph() const {
    // Create new graph with same state length as original
    // Note: parameterized flag is set automatically when adding parameterized edges
    Graph& orig_graph = const_cast<Graph&>(parent_scc_graph_->original_graph());
    Graph scc_graph(orig_graph.state_length());

    // Map old vertex pointers to vertices in new graph (by state)
    std::unordered_map<struct ptd_vertex*, std::vector<int>> vertex_state_map;

    // Step 1: Create vertices for all internal vertices in this SCC
    for (size_t i = 0; i < scc_vertex_->internal_vertices_length; ++i) {
        struct ptd_vertex* orig_vertex = scc_vertex_->internal_vertices[i];

        // Get state vector
        std::vector<int> state(orig_graph.state_length());
        for (size_t j = 0; j < orig_graph.state_length(); ++j) {
            state[j] = orig_vertex->state[j];
        }

        // Create vertex in new graph
        scc_graph.find_or_create_vertex(state);
        vertex_state_map[orig_vertex] = state;
    }

    // Step 2: Copy edges (only internal edges within this SCC)
    bool has_param_edges = false;
    size_t total_edges_copied = 0;
    size_t param_edges_copied = 0;
    for (size_t i = 0; i < scc_vertex_->internal_vertices_length; ++i) {
        struct ptd_vertex* orig_vertex = scc_vertex_->internal_vertices[i];
        std::vector<int> from_state = vertex_state_map[orig_vertex];
        Vertex from_vertex = scc_graph.find_vertex(from_state);

        // Copy edges (both regular and parameterized)
        for (size_t j = 0; j < orig_vertex->edges_length; ++j) {
            struct ptd_edge* edge = orig_vertex->edges[j];

            // Only copy if target is also in this SCC
            auto it = vertex_state_map.find(edge->to);
            if (it != vertex_state_map.end()) {
                Vertex to_vertex = scc_graph.find_vertex(it->second);
                total_edges_copied++;

                // Check if edge has parameterization
                // Note: In parameterized graphs, ALL edges have coefficients (even if values are 0)
                // So we check the graph's parameterized flag instead
                if (orig_vertex->graph->parameterized && edge->coefficients_length > 0) {
                    // Copy parameterized edge with coefficients
                    std::vector<double> coeffs(edge->coefficients,
                                              edge->coefficients + edge->coefficients_length);
                    from_vertex.add_edge_parameterized(to_vertex, edge->weight, coeffs);
                    has_param_edges = true;
                    param_edges_copied++;
                } else {
                    // Copy concrete edge
                    from_vertex.add_edge(to_vertex, edge->weight);
                }
            }
        }
    }

    // Debug output
    PTD_LOG_DEBUG("as_graph: Copied %zu edges (%zu parameterized, %zu concrete)",
                total_edges_copied, param_edges_copied, total_edges_copied - param_edges_copied);
    PTD_LOG_DEBUG("  Original graph: parameterized=%d, param_length=%zu",
                orig_graph.parameterized(), orig_graph.c_graph()->param_length);
    PTD_LOG_DEBUG("  SCC subgraph: parameterized=%d, param_length=%zu",
                scc_graph.parameterized(), scc_graph.c_graph()->param_length);

    if (total_edges_copied > 0 && param_edges_copied == 0 && orig_graph.parameterized()) {
        PTD_LOG_WARNING("as_graph: Original graph is parameterized but copied 0/%zu parameterized edges (all edges were concrete!)",
                       total_edges_copied);
        PTD_LOG_WARNING("  This indicates edges don't have coefficients_length > 0");
    }

    return scc_graph;
}

std::vector<size_t> SCCVertex::internal_vertex_indices() const {
    std::vector<size_t> indices;
    indices.reserve(size());

    for (size_t i = 0; i < scc_vertex_->internal_vertices_length; ++i) {
        struct ptd_vertex* vertex = scc_vertex_->internal_vertices[i];
        indices.push_back(vertex->index);
    }

    return indices;
}

std::string SCCVertex::hash() const {
    // Extract this SCC as a graph and hash it
    Graph scc_subgraph = as_graph();

    // Use existing graph content hash function
    struct ptd_hash_result* hash_result = ptd_graph_content_hash(scc_subgraph.c_graph());
    if (!hash_result) {
        throw std::runtime_error("SCCVertex::hash: failed to compute hash");
    }

    std::string hash_hex(hash_result->hash_hex);
    ptd_hash_destroy(hash_result);

    return hash_hex;
}

std::vector<size_t> SCCVertex::outgoing_scc_edges() const {
    std::vector<size_t> targets;
    targets.reserve(scc_vertex_->edges_length);

    for (size_t i = 0; i < scc_vertex_->edges_length; ++i) {
        struct ptd_scc_edge* edge = scc_vertex_->edges[i];
        targets.push_back(edge->to->index);
    }

    return targets;
}

} // namespace phasic
