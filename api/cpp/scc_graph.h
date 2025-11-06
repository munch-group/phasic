#pragma once

#include "../../api/c/phasic.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace phasic {

class Graph;
class SCCVertex;

/**
 * @brief Strongly Connected Component (SCC) decomposition of a graph
 *
 * Represents the condensation/quotient graph where each vertex is an SCC.
 * The graph is always a DAG (acyclic) by construction.
 *
 * Example:
 * @code
 * Graph g(5);
 * // ... build graph with cycles ...
 *
 * SCCGraph scc_graph = g.scc_decomposition();
 * std::cout << "Found " << scc_graph.n_sccs() << " SCCs" << std::endl;
 *
 * for (const auto& scc : scc_graph.sccs_in_topo_order()) {
 *     Graph scc_subgraph = scc.as_graph();
 *     // Process each SCC as separate graph
 * }
 * @endcode
 */
class SCCGraph {
public:
    /**
     * @brief Construct from C ptd_scc_graph (takes ownership)
     * @param scc_graph C struct pointer (will be freed on destruction)
     */
    explicit SCCGraph(struct ptd_scc_graph* scc_graph);

    /**
     * @brief Destructor - frees underlying C struct
     */
    ~SCCGraph();

    // Disable copy (owns C pointer)
    SCCGraph(const SCCGraph&) = delete;
    SCCGraph& operator=(const SCCGraph&) = delete;

    // Enable move
    SCCGraph(SCCGraph&& other) noexcept;
    SCCGraph& operator=(SCCGraph&& other) noexcept;

    /**
     * @brief Number of strongly connected components
     */
    size_t n_sccs() const;

    /**
     * @brief Get SCC vertex by index
     * @param index SCC index (0 to n_sccs()-1)
     * @return Reference to SCCVertex (valid while SCCGraph exists)
     */
    const SCCVertex& scc_at(size_t index) const;

    /**
     * @brief Get all SCCs in topological order
     *
     * Topological ordering ensures dependencies are processed first.
     * Essential for correct trace stitching.
     *
     * @return Vector of SCC vertices in topological order
     */
    std::vector<SCCVertex> sccs_in_topo_order() const;

    /**
     * @brief Get sizes of all SCCs
     * @return Vector of vertex counts per SCC
     */
    std::vector<size_t> scc_sizes() const;

    /**
     * @brief Get reference to original graph
     *
     * The original graph that was decomposed.
     * @return Non-owning reference to original graph
     */
    const Graph& original_graph() const;

    /**
     * @brief Compute content hash for each SCC
     *
     * Hashes the subgraph formed by each SCC's internal vertices.
     * Used for cache lookups.
     *
     * @return Vector of hex hash strings (one per SCC)
     */
    std::vector<std::string> scc_hashes() const;

    /**
     * @brief Access underlying C struct (for C API interop)
     */
    struct ptd_scc_graph* c_ptr() { return scc_graph_; }
    const struct ptd_scc_graph* c_ptr() const { return scc_graph_; }

private:
    struct ptd_scc_graph* scc_graph_;
    mutable std::unique_ptr<Graph> original_graph_wrapper_;  // Lazy init
    mutable std::vector<std::unique_ptr<SCCVertex>> scc_vertices_;  // Cache
};


/**
 * @brief Single strongly connected component vertex
 *
 * Represents one SCC in the condensation graph.
 * Contains multiple vertices from the original graph.
 */
class SCCVertex {
public:
    /**
     * @brief Construct from C ptd_scc_vertex (non-owning reference)
     * @param scc_vertex C struct pointer (not owned)
     * @param parent_scc_graph Parent SCCGraph (for original graph access)
     */
    SCCVertex(struct ptd_scc_vertex* scc_vertex, const SCCGraph* parent);

    /**
     * @brief Number of vertices in this SCC
     */
    size_t size() const;

    /**
     * @brief Index of this SCC in parent graph
     */
    size_t index() const;

    /**
     * @brief Extract this SCC as a standalone Graph object
     *
     * Creates a new graph containing only the vertices in this SCC.
     * Preserves internal edges. External edges to other SCCs are excluded.
     *
     * @return New graph representing this SCC
     */
    Graph as_graph() const;

    /**
     * @brief Get indices of internal vertices in original graph
     *
     * @return Vector of vertex indices
     */
    std::vector<size_t> internal_vertex_indices() const;

    /**
     * @brief Compute content hash of this SCC subgraph
     *
     * Hashes the structure formed by internal vertices.
     * Used for cache lookups.
     *
     * @return Hex hash string (SHA-256)
     */
    std::string hash() const;

    /**
     * @brief Get edges to other SCCs (outgoing edges)
     *
     * @return Vector of target SCC indices
     */
    std::vector<size_t> outgoing_scc_edges() const;

    /**
     * @brief Access underlying C struct
     */
    struct ptd_scc_vertex* c_ptr() { return scc_vertex_; }
    const struct ptd_scc_vertex* c_ptr() const { return scc_vertex_; }

private:
    struct ptd_scc_vertex* scc_vertex_;  // Non-owning
    const SCCGraph* parent_scc_graph_;   // Non-owning
};

} // namespace phasic
