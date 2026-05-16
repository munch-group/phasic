/*
 * MIT License
 *
 * Copyright (c) 2021 Tobias Røikjer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef PTDALGORITHMS_PTDCPP_H
#define PTDALGORITHMS_PTDCPP_H

#include <cstring>
#include <errno.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iterator>
#include <functional>

#include "../c/phasic.h"
#include "scc_graph.h"

// Throw std::runtime_error with the contents of the global ptd_err buffer
// and clear it so a later post-call ptd_err[0] check (e.g. in the pybind
// add_edge dispatcher) does not see a stale message from an earlier failure.
#define PTD_THROW_AND_CLEAR() \
    do { \
        std::string _ptd_msg((const char*)ptd_err); \
        ptd_err[0] = '\0'; \
        throw std::runtime_error(_ptd_msg); \
    } while (0)

namespace phasic {
    struct rf_graph {
        struct ptd_avl_tree *tree;
        struct ptd_graph *graph;
        size_t *references;
        struct ptd_dph_probability_distribution_context *dph_context;
        struct ptd_probability_distribution_context *ph_context;
        struct ptd_dph_probability_distribution_context *dph_context_markov;
        struct ptd_probability_distribution_context *ph_context_markov;
        int64_t granularity;
        int64_t granularity_markov;
        // When true, the destructor will not free `graph` or `tree`. Used by
        // SCCGraph::original_graph(): we need to expose the underlying
        // ptd_graph through a Graph wrapper without claiming ownership of it
        // (the original Graph still owns and will destroy it). All standard
        // constructors leave this false; only Graph::make_borrowed() flips it.
        bool borrowed;
    };


    class Vertex;

    struct Edge;

    struct ParameterizedEdge;

    class PhaseTypeDistribution;

    class Graph;

    class SccGraph;

    class Graph {
    public:
        Graph(struct ptd_graph *graph) {
            this->rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));
            this->rf_graph->references = (size_t *) malloc(sizeof(*this->rf_graph->references));
            *this->rf_graph->references = 1;
            this->rf_graph->graph = graph;
            this->rf_graph->tree = ptd_avl_tree_create(this->rf_graph->graph->state_length);
            this->rf_graph->dph_context = NULL;
            this->rf_graph->ph_context = NULL;
            this->rf_graph->dph_context_markov = NULL;
            this->rf_graph->ph_context_markov = NULL;
            this->rf_graph->borrowed = false;

            if (this->rf_graph->tree == NULL) {
                throw std::runtime_error("Failed to create ptd_avl_tree\n");
            }
        }

        Graph(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree) {
            this->rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));
            this->rf_graph->references = (size_t *) malloc(sizeof(*this->rf_graph->references));
            *this->rf_graph->references = 1;
            this->rf_graph->graph = graph;
            this->rf_graph->tree = avl_tree;
            this->rf_graph->dph_context = NULL;
            this->rf_graph->ph_context = NULL;
            this->rf_graph->dph_context_markov = NULL;
            this->rf_graph->ph_context_markov = NULL;
            this->rf_graph->borrowed = false;
        }

        // Wrap an existing ptd_graph WITHOUT taking ownership: the destructor
        // will not free the graph or create/free an AVL tree. Use only when
        // another Graph already owns the underlying ptd_graph and you need a
        // const view. Callers must ensure the owner outlives this borrowed
        // wrapper (pybind's reference_internal handles this for us at the
        // Python layer).
        static Graph make_borrowed(struct ptd_graph *graph) {
            // Build via the standard ptd_graph-only constructor, then mark
            // borrowed and discard the AVL tree it allocated. Two extra
            // small allocations, but it sidesteps having to expose another
            // private constructor.
            Graph g(graph);
            ptd_avl_tree_destroy(g.rf_graph->tree);
            g.rf_graph->tree = NULL;
            g.rf_graph->borrowed = true;
            return g;
        }

        Graph(const Graph &o) {
            this->rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));
            this->rf_graph->references = o.rf_graph->references;
            this->rf_graph->graph = o.rf_graph->graph;
            this->rf_graph->tree = o.rf_graph->tree;
            this->rf_graph->dph_context = o.rf_graph->dph_context;
            this->rf_graph->ph_context = o.rf_graph->ph_context;
            this->rf_graph->dph_context_markov = o.rf_graph->dph_context_markov;
            this->rf_graph->ph_context_markov = o.rf_graph->ph_context_markov;
            this->rf_graph->granularity = o.rf_graph->granularity;
            this->rf_graph->granularity_markov = o.rf_graph->granularity_markov;
            this->rf_graph->borrowed = o.rf_graph->borrowed;
            *(this->rf_graph->references) += 1;
        }

        // Move constructor - transfers ownership without sharing references
        // This enables clone() to return by value without triggering copy semantics
        Graph(Graph &&o) noexcept {
            this->rf_graph = o.rf_graph;
            o.rf_graph = nullptr;
        }

        Graph(size_t state_length) {
            this->rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));
            this->rf_graph->references = (size_t *) malloc(sizeof(*this->rf_graph->references));
            *this->rf_graph->references = 1;
            this->rf_graph->graph = ptd_graph_create(state_length);

            if (this->rf_graph->graph == NULL) {
                throw std::runtime_error("Failed to create ptd_graph\n");
            }

            this->rf_graph->tree = ptd_avl_tree_create(this->rf_graph->graph->state_length);

            if (this->rf_graph->tree == NULL) {
                throw std::runtime_error("Failed to create ptd_avl_tree\n");
            }

            this->rf_graph->dph_context = NULL;
            this->rf_graph->ph_context = NULL;
            this->rf_graph->dph_context_markov = NULL;
            this->rf_graph->ph_context_markov = NULL;
            this->rf_graph->borrowed = false;
        }


        ~Graph() {
            // Handle moved-from objects (rf_graph is nullptr after move)
            if (this->rf_graph == nullptr) {
                return;
            }

            *(this->rf_graph->references) -= 1;

            if (*this->rf_graph->references == 0) {
                // Last reference - destroy shared resources. The
                // probability-distribution contexts are always owned (they
                // are constructed lazily by methods on this wrapper), so
                // free them unconditionally. The graph and tree are only
                // freed when this wrapper is the owner; borrowed wrappers
                // skip those to avoid double-frees against the real owner.
                ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context);
                ptd_probability_distribution_context_destroy(this->rf_graph->ph_context);
                ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context_markov);
                ptd_probability_distribution_context_destroy(this->rf_graph->ph_context_markov);
                if (!this->rf_graph->borrowed) {
                    ptd_avl_tree_destroy(this->rf_graph->tree);
                    ptd_graph_destroy(this->rf_graph->graph);
                }
                free(this->rf_graph->references);
            }

            // Always free this instance's rf_graph struct (each copy allocates its own)
            free(this->rf_graph);
        }



//         struct Iterator 
//         {
//           using iterator_category = std::forward_iterator_tag;
//           using difference_type   = std::ptrdiff_t;
//           using value_type        = Vertex;
//           using pointer           = Vertex*;  // or also value_type*
//           using reference         = Vertex&;  // or also value_type&

//             Iterator(pointer ptr) : vertex_ptr(ptr) {}

//             reference operator*() const { return *vertex_ptr; }
//             pointer operator->() { return vertex_ptr; }
//             // const reference operator*() const { return *m_ptr; }
//             // const pointer operator->() { return m_ptr; }
        
//             // Prefix increment
//             Iterator& operator++() { 

//                 // std::vector<Vertex> vertices;

//                 // *this.vertex->index + 1

//                 &this->c_graph()->vertex_at((&vertex_ptr.index) + 1);

//                 // vertex_ptr++; 
//                 return *this; 
            
//             }  
        
//             // Postfix increment
//             Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
        
//             friend bool operator== (const Iterator& a, const Iterator& b) { return a.vertex_ptr == b.vertex_ptr; };
//             friend bool operator!= (const Iterator& a, const Iterator& b) { return a.vertex_ptr != b.vertex_ptr; };     
        
//         private:
        
//             pointer vertex_ptr;
//         };
    
//         Iterator begin() { return Iterator(&this->vertex_at(0)); }
//         Iterator end() { return Iterator(&this->vertex_at(this->c_graph()->vertices_length)); }

// //        Iterator end()   { return Iterator(&m_data[200]); } // 200 is out of bounds
//         // ConstantIterator cbegin() const { return ConstantIterator(); }
//         // ConstantIterator cend()   const { return ConstantIterator(&m_data[200]); }




        void set_param_length(size_t param_length) {
            ptd_graph_set_param_length(this->c_graph(), param_length);
            if (ptd_err[0] != '\0') {
                std::string error_msg((const char*)ptd_err);
                ptd_err[0] = '\0';  // Clear error buffer to prevent stale errors
                throw std::runtime_error(error_msg);
            }
        }

        void update_weights_parameterized(std::vector<double> scalars, bool use_log = false);

        // Callback-based weight update
        void update_weights_parameterized(
            std::vector<double> scalars,
            std::function<double(const std::vector<double>&, const std::vector<double>&)> callback
        );

        // Set the initial probability vector (IPV) after graph construction.
        // ipv must have length equal to starting_vertex().edges().size().
        // Symmetric to update_weights_parameterized but applies only to
        // starting-vertex edges (skipped by update_weights). Symbolic compute
        // graph cache survives this call (Stage A0 invariant).
        void update_ipv(std::vector<double> ipv);

        std::vector<double> expected_waiting_time(std::vector<double> rewards = std::vector<double>()) {
            double *ptr = ptd_expected_waiting_time(
                    this->c_graph(),
                    rewards.empty() ? NULL : &rewards[0]
            );

            if (ptr == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            std::vector<double> res;
            res.assign(ptr, ptr + this->c_graph()->vertices_length);
            free(ptr);

            return res;
        }

        /**
         * Compute expected sojourn time for all states or a subset
         *
         * Returns array where result[i] = expected time spent in state i
         * before absorption. Much faster than calling expected_waiting_time()
         * with unit reward vectors for each state.
         *
         * @param indices Optional vector of vertex indices to compute sojourn times for.
         *                If empty (default), computes for all vertices.
         * @return Vector of sojourn times for specified states
         * @throws std::runtime_error if computation fails
         */
        std::vector<double> expected_sojourn_time(
            const std::vector<size_t>& indices = std::vector<size_t>()
        ) {
            if (indices.empty()) {
                // Compute for all vertices
                double *ptr = ptd_expected_sojourn_time(this->c_graph());

                if (ptr == NULL) {
                    PTD_THROW_AND_CLEAR();
                }

                std::vector<double> res;
                res.assign(ptr, ptr + this->c_graph()->vertices_length);
                free(ptr);

                return res;
            } else {
                // Compute for subset
                double *ptr = ptd_expected_sojourn_time_subset(
                    this->c_graph(), indices.data(), indices.size()
                );

                if (ptr == NULL) {
                    PTD_THROW_AND_CLEAR();
                }

                std::vector<double> res;
                res.assign(ptr, ptr + indices.size());
                free(ptr);

                return res;
            }
        }

        // std::vector<double> expected_residence_time(std::vector<double> rewards = std::vector<double>()) {
        //     double *ptr = ptd_expected_residence_time(
        //             this->c_graph(),
        //             rewards.empty() ? NULL : &rewards[0]
        //     );

        //     if (ptr == NULL) {
        //         PTD_THROW_AND_CLEAR();
        //     }

        //     std::vector<double> res;
        //     res.assign(ptr, ptr + this->c_graph()->vertices_length);
        //     free(ptr);

        //     return res;
        // }

        Vertex create_vertex(std::vector<int> state = std::vector<int>());

        Vertex create_vertex(const int *state);

        Vertex *create_vertex_p(std::vector<int> state = std::vector<int>());

        Vertex *create_vertex_p(const int *state);

        Vertex find_vertex(std::vector<int> state);

        Vertex find_vertex(const int *state);

        Vertex *find_vertex_p(std::vector<int> state);

        Vertex *find_vertex_p(const int *state);

        bool vertex_exists(std::vector<int> state);

        bool vertex_exists(const int *state);

        Vertex find_or_create_vertex(std::vector<int> state);

        Vertex find_or_create_vertex(const int *state);

        Vertex *find_or_create_vertex_p(std::vector<int> state);

        Vertex *find_or_create_vertex_p(const int *state);

        Vertex starting_vertex();

        Vertex *starting_vertex_p();

        std::vector<Vertex> vertices();

        std::vector<Vertex *> vertices_p();

        Vertex vertex_at(size_t index);

        Vertex *vertex_at_p(size_t index);

        size_t vertices_length();

        size_t edges_length();

        bool parameterized();

        long double random_sample(std::vector<double> rewards = std::vector<double>()) {
            return ptd_random_sample(c_graph(), &rewards[0]);
        }

        std::vector<long double> mph_random_sample(std::vector<double> rewards, size_t vertex_rewards_length) {
            std::vector<long double> res(vertex_rewards_length);
            long double *c_res = ptd_mph_random_sample(c_graph(), &rewards[0], vertex_rewards_length);

            for (size_t i = 0; i < vertex_rewards_length; ++i) {
                res[i] = c_res[i];
            }

            free(c_res);

            return res;
        }

        long double dph_random_sample_c(double *rewards) {
            long double res = ptd_dph_random_sample(c_graph(), rewards);

            if (std::isnan(res)) {
                PTD_THROW_AND_CLEAR();
            }

            return res;
        }

        long double dph_random_sample(std::vector<double> rewards = std::vector<double>()) {
            if (rewards.empty()) {
                return this->dph_random_sample_c(NULL);
            } else {
                return this->dph_random_sample_c(&rewards[0]);
            }
        }

        long double *mdph_random_sample_c(double *rewards, size_t vertex_rewards_length) {
            long double *res = ptd_mdph_random_sample(c_graph(), rewards, vertex_rewards_length);

            if (res == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return res;
        }

        std::vector<long double> mdph_random_sample(std::vector<double> rewards, size_t vertex_rewards_length) {
            std::vector<long double> res(vertex_rewards_length);
            long double *c_res;

            if (rewards.empty()) {
                c_res = mdph_random_sample_c(NULL, vertex_rewards_length);
            } else {
                c_res = mdph_random_sample_c(&rewards[0], vertex_rewards_length);
            }

            for (size_t i = 0; i < vertex_rewards_length; ++i) {
                res[i] = c_res[i];
            }

            free(c_res);

            return res;
        }

        std::vector<long double> mdph_random_sample_c(std::vector<double> rewards, size_t vertex_rewards_length) {
            std::vector<long double> res(vertex_rewards_length);
            long double *c_res = ptd_mdph_random_sample(c_graph(), &rewards[0], vertex_rewards_length);

            for (size_t i = 0; i < vertex_rewards_length; ++i) {
                res[i] = c_res[i];
            }

            free(c_res);

            return res;
        }

        std::pair<std::vector<size_t>, std::vector<double>> random_sample_path() {
            struct ptd_sample_path *path = ptd_random_sample_path(c_graph());
            std::vector<size_t> indices(path->vertex_indices, path->vertex_indices + path->length);
            std::vector<double> times(path->entry_times, path->entry_times + path->length);
            ptd_sample_path_destroy(path);
            return {indices, times};
        }

        std::vector<double> backward_probabilities(std::vector<size_t> target_vertices) {
            double *h = ptd_backward_probabilities(
                c_graph(), &target_vertices[0], target_vertices.size()
            );
            std::vector<double> result(h, h + c_graph()->vertices_length);
            free(h);
            return result;
        }

        std::pair<std::vector<size_t>, std::vector<double>>
        random_sample_path_conditioned(std::vector<double> backward_probs) {
            struct ptd_sample_path *path = ptd_random_sample_path_conditioned(
                c_graph(), &backward_probs[0]
            );
            std::vector<size_t> indices(path->vertex_indices, path->vertex_indices + path->length);
            std::vector<double> times(path->entry_times, path->entry_times + path->length);
            ptd_sample_path_destroy(path);
            return {indices, times};
        }

        size_t random_sample_stop_vertex(double time) {
            struct ptd_vertex *res = ptd_random_sample_stop_vertex(c_graph(), time);

            return res->index;
        }

        size_t dph_random_sample_stop_vertex(int jumps) {
            struct ptd_vertex *res = ptd_dph_random_sample_stop_vertex(c_graph(), jumps);

            if (res == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return res->index;
        }

        size_t state_length() const {
            return c_graph()->state_length;
        }

        PhaseTypeDistribution phase_type_distribution();

        bool is_acyclic() {
            return ptd_graph_is_acyclic(c_graph());
        }

        void validate() {
            if (ptd_validate_graph(c_graph())) {
                PTD_THROW_AND_CLEAR();
            }
        }

        /**
         * @brief Compute strongly connected component decomposition
         *
         * Decomposes this graph into SCCs (strongly connected components).
         * Returns a condensation graph where each vertex represents an SCC.
         *
         * @return SCCGraph object (always a DAG)
         *
         * @example
         * @code
         * Graph g(5);
         * // ... build graph ...
         *
         * SCCGraph scc = g.scc_decomposition();
         * for (const auto& component : scc.sccs_in_topo_order()) {
         *     std::cout << "SCC with " << component.size() << " vertices\n";
         * }
         * @endcode
         */
        SCCGraph scc_decomposition() {
            // Access rf_graph->graph directly to avoid const overload resolution issue
            struct ptd_scc_graph* scc_c = ptd_find_strongly_connected_components(rf_graph->graph);
            if (!scc_c) {
                throw std::runtime_error("Graph::scc_decomposition: failed to compute SCC");
            }
            return SCCGraph(scc_c);
        }

        // Graph expectation_dag(std::vector<double> rewards = std::vector<double>());
        // // Graph expectation_dag(std::vector<double> rewards);
        // // Graph expectation_dag(std::vector<double> rewards) {
        // //     double *rewards_ptr = &rewards[0];
        // //     struct ptd_clone_res r = ptd_graph_expectation_dag(c_graph(), rewards_ptr);

        // //     return Graph(r.graph, r.avl_tree);
        // // }

        // Graph *expectation_dag_p(std::vector<double> rewards = std::vector<double>());
        // // Graph *expectation_dag_p(std::vector<double> rewards) {
        // //     double *rewards_ptr = &rewards[0];
        // //     struct ptd_clone_res r = ptd_graph_expectation_dag(c_graph(), rewards_ptr);

        // //     return new Graph(r.graph, r.avl_tree);
        // // }

        Graph reward_transform(std::vector<double> rewards);

        Graph *reward_transform_p(std::vector<double> rewards);

        Graph dph_reward_transform(std::vector<int> rewards) {
            struct ptd_graph *res = ptd_graph_dph_reward_transform(c_graph(), &rewards[0]);

            if (res == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return Graph(res);
        }

        Graph *dph_reward_transform_p(std::vector<int> rewards) {
            struct ptd_graph *res = ptd_graph_dph_reward_transform(c_graph(), &rewards[0]);

            if (res == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return new Graph(res);
        }

        std::vector<double> normalize() {
            double *rewards = ptd_normalize_graph(this->c_graph());

            std::vector<double> res(this->c_graph()->vertices_length);

            for (size_t i = 0; i < res.size(); ++i) {
                res[i] = rewards[i];
            }

            free(rewards);

            notify_change();

            return res;
        }

        std::vector<double> dph_normalize() {
            double *rewards = ptd_dph_normalize_graph(this->c_graph());

            std::vector<double> res(this->c_graph()->vertices_length);

            for (size_t i = 0; i < res.size(); ++i) {
                res[i] = rewards[i];
            }

            free(rewards);

            notify_change();

            return res;
        }

        void notify_change() {
            ptd_notify_change(c_graph());
            ptd_probability_distribution_context_destroy(this->rf_graph->ph_context);
            this->rf_graph->ph_context = NULL;
            ptd_probability_distribution_context_destroy(this->rf_graph->ph_context_markov);
            this->rf_graph->ph_context_markov = NULL;
            ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context);
            this->rf_graph->dph_context = NULL;
            ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context_markov);
            this->rf_graph->dph_context_markov = NULL;
        }

        double defect() {
            return ptd_defect(c_graph());
        }

        Graph clone() {
            struct ptd_clone_res r = ptd_clone_graph(c_graph(), c_avl_tree());

            if (r.graph == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return Graph(r.graph, r.avl_tree);
        }


        Graph *clone_p() {
            struct ptd_clone_res r = ptd_clone_graph(c_graph(), c_avl_tree());

            if (r.graph == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return new Graph(r.graph, r.avl_tree);
        }

        double pdf(double time, int64_t granularity = 0) {
            if (this->rf_graph->ph_context == NULL
                || this->rf_graph->granularity != granularity
                || this->rf_graph->ph_context->weight_version_at_creation != c_graph()->weight_version) {
                if (this->rf_graph->ph_context != NULL) {
                    ptd_probability_distribution_context_destroy(this->rf_graph->ph_context);
                }
                this->rf_graph->ph_context = ptd_probability_distribution_context_create(c_graph(), granularity);

                if (this->rf_graph->ph_context == NULL) {
                    PTD_THROW_AND_CLEAR();
                }

                _pdf.clear();
                _cdf.clear();
                _pdf.push_back(this->rf_graph->ph_context->pdf);
                _cdf.push_back(this->rf_graph->ph_context->cdf);
                this->rf_graph->granularity = granularity;
            }

            // Step-count cap: the forward algorithm appends to _pdf/_cdf once per
            // discretization step and would take ~granularity * time steps to reach
            // `time`. Cap at 1e9 (~16 GB of vector storage) so an over-resolved
            // auto-granularity on a graph with extreme rates fails fast instead of
            // exhausting memory. Uses ph_context->granularity (post C-side auto +
            // floor), so the check covers both explicit and granularity=0 paths.
            const double resolved_g = (double) this->rf_graph->ph_context->granularity;
            const double est_steps = resolved_g * time;
            if (est_steps > 1.0e9) {
                char msg[512];
                snprintf(msg, sizeof(msg),
                    "granularity * time = %.3e would require too many "
                    "forward-algorithm steps (cap: 1e9). granularity=%lld, "
                    "time=%g. Reduce granularity, rescale your time/rate units, "
                    "or check that no edge has an outsized rate.",
                    est_steps, (long long) this->rf_graph->ph_context->granularity, time);
                throw std::invalid_argument(msg);
            }

            while (time >= this->rf_graph->ph_context->time) {
                ptd_probability_distribution_step(
                        this->rf_graph->ph_context
                );
                _pdf.push_back(this->rf_graph->ph_context->pdf);
                _cdf.push_back(this->rf_graph->ph_context->cdf);
            }

            return _pdf[this->rf_graph->ph_context->granularity * time];
        }

        double cdf(double time, int64_t granularity = 0) {
            pdf(time, granularity);

            return _cdf[this->rf_graph->ph_context->granularity * time];
        }
        
        double dph_pmf(int jumps) {
            if (this->rf_graph->dph_context == NULL
                || this->rf_graph->dph_context->weight_version_at_creation != c_graph()->weight_version) {
                if (this->rf_graph->dph_context != NULL) {
                    ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context);
                }
                this->rf_graph->dph_context = ptd_dph_probability_distribution_context_create(c_graph());

                if (this->rf_graph->dph_context == NULL) {
                    PTD_THROW_AND_CLEAR();
                }
                _dph_pmf.clear();
                _dph_cdf.clear();
                _dph_pmf.push_back(this->rf_graph->dph_context->pmf);
                _dph_cdf.push_back(this->rf_graph->dph_context->cdf);
            }

            if (jumps > this->rf_graph->dph_context->jumps) {
                for (int i = this->rf_graph->dph_context->jumps; i < jumps; ++i) {
                    ptd_dph_probability_distribution_step(
                            this->rf_graph->dph_context
                    );
                    _dph_pmf.push_back(this->rf_graph->dph_context->pmf);
                    _dph_cdf.push_back(this->rf_graph->dph_context->cdf);
                }
            }

            return _dph_pmf[jumps];
        }

        double dph_cdf(int jumps) {
            dph_pmf(jumps);

            return _dph_cdf[jumps];
        }

        Graph laplace_transform(double theta) {
            struct ptd_clone_res r = ptd_graph_laplace_transform(
                this->c_graph(),
                this->c_avl_tree(),
                theta
            );

            if (r.graph == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return Graph(r.graph, r.avl_tree);
        }

        std::vector<double> stop_probability(double time, int64_t granularity = 0) {
            if (this->rf_graph->ph_context_markov == NULL
                || this->rf_graph->granularity_markov != granularity
                || this->rf_graph->ph_context_markov->time -
                   ((double) 1.0) / this->rf_graph->ph_context_markov->granularity >
                   time
                || this->rf_graph->ph_context_markov->weight_version_at_creation != c_graph()->weight_version) {
                if (this->rf_graph->ph_context_markov != NULL) {
                    ptd_probability_distribution_context_destroy(this->rf_graph->ph_context_markov);
                }
                this->rf_graph->ph_context_markov = ptd_probability_distribution_context_create(c_graph(), granularity);

                if (this->rf_graph->ph_context_markov == NULL) {
                    PTD_THROW_AND_CLEAR();
                }

                this->rf_graph->granularity_markov = granularity;
            }

            while (time > this->rf_graph->ph_context_markov->time) {
                ptd_probability_distribution_step(
                        this->rf_graph->ph_context_markov
                );
            }

            std::vector<double> ret(this->rf_graph->ph_context_markov->graph->vertices_length);

            for (size_t i = 0; i < this->rf_graph->ph_context_markov->graph->vertices_length; ++i) {
                ret[i] = (double) this->rf_graph->ph_context_markov->probability_at[i];
            }

            return ret;
        }

        std::vector<double> accumulated_visiting_time(double time, int64_t granularity = 0) {
            if (this->rf_graph->ph_context_markov == NULL
                || this->rf_graph->granularity_markov != granularity
                || this->rf_graph->ph_context_markov->time -
                   ((double) 1.0) / this->rf_graph->ph_context_markov->granularity >
                   time
                || this->rf_graph->ph_context_markov->weight_version_at_creation != c_graph()->weight_version) {
                if (this->rf_graph->ph_context_markov != NULL) {
                    ptd_probability_distribution_context_destroy(this->rf_graph->ph_context_markov);
                }
                this->rf_graph->ph_context_markov = ptd_probability_distribution_context_create(c_graph(), granularity);

                if (this->rf_graph->ph_context_markov == NULL) {
                    PTD_THROW_AND_CLEAR();
                }

                this->rf_graph->granularity_markov = granularity;
            }

            while (time > this->rf_graph->ph_context_markov->time) {
                ptd_probability_distribution_step(
                        this->rf_graph->ph_context_markov
                );
            }

            std::vector<double> ret(this->rf_graph->ph_context_markov->graph->vertices_length);

            for (size_t i = 0; i < this->rf_graph->ph_context_markov->graph->vertices_length; ++i) {
                ret[i] = (double) this->rf_graph->ph_context_markov->accumulated_visits[i];
                ret[i] /= this->rf_graph->ph_context_markov->granularity;
            }

            return ret;
        }

        std::vector<double> dph_stop_probability(int jumps) {
            if (this->rf_graph->dph_context_markov == NULL
                || this->rf_graph->dph_context_markov->jumps > jumps
                || this->rf_graph->dph_context_markov->weight_version_at_creation != c_graph()->weight_version) {
                if (this->rf_graph->dph_context_markov != NULL) {
                    ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context_markov);
                }
                this->rf_graph->dph_context_markov = ptd_dph_probability_distribution_context_create(c_graph());

                if (this->rf_graph->dph_context_markov == NULL) {
                    PTD_THROW_AND_CLEAR();
                }
            }

            while (jumps > this->rf_graph->dph_context_markov->jumps) {
                ptd_dph_probability_distribution_step(
                        this->rf_graph->dph_context_markov
                );
            }

            std::vector<double> ret(this->rf_graph->dph_context_markov->graph->vertices_length);

            for (size_t i = 0; i < this->rf_graph->dph_context_markov->graph->vertices_length; ++i) {
                ret[i] = (double) this->rf_graph->dph_context_markov->probability_at[i];
            }

            return ret;
        }

        std::vector<double> dph_accumulated_visits(int jumps) {
            if (this->rf_graph->dph_context_markov == NULL
                || this->rf_graph->dph_context_markov->jumps > jumps) {
                if (this->rf_graph->dph_context_markov != NULL) {
                    ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context_markov);
                }
                this->rf_graph->dph_context_markov = ptd_dph_probability_distribution_context_create(c_graph());

                if (this->rf_graph->dph_context_markov == NULL) {
                    PTD_THROW_AND_CLEAR();
                }
            }

            while (jumps > this->rf_graph->dph_context_markov->jumps) {
                ptd_dph_probability_distribution_step(
                        this->rf_graph->dph_context_markov
                );
            }

            std::vector<double> ret(this->rf_graph->dph_context_markov->graph->vertices_length);

            for (size_t i = 0; i < this->rf_graph->dph_context_markov->graph->vertices_length; ++i) {
                ret[i] = (double) this->rf_graph->dph_context_markov->accumulated_visits[i];
            }

            return ret;
        }

        // std::vector<double> dph_expected_visits(int jumps) {
        //     if (this->rf_graph->dph_context_markov == NULL
        //         || this->rf_graph->dph_context_markov->jumps > jumps) {
        //         if (this->rf_graph->dph_context_markov != NULL) {
        //             ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context_markov);
        //         }
        //         this->rf_graph->dph_context_markov = ptd_dph_probability_distribution_context_create(c_graph());

        //         if (this->rf_graph->dph_context_markov == NULL) {
        //             PTD_THROW_AND_CLEAR();
        //         }
        //     }

        //     while (jumps > this->rf_graph->dph_context_markov->jumps) {
        //         ptd_dph_probability_distribution_step(
        //                 this->rf_graph->dph_context_markov
        //         );
        //     }

        //     std::vector<double> ret(this->rf_graph->dph_context_markov->graph->vertices_length);

        //     for (size_t i = 0; i < this->rf_graph->dph_context_markov->graph->vertices_length; ++i) {
        //         ret[i] = (double) this->rf_graph->dph_context_markov->accumulated_visits[i];
        //     }

        //     return ret;
        // }

    public:
        Graph &operator=(const Graph &o) {
            if (this == &o) {
                return *this;
            }

            (*this).rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));

            *this->rf_graph->references -= 1;

            if (*this->rf_graph->references == 0) {
                ptd_avl_tree_destroy(this->rf_graph->tree);
                ptd_graph_destroy(this->rf_graph->graph);
                free(this->rf_graph->references);
                ptd_probability_distribution_context_destroy(this->rf_graph->ph_context);
                ptd_dph_probability_distribution_context_destroy(this->rf_graph->dph_context);
                this->rf_graph->dph_context = NULL;
            }

            free(this->rf_graph);

            //this->rf_graph = o.rf_graph;
            this->rf_graph = (struct rf_graph *) malloc(sizeof(*this->rf_graph));
            this->rf_graph->references = o.rf_graph->references;
            this->rf_graph->graph = o.rf_graph->graph;
            this->rf_graph->tree = o.rf_graph->tree;
            this->rf_graph->ph_context = o.rf_graph->ph_context;
            this->rf_graph->dph_context = o.rf_graph->dph_context;
            *(this->rf_graph->references) += 1;

            return *this;
        }

        struct ptd_graph *c_graph() {
            return rf_graph->graph;
        }

        const struct ptd_graph *c_graph() const {
            return rf_graph->graph;
        }

        struct ptd_avl_tree *c_avl_tree() {
            return rf_graph->tree;
        }

    private:
        Graph(struct rf_graph *rf_graph) {
            this->rf_graph = rf_graph;
            *(rf_graph->references) += 1;  // Fixed: dereference to increment count, not pointer
        }

        struct rf_graph *rf_graph;

        std::vector<double> _pdf;
        std::vector<double> _cdf;
        std::vector<double> _dph_pmf;
        std::vector<double> _dph_cdf;

        friend class VertexLinkedList;

        friend class Vertex;
    };

    class Vertex {
    private:
        Vertex(Graph &graph, int *state) : graph(graph) {
            this->vertex = ptd_vertex_create_state(graph.rf_graph->graph, state);

            if (this->vertex == NULL) {
                throw std::runtime_error("Failed to create ptd_vertex\n");
            }
        }

    public:
        Vertex(Graph &graph, struct ptd_vertex *vertex) : graph(graph) {
            this->vertex = vertex;
        }

        Vertex(const Vertex &o) : graph(o.graph) {
            this->vertex = o.vertex;
        }

        // pybind11 factory function
        static Vertex init_factory(Graph &graph, std::vector<int> state) {
            return Vertex(graph, state.data());
        }

        ~Vertex() {
        }

        void add_edge(Vertex &to, double weight);

        void add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state);

        Vertex add_aux_vertex(double rate);

        Vertex add_aux_vertex(std::vector<double> rate_coeffs);

        // Like add_aux_vertex(rate), but creates BOTH directions as
        // coefficient-less constant edges. Works on any graph
        // (parameterised or not). The trapping loop's rate is fixed
        // and independent of theta, so per-observation theta scaling
        // (e.g. exposure) does not affect it. Used by
        // joint_stop_prob_graph() to install trapping loops on
        // parameterised JSP graphs without inflating lambda_max.
        Vertex add_aux_vertex_constant(double weight);

        std::vector<int> state();

        std::vector<Edge> edges();

        std::vector<ParameterizedEdge> parameterized_edges();

        bool operator==(const Vertex &other) const {
            return vertex == other.vertex;
        }

        Vertex &operator=(const Vertex &o) {
            vertex = o.vertex;
            graph = Graph(o.graph.rf_graph);

            return *this;
        }

        struct ptd_vertex *c_vertex() {
            return vertex;
        }

        double rate() {
            return ptd_vertex_rate(vertex);
        }

        size_t edges_length() {
            return vertex->edges_length;
        }

        /**
         * Check if this vertex is an auxiliary vertex.
         *
         * Auxiliary vertices are created by add_aux_vertex() and have special
         * semantics (e.g., their return edge is always constant weight 1.0).
         *
         * @return true if vertex is auxiliary, false otherwise
         */
        bool is_aux() {
            return vertex->is_aux;
        }

        /**
         * Mark or unmark this vertex as auxiliary.
         *
         * @param value true to mark as auxiliary, false to unmark
         */
        void set_aux(bool value) {
            vertex->is_aux = value;
        }

        // NB: moved this from private to public to allow pybin11 to find it
        // @Tobias: is there a better way to do this?
        struct ptd_vertex *vertex;

    private:
        Graph &graph;

        // struct ptd_vertex *vertex;

        friend class Graph;
    };

    struct Edge {
    private:
        Edge(struct ptd_vertex *vertex, struct ptd_edge *edge, Graph &graph, double weight) : graph(graph) {
            this->_weight = weight;
            this->_edge = edge;
            this->_vertex = vertex;
        }

    private:
        Graph &graph;
        struct ptd_vertex *_vertex;
        struct ptd_edge *_edge;
        double _weight;

    public:

    // pybind11 factory function
        static Edge init_factory(struct ptd_vertex *vertex, struct ptd_edge *edge, Graph &graph, double weight) {
            Edge e = Edge(vertex, edge, graph, weight);
            e.graph = graph;
            e._vertex = vertex;
            e._edge = edge;
            return e; 
        }

        Vertex to() {
            return Vertex(graph, _vertex);
        }

	    Vertex *to_p() {
            return new Vertex(graph, _vertex);
        }

        double weight() {
            return _weight;
        }

        void update_weight(double weight) {
            ptd_edge_update_weight(_edge, weight);
            _weight = weight;
        }

        void update_to(const Vertex &v) {
            ptd_edge_update_to(_edge, v.vertex);
            _vertex = v.vertex;
        }

        // Accessor for C edge structure (for callback-based weight updates)
        struct ptd_edge *c_edge() const {
            return _edge;
        }

        // Get number of coefficients in this edge
        size_t coefficients_length() const {
            return _edge->coefficients_length;
        }

        // Get coefficient at index
        double coefficient_at(size_t index) const {
            if (index >= _edge->coefficients_length) {
                throw std::out_of_range("Coefficient index out of range");
            }
            return _edge->coefficients[index];
        }

        Edge &operator=(const Edge &o) {
            _weight = o._weight;
            _vertex = o._vertex;
            graph = o.graph;

            return *this;
        }

        friend class Vertex;

        friend struct ParameterizedEdge;
    };

    struct ParameterizedEdge : private Edge {
    private:
        ParameterizedEdge(
                struct ptd_vertex *vertex,
                struct ptd_edge *edge,
                Graph &graph,
                double weight,
                double *state
        ) : Edge(vertex, edge, graph, weight) {
            this->_state = state;
        }

    private:
        double *_state;

    public:

        // pybind11 factory function
        static ParameterizedEdge init_factory(struct ptd_vertex *vertex, struct ptd_edge *edge, Graph &graph, double weight, double *state) {
            return ParameterizedEdge(vertex, edge, graph, weight, state);
        }

        Vertex to() {
            return Vertex(graph, _vertex);
        }

	    Vertex *to_p() {
            return new Vertex(graph, _vertex);
        }

        double weight() {
            return _weight;
        }

        // Expose coefficients_length from base class
        size_t coefficients_length() const {
            return Edge::coefficients_length();
        }

        std::vector<double> edge_state(size_t requested_length) {
            std::vector<double> state;

            if (_state != NULL && _edge->coefficients_length > 0) {
                size_t actual_length = _edge->coefficients_length;

                // Return min(requested_length, actual_length) coefficients
                // This allows callers to request more than available and get what exists
                size_t n_to_return = (requested_length < actual_length) ? requested_length : actual_length;

                for (size_t i = 0; i < n_to_return; ++i) {
                    state.push_back(_state[i]);
                }
            }

            return state;
        }

        ParameterizedEdge &operator=(const ParameterizedEdge &o) {
            _weight = o._weight;
            _vertex = o._vertex;
            graph = o.graph;
            _state = o._state;

            return *this;
        }

        friend class Vertex;
    };

    class PhaseTypeDistribution {
    private:
        PhaseTypeDistribution(Graph &graph, struct ptd_phase_type_distribution *matrix) {
            this->length = matrix->length;
            this->sub_intensity_matrix = matrix->sub_intensity_matrix;
            this->initial_probability_vector = matrix->initial_probability_vector;

            for (size_t i = 0; i < matrix->length; ++i) {
                this->vertices.push_back(Vertex(graph, matrix->vertices[i]));
            }

            this->distribution = matrix;
        }

        struct ptd_phase_type_distribution *distribution;

    public:

       // pybind11 factory function
        static PhaseTypeDistribution init_factory(Graph &graph, struct ptd_phase_type_distribution *matrix) {
            return PhaseTypeDistribution(graph, matrix);
        }

        ~PhaseTypeDistribution() {
            ptd_phase_type_distribution_destroy(distribution);
        }

        struct ptd_phase_type_distribution *c_distribution() {
            return this->distribution;
        }

        size_t length;
        double **sub_intensity_matrix;
        double *initial_probability_vector;
        std::vector<Vertex> vertices;

        friend class Graph;
    };

    class AnyProbabilityDistributionContext {
    public:
        int is_discrete() {
            return _discrete;
        }

        virtual void step() {
            throw std::runtime_error("Not implemented");
        }

        virtual double pmf() {
            throw std::runtime_error("Not implemented");
        }

        virtual double pdf() {
            throw std::runtime_error("Not implemented");
        }

        virtual double cdf() {
            throw std::runtime_error("Not implemented");
        }

        virtual double time() {
            throw std::runtime_error("Not implemented");
        }

        virtual int jumps() {
            throw std::runtime_error("Not implemented");
        }

        virtual std::vector<long double> stop_probability() {
            throw std::runtime_error("Not implemented");
        }

        virtual std::vector<long double> accumulated_visits() {
            throw std::runtime_error("Not implemented");
        }

        virtual std::vector<long double> accumulated_visiting_time() {
            throw std::runtime_error("Not implemented");
        }

        virtual ~AnyProbabilityDistributionContext() {};

    protected:
        int _discrete;
    };

    class ProbabilityDistributionContext : private AnyProbabilityDistributionContext {
    public:
        ProbabilityDistributionContext(Graph &graph, int64_t granularity = 0) : graph(graph) {
            context = ptd_probability_distribution_context_create(graph.c_graph(), granularity);
            _discrete = false;
        }

        // pybind11 factory function
        static ProbabilityDistributionContext init_factory(Graph &graph, int64_t granularity = 0) {
            return ProbabilityDistributionContext(graph, granularity);
        }

        void step() {
            ptd_probability_distribution_step(context);
        }

        double pdf() {
            return context->pdf;
        }

        double cdf() {
            return context->cdf;
        }

        double time() {
            return (double) context->time;
        }

        std::vector<long double> stop_probability() {
            return std::vector<long double>(context->probability_at,
                                            context->probability_at + context->graph->vertices_length);
        }

        std::vector<long double> accumulated_visiting_time() {
            std::vector<long double> res(context->accumulated_visits,
                                         context->accumulated_visits + context->graph->vertices_length);

            for (size_t i = 0; i < context->graph->vertices_length; ++i) {
                res[i] /= (long double) context->granularity;
            }

            return res;
        }

        ~ProbabilityDistributionContext() {
            ptd_probability_distribution_context_destroy(context);
        }

    private:
        // Stored solely to keep the Graph alive while this context exists;
        // `context` was built from graph.c_graph() and references its
        // internals.
        [[maybe_unused]] Graph &graph;
        struct ptd_probability_distribution_context *context;
    };

    class DPHProbabilityDistributionContext : private AnyProbabilityDistributionContext {
    public:
        DPHProbabilityDistributionContext(Graph &graph) : graph(graph) {
            context = ptd_dph_probability_distribution_context_create(graph.c_graph());
            _discrete = true;
        }

        // pybind11 factory function
        static DPHProbabilityDistributionContext init_factory(Graph &graph) {
            return DPHProbabilityDistributionContext(graph);
        }

        void step() {
            ptd_dph_probability_distribution_step(context);
        }

        double pmf() {
            return context->pmf;
        }

        double cdf() {
            return context->cdf;
        }

        int jumps() {
            return context->jumps;
        }

        std::vector<long double> stop_probability() {
            return std::vector<long double>(context->probability_at,
                                            context->probability_at + context->graph->vertices_length);
        }


        std::vector<long double> accumulated_visits() {
            return std::vector<long double>(context->accumulated_visits,
                                            context->accumulated_visits + context->graph->vertices_length);
        }

        ~DPHProbabilityDistributionContext() {
            ptd_dph_probability_distribution_context_destroy(context);
        }

    private:
        // Stored solely to keep the Graph alive while this context exists;
        // `context` was built from graph.c_graph() and references its
        // internals.
        [[maybe_unused]] Graph &graph;
        struct ptd_dph_probability_distribution_context *context;
    };
}

#endif //PTDALGORITHMS_PTDCPP_H
