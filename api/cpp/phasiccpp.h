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
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <iterator>
#include <functional>
#include <utility>

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

    // ---- Callback-based state-space construction ------------------------------
    // Mirrors the Python `Graph(callback, ipv=..., theta_dim=...)` UI in native
    // C++. A user supplies a transition callback returning the out-transitions
    // of a given state; `Graph::from_callback` runs the breadth-first
    // state-space exploration in C++ (no Python, no GIL) and builds the graph.

    // One out-transition returned by a TransitionCallback. An edge is constant
    // when `coefficients` is empty and parameterized otherwise (matching the
    // per-edge dispatch of the Python/pybind builder).
    struct Transition {
        std::vector<int> state;           // the next state (target vertex)
        double weight;                    // constant-edge weight; vestigial (ignored) for parameterized edges
        std::vector<double> coefficients; // empty => constant edge; non-empty => parameterized edge

        Transition(std::vector<int> state, double weight)
                : state(std::move(state)), weight(weight) {}
        Transition(std::vector<int> state, double weight, std::vector<double> coefficients)
                : state(std::move(state)), weight(weight), coefficients(std::move(coefficients)) {}
    };

    // Called with a discovered vertex's state; returns that vertex's
    // out-transitions. The starting/IPV edges are supplied separately to
    // `Graph::from_callback`, so the callback only ever sees non-empty states.
    typedef std::function<std::vector<Transition>(const std::vector<int> &state)> TransitionCallback;

    // Raised by Graph::weight_formula(const std::string&) on a syntactically or
    // semantically invalid formula (mirrors Python's weight_formula.WeightFormulaError,
    // which subclasses ValueError).
    class WeightFormulaError : public std::runtime_error {
    public:
        explicit WeightFormulaError(const std::string &msg) : std::runtime_error(msg) {}
    };

    // Traditional matrix representation of a phase-type distribution, mirroring
    // the fields of Python Graph.as_matrices() (a NamedTuple). Returned by
    // Graph::as_matrices(); consumed by Graph::from_matrices(). Row i of every
    // member refers to the same transient state; `indices` are 1-based vertex
    // numbers, matching Python.
    struct MatrixRepresentation {
        std::vector<std::vector<int>> states;   // (n_states, state_length): state vector per row
        std::vector<std::vector<double>> sim;   // (n_states, n_states): sub-intensity matrix S
        std::vector<double> ipv;                // (n_states,): initial probability vector (alpha)
        std::vector<size_t> indices;            // (n_states,): 1-based vertex indices
    };

    // Result of Graph::profile() — recommended build/eval settings with reasons,
    // mirroring Python's GraphProfile (src/phasic/profile.py). The structural
    // (SCC) and eval-path tiers are ported faithfully; the measured dyn-ordering
    // probe (which times two eliminations of the largest SCC via pybind-internal
    // machinery) is NOT ported, so dyn_ordering is reported as "not probed".
    struct GraphProfile {
        size_t n_vertices = 0;
        size_t n_edges = 0;
        size_t param_length = 0;
        double max_rate = 0.0;
        double auto_granularity = 0.0;      // 2 * max_rate
        size_t n_sccs = 0;
        size_t largest_scc = 0;
        double largest_scc_frac = 0.0;
        size_t n_levels = 0;
        size_t max_level_width = 0;
        double parallelizable_frac = 0.0;
        double speedup_ceiling = 0.0;       // min(total/critical-path, cpu_count)
        unsigned cpu_count = 1;
        bool parallel_elimination = false;              // the recommended decision
        std::string parallel_elimination_reason;
        std::string dyn_ordering_reason;                // always "not probed" in C++
        std::string path;                               // "forward" or "sojourn"
        std::string path_reason;

        std::string report() const;         // multi-line human-readable summary
        std::string apply_snippet() const;  // ready-to-paste phasic.configure(...) line
    };

    // Result of Graph::discretize() — the discretized graph plus its reward
    // vector (1 for auxiliary vertices, else 0). Defined after Graph (needs the
    // complete Graph type); forward-declared here so Graph can name it.
    struct DiscretizeResult;

    // Array representation of a graph produced by Graph::serialize() and consumed
    // by Graph::from_serialized(), mirroring the fields of Python's
    // serialize()/from_serialized() dict. Row-indexed by an enumeration position
    // (0..n_vertices-1); edge rows reference those positions. Regular (constant-
    // valued but coefficient-carrying) and coefficient-less (constant_edges)
    // edges are kept separate, as in Python. Python-runtime metadata
    // (weight_mode/dyn_ordering/formula tape) is intentionally omitted.
    struct SerializedGraph {
        std::vector<std::vector<int>> states;             // (n, state_length)
        std::vector<size_t> vertex_indices;               // (n,) underlying C vertex indices
        std::vector<std::vector<double>> edges;           // [from, to, weight]
        std::vector<std::vector<double>> constant_edges;  // [from, to, weight] (coefficient-less)
        std::vector<std::vector<double>> start_edges;     // [to, weight]
        std::vector<std::vector<double>> param_edges;     // [from, to, coeff0, coeff1, ...]
        size_t param_length = 0;
        size_t state_length = 0;
        size_t n_vertices = 0;
    };

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
            // Propagate the C++-side members. The forward-state caches (_pdf/_cdf/
            // …) MUST be copied, not left empty: the copy shares o's rf_graph and
            // thus its (possibly already-stepped) ph_context, so an empty cache
            // here would make pdf()/cdf() index out of bounds. The formula string,
            // epoch metadata, and construction callback carry semantic state.
            this->_pdf = o._pdf;
            this->_cdf = o._cdf;
            this->_dph_pmf = o._dph_pmf;
            this->_dph_cdf = o._dph_cdf;
            this->_weight_formula_src = o._weight_formula_src;
            this->_epoch_state_index = o._epoch_state_index;
            this->_n_epochs = o._n_epochs;
            this->_base_param_length = o._base_param_length;
            this->_callback = o._callback;
        }

        // Move constructor - transfers ownership without sharing references
        // This enables clone() to return by value without triggering copy semantics
        Graph(Graph &&o) noexcept {
            this->rf_graph = o.rf_graph;
            o.rf_graph = nullptr;
            this->_pdf = std::move(o._pdf);
            this->_cdf = std::move(o._cdf);
            this->_dph_pmf = std::move(o._dph_pmf);
            this->_dph_cdf = std::move(o._dph_cdf);
            this->_weight_formula_src = std::move(o._weight_formula_src);
            this->_epoch_state_index = o._epoch_state_index;
            this->_n_epochs = o._n_epochs;
            this->_base_param_length = o._base_param_length;
            this->_callback = std::move(o._callback);
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

        // Build a graph from a transition callback, mirroring the Python
        // `Graph(callback, ipv=..., theta_dim=...)` UI. Runs the breadth-first
        // state-space exploration entirely in C++ (no Python, no GIL).
        //
        //   state_length : length of each state vector (as in Graph(size_t)).
        //   ipv          : starting-vertex edges as (state, probability) pairs.
        //                  Probabilities are constant weights and must be
        //                  positive with sum <= 1 (a defect < 1 is allowed).
        //   callback     : returns the out-transitions of each discovered
        //                  (non-empty) state. An edge is parameterized iff its
        //                  Transition carries a non-empty coefficient vector;
        //                  a graph must not mix constant and parameterized
        //                  interior edges (enforced by the edge-mode lock).
        //   param_length : mirrors theta_dim; set before any edge is added.
        //                  Pass 0 (default) for constant graphs or to let the
        //                  C core infer it from the first parameterized edge
        //                  (set_param_length(0) is itself an error).
        static Graph from_callback(
                size_t state_length,
                const std::vector<std::pair<std::vector<int>, double>> &ipv,
                const TransitionCallback &callback,
                size_t param_length = 0);

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

        // Rewards are one per vertex; the C reward routines loop over
        // vertices_length, so a wrong-length vector reads out of bounds. Throw
        // instead. (expected_waiting_time additionally allows an empty vector =
        // unit rewards, so it does its own check.)
        void _check_reward_length(size_t n, const char *fn) {
            if (n != (size_t) this->c_graph()->vertices_length) {
                throw std::runtime_error(
                    std::string(fn) + ": rewards length " + std::to_string(n) +
                    " must equal the number of vertices " +
                    std::to_string((size_t) this->c_graph()->vertices_length));
            }
        }

        std::vector<double> expected_waiting_time(std::vector<double> rewards = std::vector<double>()) {
            // A reward vector must have one entry per vertex; ptd_expected_waiting_time
            // reads vertices_length entries, so a short vector reads out of bounds.
            if (!rewards.empty() && rewards.size() != (size_t) this->c_graph()->vertices_length) {
                throw std::runtime_error(
                    "expected_waiting_time: rewards length " + std::to_string(rewards.size()) +
                    " must equal the number of vertices " +
                    std::to_string((size_t) this->c_graph()->vertices_length));
            }
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

#ifdef PHASIC_B3_VALIDATORS
        // B3 de-risk validators (-DPHASIC_B3_VALIDATORS only; absent in production).
        // (E[T], forward-mode dE[T]/d(edge weight) per _off input, self-contained
        // central diff per input).
        std::tuple<double, std::vector<double>, std::vector<double>>
        debug_fwdmode_grad() {
            double ewt = 0.0; double *fwd = nullptr, *cd = nullptr; size_t ni = 0;
            if (ptd_debug_fwdmode_grad(this->c_graph(), &ewt, &fwd, &cd, &ni) != 0) {
                PTD_THROW_AND_CLEAR();
            }
            std::vector<double> f(fwd, fwd + ni), c(cd, cd + ni);
            free(fwd); free(cd);
            return std::make_tuple(ewt, f, c);
        }

        // B3 Batch-1 (non-shippable): (E[T], reverse-mode theta-adjoint
        // dE[T]/d(edge weight) per _off input).
        std::tuple<double, std::vector<double>>
        debug_reverse_grad() {
            double ewt = 0.0; double *g = nullptr; size_t ni = 0;
            if (ptd_debug_reverse_grad(this->c_graph(), &ewt, &g, &ni) != 0) {
                PTD_THROW_AND_CLEAR();
            }
            std::vector<double> gv(g, g + ni);
            free(g);
            return std::make_tuple(ewt, gv);
        }

        // B3 Batch-2 (production): (E[T], exact d(E[T])/dtheta) for a
        // continuous/linear/monolithic parameterized graph. Returns success flag
        // as an empty grad vector when the exact path is not applicable (caller
        // falls back to FD) rather than throwing.
        std::tuple<double, std::vector<double>>
        moment0_grad_theta() {
            size_t P = static_cast<size_t>(this->c_graph()->param_length);
            std::vector<double> dtheta(P, 0.0);
            double ewt = 0.0;
            int rc = ptd_moment0_grad_theta(this->c_graph(), &ewt,
                                            P ? dtheta.data() : nullptr);
            if (rc != 0) return std::make_tuple(0.0, std::vector<double>());
            return std::make_tuple(ewt, dtheta);
        }
#endif /* PHASIC_B3_VALIDATORS */

        // B3 Batch-3 (production): exact moment-vector Jacobian d[m]/dtheta,
        // returned FLAT (row-major nr_moments*param_length). Empty => not
        // applicable (caller falls back to FD).
        std::vector<double> moments_grad_theta(int nr_moments) {
            size_t P = static_cast<size_t>(this->c_graph()->param_length);
            if (P == 0 || nr_moments < 1) return std::vector<double>();
            std::vector<double> J(static_cast<size_t>(nr_moments) * P, 0.0);
            int rc = ptd_moments_grad_theta(this->c_graph(), nr_moments, J.data());
            if (rc != 0) return std::vector<double>();
            return J;
        }

        // B3 discrete/was_dph extension (production): exact d[discrete m]/dtheta
        // Jacobian for a discrete (is_discrete) parameterized graph, covering both
        // was_dph=True (Graph.discretize(), renormalised edges) and was_dph=False
        // (native DPH). theta must match what the caller most recently passed to
        // update_weights(). Returned FLAT (row-major nr_moments*param_length).
        // Empty => not applicable (caller falls back to FD).
        std::vector<double> moments_grad_theta_dph(int nr_moments, std::vector<double> theta) {
            size_t P = static_cast<size_t>(this->c_graph()->param_length);
            if (P == 0 || nr_moments < 1 || theta.size() != P) return std::vector<double>();
            std::vector<double> J(static_cast<size_t>(nr_moments) * P, 0.0);
            int rc = ptd_moments_grad_theta_dph(this->c_graph(), nr_moments,
                                                theta.data(), theta.size(), J.data());
            if (rc != 0) return std::vector<double>();
            return J;
        }

        // ------------------------------------------------------------------
        // Python-API-name parity (additive).
        //
        // The following methods give the C++ phasic::Graph the same method
        // names as the Python Graph, so the two APIs read identically in user
        // code and docs (e.g. graph.expectation() instead of
        // graph.expected_waiting_time()[0]). They are either faithful ports of
        // the moment helpers that previously lived only as pybind free
        // functions in src/cpp/phasic_pybind.cpp (_moments/_expectation/...),
        // or thin forwarders over already-public methods declared elsewhere in
        // this class. Nothing here changes existing behaviour and every
        // existing C++ name remains valid. SVGD/JAX/inference methods are
        // intentionally excluded.
        // ------------------------------------------------------------------

        // Raw moments of the (reward-weighted) phase-type distribution:
        // moments(power)[k-1] = E[T^k] for k = 1..power. Mirrors Python
        // Graph.moments(power, rewards). rewards, when given, must have one
        // entry per vertex.
        std::vector<double> moments(int power, std::vector<double> rewards = std::vector<double>()) {
            if (!rewards.empty() && rewards.size() != this->c_graph()->vertices_length) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) this->c_graph()->vertices_length, (int) rewards.size());
                throw std::runtime_error(message);
            }
            if (power <= 0) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: power must be a strictly positive integer. Got %i", power);
                throw std::runtime_error(message);
            }

            std::vector<double> res(power);
            std::vector<double> rewards2 = this->expected_waiting_time(rewards);
            std::vector<double> rewards3(rewards2.size());
            res[0] = rewards2[0];

            for (int i = 1; i < power; ++i) {
                if (!rewards.empty()) {
                    for (int j = 0; j < (int) rewards2.size(); ++j) {
                        rewards3[j] = rewards2[j] * rewards[j];
                    }
                } else {
                    rewards3 = rewards2;
                }

                rewards2 = this->expected_waiting_time(rewards3);

                long factorial = 1;
                for (int k = 2; k <= i + 1; ++k) {
                    factorial *= k;
                }
                res[i] = (double) factorial * rewards2[0];
            }

            return res;
        }

        // Expectation E[T] (first moment). Mirrors Python Graph.expectation();
        // equal to expected_waiting_time()[0].
        double expectation(std::vector<double> rewards = std::vector<double>()) {
            return this->moments(1, rewards)[0];
        }

        // Variance Var[T]. Mirrors Python Graph.variance().
        double variance(std::vector<double> rewards = std::vector<double>()) {
            std::vector<double> exp = this->expected_waiting_time(rewards);
            std::vector<double> second;

            if (rewards.empty()) {
                second = this->expected_waiting_time(exp);
            } else {
                std::vector<double> new_rewards(exp.size());
                for (int i = 0; i < (int) exp.size(); ++i) {
                    new_rewards[i] = exp[i] * rewards[i];
                }
                second = this->expected_waiting_time(new_rewards);
            }

            return (2 * second[0] - exp[0] * exp[0]);
        }

        // Covariance for multivariate phase-type. Mirrors Python
        // Graph.covariance(). Both reward vectors are required and must have
        // one entry per vertex (covariance is undefined without them).
        double covariance(std::vector<double> rewards1 = std::vector<double>(),
                          std::vector<double> rewards2 = std::vector<double>()) {
            if (rewards1.size() != this->c_graph()->vertices_length ||
                rewards2.size() != this->c_graph()->vertices_length) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: covariance requires two reward vectors of length %i "
                    "(one entry per vertex); got %i and %i",
                    (int) this->c_graph()->vertices_length,
                    (int) rewards1.size(), (int) rewards2.size());
                throw std::runtime_error(message);
            }

            std::vector<double> exp1 = this->expected_waiting_time(rewards1);
            std::vector<double> exp2 = this->expected_waiting_time(rewards2);

            std::vector<double> new_rewards(exp1.size());

            for (int i = 0; i < (int) exp1.size(); ++i) {
                new_rewards[i] = exp1[i] * rewards2[i];
            }
            std::vector<double> second1 = this->expected_waiting_time(new_rewards);

            for (int i = 0; i < (int) exp1.size(); ++i) {
                new_rewards[i] = exp2[i] * rewards1[i];
            }
            std::vector<double> second2 = this->expected_waiting_time(new_rewards);

            return (second1[0] + second2[0] - exp1[0] * exp2[0]);
        }

        // Discrete-phase-type expectation (mean number of jumps). Mirrors
        // Python Graph.expectation_discrete().
        double expectation_discrete(std::vector<double> rewards = std::vector<double>()) {
            return this->moments(1, rewards)[0];
        }

        // Discrete-phase-type variance. Mirrors Python Graph.variance_discrete().
        double variance_discrete(std::vector<double> rewards = std::vector<double>()) {
            if (!rewards.empty() && rewards.size() != this->c_graph()->vertices_length) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) this->c_graph()->vertices_length, (int) rewards.size());
                throw std::runtime_error(message);
            }
            if (rewards.empty()) {
                std::vector<double> m = this->moments(2);
                return m[1] - 2 * m[0];
            } else {
                std::vector<double> sq_rewards(rewards.size());
                for (int i = 0; i < (int) rewards.size(); ++i) {
                    sq_rewards[i] = rewards[i] * rewards[i];
                }
                std::vector<double> momentsr = this->moments(2, rewards);
                std::vector<double> momentsrr = this->moments(1, sq_rewards);
                return momentsr[1] - momentsr[0] * momentsr[0] - momentsrr[0];
            }
        }

        // Discrete-phase-type covariance. Mirrors Python
        // Graph.covariance_discrete(). Both reward vectors are required.
        double covariance_discrete(std::vector<double> rewards1 = std::vector<double>(),
                                   std::vector<double> rewards2 = std::vector<double>()) {
            if (rewards1.size() != this->c_graph()->vertices_length ||
                rewards2.size() != this->c_graph()->vertices_length) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: covariance_discrete requires two reward vectors of "
                    "length %i (one entry per vertex); got %i and %i",
                    (int) this->c_graph()->vertices_length,
                    (int) rewards1.size(), (int) rewards2.size());
                throw std::runtime_error(message);
            }

            std::vector<double> sq_rewards(rewards1.size());
            for (int i = 0; i < (int) rewards1.size(); ++i) {
                sq_rewards[i] = rewards1[i] * rewards2[i];
            }

            std::vector<double> rw1to2(rewards1.size());
            std::vector<double> rw2to1(rewards2.size());
            std::vector<double> exp1 = this->expected_waiting_time(rewards1);
            std::vector<double> exp2 = this->expected_waiting_time(rewards2);

            for (int i = 0; i < (int) rewards1.size(); ++i) {
                rw1to2[i] = exp1[i] * rewards2[i];
                rw2to1[i] = exp2[i] * rewards1[i];
            }

            return this->expected_waiting_time(rw1to2)[0] +
                   this->moments(1, sq_rewards)[0] -
                   this->moments(1, rewards1)[0] *
                   this->moments(1, rewards2)[0];
        }

        // --- Same-name forwarders over existing C++ methods ---

        // Update parameterized edge weights from theta (linear/log inner
        // product). Python name for update_weights_parameterized.
        void update_weights(std::vector<double> scalars, bool use_log = false) {
            this->update_weights_parameterized(scalars, use_log);
        }
        // Callback-based weight update. Python name for the callback overload
        // of update_weights_parameterized.
        void update_weights(
            std::vector<double> scalars,
            std::function<double(const std::vector<double>&, const std::vector<double>&)> callback) {
            this->update_weights_parameterized(scalars, callback);
        }

        // Discrete reward transform. Python name for dph_reward_transform.
        Graph reward_transform_discrete(std::vector<int> rewards) {
            return this->dph_reward_transform(rewards);
        }

        // Per-state accumulated visits after `jumps` steps (discrete). Python
        // name for dph_accumulated_visits.
        std::vector<double> accumulated_visits(int jumps) {
            return this->dph_accumulated_visits(jumps);
        }

        // Discrete PMF / CDF at `jumps`. Python names for dph_pmf / dph_cdf.
        double pdf_discrete(int jumps) {
            return this->dph_pmf(jumps);
        }
        double cdf_discrete(int jumps) {
            return this->dph_cdf(jumps);
        }

        // Deep copy. Python name (alongside the existing clone()).
        Graph copy() {
            return this->clone();
        }

        // Per-state occupancy probability at `time`. Python name (alias) for
        // stop_probability; matches Python's state_probability = stop_probability.
        std::vector<double> state_probability(double time, int64_t granularity = 0) {
            return this->stop_probability(time, granularity);
        }

        // Sample a full path through the Markov chain: (vertex indices, entry
        // times). Python name for random_sample_path.
        std::pair<std::vector<size_t>, std::vector<double>> sample_path() {
            return this->random_sample_path();
        }

        // Sample a path conditioned on backward probabilities. Python name for
        // random_sample_path_conditioned.
        std::pair<std::vector<size_t>, std::vector<double>> sample_path_conditioned(
            std::vector<double> backward_probs) {
            return this->random_sample_path_conditioned(backward_probs);
        }

        // Draw n independent samples from the (reward-weighted) distribution.
        // Python name for repeated random_sample. rewards, when given, must
        // have one entry per vertex; omitted -> unit rewards (NULL to the C
        // sampler, which treats it as unit weights).
        std::vector<long double> sample(size_t n, std::vector<double> rewards = std::vector<double>()) {
            if (!rewards.empty() && rewards.size() != this->c_graph()->vertices_length) {
                char message[1024];
                snprintf(message, 1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) this->c_graph()->vertices_length, (int) rewards.size());
                throw std::runtime_error(message);
            }
            std::vector<long double> res(n);
            double *rw = rewards.empty() ? NULL : &rewards[0];
            for (size_t i = 0; i < n; ++i) {
                res[i] = ptd_random_sample(this->c_graph(), rw);
            }
            return res;
        }

        // Traditional matrix representation (SIM, IPV, states, 1-based indices).
        // Python name for the matrix export; mirrors Graph.as_matrices()
        // field-for-field. Vertices may be reordered relative to vertices();
        // `indices` gives the 1-based vertex numbers. (Inline: uses only the C
        // phase-type-distribution struct, no phasic::Vertex.)
        MatrixRepresentation as_matrices() {
            struct ptd_phase_type_distribution *dist =
                ptd_graph_as_phase_type_distribution(this->c_graph());
            if (dist == NULL) {
                throw std::runtime_error(
                    "Graph::as_matrices: failed to build phase-type distribution");
            }

            size_t n = dist->length;
            size_t sl = this->state_length();

            MatrixRepresentation rep;
            rep.states.assign(n, std::vector<int>(sl));
            rep.sim.assign(n, std::vector<double>(n));
            rep.ipv.assign(n, 0.0);
            rep.indices.assign(n, 0);

            for (size_t i = 0; i < n; ++i) {
                rep.ipv[i] = dist->initial_probability_vector[i];
                rep.indices[i] = dist->vertices[i]->index + 1;
                for (size_t j = 0; j < n; ++j) {
                    rep.sim[i][j] = dist->sub_intensity_matrix[i][j];
                }
                for (size_t j = 0; j < sl; ++j) {
                    rep.states[i][j] = dist->vertices[i]->state[j];
                }
            }

            ptd_phase_type_distribution_destroy(dist);
            return rep;
        }

        // Per-vertex state matrix over vertices(): row i is vertex i's state.
        // Python name for Graph.states(). (Defined out-of-line in
        // phasiccpp.cpp: uses phasic::Vertex, which is completed after Graph.)
        std::vector<std::vector<int>> states();

        // Build a graph from the traditional (IPV, SIM[, states]) matrix form.
        // Python name for Graph.from_matrices(); states defaults to
        // [0],[1],[2],... when omitted. (Defined out-of-line in phasiccpp.cpp:
        // uses phasic::Vertex.)
        static Graph from_matrices(
            std::vector<double> ipv,
            std::vector<std::vector<double>> sim,
            std::vector<std::vector<int>> states = std::vector<std::vector<int>>());

        // ---- Tier 2 additions (graph queries / transforms) ----

        // Reward vector (length n_vertices): 1.0 for each vertex with an edge to
        // an absorbing state (a vertex with no outgoing edges), else 0.0. Python
        // name for Graph.absorbing_state_rewards(). Used to read a Laplace-
        // transform value: laplace_transform(theta).expectation(rewards).
        std::vector<double> absorbing_state_rewards() {
            struct ptd_graph *g = this->c_graph();
            size_t n = g->vertices_length;
            std::vector<double> rewards(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                struct ptd_vertex *v = g->vertices[i];
                for (size_t k = 0; k < v->edges_length; ++k) {
                    if (v->edges[k]->to->edges_length == 0) {
                        rewards[i] = 1.0;
                        break;
                    }
                }
            }
            return rewards;
        }

        // Probability that a trajectory from the starting vertex visits at least
        // one vertex with reward > 0 before absorption:
        //   p = sum_v alpha_v * P(reach a rewarded vertex | start at v).
        // Python name for Graph.reward_visit_probability() (concrete-theta path;
        // the Python JAX/FFI path has no C++ equivalent). If theta is non-empty
        // the edge weights are updated from it first, and — mirroring Python —
        // that update is NOT rolled back.
        double reward_visit_probability(
            std::vector<double> rewards,
            std::vector<double> theta = std::vector<double>()) {
            size_t n_v = this->c_graph()->vertices_length;
            if (rewards.size() != n_v) {
                char message[1024];
                snprintf(message, 1024,
                    "reward_visit_probability: rewards must have length "
                    "n_vertices=%i; got %i", (int) n_v, (int) rewards.size());
                throw std::runtime_error(message);
            }

            std::vector<size_t> rewarded;
            for (size_t i = 0; i < n_v; ++i) {
                if (rewards[i] > 0.0) rewarded.push_back(i);
            }
            if (rewarded.empty()) {
                return 0.0;
            }

            if (!theta.empty()) {
                this->update_weights(theta);
            }

            std::vector<double> h = this->backward_probabilities(rewarded);

            // alpha (IPV): starting-vertex edge weights, keyed by target index.
            std::vector<double> alpha(n_v, 0.0);
            struct ptd_vertex *sv = this->c_graph()->starting_vertex;
            for (size_t k = 0; k < sv->edges_length; ++k) {
                double w = sv->edges[k]->weight;
                if (w > 0.0) {
                    alpha[sv->edges[k]->to->index] = w;
                }
            }

            double p = 0.0;
            for (size_t i = 0; i < n_v; ++i) {
                p += alpha[i] * h[i];
            }
            return p;
        }

        // Continue the callback-based build from vertex `vertex_index` onward
        // (and any vertices discovered thereafter). Python name for
        // Graph.extend(): the same breadth-first exploration as from_callback,
        // but starting mid-graph, so a caller can hand-add vertices/edges and
        // then explore their successors. Unlike Python (which tracks the
        // last-built length), the resume index is explicit here. Defined
        // out-of-line in phasiccpp.cpp (uses phasic::Vertex).
        void extend(const TransitionCallback &callback, size_t vertex_index);
        // Overload using the graph's stored construction callback (set by
        // from_callback). Raises if the graph has no stored callback.
        void extend(size_t vertex_index);

        // Return a discretized copy of this graph (original unchanged): an
        // auxiliary vertex is added per transient state and the graph is
        // normalized, producing a discrete phase-type distribution. Python name
        // for Graph.discretize(); the returned DiscretizeResult carries the new
        // graph and its reward vector (1 for aux vertices) — Python attaches the
        // rewards as an attribute, which C++ cannot. `rate` must be in (0, 1)
        // (scalar overload) or a callback state->rate. Parameterized graphs are
        // not supported (the Python path widens the coefficient layout, which has
        // no C++ equivalent) and raise. Defined out-of-line in phasiccpp.cpp.
        DiscretizeResult discretize(double rate, bool skip_existing = false);
        DiscretizeResult discretize(
            std::function<double(const std::vector<int>&)> rate_fn,
            bool skip_existing = false);

        // Analyse this (parameterized) graph and recommend build/eval settings
        // with reasons — parallel_elimination (from the SCC structure) and the
        // evaluation path (from max_rate -> auto-granularity). Python name for
        // Graph.profile(); `theta` (default all-ones) sets the reference point for
        // max_rate. The measured dyn-ordering probe is not ported (dyn_ordering is
        // reported "not probed"). Defined out-of-line in phasiccpp.cpp.
        GraphProfile profile(std::vector<double> theta = std::vector<double>());

        // Compile a per-edge weight-formula string and install it as the graph's
        // weight tape, so update_weights(theta) evaluates it per edge in C (the
        // fast path). Python name for the `graph.weight_formula = "..."` setter.
        // t0,t1,... = theta; c0,c1,... = the edge's coefficients; functions:
        // exp log sqrt logistic pow delta and or not select. Comparison/delta/
        // boolean conditions and select's condition must be theta-independent (a
        // theta-dependent condition raises). Throws WeightFormulaError on an
        // invalid formula. Defined out-of-line in phasiccpp.cpp.
        void weight_formula(const std::string &formula);
        // Getter: the formula string most recently set (empty if none). Python
        // name parity for reading the `weight_formula` property.
        std::string weight_formula() const;

        // Serialize the graph to an array representation (states + edges + params).
        // Python name for Graph.serialize(); returns a SerializedGraph rather than
        // Python's numpy dict. Round-trips via from_serialized(). Defined
        // out-of-line in phasiccpp.cpp (uses phasic::Vertex/Edge).
        SerializedGraph serialize();

        // Reconstruct a graph from serialize() output. Python name for
        // Graph.from_serialized(); rebuilds vertices + parameterized/regular/start
        // edges (coefficient-less constant_edges are not re-added, matching
        // Python). Defined out-of-line in phasiccpp.cpp.
        static Graph from_serialized(const SerializedGraph &data);

        // Add an epoch boundary at `time`, returning a NEW graph (original
        // unchanged) with a widened state/coefficient layout: epoch-transition
        // edges are wired from stop_probability(time)/accumulated_visiting_time(time)
        // and the next epoch's state space is explored via a callback. Python name
        // for Graph.add_epoch(). `callback` defaults to the graph's stored
        // construction callback (set by from_callback); pass one explicitly for a
        // manually-built graph or to override. Raises if neither is available.
        // Supports the no-StateIndexer path; the base graph should be
        // parameterized and have current weights (call update_weights first).
        // Epoch metadata is carried on the returned graph so a further add_epoch
        // chains. Defined out-of-line in phasiccpp.cpp.
        Graph add_epoch(double time, const TransitionCallback &callback = TransitionCallback());

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
            _check_reward_length(rewards.size(), "dph_reward_transform");
            struct ptd_graph *res = ptd_graph_dph_reward_transform(c_graph(), &rewards[0]);

            if (res == NULL) {
                PTD_THROW_AND_CLEAR();
            }

            return Graph(res);
        }

        Graph *dph_reward_transform_p(std::vector<int> rewards) {
            _check_reward_length(rewards.size(), "dph_reward_transform");
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

            // Propagate the C++-side members (see the copy constructor).
            this->_pdf = o._pdf;
            this->_cdf = o._cdf;
            this->_dph_pmf = o._dph_pmf;
            this->_dph_cdf = o._dph_cdf;
            this->_weight_formula_src = o._weight_formula_src;
            this->_epoch_state_index = o._epoch_state_index;
            this->_n_epochs = o._n_epochs;
            this->_base_param_length = o._base_param_length;
            this->_callback = o._callback;

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
        std::string _weight_formula_src;  // last formula set via weight_formula(str)
        // add_epoch metadata (mirrors the Python attributes). Set on the graph
        // returned by add_epoch; read by a subsequent add_epoch to chain epochs.
        int _epoch_state_index = -1;      // -1 = no epoch dimension yet
        int _n_epochs = 0;                // number of epoch boundaries added
        int _base_param_length = -1;      // -1 = unset (use param_length())
        // Construction callback stored by from_callback (empty for a graph built
        // manually or via Graph(size_t)); lets extend()/add_epoch() default to it,
        // mirroring Python's stored self._callback.
        TransitionCallback _callback;

        friend class VertexLinkedList;

        friend class Vertex;
    };

    // Definition deferred until here so the Graph member is a complete type.
    struct DiscretizeResult {
        Graph graph;                 // the discretized copy (original unchanged)
        std::vector<int> rewards;    // 1 for auxiliary vertices, else 0
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

        // Append a coefficient-less CONSTANT edge (this -> to) via direct
        // ptd_edge struct manipulation, bypassing the EDGE_MODE_PARAMETERIZED
        // lock (mirrors add_aux_vertex_constant / GraphBuilder::build). Its
        // weight stays fixed across update_weights (coefficients_length == 0).
        // Used by Graph.from_serialized to rebuild the aux back-edges of a
        // discretize()/joint-stop-prob parameterised graph.
        void add_edge_constant(Vertex &to, double weight);

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
