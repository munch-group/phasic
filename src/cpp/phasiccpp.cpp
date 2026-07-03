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

#include <stdexcept>
#include <sstream>
#include <stack>
#include <cerrno>
#include <cstring>
#include <vector>
#include <thread>       // std::thread::hardware_concurrency (Graph::profile)
#include <algorithm>    // std::max / std::min (Graph::profile)
#include <memory>       // std::unique_ptr (weight_formula AST)
#include <map>          // std::map (weight_formula const dedup / serialize)
#include <set>          // std::set (serialize param-edge pairs)
#include <cctype>       // isdigit/isalpha/... (weight_formula tokenizer)
#include "phasiccpp.h"

/* While it seems very strange to have this in a C file, the R code
 * has very strange linking behavior, and we therefore sometimes include
 * the same C file...
 */

#ifndef PTDALGORITHMS_PTDCPP_C
#define PTDALGORITHMS_PTDCPP_C

static void assert_same_length(std::vector<int> state, struct ptd_graph *graph) {
    if (state.size() != graph->state_length) {
        std::stringstream message;
        message << "Vector `state` argument must have same size as graph state length. Was '";
        message << state.size() << "' expected '" << graph->state_length << "'" << std::endl;

        throw std::invalid_argument(message.str());
    }
}

static int *force_same_length(std::vector<int> state, struct ptd_graph *graph) {
    int *res = (int *) calloc(graph->state_length, sizeof(int));
  
    for (size_t i = 0; i < state.size(); ++i) {
        res[i] = state[i];
    }
  
    return res;
}

phasic::Vertex phasic::Graph::create_vertex(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex res = create_vertex(c);

    return res;
}


phasic::Vertex phasic::Graph::create_vertex(const int *state) {
    struct ptd_vertex *c_vertex = ptd_vertex_create_state(c_graph(), (int*)state);
  
    Vertex vertex = phasic::Vertex(*this, c_vertex);
    notify_change();

    return vertex;
}

phasic::Vertex *phasic::Graph::create_vertex_p(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex *res = create_vertex_p(c);

    return res;
}


phasic::Vertex *phasic::Graph::create_vertex_p(const int *state) {
    struct ptd_vertex *c_vertex = ptd_vertex_create_state(c_graph(), (int*)state);

    Vertex *vertex = new phasic::Vertex(*this, c_vertex);
    notify_change();
    
    return vertex;
}

phasic::Vertex phasic::Graph::find_vertex(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex res = find_vertex(c);

    free(c);

    return res;
}


phasic::Vertex phasic::Graph::find_vertex(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    if (node == NULL) {
        throw std::runtime_error(
                "No such vertex found\n"
        );
    }

    return phasic::Vertex(*this, (struct ptd_vertex *) node->entry);
}

phasic::Vertex *phasic::Graph::find_vertex_p(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex *res = find_vertex_p(c);

    free(c);

    return res;
}

phasic::Vertex *phasic::Graph::find_vertex_p(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    if (node == NULL) {
        throw std::runtime_error(
                "No such vertex found\n"
        );
    }

    return new phasic::Vertex(*this, (struct ptd_vertex *) node->entry);
}

bool phasic::Graph::vertex_exists(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return vertex_exists(&state[0]);
}

bool phasic::Graph::vertex_exists(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    return (node != NULL);
}

phasic::Vertex phasic::Graph::find_or_create_vertex(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return find_or_create_vertex(&state[0]);
}

phasic::Vertex phasic::Graph::find_or_create_vertex(const int *state) {
    notify_change();

    return phasic::Vertex(*this, ptd_find_or_create_vertex(c_graph(), c_avl_tree(), state));
}

phasic::Vertex *phasic::Graph::find_or_create_vertex_p(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return find_or_create_vertex_p(&state[0]);
}

phasic::Vertex *phasic::Graph::find_or_create_vertex_p(const int *state) {
    notify_change();

    return new phasic::Vertex(*this, ptd_find_or_create_vertex(c_graph(), c_avl_tree(), state));
}

phasic::Vertex phasic::Graph::starting_vertex() {
    return Vertex(*this, this->rf_graph->graph->starting_vertex);
}

phasic::Vertex *phasic::Graph::starting_vertex_p() {
    return new Vertex(*this, this->rf_graph->graph->starting_vertex);
}

phasic::Graph phasic::Graph::from_callback(
        size_t state_length,
        const std::vector<std::pair<std::vector<int>, double>> &ipv,
        const phasic::TransitionCallback &callback,
        size_t param_length) {

    if (ipv.empty()) {
        throw std::invalid_argument("from_callback: ipv must be non-empty");
    }

    // Validate the IPV probabilities up front (mirrors the Python
    // _callback/_validate_ipv check): each must be positive and the total must
    // not exceed 1 (a defect < 1 is allowed).
    double ipv_sum = 0.0;
    for (const auto &entry : ipv) {
        if (entry.second <= 0.0) {
            throw std::invalid_argument("from_callback: ipv probabilities must be positive");
        }
        ipv_sum += entry.second;
    }
    if (ipv_sum > 1.0 + 1e-9) {
        throw std::invalid_argument("from_callback: ipv probabilities must sum to <= 1");
    }

    phasic::Graph graph(state_length);

    // Fix param_length before ANY edge is added. 0 is the skip sentinel:
    // ptd_graph_set_param_length(graph, 0) is a hard error, and for constant
    // graphs the C core infers/locks param_length from the first edge anyway.
    if (param_length > 0) {
        graph.set_param_length(param_length);
    }

    // Bootstrap: starting-vertex (IPV) edges. IPV edges never lock the edge
    // mode or param_length in the C core, so they are added as constant
    // weights (probabilities), matching the Python/pybind driver.
    {
        phasic::Vertex start = graph.starting_vertex();
        for (const auto &entry : ipv) {
            phasic::Vertex child = graph.find_or_create_vertex(entry.first);
            start.add_edge(child, entry.second);
        }
    }

    // Breadth-first worklist over the graph's own vertex array.
    // find_or_create_vertex appends newly discovered states and deduplicates
    // via the graph's AVL tree, so vertices_length() grows in place until the
    // reachable state space is exhausted. A ptd_vertex* stays valid across
    // this growth (only the array of pointers is reallocated, not the vertex
    // objects), so `vertex` remains usable after find_or_create_vertex.
    for (size_t index = 1; index < graph.vertices_length(); ++index) {
        phasic::Vertex vertex = graph.vertex_at(index);
        std::vector<phasic::Transition> transitions = callback(vertex.state());

        for (const auto &t : transitions) {
            phasic::Vertex child = graph.find_or_create_vertex(t.state);
            if (t.coefficients.empty()) {
                vertex.add_edge(child, t.weight);
            } else {
                vertex.add_edge_parameterized(child, t.weight, t.coefficients);
            }
        }
    }

    return graph;  // NRVO / move ctor (phasiccpp.h) — leak-free
}

std::vector<phasic::Vertex> phasic::Graph::vertices() {
    std::vector<Vertex> vec;

    for (size_t i = 0; i < c_graph()->vertices_length; ++i) {
        vec.push_back(Vertex(*this, c_graph()->vertices[i]));
    }

    return vec;
}

std::vector<phasic::Vertex *> phasic::Graph::vertices_p() {
    std::vector<Vertex *> vec;

    for (size_t i = 0; i < c_graph()->vertices_length; ++i) {
        vec.push_back(new Vertex(*this, c_graph()->vertices[i]));
    }

    return vec;
}

phasic::Vertex phasic::Graph::vertex_at(size_t index) {
    return Vertex(*this, c_graph()->vertices[index]);
}

phasic::Vertex *phasic::Graph::vertex_at_p(size_t index) {
    return new Vertex(*this, c_graph()->vertices[index]);
}

size_t phasic::Graph::vertices_length() {
    return c_graph()->vertices_length;
}

size_t phasic::Graph::edges_length() {
    size_t total = 0;
    struct ptd_graph *graph = c_graph();
    for (size_t i = 0; i < graph->vertices_length; ++i) {
        total += graph->vertices[i]->edges_length;
    }
    return total;
}

bool phasic::Graph::parameterized() {
    return c_graph()->parameterized;
}

// Python-API-name parity: per-vertex state matrix over vertices() (row i is
// vertex i's state). Mirrors the pybind _states helper. Defined here (not
// inline in the header) only so it can predate the phasic::Vertex definition.
std::vector<std::vector<int>> phasic::Graph::states() {
    struct ptd_graph *graph = c_graph();
    size_t sl = graph->state_length;
    std::vector<std::vector<int>> res(graph->vertices_length, std::vector<int>(sl));
    for (size_t i = 0; i < graph->vertices_length; ++i) {
        for (size_t j = 0; j < sl; ++j) {
            res[i][j] = graph->vertices[i]->state[j];
        }
    }
    return res;
}

// Python-API-name parity: build a graph from the traditional (IPV, SIM[,
// states]) matrix form. Faithful port of the pybind from_matrices lambda; uses
// the leak-free by-value vertex API instead of the *_p variants the lambda used.
phasic::Graph phasic::Graph::from_matrices(
    std::vector<double> ipv,
    std::vector<std::vector<double>> sim,
    std::vector<std::vector<int>> states) {
    size_t n = ipv.size();

    if (sim.size() != n) {
        throw std::runtime_error("SIM must be square and have same dimension as IPV length");
    }
    for (size_t i = 0; i < n; ++i) {
        if (sim[i].size() != n) {
            throw std::runtime_error("SIM must be square and have same dimension as IPV length");
        }
    }

    bool have_states = !states.empty();
    size_t state_dim = 1;
    if (have_states) {
        if (states.size() != n) {
            throw std::runtime_error("states must have same number of rows as IPV length");
        }
        state_dim = states[0].size();
    }

    Graph graph(state_dim);
    std::vector<Vertex> vertices;
    vertices.reserve(n);

    int s = 0;
    if (have_states) {
        for (size_t i = 0; i < n; ++i) {
            std::vector<int> state(state_dim);
            for (size_t j = 0; j < state_dim; ++j) {
                state[j] = states[i][j];
            }
            vertices.push_back(graph.find_or_create_vertex(state));
        }
    } else {
        for (s = 0; s < (int) n; ++s) {
            std::vector<int> state = {s};
            vertices.push_back(graph.find_or_create_vertex(state));
        }
    }

    // Absorbing vertex: state (state_dim copies of s) mirrors the pybind lambda
    // (s == n for default states, s == 0 when explicit states are supplied).
    std::vector<int> absorbing_state(state_dim, s);
    Vertex absorbing = graph.find_or_create_vertex(absorbing_state);

    Vertex start = graph.starting_vertex();
    double sum_ipv = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (ipv[i] > 0) {
            start.add_edge(vertices[i], ipv[i]);
            sum_ipv += ipv[i];
        }
    }

    if (sum_ipv < 0.99999) {
        throw std::runtime_error("Initial probability vector does not sum to one\n");
    }

    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (i != j && sim[i][j] > 0) {
                vertices[i].add_edge(vertices[j], sim[i][j]);
                row_sum += sim[i][j];
            }
        }
        double exit_rate = -(sim[i][i] + row_sum);
        if (exit_rate > 0.000001) {
            vertices[i].add_edge(absorbing, exit_rate);
        }
    }

    return graph;
}

// Python-API-name parity: continue the callback-based build from `vertex_index`
// onward. Same breadth-first loop as from_callback (find_or_create_vertex
// appends and deduplicates; ptd_vertex* stays valid across growth), but resuming
// mid-graph instead of from vertex 1. Defined here (not inline) because it uses
// phasic::Vertex, which is completed after the Graph class.
void phasic::Graph::extend(const phasic::TransitionCallback &callback, size_t vertex_index) {
    for (size_t index = vertex_index; index < vertices_length(); ++index) {
        Vertex vertex = vertex_at(index);
        std::vector<Transition> transitions = callback(vertex.state());

        for (const auto &t : transitions) {
            Vertex child = find_or_create_vertex(t.state);
            if (t.coefficients.empty()) {
                vertex.add_edge(child, t.weight);
            } else {
                vertex.add_edge_parameterized(child, t.weight, t.coefficients);
            }
        }
    }
}

// Python-API-name parity: discretize (callback-rate core). Faithful port of the
// Python discretize() non-parameterized path: clone, add an auxiliary vertex per
// transient state, normalize, and return the reward vector marking aux vertices.
// The `verts` snapshot is taken before the loop so newly-added aux vertices are
// not themselves processed (matching Python's `for v in new_graph.vertices()`).
phasic::DiscretizeResult phasic::Graph::discretize(
    std::function<double(const std::vector<int>&)> rate_fn, bool skip_existing) {
    if (this->parameterized()) {
        throw std::runtime_error(
            "discretize: parameterized graphs are not supported in the C++ API. "
            "The Python parameterized path widens the per-edge coefficient layout "
            "(_rebuild_with_wider_layout), which has no C++ equivalent. Build a "
            "discrete graph directly, or discretize in Python.");
    }

    Graph new_graph = this->clone();
    std::vector<Vertex> verts = new_graph.vertices();  // snapshot before adding aux
    size_t start_index = new_graph.starting_vertex().c_vertex()->index;
    size_t vlength = new_graph.vertices_length();
    std::vector<size_t> aux_indices;

    for (Vertex &vertex : verts) {
        if (vertex.c_vertex()->index == start_index || vertex.edges_length() == 0) {
            continue;
        }

        if (skip_existing) {
            bool has_aux = false, is_aux = false;
            std::vector<Edge> ve = vertex.edges();
            for (Edge &edge : ve) {
                Vertex to = edge.to();
                std::vector<int> tst = to.state();
                int s = 0;
                for (int x : tst) s += x;
                if (s == 0 && to.edges_length() > 0 &&
                    to.edges()[0].to().c_vertex()->index == vertex.c_vertex()->index) {
                    has_aux = true;
                    aux_indices.push_back(to.c_vertex()->index);
                    vlength -= 1;
                    break;
                }
            }
            std::vector<int> vst = vertex.state();
            int vs = 0;
            for (int x : vst) vs += x;
            if (vs == 0) is_aux = true;
            if (has_aux || is_aux) continue;
        }

        double _rate = rate_fn(vertex.state());
        Vertex aux = vertex.add_aux_vertex(_rate);
        aux.set_aux(true);
        aux_indices.push_back(aux.c_vertex()->index);
    }

    std::vector<int> rewards(vlength + aux_indices.size(), 0);
    for (size_t idx : aux_indices) {
        rewards[idx] = 1;
    }

    new_graph.normalize();

    return DiscretizeResult{ std::move(new_graph), std::move(rewards) };
}

// Python-API-name parity: discretize (scalar-rate overload). Validates the rate
// lies in (0, 1) then delegates to the callback core.
phasic::DiscretizeResult phasic::Graph::discretize(double rate, bool skip_existing) {
    if (rate <= 0.0 || rate >= 1.0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "discretize: rate must be in (0, 1), got %g", rate);
        throw std::runtime_error(msg);
    }
    return this->discretize(
        [rate](const std::vector<int>&) { return rate; }, skip_existing);
}

// Python-API-name parity: structural + eval-path graph profile. Faithful port of
// the cheap tiers of src/phasic/profile.py (SCC structure -> parallel_elimination
// recommendation; max_rate -> eval-path recommendation). The measured
// dyn-ordering probe is not ported (it needs pybind-internal elimination-timing
// plumbing); dyn_ordering is reported as "not probed".
phasic::GraphProfile phasic::Graph::profile(std::vector<double> theta) {
    // Recommendation thresholds — keep in sync with profile.py.
    const double PARALLEL_MIN_FRAC = 0.25;
    const double PARALLEL_MAX_DOMINANCE = 0.5;
    const double PATH_GRANULARITY_WARN = 1e5;

    unsigned cpu = std::thread::hardware_concurrency();
    if (cpu == 0) cpu = 1;

    // --- SCC structure (Tarjan O(V+E)) ---
    // Use the C SCC API directly rather than the C++ SCCGraph wrapper: SCCGraph's
    // methods are defined in scc_graph.cpp and compiled -fvisibility=hidden into
    // the extension, so a translation unit that includes phasiccpp.cpp WITHOUT
    // also linking scc_graph.cpp (e.g. the cppimport test fixture) cannot resolve
    // them. The C functions are exported and the struct fields carry the same
    // information (internal_vertices_length == SCC size; edge->to->index == the
    // target SCC's position, which equals its array index).
    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(this->c_graph());
    if (scc == NULL) {
        throw std::runtime_error("profile: SCC decomposition failed");
    }
    size_t n = scc->vertices_length;

    std::vector<size_t> sizes(n);
    for (size_t i = 0; i < n; ++i) sizes[i] = scc->vertices[i]->internal_vertices_length;
    size_t total = 0;
    for (size_t s : sizes) total += s;
    if (total == 0) total = 1;

    // Sink-first levels: level_of[i] = 0 if no outgoing edges, else
    // 1 + max(level_of[target]). Memoised DFS, mirroring compute_scc_levels().
    std::vector<std::vector<size_t>> outgoing(n);
    for (size_t i = 0; i < n; ++i) {
        struct ptd_scc_vertex *sv = scc->vertices[i];
        outgoing[i].reserve(sv->edges_length);
        for (size_t e = 0; e < sv->edges_length; ++e) {
            outgoing[i].push_back(sv->edges[e]->to->index);
        }
    }
    ptd_scc_graph_destroy(scc);
    std::vector<int> level_of(n, -1);
    std::function<int(size_t)> level = [&](size_t i) -> int {
        if (level_of[i] >= 0) return level_of[i];
        if (outgoing[i].empty()) { level_of[i] = 0; return 0; }
        int m = 0;
        for (size_t j : outgoing[i]) m = std::max(m, level(j));
        level_of[i] = 1 + m;
        return level_of[i];
    };
    for (size_t i = 0; i < n; ++i) level(i);

    int max_level = 0;
    for (int lv : level_of) max_level = std::max(max_level, lv);
    std::vector<std::vector<size_t>> levels(max_level + 1);
    for (size_t i = 0; i < n; ++i) levels[level_of[i]].push_back(i);

    size_t max_width = 0;
    for (auto &lv : levels) max_width = std::max(max_width, lv.size());
    size_t parallel_vertices = 0;
    for (auto &lv : levels)
        if (lv.size() >= 2)
            for (size_t i : lv) parallel_vertices += sizes[i];
    size_t critical = 0;
    for (auto &lv : levels) {
        size_t mx = 0;
        for (size_t i : lv) mx = std::max(mx, sizes[i]);
        critical += mx;
    }
    if (critical == 0) critical = 1;

    double parallelizable_frac = (double) parallel_vertices / (double) total;
    double speedup_inf = (double) total / (double) critical;
    double speedup = std::min(speedup_inf, (double) cpu);
    size_t largest = 0;
    for (size_t s : sizes) largest = std::max(largest, s);
    double largest_frac = (double) largest / (double) total;

    // --- max_rate (drives auto-granularity) ---
    // Only recompute weights from theta when the graph is actually parameterized
    // (update_weights refuses on a constant graph); a constant graph keeps its
    // constructed weights, which give the same max_rate. Mirrors the intent of
    // profile.py's `_max_rate` (its graphs are parameterized, so it always
    // updates; a C++ caller may hand a constant graph).
    size_t pl = this->c_graph()->param_length;
    if (this->parameterized()) {
        std::vector<double> th;
        if (theta.empty()) {
            th.assign(pl, 1.0);
        } else {
            if (theta.size() != pl) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                    "profile: theta must have length param_length=%i; got %i",
                    (int) pl, (int) theta.size());
                throw std::runtime_error(msg);
            }
            th = theta;
        }
        this->update_weights(th);
    }
    double mr = 0.0;
    for (Vertex v : this->vertices()) {
        double r = (double) v.rate();
        if (r > mr) mr = r;
    }
    double auto_g = 2.0 * mr;

    // --- assemble ---
    GraphProfile p;
    p.n_vertices = this->vertices_length();
    p.n_edges = this->edges_length();
    p.param_length = pl;
    p.max_rate = mr;
    p.auto_granularity = auto_g;
    p.n_sccs = n;
    p.largest_scc = largest;
    p.largest_scc_frac = largest_frac;
    p.n_levels = levels.size();
    p.max_level_width = max_width;
    p.parallelizable_frac = parallelizable_frac;
    p.speedup_ceiling = speedup;
    p.cpu_count = cpu;

    char buf[512];

    if (cpu <= 1) {
        p.parallel_elimination = false;
        p.parallel_elimination_reason = "leave OFF (single core available)";
    } else if (max_width < 2) {
        p.parallel_elimination = false;
        p.parallel_elimination_reason =
            "leave OFF (condensation is a chain/one SCC — nothing independent "
            "to run in parallel)";
    } else if (parallelizable_frac < PARALLEL_MIN_FRAC) {
        p.parallel_elimination = false;
        snprintf(buf, sizeof(buf),
            "leave OFF (only %.0f%% of work is at width>=2 levels)",
            parallelizable_frac * 100.0);
        p.parallel_elimination_reason = buf;
    } else if (largest_frac > PARALLEL_MAX_DOMINANCE) {
        p.parallel_elimination = false;
        snprintf(buf, sizeof(buf),
            "marginal (one SCC is %.0f%% of vertices — Amdahl-bound, ceiling ~%.1fx)",
            largest_frac * 100.0, speedup);
        p.parallel_elimination_reason = buf;
    } else {
        p.parallel_elimination = true;
        snprintf(buf, sizeof(buf),
            "RECOMMEND ON (%zu independent SCCs at the widest level, %u cores; "
            "ceiling ~%.1fx; parallelizable_frac %.0f%%)",
            max_width, cpu, speedup, parallelizable_frac * 100.0);
        p.parallel_elimination_reason = buf;
    }

    p.dyn_ordering_reason =
        "not probed (the measured dyn-ordering probe is not implemented in the "
        "C++ API; run graph.profile() in Python for it)";

    if (auto_g >= PATH_GRANULARITY_WARN) {
        p.path = "sojourn";
        snprintf(buf, sizeof(buf),
            "forward-PDF will be SLOW per eval (max_rate %.3g -> granularity %.3g); "
            "IF your likelihood is over a joint/discrete index rather than continuous "
            "times, use the max_rate-independent joint/sojourn path", mr, auto_g);
        p.path_reason = buf;
    } else {
        p.path = "forward";
        snprintf(buf, sizeof(buf),
            "forward-PDF OK (max_rate %.3g -> granularity %.3g)", mr, auto_g);
        p.path_reason = buf;
    }

    return p;
}

std::string phasic::GraphProfile::apply_snippet() const {
    if (parallel_elimination) {
        return "phasic.configure(parallel_elimination=True)";
    }
    return "# defaults are fine for this graph";
}

std::string phasic::GraphProfile::report() const {
    std::ostringstream os;
    char buf[256];
    snprintf(buf, sizeof(buf),
        "phasic graph profile — %zu vertices, %zu edges, param_length=%zu",
        n_vertices, n_edges, param_length);
    os << buf << "\n";
    snprintf(buf, sizeof(buf),
        "  SCC structure : %zu SCCs, largest %zu (%.0f%%), %zu levels, widest level %zu",
        n_sccs, largest_scc, largest_scc_frac * 100.0, n_levels, max_level_width);
    os << buf << "\n";
    os << "  parallel_elim : " << parallel_elimination_reason << "\n";
    os << "  dyn_ordering  : " << dyn_ordering_reason << "\n";
    os << "  eval path     : " << path_reason << "\n";
    os << "  -> " << apply_snippet();
    return os.str();
}

// ===========================================================================
// weight_formula: port of src/phasic/weight_formula.py (tokenizer -> recursive-
// descent parser -> theta-independence guard -> bytecode compiler). Produces the
// exact same integer opcode tape + float const pool the Python front-end does,
// so the shared C VM (ptd_weight_tape_eval, consulted by ptd_graph_update_weights)
// evaluates identical per-edge weights. Opcodes MUST match weight_formula.OPCODES.
// ===========================================================================
namespace {

enum WFOp {
    WF_PUSH_THETA = 0, WF_PUSH_COEFF = 1, WF_PUSH_CONST = 2,
    WF_ADD = 3, WF_SUB = 4, WF_MUL = 5, WF_DIV = 6, WF_POW = 7,
    WF_NEG = 8, WF_EXP = 9, WF_LOG = 10, WF_SQRT = 11, WF_LOGISTIC = 12,
    WF_EQ = 13, WF_NE = 14, WF_LT = 15, WF_GT = 16, WF_LE = 17, WF_GE = 18,
    WF_AND = 19, WF_OR = 20, WF_NOT = 21, WF_SELECT = 22
};

struct WFToken { std::string kind; std::string text; };  // kind: NUMBER|NAME|OP|EOF

enum class WFTag { Const, Theta, Coeff, Unop, Binop, Select };
struct WFNode {
    WFTag tag;
    double cval = 0.0;   // Const
    int idx = 0;         // Theta / Coeff index
    std::string op;      // Unop/Binop op name
    std::vector<std::unique_ptr<WFNode>> kids;
};
typedef std::unique_ptr<WFNode> WFNodeP;

WFNodeP wf_node(WFTag t) { WFNodeP n(new WFNode()); n->tag = t; return n; }

phasic::WeightFormulaError wf_err(const std::string &msg) { return phasic::WeightFormulaError(msg); }

std::vector<WFToken> wf_tokenize(const std::string &src) {
    std::vector<WFToken> toks;
    size_t pos = 0, n = src.size();
    while (pos < n) {
        unsigned char c = (unsigned char) src[pos];
        if (isspace(c)) { pos++; continue; }
        // NUMBER: (\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?
        if (isdigit(c) || (c == '.' && pos + 1 < n && isdigit((unsigned char) src[pos + 1]))) {
            size_t start = pos;
            while (pos < n && isdigit((unsigned char) src[pos])) pos++;
            if (pos < n && src[pos] == '.') {
                pos++;
                while (pos < n && isdigit((unsigned char) src[pos])) pos++;
            }
            if (pos < n && (src[pos] == 'e' || src[pos] == 'E')) {
                size_t save = pos; pos++;
                if (pos < n && (src[pos] == '+' || src[pos] == '-')) pos++;
                if (pos < n && isdigit((unsigned char) src[pos])) {
                    while (pos < n && isdigit((unsigned char) src[pos])) pos++;
                } else {
                    pos = save;  // not a valid exponent; leave 'e' for the tokenizer to reject
                }
            }
            toks.push_back({"NUMBER", src.substr(start, pos - start)});
            continue;
        }
        // NAME: [A-Za-z_][A-Za-z0-9_]*
        if (isalpha(c) || c == '_') {
            size_t start = pos;
            while (pos < n && (isalnum((unsigned char) src[pos]) || src[pos] == '_')) pos++;
            toks.push_back({"NAME", src.substr(start, pos - start)});
            continue;
        }
        // OP: two-char operators first
        if (pos + 1 < n) {
            std::string two = src.substr(pos, 2);
            if (two == "**" || two == "==" || two == "!=" || two == "<=" || two == ">=") {
                toks.push_back({"OP", two}); pos += 2; continue;
            }
        }
        if (c && strchr("+-*/<>(),", (int) c)) {
            toks.push_back({"OP", std::string(1, (char) c)}); pos++; continue;
        }
        throw wf_err("weight_formula: invalid character '" + std::string(1, (char) c) +
                     "' at position " + std::to_string(pos) + " in '" + src + "'");
    }
    toks.push_back({"EOF", ""});
    return toks;
}

// arity of the callable functions
int wf_func_arity(const std::string &name) {
    if (name == "exp" || name == "log" || name == "sqrt" || name == "logistic" || name == "not") return 1;
    if (name == "pow" || name == "delta" || name == "and" || name == "or") return 2;
    if (name == "select") return 3;
    return -1;
}

struct WFParser {
    std::string src;
    std::vector<WFToken> toks;
    size_t i;
    explicit WFParser(const std::string &s) : src(s), toks(wf_tokenize(s)), i(0) {}

    const WFToken &peek() { return toks[i]; }
    WFToken next() { return toks[i++]; }
    bool peek_is(const std::string &op) {
        return toks[i].kind == "OP" && toks[i].text == op;
    }
    void expect_op(const std::string &op) {
        WFToken t = next();
        if (!(t.kind == "OP" && t.text == op)) {
            std::string found = t.text.empty() ? "end of input" : t.text;
            throw wf_err("weight_formula: expected '" + op + "' but found '" + found +
                         "' in '" + src + "'");
        }
    }

    WFNodeP parse() {
        WFNodeP node = comparison();
        if (peek().kind != "EOF") {
            throw wf_err("weight_formula: unexpected trailing '" + peek().text +
                         "' in '" + src + "'");
        }
        return node;
    }

    WFNodeP binop(const std::string &opname, WFNodeP l, WFNodeP r) {
        WFNodeP n = wf_node(WFTag::Binop);
        n->op = opname;
        n->kids.push_back(std::move(l));
        n->kids.push_back(std::move(r));
        return n;
    }

    WFNodeP comparison() {
        WFNodeP left = addsub();
        while (peek().kind == "OP") {
            const std::string &v = peek().text;
            std::string opn;
            if (v == "==") opn = "eq"; else if (v == "!=") opn = "ne";
            else if (v == "<") opn = "lt"; else if (v == ">") opn = "gt";
            else if (v == "<=") opn = "le"; else if (v == ">=") opn = "ge";
            else break;
            next();
            left = binop(opn, std::move(left), addsub());
        }
        return left;
    }
    WFNodeP addsub() {
        WFNodeP left = muldiv();
        while (peek().kind == "OP" && (peek().text == "+" || peek().text == "-")) {
            std::string opn = (next().text == "+") ? "add" : "sub";
            left = binop(opn, std::move(left), muldiv());
        }
        return left;
    }
    WFNodeP muldiv() {
        WFNodeP left = unary();
        while (peek().kind == "OP" && (peek().text == "*" || peek().text == "/")) {
            std::string opn = (next().text == "*") ? "mul" : "div";
            left = binop(opn, std::move(left), unary());
        }
        return left;
    }
    WFNodeP unary() {
        if (peek_is("-")) {
            next();
            WFNodeP n = wf_node(WFTag::Unop);
            n->op = "neg";
            n->kids.push_back(unary());
            return n;
        }
        return power();
    }
    WFNodeP power() {
        WFNodeP base = atom();
        if (peek_is("**")) {
            next();
            return binop("pow", std::move(base), unary());  // right-assoc; exponent may be unary
        }
        return base;
    }
    WFNodeP atom() {
        WFToken t = next();
        if (t.kind == "NUMBER") {
            WFNodeP n = wf_node(WFTag::Const);
            n->cval = strtod(t.text.c_str(), NULL);
            return n;
        }
        if (t.kind == "OP" && t.text == "(") {
            WFNodeP node = comparison();
            expect_op(")");
            return node;
        }
        if (t.kind == "NAME") {
            if (peek_is("(")) return call(t.text);
            // t<i> / c<j> variable reference
            if (t.text.size() >= 2 && (t.text[0] == 't' || t.text[0] == 'c')) {
                bool all_digits = true;
                for (size_t k = 1; k < t.text.size(); ++k)
                    if (!isdigit((unsigned char) t.text[k])) { all_digits = false; break; }
                if (all_digits) {
                    WFNodeP n = wf_node(t.text[0] == 't' ? WFTag::Theta : WFTag::Coeff);
                    n->idx = (int) strtol(t.text.c_str() + 1, NULL, 10);
                    return n;
                }
            }
            if (wf_func_arity(t.text) >= 0) {
                throw wf_err("weight_formula: function '" + t.text +
                             "' used without arguments in '" + src + "'");
            }
            throw wf_err("weight_formula: unknown identifier '" + t.text + "' in '" + src +
                         "'. Use t<i> for theta, c<j> for coefficients, or a known function "
                         "(exp, log, sqrt, logistic, pow, delta, select, and, or, not).");
        }
        std::string what = t.text.empty() ? "end of input" : t.text;
        throw wf_err("weight_formula: unexpected '" + what + "' in '" + src + "'");
    }
    WFNodeP call(const std::string &name) {
        int arity = wf_func_arity(name);
        if (arity < 0)
            throw wf_err("weight_formula: unknown function '" + name + "' in '" + src + "'");
        expect_op("(");
        std::vector<WFNodeP> args;
        if (!peek_is(")")) {
            args.push_back(comparison());
            while (peek_is(",")) { next(); args.push_back(comparison()); }
        }
        expect_op(")");
        if ((int) args.size() != arity) {
            throw wf_err("weight_formula: " + name + "() takes " + std::to_string(arity) +
                         " argument(s) but got " + std::to_string(args.size()) + " in '" + src + "'");
        }
        if (name == "exp" || name == "log" || name == "sqrt" || name == "logistic" || name == "not") {
            WFNodeP n = wf_node(WFTag::Unop); n->op = name;
            n->kids.push_back(std::move(args[0]));
            return n;
        }
        if (name == "pow") return binop("pow", std::move(args[0]), std::move(args[1]));
        if (name == "delta") return binop("eq", std::move(args[0]), std::move(args[1]));  // sugar for ==
        if (name == "and" || name == "or") return binop(name, std::move(args[0]), std::move(args[1]));
        if (name == "select") {
            WFNodeP n = wf_node(WFTag::Select);
            n->kids.push_back(std::move(args[0]));
            n->kids.push_back(std::move(args[1]));
            n->kids.push_back(std::move(args[2]));
            return n;
        }
        throw wf_err("weight_formula: internal error on function '" + name + "'");
    }
};

bool wf_uses_theta(const WFNode *n) {
    switch (n->tag) {
        case WFTag::Theta: return true;
        case WFTag::Const:
        case WFTag::Coeff: return false;
        default:
            for (const WFNodeP &k : n->kids) if (wf_uses_theta(k.get())) return true;
            return false;
    }
}

void wf_check_theta_indep(const WFNode *n, const std::string &where, const std::string &src) {
    if (wf_uses_theta(n)) {
        throw wf_err("weight_formula: " + where + " must be theta-independent (it may use c<j> "
                     "and constants but not t<i>) in '" + src + "'. Comparisons/indicators are "
                     "non-differentiable and SVGD gradients are finite differences, so a "
                     "theta-dependent condition would sit on the gradient path. For smooth "
                     "theta-gating use logistic(...) instead.");
    }
}

bool wf_is_comparison(const std::string &op) {
    return op == "eq" || op == "ne" || op == "lt" || op == "gt" || op == "le" || op == "ge";
}

void wf_enforce_guard(const WFNode *n, const std::string &src) {
    switch (n->tag) {
        case WFTag::Const: case WFTag::Theta: case WFTag::Coeff:
            return;
        case WFTag::Unop:
            if (n->op == "not") wf_check_theta_indep(n->kids[0].get(), "the operand of not()", src);
            wf_enforce_guard(n->kids[0].get(), src);
            return;
        case WFTag::Binop:
            if (wf_is_comparison(n->op)) {
                wf_check_theta_indep(n->kids[0].get(), "the left operand of a comparison", src);
                wf_check_theta_indep(n->kids[1].get(), "the right operand of a comparison", src);
            } else if (n->op == "and" || n->op == "or") {
                wf_check_theta_indep(n->kids[0].get(), "an operand of " + n->op + "()", src);
                wf_check_theta_indep(n->kids[1].get(), "an operand of " + n->op + "()", src);
            }
            wf_enforce_guard(n->kids[0].get(), src);
            wf_enforce_guard(n->kids[1].get(), src);
            return;
        case WFTag::Select:
            wf_check_theta_indep(n->kids[0].get(), "the condition of select()", src);
            wf_enforce_guard(n->kids[0].get(), src);
            wf_enforce_guard(n->kids[1].get(), src);
            wf_enforce_guard(n->kids[2].get(), src);
            return;
    }
}

int wf_unop_opcode(const std::string &op) {
    if (op == "neg") return WF_NEG; if (op == "exp") return WF_EXP; if (op == "log") return WF_LOG;
    if (op == "sqrt") return WF_SQRT; if (op == "logistic") return WF_LOGISTIC; if (op == "not") return WF_NOT;
    return -1;
}
int wf_binop_opcode(const std::string &op) {
    if (op == "add") return WF_ADD; if (op == "sub") return WF_SUB; if (op == "mul") return WF_MUL;
    if (op == "div") return WF_DIV; if (op == "pow") return WF_POW;
    if (op == "eq") return WF_EQ; if (op == "ne") return WF_NE; if (op == "lt") return WF_LT;
    if (op == "gt") return WF_GT; if (op == "le") return WF_LE; if (op == "ge") return WF_GE;
    if (op == "and") return WF_AND; if (op == "or") return WF_OR;
    return -1;
}

int wf_const_index(double v, std::vector<double> &consts, std::map<double, int> &cmap) {
    std::map<double, int>::iterator it = cmap.find(v);
    if (it != cmap.end()) return it->second;
    int ix = (int) consts.size();
    consts.push_back(v);
    cmap[v] = ix;
    return ix;
}

void wf_emit(const WFNode *n, std::vector<int> &ops, std::vector<double> &consts,
             std::map<double, int> &cmap, int &n_theta, int &n_coeff) {
    switch (n->tag) {
        case WFTag::Const:
            ops.push_back(WF_PUSH_CONST); ops.push_back(wf_const_index(n->cval, consts, cmap));
            return;
        case WFTag::Theta:
            n_theta = std::max(n_theta, n->idx + 1);
            ops.push_back(WF_PUSH_THETA); ops.push_back(n->idx);
            return;
        case WFTag::Coeff:
            n_coeff = std::max(n_coeff, n->idx + 1);
            ops.push_back(WF_PUSH_COEFF); ops.push_back(n->idx);
            return;
        case WFTag::Unop:
            wf_emit(n->kids[0].get(), ops, consts, cmap, n_theta, n_coeff);
            ops.push_back(wf_unop_opcode(n->op));
            return;
        case WFTag::Binop:
            wf_emit(n->kids[0].get(), ops, consts, cmap, n_theta, n_coeff);
            wf_emit(n->kids[1].get(), ops, consts, cmap, n_theta, n_coeff);
            ops.push_back(wf_binop_opcode(n->op));
            return;
        case WFTag::Select:
            wf_emit(n->kids[0].get(), ops, consts, cmap, n_theta, n_coeff);
            wf_emit(n->kids[1].get(), ops, consts, cmap, n_theta, n_coeff);
            wf_emit(n->kids[2].get(), ops, consts, cmap, n_theta, n_coeff);
            ops.push_back(WF_SELECT);
            return;
    }
}

int wf_stack_depth(const WFNode *n) {
    switch (n->tag) {
        case WFTag::Const: case WFTag::Theta: case WFTag::Coeff:
            return 1;
        case WFTag::Unop:
            return wf_stack_depth(n->kids[0].get());
        case WFTag::Binop:
            return std::max(wf_stack_depth(n->kids[0].get()), 1 + wf_stack_depth(n->kids[1].get()));
        case WFTag::Select:
            return std::max(wf_stack_depth(n->kids[0].get()),
                            std::max(1 + wf_stack_depth(n->kids[1].get()),
                                     2 + wf_stack_depth(n->kids[2].get())));
    }
    return 1;
}

}  // anonymous namespace

// Python-API-name parity: compile a per-edge weight formula string and install it
// as the graph's weight tape, so update_weights(theta) evaluates it per edge in C.
// Mirrors Python's `graph.weight_formula = "..."`. See weight_formula.py.
void phasic::Graph::weight_formula(const std::string &formula) {
    // non-empty check (mirrors compile_formula)
    bool blank = true;
    for (char ch : formula) if (!isspace((unsigned char) ch)) { blank = false; break; }
    if (blank) throw WeightFormulaError("weight_formula must be a non-empty string");

    WFParser parser(formula);
    WFNodeP ast = parser.parse();
    wf_enforce_guard(ast.get(), formula);

    std::vector<int> ops;
    std::vector<double> consts;
    std::map<double, int> cmap;
    int n_theta = 0, n_coeff = 0;
    wf_emit(ast.get(), ops, consts, cmap, n_theta, n_coeff);
    int stack_depth = wf_stack_depth(ast.get());

    struct ptd_weight_tape *tape = ptd_weight_tape_create(
        ops.data(), ops.size(),
        consts.empty() ? NULL : consts.data(), consts.size(),
        (size_t) stack_depth, (size_t) n_theta, (size_t) n_coeff);
    if (tape == NULL) {
        throw std::runtime_error("weight_formula: failed to allocate weight tape");
    }
    ptd_graph_set_weight_tape(this->c_graph(), tape);  // takes ownership, frees any prior
    this->_weight_formula_src = formula;
}

std::string phasic::Graph::weight_formula() const {
    return this->_weight_formula_src;
}

// Python-API-name parity: serialize the graph to arrays. Faithful port of the
// topological part of Python Graph.serialize() (parameterized edges are pulled
// out first and their (from,to) pairs excluded from the regular/constant edge
// lists; coefficient-less edges go to constant_edges). theta_dim = the graph's
// param_length. Python-runtime metadata (weight_mode/dyn_ordering/tape) is not
// carried.
phasic::SerializedGraph phasic::Graph::serialize() {
    std::vector<Vertex> verts = this->vertices();
    size_t n = verts.size();
    size_t sl = this->state_length();
    size_t pl = this->c_graph()->param_length;  // theta_dim

    SerializedGraph out;
    out.state_length = sl;
    out.n_vertices = n;
    out.param_length = pl;
    out.states.assign(n, std::vector<int>(sl));
    out.vertex_indices.assign(n, 0);

    std::map<size_t, size_t> idx_to_enum;  // C vertex index -> enumeration position
    for (size_t i = 0; i < n; ++i) {
        std::vector<int> st = verts[i].state();
        for (size_t j = 0; j < sl; ++j) out.states[i][j] = st[j];
        size_t vi = verts[i].c_vertex()->index;
        out.vertex_indices[i] = vi;
        idx_to_enum[vi] = i;
    }

    Vertex start = this->starting_vertex();
    size_t start_vidx = start.c_vertex()->index;

    // Parameterized edges first (and record their (from,to) pairs).
    std::set<std::pair<size_t, size_t>> param_pairs;
    if (pl > 0) {
        for (size_t i = 0; i < n; ++i) {
            if (verts[i].c_vertex()->index == start_vidx) continue;
            std::vector<ParameterizedEdge> pes = verts[i].parameterized_edges();
            for (ParameterizedEdge &pe : pes) {
                size_t to_vi = pe.to().c_vertex()->index;
                std::map<size_t, size_t>::iterator it = idx_to_enum.find(to_vi);
                if (it == idx_to_enum.end()) continue;
                size_t to = it->second;
                size_t clen = pe.coefficients_length();
                std::vector<double> es = pe.edge_state(clen);
                if (!es.empty()) {
                    std::vector<double> row;
                    row.reserve(2 + es.size());
                    row.push_back((double) i);
                    row.push_back((double) to);
                    for (double c : es) row.push_back(c);
                    out.param_edges.push_back(row);
                    param_pairs.insert(std::make_pair(i, to));
                }
            }
        }
    }

    // Regular (coefficient-carrying) and constant (coefficient-less) edges.
    for (size_t i = 0; i < n; ++i) {
        if (verts[i].c_vertex()->index == start_vidx) continue;
        std::vector<Edge> es = verts[i].edges();
        for (Edge &e : es) {
            size_t to_vi = e.to().c_vertex()->index;
            std::map<size_t, size_t>::iterator it = idx_to_enum.find(to_vi);
            if (it == idx_to_enum.end()) continue;
            size_t to = it->second;
            if (param_pairs.count(std::make_pair(i, to))) continue;
            std::vector<double> row;
            row.push_back((double) i);
            row.push_back((double) to);
            row.push_back(e.weight());
            if (e.coefficients_length() == 0) out.constant_edges.push_back(row);
            else out.edges.push_back(row);
        }
    }

    // Starting-vertex edges (never parameterized).
    std::vector<Edge> ses = start.edges();
    for (Edge &e : ses) {
        size_t to_vi = e.to().c_vertex()->index;
        std::map<size_t, size_t>::iterator it = idx_to_enum.find(to_vi);
        if (it == idx_to_enum.end()) continue;
        std::vector<double> row;
        row.push_back((double) it->second);
        row.push_back(e.weight());
        out.start_edges.push_back(row);
    }

    return out;
}

// Python-API-name parity: rebuild a graph from serialize() output. Faithful port
// of Python Graph.from_serialized(): create the vertices (reusing the starting
// vertex where the recorded C index matches), then add parameterized, regular,
// and starting edges. Coefficient-less constant_edges are not re-added (matching
// Python).
phasic::Graph phasic::Graph::from_serialized(const SerializedGraph &data) {
    size_t n = data.n_vertices;
    size_t sl = data.state_length;
    size_t pl = data.param_length;

    size_t coeff_len = pl;
    if (!data.param_edges.empty() && data.param_edges[0].size() >= 2) {
        coeff_len = data.param_edges[0].size() - 2;
    }

    Graph graph(sl);
    // Pin param_length up front only when edges carry MORE coefficients than
    // param_length (decoupled models), matching Python; otherwise let the first
    // parameterized edge lock it.
    if (pl > 0 && coeff_len > pl) {
        graph.set_param_length(pl);
    }

    Vertex start = graph.starting_vertex();
    size_t start_c_idx = start.c_vertex()->index;

    std::vector<Vertex> idx_to_vertex;
    idx_to_vertex.reserve(n);
    for (size_t idx = 0; idx < n; ++idx) {
        if (data.vertex_indices[idx] == start_c_idx) {
            idx_to_vertex.push_back(start);
        } else {
            idx_to_vertex.push_back(graph.find_or_create_vertex(data.states[idx]));
        }
    }

    for (const std::vector<double> &row : data.param_edges) {
        size_t from = (size_t) row[0];
        size_t to = (size_t) row[1];
        std::vector<double> es(row.begin() + 2, row.end());
        idx_to_vertex[from].add_edge_parameterized(idx_to_vertex[to], 0.0, es);
    }
    for (const std::vector<double> &row : data.edges) {
        size_t from = (size_t) row[0];
        size_t to = (size_t) row[1];
        idx_to_vertex[from].add_edge(idx_to_vertex[to], row[2]);
    }
    for (const std::vector<double> &row : data.start_edges) {
        size_t to = (size_t) row[0];
        start.add_edge(idx_to_vertex[to], row[1]);
    }

    // Ensure param_length is set after all edges (mirrors Python's trailing
    // set_param_length wrapped in try/except: it can fail if the value is already
    // locked, which is fine).
    if (pl > 0) {
        try {
            graph.set_param_length(pl);
        } catch (const std::exception &) {
            // already locked by the first parameterized edge — matches Python
        }
    }

    return graph;
}

phasic::PhaseTypeDistribution phasic::Graph::phase_type_distribution() {
    struct ptd_phase_type_distribution *matrix = ptd_graph_as_phase_type_distribution(this->rf_graph->graph);

    if (matrix == NULL) {
        char msg[1024];

        snprintf(msg, 1024, "Failed to make sub-intensity matrix: %s \n", std::strerror(errno));

        throw new std::runtime_error(
                msg
        );
    }

    return PhaseTypeDistribution(*this, matrix);
}

void phasic::Vertex::add_edge(Vertex &to, double weight) {
    if (this->vertex == to.vertex) {
        throw new std::invalid_argument(
                "The edge to add is between the same vertex\n"
        );
    }

    // EDGE MODE LOCKING: Lock to CONSTANT mode on first non-IPV edge with scalar syntax
    // IPV (starting vertex) edges don't affect mode locking
    if (this->vertex != this->vertex->graph->starting_vertex) {
        if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
            // First non-IPV edge: lock to CONSTANT mode
            this->vertex->graph->edge_mode = PTD_EDGE_MODE_CONSTANT;
        } else if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_PARAMETERIZED) {
            // Graph is locked to PARAMETERIZED, reject scalar syntax
            throw std::runtime_error(
                "Cannot mix constant and parameterized edges. "
                "Graph mode is PARAMETERIZED (locked by first non-IPV edge using array syntax). "
                "This edge uses scalar syntax. "
                "Use add_edge(vertex, [coefficients]) for parameterized edges."
            );
        }
    }

    graph.notify_change();

    // Constant edge: single-element coefficient array
    double coeff = weight;
    struct ptd_edge *result = ptd_graph_add_edge(this->vertex, to.vertex, &coeff, 1);

    if (result == NULL) {
        PTD_THROW_AND_CLEAR();
    }
}


void phasic::Vertex::add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state) {
    if (this->vertex == to.vertex) {
        throw new std::invalid_argument(
                "The edge to add is between the same vertex\n"
        );
    }

    // EDGE MODE LOCKING: Lock to PARAMETERIZED mode on first non-IPV edge with array syntax
    // IPV (starting vertex) edges don't affect mode locking
    if (this->vertex != this->vertex->graph->starting_vertex) {
        if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
            // First non-IPV edge: lock to PARAMETERIZED mode
            this->vertex->graph->edge_mode = PTD_EDGE_MODE_PARAMETERIZED;
        } else if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_CONSTANT) {
            // Graph is locked to CONSTANT, reject array syntax
            throw std::runtime_error(
                "Cannot mix constant and parameterized edges. "
                "Graph mode is CONSTANT (locked by first non-IPV edge using scalar syntax). "
                "This edge uses array syntax. "
                "Use add_edge(vertex, scalar) for constant edges."
            );
        }
    }

    size_t state_length = edge_state.size();
    double *state = (double *) calloc(state_length, sizeof(*state));

    for (size_t i = 0; i < state_length; ++i) {
        state[i] = edge_state[i];
    }

    graph.notify_change();

    // Unified API: use coefficient array directly
    struct ptd_edge *result = ptd_graph_add_edge(this->vertex, to.vertex, state, state_length);

    free(state);

    if (result == NULL) {
        PTD_THROW_AND_CLEAR();
    }
}


phasic::Vertex phasic::Vertex::add_aux_vertex(double rate) {
    // Create all-zero state vector
    size_t state_len = this->vertex->graph->state_length;
    std::vector<int> zero_state(state_len, 0);

    // create the aux vertex
    Vertex aux = graph.create_vertex(zero_state);

    // Edge 1: FROM aux TO this vertex with constant weight 1.0
    // Create edge manually to bypass validation (always constant weight 1.0)
    struct ptd_edge *edge1 = (struct ptd_edge *)malloc(sizeof(*edge1));
    if (edge1 == NULL) {
        throw std::runtime_error("Failed to allocate edge");
    }

    edge1->to = this->vertex;
    edge1->weight = 1.0;
    edge1->coefficients_length = 0;  // No coefficients - pure constant
    edge1->coefficients = NULL;
    edge1->should_free_coefficients = false;

    // Add edge to aux vertex's edge list
    struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
        aux.vertex->edges,
        (aux.vertex->edges_length + 1) * sizeof(struct ptd_edge *)
    );
    if (new_edges == NULL) {
        free(edge1);
        throw std::runtime_error("Failed to allocate edge array");
    }
    aux.vertex->edges = new_edges;
    aux.vertex->edges[aux.vertex->edges_length] = edge1;
    aux.vertex->edges_length++;

    // Edge 2: FROM this vertex TO aux with given rate (constant)
    // Use normal add_edge for proper validation
    this->add_edge(aux, rate);

    graph.notify_change();

    return aux;
}


phasic::Vertex phasic::Vertex::add_aux_vertex(std::vector<double> rate_coeffs) {
    // Create all-zero state vector
    size_t state_len = this->vertex->graph->state_length;
    std::vector<int> zero_state(state_len, 0);

    // create the aux vertex
    Vertex aux = graph.create_vertex(zero_state);

    // Edge 1: FROM aux TO this vertex with constant weight 1.0
    // Create edge manually to bypass validation (always constant weight 1.0)
    struct ptd_edge *edge1 = (struct ptd_edge *)malloc(sizeof(*edge1));
    if (edge1 == NULL) {
        throw std::runtime_error("Failed to allocate edge");
    }

    edge1->to = this->vertex;
    edge1->weight = 1.0;
    edge1->coefficients_length = 0;  // No coefficients - pure constant
    edge1->coefficients = NULL;
    edge1->should_free_coefficients = false;

    // Add edge to aux vertex's edge list
    struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
        aux.vertex->edges,
        (aux.vertex->edges_length + 1) * sizeof(struct ptd_edge *)
    );
    if (new_edges == NULL) {
        free(edge1);
        throw std::runtime_error("Failed to allocate edge array");
    }
    aux.vertex->edges = new_edges;
    aux.vertex->edges[aux.vertex->edges_length] = edge1;
    aux.vertex->edges_length++;

    // Edge 2: FROM this vertex TO aux with given rate (parameterized)
    // Use normal add_edge_parameterized for proper validation
    this->add_edge_parameterized(aux, 0.0, rate_coeffs);

    graph.notify_change();

    return aux;
}


phasic::Vertex phasic::Vertex::add_aux_vertex_constant(double weight) {
    // Create an aux vertex and bidirectional coefficient-less constant
    // edges. Both directions have coefficients_length == 0, so
    // ptd_graph_update_weights skips them (see src/c/phasic.c around
    // L4435-4439). The weights are hardcoded and remain constant
    // regardless of any later update_weights() call.
    //
    // This bypasses the EDGE_MODE_PARAMETERIZED lock for the v->aux
    // direction by manipulating the ptd_edge struct directly,
    // mirroring the trick used in add_aux_vertex(double rate) for the
    // aux->v return edge. Used by joint_stop_prob_graph() to install
    // trapping loops on parameterised JSP graphs without making the
    // trapping rate depend on theta (which would blow up λ_max under
    // per-observation exposure scaling).
    if (!(weight > 0.0)) {
        throw std::invalid_argument(
            "add_aux_vertex_constant: weight must be strictly positive"
        );
    }

    size_t state_len = this->vertex->graph->state_length;
    std::vector<int> zero_state(state_len, 0);

    // create the aux vertex
    Vertex aux = graph.create_vertex(zero_state);

    // Helper lambda: append a manually constructed constant edge to a
    // vertex's edge list, bypassing all add_edge validation.
    auto append_constant_edge =
        [](struct ptd_vertex *from_v, struct ptd_vertex *to_v, double w) {
            struct ptd_edge *edge =
                (struct ptd_edge *)malloc(sizeof(*edge));
            if (edge == NULL) {
                throw std::runtime_error("Failed to allocate edge");
            }
            edge->to = to_v;
            edge->weight = w;
            edge->coefficients_length = 0;
            edge->coefficients = NULL;
            edge->should_free_coefficients = false;

            struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
                from_v->edges,
                (from_v->edges_length + 1) * sizeof(struct ptd_edge *)
            );
            if (new_edges == NULL) {
                free(edge);
                throw std::runtime_error("Failed to allocate edge array");
            }
            from_v->edges = new_edges;
            from_v->edges[from_v->edges_length] = edge;
            from_v->edges_length++;
        };

    // Edge 1: aux -> this vertex (constant weight, coefficient-less).
    append_constant_edge(aux.vertex, this->vertex, weight);

    // Edge 2: this vertex -> aux (constant weight, coefficient-less).
    // Bypasses the EDGE_MODE_PARAMETERIZED lock; this is the whole
    // point of the new method.
    append_constant_edge(this->vertex, aux.vertex, weight);

    graph.notify_change();

    return aux;
}


std::vector<int> phasic::Vertex::state() {
    return std::vector<int>(
            this->vertex->state,
            this->vertex->state + this->vertex->graph->state_length
    );
}

std::vector<phasic::Edge> phasic::Vertex::edges() {
    std::vector<Edge> vector;

    for (size_t i = 0; i < this->vertex->edges_length; ++i) {
        Edge edge_i(
                this->vertex->edges[i]->to,
                this->vertex->edges[i],
                graph,
                this->vertex->edges[i]->weight
        );

        vector.push_back(edge_i);
    }

    return vector;
}

std::vector<phasic::ParameterizedEdge> phasic::Vertex::parameterized_edges() {
    std::vector<ParameterizedEdge> vector;

    for (size_t i = 0; i < this->vertex->edges_length; ++i) {
        // Include edges with coefficient arrays (parameterized in unified interface)
        // This includes single-parameter edges (coefficients_length == 1)
        if (this->vertex->edges[i]->coefficients_length >= 1) {
            ParameterizedEdge edge_i(
                    this->vertex->edges[i]->to,
                    this->vertex->edges[i],
                    graph,
                    this->vertex->edges[i]->weight,
                    this->vertex->edges[i]->coefficients
            );

            vector.push_back(edge_i);
        }
    }

    return vector;
}

// phasic::Graph phasic::Graph::expectation_dag(std::vector<double> rewards) {
//     struct ptd_clone_res res = ptd_graph_expectation_dag(this->c_graph(), &rewards[0]);

//     if (res.graph == NULL) {
//         throw std::runtime_error((char *) ptd_err);
//     }

//     return Graph(res.graph, res.avl_tree);
// }

// phasic::Graph *phasic::Graph::expectation_dag_p(std::vector<double> rewards) {
//   struct ptd_clone_res res = ptd_graph_expectation_dag(this->c_graph(), &rewards[0]);
  
//   if (res.graph == NULL) {
//     throw std::runtime_error((char *) ptd_err);
//   }
  
//   return new Graph(res.graph, res.avl_tree);
// }

phasic::Graph phasic::Graph::reward_transform(std::vector<double> rewards) {
    struct ptd_graph *res = ptd_graph_reward_transform(this->c_graph(), &rewards[0]);

    if (res == NULL) {
        PTD_THROW_AND_CLEAR();
    }

    return Graph(res);
}

phasic::Graph *phasic::Graph::reward_transform_p(std::vector<double> rewards) {
  struct ptd_graph *res = ptd_graph_reward_transform(this->c_graph(), &rewards[0]);

  if (res == NULL) {
    PTD_THROW_AND_CLEAR();
  }

  return new Graph(res);
}

void phasic::Graph::update_weights_parameterized(std::vector<double> scalars, bool use_log) {
    ptd_graph_update_weights(
            this->c_graph(),
            &scalars[0],
            scalars.size(),
            use_log
    );

    // Check if error occurred and throw exception
    if (ptd_err[0] != '\0') {
        std::string error_msg((const char*)ptd_err);
        ptd_err[0] = '\0';  // Clear error
        throw std::runtime_error(error_msg);
    }

    notify_change();
}

void phasic::Graph::update_ipv(std::vector<double> ipv) {
    ptd_graph_update_ipv(
            this->c_graph(),
            ipv.empty() ? NULL : &ipv[0],
            ipv.size()
    );

    if (ptd_err[0] != '\0') {
        std::string error_msg((const char*)ptd_err);
        ptd_err[0] = '\0';
        throw std::runtime_error(error_msg);
    }

    notify_change();
}

// Callback-based weight update
void phasic::Graph::update_weights_parameterized(
    std::vector<double> scalars,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> callback
) {
    // Validate graph has edges
    struct ptd_graph *graph = this->c_graph();
    if (graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
        throw std::runtime_error(
            "Cannot call update_weights() on empty graph (no edges added yet). "
            "Add edges using add_edge() before calling update_weights()."
        );
    }

    // Validate graph is parameterized
    if (graph->edge_mode == PTD_EDGE_MODE_CONSTANT) {
        throw std::runtime_error(
            "Cannot call update_weights() on constant graph. "
            "Graph has constant edges (created with scalar syntax: add_edge(v, 3.0)). "
            "Use parameterized edges (array syntax: add_edge(v, [3.0])) if you need to update weights."
        );
    }

    // NOTE: Parameter length is NOT validated against graph->param_length in callback mode.
    // The callback owns the theta→weight mapping and receives both theta and the edge's
    // coefficient array as-is. This allows theta to be longer than param_length (e.g.
    // PSMC-style epoch models where each edge uses a different slice of theta) or shorter
    // (e.g. constant callbacks that ignore theta entirely). Coefficient length is likewise
    // not validated here — the callback handles indexing.

    // Iterate through all vertices and edges
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *vertex = graph->vertices[i];

        // Skip starting vertex edges
        if (vertex == graph->starting_vertex) {
            continue;
        }

        for (size_t j = 0; j < vertex->edges_length; j++) {
            struct ptd_edge *edge = vertex->edges[j];

            // Skip edges with no coefficients (pure constant edges)
            if (edge->coefficients_length == 0) {
                continue;
            }

            // Get coefficient vector
            std::vector<double> coeffs(edge->coefficients_length);
            for (size_t k = 0; k < edge->coefficients_length; k++) {
                coeffs[k] = edge->coefficients[k];
            }

            // Call callback to compute weight
            double new_weight = callback(scalars, coeffs);

            // Validate weight is positive
            if (new_weight <= 0.0) {
                throw std::runtime_error(
                    "Callback returned non-positive weight: " + std::to_string(new_weight) +
                    ". All edge weights must be strictly positive."
                );
            }

            // Update edge weight directly
            edge->weight = new_weight;
        }
    }

    notify_change();
}

#endif