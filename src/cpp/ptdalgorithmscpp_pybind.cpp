


#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

// FIXME: Had to:
// cd ~/miniconda3/envs/phasetype/include
// ln -s eigen3/Eigen
#include <eigen3/Eigen/Core>

#include "ptdalgorithmscpp.h"

#include <deque>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using std::deque;
using std::vector;
using std::tuple;
using std::deque;
using std::endl;


PYBIND11_MODULE(ptdalgorithmscpp_pybind, m) {

  m.doc() = "These are the docs";


  /* Bind MatrixXd (or some other Eigen type) to Python */
  typedef Eigen::MatrixXd Matrix;

  typedef Matrix::Scalar Scalar;
  constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

  py::class_<Matrix>(m, "Matrix", py::buffer_protocol())

    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<Scalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));

        auto map = Eigen::Map<Matrix, 0, Strides>(
            static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return Matrix(map);
    }))

    .def_buffer([](Matrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                /* Pointer to buffer */
        sizeof(Scalar),                          /* Size of one scalar */
        py::format_descriptor<Scalar>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { m.rows(), m.cols() },                  /* Buffer dimensions */
        { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
          sizeof(Scalar) * (rowMajor ? 1 : m.rows()) }
                                                 /* Strides (in bytes) for each index */
    );
 })
  ;


  py::class_<ptdalgorithms::Graph>(m, "Graph")

    .def(py::init<int>(), py::arg("state_length"), R"delim(
        Create a graph representing a phase-type distribution

        @description
        `create_graph` creates a graph representing a phase-type distribution.
        This is the primary entry-point of the library.

        @details
        There will *always* be a starting vertex added to
        the graph.

        Notice that when the library functions are invoked on
        this object, the object is *mutated*, i.e. changed, which
        may be surprising considering the normal behavior of R
        objects.

        @return Simple reference to a CPP object.

        @param state_length The length of the integer vector used to represent and reference a state.

        @examples
        graph <- create_graph(4)
        [Rcpp::export]]

        Parameters
        ----------
      )delim")

    .def(py::init<struct ::ptd_graph* >(), py::arg("ptd_graph"), R"delim(

      )delim")

    .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"), R"delim(

      )delim")
      
    .def(py::init<const ptdalgorithms::Graph>(), py::arg("o"), R"delim(

      )delim")

    .def("update_weights_parameterized", &ptdalgorithms::Graph::update_weights_parameterized, py::arg("rewards"), R"delim(
//' Updates all parameterized edges of the graph by given scalars.
//' 
//' @description
//' Given a vector of scalars, computes a new weight of
//' the parameterized edges in the graph by a simple inner
//' product of the edge state vector and the scalar vector.
//' 
//' @details
//' A faster and simpler version to compute new moments, when
//' the user wants to try multiple different weights.
//'
//' 
//' @param phase_type_graph A reference to the graph created by [ptdalgorithms::create_graph()]
//' @param scalars A numeric vector of multiplies for the edge states.
//' 
//' @seealso [ptdalgorithms::expected_waiting_time()]
//' @seealso [ptdalgorithms::add_edge()]
//' 
//' @examples
//' graph <- create_graph(4)
//' v1 <- find_or_create_vertex(graph, c(1,2,1,0))
//' v2 <- find_or_create_vertex(graph, c(2,0,1,0))
//' add_edge(starting_vertex(graph), v1, 5)
//' add_edge(v1, v2, 0, c(5,2))
//' edges(starting_vertex(graph))[[1]]$weight # => 5
//' edges(v1)[[1]]$weight # => 0
//' graph_update_weights_parameterized(graph, c(9,7))
//' edges(starting_vertex(graph))[[1]]$weight # => 5
//' edges(v1)[[1]]$weight # => 59
      )delim")

    .def("expected_waiting_time", &ptdalgorithms::Graph::expected_waiting_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(

      )delim")
      
    .def("expected_residence_time", &ptdalgorithms::Graph::expected_residence_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(

      )delim")
      
    .def("create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("find_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("vertex_exists", static_cast<bool (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::vertex_exists), py::arg("state"), 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("find_or_create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
//' Find or create a vertex matching `state`
//' 
//' @description
//' Finds a vertex by the `state` parameter. If no such
//' vertex exists, it creates the vertex and adds it to
//' the graph object instead.
//' 
//' @details
//' A faster and simpler version of calling [ptdalgorithms::find_vertex()] and  [ptdalgorithms::create_vertex()]
//' 
//' @return The newly found or inserted vertex in the graph
//' 
//' @param phase_type_graph A reference to the graph created by [ptdalgorithms::create_graph()]
//' @param state An integer vector of what vertex to look for. Has length as given by `state_length` in  [ptdalgorithms::create_graph()]
//' 
//' @examples
//' graph <- create_graph(4)
//' find_or_create_vertex(graph, c(1,2,1,0)) # Adds and returns the vertex
//' find_or_create_vertex(graph, c(1,2,1,0)) # Only returns the vertex
//' # `graph` is now changed permanently
      )delim")
      
    .def("focv", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Alias for find_or_create_vertex
      )delim")      

    .def("starting_vertex", &ptdalgorithms::Graph::starting_vertex, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("vertices", &ptdalgorithms::Graph::vertices, 
      py::return_value_policy::copy, R"delim(
//' Obtain a list of all vertices in the graph
//' 
//' @description
//' Returns all vertices that have been added to the
//' graph from either calling `find_or_create_vertex` or
//' `create_vertex`. The first vertex in the list is
//' *always* the starting vertex [ptdalgorithms::starting_vertex()].
//' Importantly, for speed, use [ptdalgorithms::vertices_length()] to get the number
//' of added vertices, and use [ptdalgorithms::vertex_at()] to
//' get a vertex at a particular index.
//' 
//' @details
//' The list of vertices contains any added vertex, even
//' if it does not have any in-going / out-going edges.
//' 
//' @param phase_type_graph A reference to the graph created by [ptdalgorithms::create_graph()]
//' 
//' @seealso [ptdalgorithms::starting_vertex()]
//' @seealso [ptdalgorithms::vertices_length()]
//' @seealso [ptdalgorithms::vertex_at()]
//' 
//' @examples
//' graph <- create_graph(4)
//' vertex_a <- find_or_create_vertex(graph, c(1,2,1,0))
//' vertex_b <- find_or_create_vertex(graph, c(2,0,1,0))
//' vertices(graph)[[1]] == starting_vertex(graph)
//' vertices(graph)[[2]] == vertex_at(graph, 2)
//' vertices_length(graph) == 3
      )delim")
      
    .def("vertex_at", &ptdalgorithms::Graph::vertex_at, py::arg("index"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("vertices_length", &ptdalgorithms::Graph::vertices_length, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")

    .def("states",
      [](ptdalgorithms::Graph &graph) {

          std::vector<ptdalgorithms::Vertex> ver = graph.vertices();

          int rows = ver.size();
          int cols = graph.state_length();
          Matrix states = Matrix(rows, cols);

          // int **states = new int*[rows]; 
          // for (int i = 0; i < rows; ++i) states[i] = new int[cols];

          for (size_t i = 0; i < ver.size(); i++) {
              for (size_t j = 0; j < graph.state_length(); j++) {
                  states(i, j) = ver[i].state()[j];
              }
          }

          return states;
      }, py::return_value_policy::copy, R"delim(

      )delim")

    .def("__repr__",
      [](ptdalgorithms::Graph &g) {
          return "<ptdalgorithms.Graph (" + std::to_string(g.vertices_length()) + " vertices)>";
      }, py::return_value_policy::move, R"delim(

      )delim")
      
    .def("random_sample", &ptdalgorithms::Graph::random_sample, py::arg("rewards"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("mph_random_sample", static_cast<std::vector<long double> (ptdalgorithms::Graph::*)(std::vector<double>, size_t)>(&ptdalgorithms::Graph::mph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("random_sample_stop_vertex", &ptdalgorithms::Graph::random_sample_stop_vertex, py::arg("time"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("dph_random_sample_stop_vertex", &ptdalgorithms::Graph::dph_random_sample_stop_vertex, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("state_length", &ptdalgorithms::Graph::state_length, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("is_acyclic", &ptdalgorithms::Graph::is_acyclic, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("validate", &ptdalgorithms::Graph::validate, R"delim(

      )delim")
      
    .def("expectation_dag", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::expectation_dag), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("reward_transform", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("dph_reward_transform", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::dph_reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("normalize", &ptdalgorithms::Graph::normalize, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("dph_normalize", &ptdalgorithms::Graph::dph_normalize, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("notify_change", &ptdalgorithms::Graph::notify_change, R"delim(

      )delim")
      
    .def("defect", &ptdalgorithms::Graph::defect, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("clone", &ptdalgorithms::Graph::clone, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("pdf", &ptdalgorithms::Graph::pdf, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("cdf", &ptdalgorithms::Graph::cdf, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("dph_pmf", &ptdalgorithms::Graph::dph_pmf, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("dph_cdf", &ptdalgorithms::Graph::dph_cdf, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")

    .def("stop_probability", &ptdalgorithms::Graph::stop_probability, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(

      )delim")

    .def("accumulated_visiting_time", &ptdalgorithms::Graph::accumulated_visiting_time, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(

      )delim")

    .def("dph_stop_probability", &ptdalgorithms::Graph::dph_stop_probability, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")

    .def("dph_accumulated_visits", &ptdalgorithms::Graph::dph_accumulated_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")

    .def("dph_expected_visits", &ptdalgorithms::Graph::dph_expected_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(

      )delim")
    ;


  py::class_<ptdalgorithms::Vertex>(m, "Vertex", R"delim(

      )delim")

    .def(py::init(&ptdalgorithms::Vertex::init_factory), R"delim(

      )delim")
      
    .def("add_edge", &ptdalgorithms::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(
//' Adds an edge between two vertices in the graph
//' 
//' @description
//' The graph represents transitions between states as
//' a weighted direction edge between two vertices.
//' 
//' @seealso [ptdalgorithms::expected_waiting_time()]
//' @seealso [ptdalgorithms::moments()]
//' @seealso [ptdalgorithms::variance()]
//' @seealso [ptdalgorithms::covariance()]
//' @seealso [ptdalgorithms::graph_update_weights_parameterized()]
//' 
//' @param phase_type_vertex_from The vertex that transitions from
//' @param phase_type_vertex_to The vertex that transitions to
//' @param weight The weight of the edge, i.e. the transition rate
//' @param parameterized_edge_state Optional. Associate a numeric vector to an edge, for faster computations of moments when weights are changed.
//' 
//' @examples
//' graph <- create_graph(4)
//' vertex_a <- find_or_create_vertex(graph, c(1,2,1,0))
//' vertex_b <- find_or_create_vertex(graph, c(2,0,1,0))
//' add_edge(vertex_a, vertex_b, 1.5)
      )delim")

    .def("ae", &ptdalgorithms::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(
      Alias for add_edge
      )delim")

    .def("__repr__",
      [](ptdalgorithms::Vertex &v) {

        std::ostringstream s;
        s << "<ptdalgorithms.Vertex [";
        std::vector<int> state = v.state();
        for (auto i(state.begin()); i != state.end(); i++) {
            if (state.begin() != i) s << ", ";
            s << *i;
        }
        s << "]";
        // std::vector<ptdalgorithms::Edge> edges = v.edges();
        // for (auto e(edges.begin()); e != edges.end(); e++) {
        //   std::vector<int> state = e->to().state();
        //   s << std::endl << " " << e->weight() << " -> [";
        //   for (auto i(state.begin()); i != state.end(); i++) {
        //     if (state.begin() != i) s << ", ";
        //     s << *i;
        //   }
        //   s << "]";
        // }
        // s << std::endl << ">";
        return s.str();
      }, R"delim(

      )delim")

    .def("add_edge_parameterized", &ptdalgorithms::Vertex::add_edge_parameterized, py::arg("to"), py::arg("weight"), py::arg("edge_state"), R"delim(

      )delim")
      
    .def("state", &ptdalgorithms::Vertex::state, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("edges", &ptdalgorithms::Vertex::edges, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def(py::self == py::self)
    .def("__assign__", [](ptdalgorithms::Vertex &v, const ptdalgorithms::Vertex &o) {
          return v = o;
    }, py::is_operator(), 
    py::return_value_policy::move, R"delim(

      )delim")
      
    // .def("c_vertex", &ptdalgorithms::Vertex::c_vertex, R"delim(

    //   )delim")
      
    .def("rate", &ptdalgorithms::Vertex::rate, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    ;


  py::class_<ptdalgorithms::Edge>(m, "Edge", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::Edge::init_factory), R"delim(

      )delim")
      
    // .def(py::init<struct ::ptd_vertex*, struct ::ptd_edge*, &ptdalgorithms::Graph, double >(), py::arg("vertex"), py::arg("edge"), py::arg("graph"), py::arg("weight"))
    .def("to", &ptdalgorithms::Edge::to, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("weight", &ptdalgorithms::Edge::weight, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("update_weight", &ptdalgorithms::Edge::update_weight, R"delim(

      )delim")
      
    .def("__assign__", [](ptdalgorithms::Edge &e, const ptdalgorithms::Edge &o) {
          return e = o;
    }, py::is_operator(), R"delim(

      )delim")
      
    ;


  py::class_<ptdalgorithms::ParameterizedEdge>(m, "ParameterizedEdge", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::ParameterizedEdge::init_factory), R"delim(

      )delim")
      
    .def("to", &ptdalgorithms::ParameterizedEdge::to, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("weight", &ptdalgorithms::ParameterizedEdge::weight, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("edge_state", &ptdalgorithms::ParameterizedEdge::edge_state, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("__assign__", [](ptdalgorithms::ParameterizedEdge &e, const ptdalgorithms::ParameterizedEdge &o) {
          return e = o;
    }, py::is_operator(), py::return_value_policy::move, R"delim(

      )delim")
      
    ;

  py::class_<ptdalgorithms::PhaseTypeDistribution>(m, "PhaseTypeDistribution", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::PhaseTypeDistribution::init_factory), R"delim(

      )delim")
      
    // .def("c_distribution", &ptdalgorithms::PhaseTypeDistribution::c_distribution, R"delim(

    //   )delim")
      
    .def_readwrite("length", &ptdalgorithms::PhaseTypeDistribution::length, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def_readwrite("vertices", &ptdalgorithms::PhaseTypeDistribution::vertices,
       py::return_value_policy::reference_internal, R"delim(

      )delim")      
    ;


  py::class_<ptdalgorithms::AnyProbabilityDistributionContext>(m, "AnyProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init<>(), R"delim(

      )delim")
      
    .def("is_discrete", &ptdalgorithms::AnyProbabilityDistributionContext::is_discrete, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("step", &ptdalgorithms::AnyProbabilityDistributionContext::step, R"delim(

      )delim")
      
    .def("pmf", &ptdalgorithms::AnyProbabilityDistributionContext::pmf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("pdf", &ptdalgorithms::AnyProbabilityDistributionContext::pdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("cdf", &ptdalgorithms::AnyProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("time", &ptdalgorithms::AnyProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("jumps", &ptdalgorithms::AnyProbabilityDistributionContext::jumps, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("stop_probability", &ptdalgorithms::AnyProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("accumulated_visits", &ptdalgorithms::AnyProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("accumulated_visiting_time", &ptdalgorithms::AnyProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
    ;


  py::class_<ptdalgorithms::ProbabilityDistributionContext>(m, "ProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::ProbabilityDistributionContext::init_factory), R"delim(

      )delim")
      
    .def("step", &ptdalgorithms::ProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("pdf", &ptdalgorithms::ProbabilityDistributionContext::pdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("cdf", &ptdalgorithms::ProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("time", &ptdalgorithms::ProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("time", &ptdalgorithms::ProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("stop_probability", &ptdalgorithms::ProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("accumulated_visiting_time", &ptdalgorithms::ProbabilityDistributionContext::accumulated_visiting_time, 
      py::return_value_policy::copy, R"delim(

      )delim")
    ;


  py::class_<ptdalgorithms::DPHProbabilityDistributionContext>(m, "DPHProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::DPHProbabilityDistributionContext::init_factory), R"delim(

      )delim")
      
    .def("step", &ptdalgorithms::DPHProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("pmf", &ptdalgorithms::DPHProbabilityDistributionContext::pmf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("cdf", &ptdalgorithms::DPHProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("jumps", &ptdalgorithms::DPHProbabilityDistributionContext::jumps, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("stop_probability", &ptdalgorithms::DPHProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("accumulated_visits", &ptdalgorithms::DPHProbabilityDistributionContext::accumulated_visits, 
      py::return_value_policy::copy, R"delim(

      )delim")
    ;
}