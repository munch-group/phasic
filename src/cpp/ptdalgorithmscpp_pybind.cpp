

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

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
  py::class_<ptdalgorithms::Graph>(m, "Graph")
  .def(py::init<int>(), py::arg("state_length"))
  .def(py::init<struct ::ptd_graph* >(), py::arg("ptd_graph"))
  .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
  .def(py::init<const ptdalgorithms::Graph>(), py::arg("o"))
  .def("update_weights_parameterized", &ptdalgorithms::Graph::update_weights_parameterized, py::arg("rewards"))
  .def("expected_waiting_time", &ptdalgorithms::Graph::expected_waiting_time, py::arg("rewards"))
  .def("expected_residence_time", &ptdalgorithms::Graph::expected_residence_time, py::arg("rewards"))
  .def("create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::create_vertex), py::arg("state"))
  .def("find_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_vertex), py::arg("state"))
  .def("vertex_exists", static_cast<bool (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::vertex_exists), py::arg("state"))
  .def("find_or_create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_or_create_vertex), py::arg("state"))
  .def("starting_vertex", &ptdalgorithms::Graph::starting_vertex)
  .def("vertices", &ptdalgorithms::Graph::vertices)
  .def("vertex_at", &ptdalgorithms::Graph::vertex_at, py::arg("index"))
  .def("vertices_length", &ptdalgorithms::Graph::vertices_length)
  .def("__repr__",
    [](ptdalgorithms::Graph &g) {
        return "<ptdalgorithms.Graph (" + std::to_string(g.vertices_length()) + " vertices)>";
    })
  .def("random_sample", &ptdalgorithms::Graph::random_sample, py::arg("rewards"))
  .def("mph_random_sample", static_cast<std::vector<long double> (ptdalgorithms::Graph::*)(std::vector<double>, size_t)>(&ptdalgorithms::Graph::mph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"))
  .def("random_sample_stop_vertex", &ptdalgorithms::Graph::random_sample_stop_vertex, py::arg("time"))
  .def("dph_random_sample_stop_vertex", &ptdalgorithms::Graph::dph_random_sample_stop_vertex, py::arg("jumps"))
  .def("state_length", &ptdalgorithms::Graph::state_length)
  .def("is_acyclic", &ptdalgorithms::Graph::is_acyclic)
  .def("validate", &ptdalgorithms::Graph::validate)
  .def("expectation_dag", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::expectation_dag), py::arg("rewards"))
  .def("reward_transform", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::reward_transform), py::arg("rewards"))
  .def("dph_reward_transform", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::dph_reward_transform), py::arg("rewards"))
  .def("normalize", &ptdalgorithms::Graph::normalize)
  .def("dph_normalize", &ptdalgorithms::Graph::dph_normalize)
  .def("notify_change", &ptdalgorithms::Graph::notify_change)
  .def("defect", &ptdalgorithms::Graph::defect)
  .def("clone", &ptdalgorithms::Graph::clone)
  .def("pdf", &ptdalgorithms::Graph::pdf, py::arg("time"), py::arg("granularity") = 0)
  .def("cdf", &ptdalgorithms::Graph::cdf, py::arg("time"), py::arg("granularity") = 0)
  .def("dph_pmf", &ptdalgorithms::Graph::dph_pmf, py::arg("jumps"))
  .def("dph_cdf", &ptdalgorithms::Graph::dph_cdf, py::arg("jumps"))
  .def("stop_probability", &ptdalgorithms::Graph::stop_probability, py::arg("time"), py::arg("granularity") = 0)
  .def("accumulated_visiting_time", &ptdalgorithms::Graph::accumulated_visiting_time, py::arg("time"), py::arg("granularity") = 0)
  .def("dph_stop_probability", &ptdalgorithms::Graph::dph_stop_probability, py::arg("jumps"))
  .def("dph_accumulated_visits", &ptdalgorithms::Graph::dph_accumulated_visits, py::arg("jumps"))
  .def("dph_expected_visits", &ptdalgorithms::Graph::dph_expected_visits, py::arg("jumps"))
    ;
  py::class_<ptdalgorithms::Vertex>(m, "Vertex")

    .def(py::init(&ptdalgorithms::Vertex::init_factory))
    .def("add_edge", &ptdalgorithms::Vertex::add_edge, py::arg("to"), py::arg("weight"))
    .def("add_edge_parameterized", &ptdalgorithms::Vertex::add_edge_parameterized, py::arg("to"), py::arg("weight"), py::arg("edge_state"))
    .def("state", &ptdalgorithms::Vertex::state)
    .def("edges", &ptdalgorithms::Vertex::edges)
    .def(py::self == py::self)
    .def("__assign__", [](ptdalgorithms::Vertex &v, const ptdalgorithms::Vertex &o) {
          return v = o;
    }, py::is_operator())
    .def("c_vertex", &ptdalgorithms::Vertex::c_vertex)
    .def("rate", &ptdalgorithms::Vertex::rate)
;
  py::class_<ptdalgorithms::Edge>(m, "Edge")

    .def(py::init(&ptdalgorithms::Edge::init_factory))
    // .def(py::init<struct ::ptd_vertex*, struct ::ptd_edge*, &ptdalgorithms::Graph, double >(), py::arg("vertex"), py::arg("edge"), py::arg("graph"), py::arg("weight"))


    ;

}