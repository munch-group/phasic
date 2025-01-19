

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  py::class_<ptdalgorithms::Graph>(m, "Graph")
  .def(py::init<int>(), py::arg("state_length"))
  .def(py::init<struct ::ptd_graph* >(), py::arg("ptd_graph"))
  .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
  .def(py::init<const ptdalgorithms::Graph>(), py::arg("o"))
//   .def("update_weights_parameterized", &ptdalgorithms::Graph::update_weights_parameterized, py::arg("rewards") = std::vector<double>{})
  .def("update_weights_parameterized", &ptdalgorithms::Graph::update_weights_parameterized, py::arg("rewards") = std::vector<double>())
  .def("expected_waiting_time", &ptdalgorithms::Graph::expected_waiting_time, py::arg("rewards") = std::vector<double>())
  .def("expected_residence_time", &ptdalgorithms::Graph::expected_residence_time, py::arg("rewards") = std::vector<double>())
  .def("create_vertex", &ptdalgorithms::Graph::create_vertex, py::arg("state") = std::vector<int>())
  .def("find_vertex", &ptdalgorithms::Graph::find_vertex, py::arg("state"))
//   .def("vertex_exists", &ptdalgorithms::Graph::vertex_exists, py::arg("state"))
//   .def("find_or_create_vertex", &ptdalgorithms::Graph::find_or_create_vertex, py::arg("state"))
  .def("starting_vertex", &ptdalgorithms::Graph::starting_vertex)
  .def("vertices", &ptdalgorithms::Graph::vertices)
  .def("vertex_at", &ptdalgorithms::Graph::vertex_at, py::arg("index"))

  .def("vertices_length", &ptdalgorithms::Graph::vertices_length)
  .def("__repr__",
    [](ptdalgorithms::Graph &g) {
        return "<ptdalgorithms.Graph (" + std::to_string(g.vertices_length()) + " vertices)>";
    })
  .def("random_sample", &ptdalgorithms::Graph::random_sample, py::arg("rewards") = std::vector<double>())


    ;
}


// .def(
//     "create_vertex",
//     [](const int *state) {
//         return create_vertex(state);
//     },
//     "Want to call from Python");

// py::class_<Pet>(m, "Pet")
//     // ...
//     .def("set",
//         [](Pet &self, const std::string &s)
//         {
//             aString as = aString(s);
//             self.set(as);
//         });

// .def(
//     "create_vertex",
//     [](float a, std::string const& s) {
//         return create_vertex(a, s);
//     },
//     "Want to call from Python");

        // Vertex create_vertex(std::vector<int> state = std::vector<int>());

        // Vertex create_vertex(const int *state);


//   py::class_<ptdalgorithms::Vertex>(m, "Vertex")
//       .def(py::init<int  >(), py::arg(""))

//     ;

