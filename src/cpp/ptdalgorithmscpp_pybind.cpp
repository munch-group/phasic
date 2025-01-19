

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
      .def("update_weights_parameterized", &ptdalgorithms::Graph::update_weights_parameterized, py::arg("rewards") = std::vector<double>{})
      .def("expected_waiting_time", &ptdalgorithms::Graph::expected_waiting_time, py::arg("rewards") = std::vector<double>{})
      .def("expected_residence_time", &ptdalgorithms::Graph::expected_residence_time, py::arg("rewards") = std::vector<double>{})

    ;
//   py::class_<ptdalgorithms::Vertex>(m, "Vertex")
//       .def(py::init<int  >(), py::arg(""))

//     ;

}