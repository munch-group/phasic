#!/usr/bin/env python3
"""
Script to update all docstrings in phasic_pybind.cpp to numpy format.
"""

import re
from pathlib import Path

# Read the file
file_path = Path("/Users/kmt/phasic/src/cpp/phasic_pybind.cpp")
content = file_path.read_text()

# Dictionary of docstring replacements
# Maps method signature patterns to proper numpy-formatted docstrings

docstring_fixes = {
    # MatrixRepresentation class
    r'(\.def\(py::init<const MatrixRepresentation>\(\), py::arg\("graph"\), R"delim\(\s+\)delim")': r'''.def(py::init<const MatrixRepresentation>(), py::arg("graph"), R"delim(
Construct MatrixRepresentation from a Graph object.

Parameters
----------
graph : Graph
    The phase-type graph to convert to matrix representation.
      )delim"''',

    r'(\.def_readwrite\("states", &MatrixRepresentation::states, R"delim\(\s+\)delim")': r'''.def_readwrite("states", &MatrixRepresentation::states, R"delim(
State matrix where each row represents a vertex state.

Returns
-------
iMatrix
    Integer matrix of size (n_vertices, state_length).
      )delim"''',

    r'(\.def_readwrite\("sim", &MatrixRepresentation::sim, R"delim\(\s+\)delim")': r'''.def_readwrite("sim", &MatrixRepresentation::sim, R"delim(
Sub-intensity matrix of the phase-type distribution.

Returns
-------
dMatrix
    Float matrix of size (n_vertices, n_vertices) representing transition rates.
      )delim"''',

    r'(\.def_readwrite\("ipv", &MatrixRepresentation::ipv, R"delim\(\s+\)delim")': r'''.def_readwrite("ipv", &MatrixRepresentation::ipv, R"delim(
Initial probability vector of the phase-type distribution.

Returns
-------
list of float
    Vector of length n_vertices with initial probabilities.
      )delim"''',

    r'(\.def_readwrite\("indices", &MatrixRepresentation::indices, R"delim\(\s+\)delim")': r'''.def_readwrite("indices", &MatrixRepresentation::indices, R"delim(
Vertex indices mapping matrix rows to graph vertices.

Returns
-------
list of int
    Vector of length n_vertices with 1-indexed vertex numbers.
      )delim"''',

    # vertex_exists
    r'(\.def\("vertex_exists".*?py::return_value_policy::reference_internal, R"delim\(\s+\)delim")': r'''.def("vertex_exists", static_cast<bool (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::vertex_exists), py::arg("state"),
      py::return_value_policy::reference_internal, R"delim(
Check if a vertex with the given state exists in the graph.

Parameters
----------
state : list of int
    Integer sequence defining the state to search for.

Returns
-------
bool
    True if vertex exists, False otherwise.
      )delim"''',

    # __repr__
    r'(\}, py::return_value_policy::move,\s+R"delim\(\s+\)delim"\))\s+(\.def\("param_length")': r''', py::return_value_policy::move,
      R"delim(
String representation of the Graph object.

Returns
-------
str
    String in format "<Graph (N vertices)>".
      )delim")

    .def("param_length"''',

    # vertex_at overload without docstring
    r'(\.def\("vertex_at",\[]\(phasic::Graph &graph, double index\).*?\}, py::return_value_policy::reference_internal\))': r'''.def("vertex_at",[](phasic::Graph &graph, double index) {
      return graph.vertex_at_p((int) index);

    }, py::return_value_policy::reference_internal, R"delim(
Get vertex at given index (float overload, casts to int).

Parameters
----------
index : float
    Vertex index (will be cast to int).

Returns
-------
Vertex
    The vertex at the given index.
      )delim")''',

    # state_length
    r'(\.def\("state_length", &phasic::Graph::state_length,\s*)$': r'''.def("state_length", &phasic::Graph::state_length,
      py::return_value_policy::copy, R"delim(
Get the state vector length for this graph.

Returns
-------
int
    The length of state vectors for all vertices in this graph.
      )delim")
''',
}

# Apply fixes
for pattern, replacement in docstring_fixes.items():
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
file_path.write_text(content)
print(f"Updated {len(docstring_fixes)} docstring patterns in {file_path}")
