#include <pybind11/pybind11.h>
#include "phasiccpp.h"
namespace py = pybind11;
#ifdef ANCHOR
// simulated header-scope anchor: internal-linkage object in every TU that includes the header
namespace phasic { namespace detail { int abi_guard_anchor(); } }
namespace { const int phasic_anchor_ = phasic::detail::abi_guard_anchor(); }
#endif
// accessor-only: takes a Graph by reference, no phasic::Graph constructor instantiated
size_t nv(phasic::Graph &g) { return (size_t) g.c_graph()->vertices_length; }
PYBIND11_MODULE(MODNAME, m) { m.def("nv", &nv); }
