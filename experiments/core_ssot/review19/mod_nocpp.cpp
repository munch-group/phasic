#include <pybind11/pybind11.h>
#include "phasiccpp.h"
namespace py = pybind11;
phasic::Graph chain(int n) {
  phasic::Graph g((size_t)1);
  struct ptd_graph *cg = g.c_graph();
  struct ptd_vertex *prev = cg->starting_vertex;
  for (int i = 1; i <= n; ++i) { int s[1] = {i};
    struct ptd_vertex *v = ptd_find_or_create_vertex(cg, g.c_avl_tree(), s);
    { double c[1]={1.0}; ptd_graph_add_edge(prev, v, c, 1); } prev = v; }
  return g;
}
PYBIND11_MODULE(mod_nocpp, m) { m.def("chain", &chain); }
