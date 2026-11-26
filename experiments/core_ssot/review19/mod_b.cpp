#include <pybind11/pybind11.h>
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
extern "C" int ptd_probe_check(int fp);
#ifndef FP
#define FP 42
#endif
namespace { struct Guard { Guard() {
  int ok = ptd_probe_check(FP);
  std::fprintf(stderr, "[mod_b static init] check=%d\n", ok);
  if (!ok) {
#if MODE == 1
    throw std::runtime_error("layout mismatch");
#elif MODE == 2
    PyErr_SetString(PyExc_ImportError, "mod_b: layout mismatch, rebuild against this phasic");
#endif
  }
}} g_guard; }
PYBIND11_MODULE(MODNAME, m) { m.def("hello", []{ return 1; }); }
