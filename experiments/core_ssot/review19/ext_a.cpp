#include <pybind11/pybind11.h>
extern "C" __attribute__((visibility("default"))) int ptd_probe_check(int fp) { return fp == 42; }
PYBIND11_MODULE(ext_a, m) {}
