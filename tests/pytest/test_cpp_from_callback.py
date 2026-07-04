"""End-to-end tests for the native C++ callback constructor
``phasic::Graph::from_callback`` (mirrors the Python ``Graph(callback,
ipv=..., theta_dim=...)`` UI).

The C++ code lives in the installed ``phasiccpp.h`` / ``phasiccpp.cpp``; these
tests compile a small fixture module against it via ``cppimport`` (the same
linkage recipe used by ``docs/pages/tutorial/cpp_state_spaces.cpp``), then check
that graphs built through ``from_callback`` are identical to (a) a hand-written
C++ build and (b) the Python callback UI, for both constant and parameterized
edges.

Requires a working C++ toolchain and an installed phasic (``pixi run
install-dev``); this is the standard test environment.
"""
import os

import numpy as np
import pytest

import phasic  # import before creating numpy arrays used by the x64 FFI
from phasic import Graph

# The cppimport fixture below links phasiccpp.cpp and references phasic's C
# API symbols — including the thread-local ``ptd_err``. Those must resolve
# from the already-loaded ``phasic_pybind`` extension at dlopen time, but
# CPython loads extensions with ``RTLD_LOCAL | RTLD_NOW``, so the extension's
# symbols are not in the global scope. Function symbols would bind lazily, but
# ``ptd_err`` is a TLS symbol that must bind eagerly at load, so the fixture
# aborts with ``undefined symbol: ptd_err``. Re-open the extension with
# ``RTLD_GLOBAL`` to promote its symbols into the global namespace; this works
# even though ``phasic`` was already imported under the default flags.
if hasattr(os, "RTLD_GLOBAL"):
    import ctypes as _ctypes
    import glob as _glob

    _pb = _glob.glob(
        os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so")
    )
    if _pb:
        _ctypes.CDLL(_pb[0], mode=os.RTLD_NOW | os.RTLD_GLOBAL)

cppimport = pytest.importorskip("cppimport")

_FIXTURE = os.path.join(os.path.dirname(__file__), "_cpp_from_callback_fixture.cpp")

# Shared PDF comparison grid / granularity.
GRID = np.linspace(0.05, 6.0, 40)
GRAN = 100


@pytest.fixture(scope="module")
def cpp():
    """Compile and load the C++ fixture module once per test module."""
    return cppimport.imp_from_filepath(_FIXTURE, "_cpp_from_callback_fixture")


def _coalescent_callback(state, nr_samples):
    """Python coalescent transitions (non-empty states only), mirroring the
    C++ fixture. ``nr_samples`` arrives as float (Graph coerces int kwargs)."""
    n = int(nr_samples)
    out = []
    for i in range(n):
        for j in range(i, n):
            if i + j + 1 >= n:
                continue
            same = i == j
            s = [int(x) for x in state]
            if same and s[i] < 2:
                continue
            if (not same) and (s[i] < 1 or s[j] < 1):
                continue
            s[i] -= 1
            s[j] -= 1
            s[i + j + 1] += 1
            w = (s[i] + 1) * (s[i] + 2) / 2.0 if same else (s[i] + 1) * (s[j] + 1)
            out.append((np.array(s), w))
    return out


def _python_coalescent(n):
    initial = [n] + [0] * (n - 1)
    return Graph(_coalescent_callback, ipv=[(initial, 1.0)], nr_samples=n)


@pytest.mark.parametrize("n", [3, 5, 6])
def test_from_callback_matches_python_ui(cpp, n):
    """from_callback must produce the same graph and PDF as the Python UI."""
    g_cb = Graph(cpp.coalescent_from_callback(n))
    g_py = _python_coalescent(n)

    assert g_cb.vertices_length() == g_py.vertices_length()
    assert g_cb.edges_length() == g_py.edges_length()

    pdf_cb = np.asarray(g_cb.pdf(GRID, granularity=GRAN))
    pdf_py = np.asarray(g_py.pdf(GRID, granularity=GRAN))
    assert np.allclose(pdf_cb, pdf_py, atol=1e-12, rtol=0)


def test_from_callback_parameterized(cpp):
    """Parameterized from_callback: theta=1 reproduces the constant coalescent
    (rate == coefficient); theta scaling changes the PDF."""
    n = 5
    g_const = Graph(cpp.coalescent_from_callback(n))
    g_par = Graph(cpp.coalescent_param_from_callback(n))

    assert g_par.parameterized()
    assert g_par.vertices_length() == g_const.vertices_length()
    assert g_par.edges_length() == g_const.edges_length()

    pdf_const = np.asarray(g_const.pdf(GRID, granularity=GRAN))

    g_par.update_weights(np.array([1.0]))
    pdf_par1 = np.asarray(g_par.pdf(GRID, granularity=GRAN))
    assert np.allclose(pdf_par1, pdf_const, atol=1e-10, rtol=0)

    g_par.update_weights(np.array([2.0]))
    pdf_par2 = np.asarray(g_par.pdf(GRID, granularity=GRAN))
    assert not np.allclose(pdf_par2, pdf_const, atol=1e-6)


def test_from_callback_empty_ipv_raises(cpp):
    """An empty IPV is rejected (fail-loud, no silent fallback)."""
    with pytest.raises(Exception):
        cpp.build_empty_ipv()
