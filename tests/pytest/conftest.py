"""Pytest configuration for the phasic test suite.

Import phasic FIRST -- before pytest imports any test module that might import
jax. phasic sets ``JAX_ENABLE_X64=1`` at import (the XLA-FFI handlers require
float64 buffers); JAX reads that flag only once, when it is first imported, and
then locks it for the process. conftest.py is imported by pytest before any
test module in this directory tree, so importing phasic here guarantees x64 is
enabled process-wide regardless of test selection / collection order.

Without this, a module that does ``import jax`` (e.g. via
``pytest.importorskip("jax")``) before importing phasic -- when collected ahead
of the cross-backend gates -- locks JAX into float32 and trips the x64
assertion in ``_gate_backend.py`` (a collection error).
"""
import phasic  # noqa: F401  -- MUST be first; enables jax_enable_x64 process-wide
