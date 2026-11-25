"""reviewB p1: (a) importing phasic.cache loads the native extension;
(b) a subprocess spawned after `import phasic` inherits OMP_NUM_THREADS,
so an import-order check in such a subprocess is vacuous unless the test
removes OMP_NUM_THREADS from the child environment.

Run with HOME pointing at a scratch dir. Builds no graph, writes no cache.
"""
import importlib.util
import os
import subprocess
import sys
import types

# Locate the installed package without importing it.
spec = importlib.util.find_spec("phasic")
pkgdir = os.path.dirname(spec.origin)
print("pkgdir:", pkgdir)

CHILD_A = r'''
import os, sys, types
pkgdir = sys.argv[1]
os.environ.pop("OMP_NUM_THREADS", None)
# Stub the package so phasic/__init__.py does NOT run: this isolates what a
# module-level `from .cache import ...` (or `from . import cache`) inside
# graph_cache would pull in, at the point graph_cache is imported
# (__init__.py:53, before OMP_NUM_THREADS is set at :107-108).
stub = types.ModuleType("phasic"); stub.__path__ = [pkgdir]
sys.modules["phasic"] = stub
import phasic.cache
print("A: phasic.phasic_pybind loaded by importing phasic.cache:",
      "phasic.phasic_pybind" in sys.modules)
print("A: OMP_NUM_THREADS at that moment:", os.environ.get("OMP_NUM_THREADS"))
'''

r = subprocess.run([sys.executable, "-c", CHILD_A, pkgdir],
                   capture_output=True, text=True)
print(r.stdout.strip()); print(r.stderr.strip()[-500:])

# (b) inheritance
env_before = os.environ.get("OMP_NUM_THREADS")
print("B: OMP_NUM_THREADS in parent before import phasic:", env_before)
import phasic  # noqa: E402,F401
print("B: OMP_NUM_THREADS in parent after import phasic:",
      os.environ.get("OMP_NUM_THREADS"))
r = subprocess.run([sys.executable, "-c",
                    "import os; print(os.environ.get('OMP_NUM_THREADS'))"],
                   capture_output=True, text=True)
print("B: child (default env inheritance) sees OMP_NUM_THREADS =",
      r.stdout.strip())
