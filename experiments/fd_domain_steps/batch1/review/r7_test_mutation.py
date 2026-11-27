"""Review probe r7: mutation testing of test_fd_domain_steps.py's value tests.
Usage: python r7_test_mutation.py <mutation> [pytest -k expr]
  zero_alt : the alternative quotient returns 0 (band/declared slots get a zero gradient)
  half_alt : the alternative quotient is halved (a 2x error)
  none     : no mutation (control)
A test that still PASSES under a mutation does not check the value it claims to."""
import sys, os
import pytest

MUT = sys.argv[1]
KEXPR = sys.argv[2] if len(sys.argv) > 2 else None


class Mutator:
    def pytest_configure(self, config):
        import jax
        jax.config.update("jax_enable_x64", True)
        import phasic  # noqa
        from phasic import _fd_steps
        orig = _fd_steps.alt_ratio
        if MUT == "zero_alt":
            _fd_steps.alt_ratio = lambda fp, fm, d: 0.0 * orig(fp, fm, d)
        elif MUT == "half_alt":
            _fd_steps.alt_ratio = lambda fp, fm, d: 0.5 * orig(fp, fm, d)


here = os.path.dirname(os.path.abspath(__file__))
tests = os.path.join(here, "../../../../tests/pytest/test_fd_domain_steps.py")
args = [tests, "-p", "no:cacheprovider", "-q", "-rA", "--no-header", "-o", "addopts="]
if KEXPR:
    args += ["-k", KEXPR]
sys.exit(pytest.main(args, plugins=[Mutator()]))
