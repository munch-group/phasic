"""Run the existing BetaPrior-using test files with the prototype active."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import proto
import pytest
import phasic.svgd as S
print('patched:', hasattr(S, '_resolve_param_transform'))
sys.exit(pytest.main(['-q', '-x', '-p', 'no:cacheprovider', '--timeout=500', '-o', 'addopts=',
                      'tests/pytest/test_beta_prior.py', 'tests/pytest/test_svgd_tied_export.py']))
