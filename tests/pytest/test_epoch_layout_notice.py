"""The epoch-fit notice about per-epoch expansion of prior and fixed.

It must appear only when something was expanded (two or more epochs and a
scalar fixed value, a per-parameter prior list, a single prior, or no
prior), say what was expanded, and name the explicit arguments that avoid
it.
"""
import warnings

import numpy as np
import pytest

from phasic import GaussPrior, _epoch_layout_notice
from phasic.svgd_config import SvgdAssumptionWarning

from test_epoch_normalization import ROWS_N2, _jc  # noqa: E402
from test_notebook_model_recovery import coalescent_indexed  # noqa: E402


def _gp():
    return GaussPrior(ci=[0.5, 10.0])


# ------------------------------------------------------------ helper
def test_single_epoch_never_notifies():
    assert _epoch_layout_notice(None, [(1, 0.7)], 1, 2) is None


def test_explicit_per_epoch_forms_do_not_notify():
    prior = [_gp(), None, _gp(), None, _gp(), None]
    assert _epoch_layout_notice(prior, [(1, [0.7, 0.7, 0.7])], 3, 2) is None


def test_scalar_fixed_and_short_prior_are_named_with_remedy():
    msg = _epoch_layout_notice([_gp(), None], [(1, 0.7)], 3, 2)
    assert "3 epochs" in msg and "6 values" in msg
    assert "fixes parameter 1 at 0.7 in every epoch" in msg
    assert "the 2 priors are repeated for every epoch" in msg
    assert "fixed=[(1, [0.7, 0.7, 0.7])]" in msg
    assert "prior= a list of 6 priors" in msg
    assert "quiet_assumptions=True" in msg


def test_no_prior_and_single_prior():
    msg = _epoch_layout_notice(None, None, 2, 2)
    assert "default data-informed prior" in msg
    assert "prior= a list of 4 priors" in msg
    assert "fixed=[" not in msg
    msg = _epoch_layout_notice(_gp(), None, 2, 2)
    assert "single prior is used for every parameter" in msg


# ------------------------------------------------------------ through svgd
@pytest.fixture(scope="module")
def jc2():
    return _jc(coalescent_indexed(2), reward_limit=3)


def _notices(jc, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jc.svgd(ROWS_N2, n_iterations=1, n_particles=4, seed=0,
                progress=False, **kw)
    return [str(x.message) for x in w
            if issubclass(x.category, SvgdAssumptionWarning)
            and "epoch" in str(x.message)]


def test_svgd_two_epochs_scalar_fixed_notifies(jc2):
    msgs = _notices(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, 0.7)],
                    prior=[_gp(), None])
    assert len(msgs) == 1, msgs
    assert "fixed=[(1, [0.7, 0.7])]" in msgs[0]


def test_svgd_two_epochs_explicit_forms_silent(jc2):
    msgs = _notices(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, [0.7, 0.7])],
                    prior=[_gp(), None, _gp(), None])
    assert msgs == []


def test_svgd_one_epoch_silent(jc2):
    msgs = _notices(jc2, epoch_starts=[0.0], fixed=[(1, 0.7)],
                    prior=[_gp(), None])
    assert msgs == []
