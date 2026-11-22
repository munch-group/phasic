"""Epoch fits: the short notice about per-epoch expansion of prior and fixed,
and the full records it points to in effective_options().

The notice must appear only when something was expanded (two or more
epochs and a scalar fixed value, a per-parameter prior list, a single
prior, or no prior), name exactly those options, and stay short. The
records hold the explanation and the explicit arguments that avoid it.
Plan of record: svgd-notices-plan.md (batch B2).
"""
import io
import warnings

import numpy as np
import pytest

from phasic import GaussPrior, _epoch_layout_records
from phasic.svgd_config import SvgdAssumptionWarning

from test_epoch_normalization import ROWS_N2, _jc  # noqa: E402
from test_notebook_model_recovery import coalescent_indexed  # noqa: E402


def _gp():
    return GaussPrior(ci=[0.5, 10.0], lower=0.0)


# ------------------------------------------------------------ helper
def test_single_epoch_records_nothing():
    rec = _epoch_layout_records(None, [(1, 0.7)], 1, 2)
    assert rec == {'layout': None, 'prior': None, 'fixed': None}


def test_explicit_per_epoch_forms_are_not_expanded():
    prior = [_gp(), None, _gp(), None, _gp(), None]
    rec = _epoch_layout_records(prior, [(1, [0.7, 0.7, 0.7])], 3, 2)
    assert rec['prior'] is None and rec['fixed'] is None
    assert "3 epochs" in rec['layout'] and "6 values" in rec['layout']


def test_scalar_fixed_and_short_prior_records():
    rec = _epoch_layout_records([_gp(), None], [(0, 5.0), (1, [0.7, 0.8, 0.9])],
                                3, 2)
    note, explicit = rec['fixed']
    assert "fixes parameter 0 at 5.0 in every epoch" in note
    assert "parameter 1" not in note           # already per-epoch
    assert explicit.startswith("fixed=[(0, [5.0, 5.0, 5.0]), (1, [0.7, 0.8, 0.9])]")
    note, explicit = rec['prior']
    assert "Your 2 priors are repeated for every epoch" in note
    assert "a list of 6 priors (your 2 repeated 3 times)" in explicit


def test_no_prior_and_single_prior_records():
    note, explicit = _epoch_layout_records(None, None, 2, 2)['prior']
    assert "default data-informed prior" in note
    assert "a list of 4 priors" in explicit
    note, _ = _epoch_layout_records(_gp(), None, 2, 2)['prior']
    assert "single prior is used for every parameter" in note


# ------------------------------------------------------------ through svgd
@pytest.fixture(scope="module")
def jc2():
    return _jc(coalescent_indexed(2), reward_limit=3)


def _fit(jc, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = jc.svgd(ROWS_N2, n_iterations=1, n_particles=4, seed=0,
                    progress=False, **kw)
    msgs = [str(x.message) for x in w
            if issubclass(x.category, SvgdAssumptionWarning)
            and ("epoch" in str(x.message) or "theta_dim" in str(x.message))]
    return s, msgs


def test_two_epochs_scalar_fixed_short_notice_and_full_record(jc2):
    s, msgs = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, 0.7)],
                   prior=[_gp(), None])
    assert len(msgs) == 1, msgs
    m = msgs[0]
    assert m == ("svgd: prior and fixed applied to each of 2 epochs. "
                 "See .effective_options('prior', 'fixed') on the returned "
                 "SVGD object.")
    assert len(m) <= 160
    buf = io.StringIO()
    d = s.effective_options('prior', 'fixed', file=buf, return_dict=True)
    text = buf.getvalue()
    flat = " ".join(text.split())
    assert "fixed=[(1, [0.7, 0.7])]" in flat
    assert "a list of 4 priors (your 2 repeated 2 times)" in flat
    assert all(len(ln) <= 80 for ln in text.splitlines())
    assert d['fixed']['status'] == 'user'
    td = s.effective_options(return_dict=True, file=io.StringIO())['theta_dim']
    assert td['status'] == 'inferred' and td['value'] == 4
    assert "ordered epoch by epoch" in td['note']


def test_two_epochs_only_fixed_expanded(jc2):
    _, msgs = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, 0.7)],
                   prior=[_gp(), None, _gp(), None])
    assert msgs == ["svgd: fixed applied to each of 2 epochs. See "
                    ".effective_options('fixed') on the returned SVGD object."]


def test_two_epochs_explicit_forms_silent(jc2):
    _, msgs = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, [0.7, 0.7])],
                   prior=[_gp(), None, _gp(), None])
    assert msgs == []


def test_one_epoch_silent(jc2):
    _, msgs = _fit(jc2, epoch_starts=[0.0], fixed=[(1, 0.7)],
                   prior=[_gp(), None])
    assert msgs == []


def test_explicit_form_keeps_full_precision():
    _, explicit = _epoch_layout_records(None, [(1, 0.0123456789)], 2, 2)['fixed']
    assert "fixed=[(1, [0.0123456789, 0.0123456789])]" in explicit


def test_tuple_prior_is_described_as_per_parameter_list():
    note, _ = _epoch_layout_records((_gp(), None), None, 3, 2)['prior']
    assert "Your 2 priors are repeated for every epoch" in note


@pytest.mark.parametrize("prior", [
    (GaussPrior(ci=[0.5, 10.0], lower=0.0), None),
    (GaussPrior(ci=[0.5, 10.0], lower=0.0), None, GaussPrior(ci=[0.5, 10.0], lower=0.0), None),
])
def test_tuple_prior_works_like_list_on_epoch_fit(jc2, prior):
    s, _ = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, [0.7, 0.7])],
                prior=prior)
    s_list, _ = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, [0.7, 0.7])],
                     prior=list(prior))
    assert np.allclose(np.asarray(s.particles), np.asarray(s_list.particles))


def test_overridden_theta_dim_is_announced(jc2):
    s, msgs = _fit(jc2, epoch_starts=[0.0, 0.5], theta_dim=2,
                   fixed=[(1, [0.7, 0.7])], prior=[_gp(), None, _gp(), None])
    assert msgs == ["svgd: theta_dim=2 overridden to 4. See "
                    ".effective_options('theta_dim') on the returned SVGD "
                    "object."]
    td = s.effective_options(return_dict=True, file=io.StringIO())['theta_dim']
    assert td['status'] == 'forced' and "ordered epoch by epoch" in td['note']


def test_default_epoch_prior_record(jc2):
    s, msgs = _fit(jc2, epoch_starts=[0.0, 0.5], fixed=[(1, [0.7, 0.7])])
    assert msgs == ["svgd: prior applied to each of 2 epochs. See "
                    ".effective_options('prior') on the returned SVGD object."]
    buf = io.StringIO()
    s.effective_options('prior', file=buf)
    assert "data-informed GaussPrior per epoch" in buf.getvalue()
