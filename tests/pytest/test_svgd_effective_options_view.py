"""effective_options(): value formatting, the table, and the single-option view.

Plan of record: svgd-notices-plan.md (batch B1).
"""
import io

import numpy as np
import pytest

from phasic.svgd import GaussPrior, Adam, ExpStepSize, SVGD
from phasic.svgd_config import SvgdOptionsLedger, format_option_value


# ------------------------------------------------------------ formatter
def test_default_repr_objects_show_module_and_class():
    assert format_option_value(GaussPrior(ci=[1, 2])) == "svgd.GaussPrior"
    assert format_option_value(Adam(learning_rate=0.1)) == "svgd.Adam"
    assert format_option_value(ExpStepSize(first_step=0.1, last_step=0.01,
                                           tau=10)) == "svgd.ExpStepSize"


def test_containers_recurse():
    assert (format_option_value([GaussPrior(ci=[1, 2]), None])
            == "[svgd.GaussPrior, None]")
    assert format_option_value([(1, 0.5)]) == "[(1, 0.5)]"
    assert format_option_value((3,)) == "(3,)"


def test_plain_values_and_functions_and_arrays():
    assert format_option_value('auto') == "'auto'"
    assert format_option_value(None) == "None"
    assert format_option_value(0.03) == "0.03"

    def my_prior(t):
        return 0.0
    assert format_option_value(my_prior).startswith("function ")
    assert "my_prior" in format_option_value(my_prior)
    assert format_option_value(np.arange(10)) == "array of 10 values"
    assert format_option_value(np.array([1.0, 2.0])) == "[1., 2.]"


# ------------------------------------------------------------ table
def _ledger():
    led = SvgdOptionsLedger()
    led.set_default('n_particles', None)
    led.set_inferred('preconditioner', 'jacobian', 'joint-probability model',
                     display="'auto' -> 'jacobian'",
                     note="On a joint-probability model 'auto' builds the "
                          "probability-Jacobian preconditioner. " * 4,
                     explicit="preconditioner='jacobian'")
    led.set_user('prior', [GaussPrior(ci=[1, 2]), None])
    return led


def test_table_uses_display_and_class_names_without_cutting():
    text = _ledger().render()
    assert "'auto' -> 'jacobian'" in text
    assert "[svgd.GaussPrior, None]" in text
    assert "object at 0x" not in text
    assert "..." not in text


def test_table_cuts_only_beyond_cap_and_says_where_to_look():
    led = SvgdOptionsLedger()
    led.set_user('fixed', [(i, 0.5) for i in range(20)])
    text = led.render()
    assert "..." in text
    assert "effective_options('<option>')" in text
    row = [ln for ln in text.splitlines() if " fixed " in ln][0]
    assert len(row) < 100


# ------------------------------------------------------------ single view
def test_single_option_view_wraps_and_never_cuts():
    led = _ledger()
    text = led.render_option('preconditioner')
    lines = text.splitlines()
    assert lines[0] == 'preconditioner'
    assert all(len(ln) <= 80 for ln in lines), lines
    assert "value:" in text and "'auto' -> 'jacobian'" in text
    assert "status:" in text and "inferred (joint-probability model)" in text
    assert "explicit:" in text and "preconditioner='jacobian'" in text
    # the whole note survives, only re-wrapped
    flat = " ".join(text.split())
    assert ("On a joint-probability model 'auto' builds the "
            "probability-Jacobian preconditioner.") in flat
    assert flat.count("probability-Jacobian preconditioner.") == 4


def test_single_option_view_without_note():
    text = _ledger().render_option('n_particles')
    assert "details:" not in text and "explicit:" not in text
    assert "default" in text


def test_unknown_option_raises_with_valid_names():
    with pytest.raises(ValueError, match="unknown option 'nope'.*preconditioner"):
        _ledger().render_option('nope')


def test_annotate_keeps_status():
    led = _ledger()
    led.annotate('prior', note="repeated per epoch", explicit="prior=[...]")
    d = led.as_dict()['prior']
    assert d['status'] == 'user'
    assert d['note'] == "repeated per epoch"
    with pytest.raises(KeyError):
        led.annotate('nope', note="x")


# ------------------------------------------------------------ SVGD method
class _Stub:
    effective_options = SVGD.effective_options


def test_effective_options_names_print_blocks_and_subset_dict():
    s = _Stub()
    s._options_ledger = _ledger()
    buf = io.StringIO()
    d = s.effective_options('preconditioner', 'prior', file=buf,
                            return_dict=True)
    out = buf.getvalue()
    assert out.startswith('preconditioner\n')
    assert '\n\nprior\n' in out
    assert set(d) == {'preconditioner', 'prior'}
    assert d['preconditioner']['value'] == 'jacobian'

    buf = io.StringIO()
    assert s.effective_options(file=buf) is None
    assert buf.getvalue().startswith("SVGD options in effect")

    with pytest.raises(ValueError):
        s.effective_options('nope', file=io.StringIO())
    with pytest.raises(TypeError):
        s.effective_options(3, file=io.StringIO())
