"""Notebook progress bars that remove only themselves when they finish.

``vscodenb``'s ``HTMLProgressBar`` clears itself with
``IPython.display.clear_output()``, which wipes the WHOLE cell output --
including any warnings or prints emitted while the loop ran. This
subclass instead blanks only the display handles the bars created.
"""

from vscodenb.progress import HTMLProgressBar, _get_display_handles


class _SelfClearingProgressBar(HTMLProgressBar):

    def _clear_display(self):
        """Blank this bar's own display outputs, leaving the rest of the cell."""
        if self._use_html:
            from IPython.display import HTML
            for handle in _get_display_handles().values():
                handle.update(HTML(''))


def pqdm(iterable=None, total=None, desc='', persist=False, **kwargs):
    """``vscodenb.pqdm`` that clears only its own output on completion."""
    return _SelfClearingProgressBar(iterable=iterable, total=total, desc=desc,
                                    persist=persist, **kwargs)


def prange(n, desc='', persist=False, **kwargs):
    """``vscodenb.prange`` that clears only its own output on completion."""
    return _SelfClearingProgressBar(iterable=range(n), total=n, desc=desc,
                                    persist=persist, **kwargs)
