from __future__ import annotations

from functools import wraps, partial
import time
from pathlib import Path
from IPython.display import display, Markdown, HTML
import base64

from typing import Any
from collections.abc import Callable


def hand_off(target_func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that forwards all parameters to another function.

    Parameters
    ----------
    target_func : Callable[..., Any]
        The function to forward calls to.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        A decorator that wraps the decorated function to call ``target_func``.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return target_func(*args, **kwargs)
        return wrapper
    return decorator

def download_link(notebook_path: str | Path) -> None:
    """Display an HTML download link for a Jupyter notebook.

    Creates a base64-encoded download link and displays it as right-aligned
    HTML in a Jupyter notebook environment.

    Parameters
    ----------
    notebook_path : str | Path
        Path to the notebook file to create a download link for.
    """
    nb_path = Path(notebook_path)
    b64 = base64.b64encode(nb_path.read_bytes()).decode()
    display(HTML(f'''<div style="text-align:right;">
  <a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}" title="Download notebook" style="text-decoration:none;">
    Download notebook
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  </a>
</div>'''))
