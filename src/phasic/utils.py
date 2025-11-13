
from functools import wraps, partial

def hand_off(target_func):
    """Decorator that forwards all parameters to a another function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return target_func(*args, **kwargs)
        return wrapper
    return decorator


# Auto-detect notebook environment and setup tqdm wrappers
try:
    from tqdm.notebook import tqdm as notebook_tqdm, trange as notebook_trange
    HAS_NOTEBOOK_TQDM = True
except ImportError:
    HAS_NOTEBOOK_TQDM = False
    notebook_tqdm = None
    notebook_trange = None

from tqdm import tqdm as std_tqdm, trange as std_trange


def _is_notebook():
    """Detect if running in Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return 'ZMQInteractiveShell' in str(type(shell))
    except (ImportError, NameError):
        return False


# Select appropriate tqdm based on environment
if HAS_NOTEBOOK_TQDM and _is_notebook():
    _base_tqdm = notebook_tqdm
    _base_trange = notebook_trange
else:
    _base_tqdm = std_tqdm
    _base_trange = std_trange

# Create wrappers - notebook widgets don't support bar_format parameter
if HAS_NOTEBOOK_TQDM and _is_notebook():
    # Notebook: use native widgets (thin, sleek style matching VS Code)
    pqdm = partial(_base_tqdm)
    prange = partial(_base_trange)
else:
    # Terminal: use custom bar_format for consistent styling
    pqdm = partial(_base_tqdm, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")
    prange = partial(_base_trange, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")
