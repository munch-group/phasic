
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

# Create wrappers with sensible defaults matching cpu_monitor.py style
pqdm = partial(_base_tqdm, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}", leave=False)
prange = partial(_base_trange, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}", leave=False)
