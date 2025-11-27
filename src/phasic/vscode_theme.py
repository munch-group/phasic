import json5 as json
import os
import platform
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Any, TypeVar, List, Tuple, Dict, Union
from collections.abc import Sequence, MutableSequence, Callable

from .logging_config import get_logger
logger = get_logger(__name__)


def get_vscode_settings_path() -> Path:
    """Get the path to VS Code's user settings.json based on the OS."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "Code" / "User" / "settings.json"
    else:  # Linux
        return Path.home() / ".config" / "Code" / "User" / "settings.json"


def get_vscode_theme() -> str | None:
    """Parse VS Code settings and return the workbench.colorTheme value."""
    settings_path = get_vscode_settings_path()
    
    if not settings_path.exists():
        print(f"Settings file not found at: {settings_path}")
        return None
    
    with open(settings_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove single-line comments (// ...) and multi-line comments (/* ... */)
    # This handles JSONC (JSON with Comments)
    import re
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    settings = json.loads(content)
    return settings.get("workbench.colorTheme", "Default Dark Modern")


def is_vscode_dark_theme() -> bool:
    """Determine if a given VS Code theme name is dark or light."""
    return 'dark' in get_vscode_theme().lower()


def set_phasic_theme(dark:bool=None):
    """
    Set the default theme for the graph plotter.
    The theme can be either 'dark' or 'light'. The default theme is autodetected.
    NOTEBOOK_THEME environment variable can be used to override the theme.

    Parameters
    ----------
    theme : 
        _description_
    """

    if dark is None:
        dark = is_vscode_dark_theme()
        logger.debug(f"No theme specified, autodetected VS Code theme as {'dark' if dark else 'light'}.")

    env_theme = os.environ.get('NOTEBOOK_THEME', None)
    if env_theme is not None:
        dark = env_theme.lower() == 'dark'
        logger.debug(f"Overriding theme from NOTEBOOK_THEME environment variable: {'dark' if dark else 'light'}.")
        print("Overriding theme from NOTEBOOK_THEME environment variable.", sys.stderr)

    if dark:
        logger.debug(f"Setting matplotlib style to dark_background.")
        plt.style.use('dark_background')
    else:
        logger.debug(f"Setting matplotlib style to default.")
        plt.style.use('default')

    if dark:
        logger.debug(f"Updating rcParams for dark theme.")
        plt.rcParams.update({
            'figure.facecolor': '#1F1F1F', 
            'axes.facecolor': '#1F1F1F',
            'grid.linewidth': 0.4,
            'grid.alpha': 0.3,
            })
    else:
        logger.debug(f"Updating rcParams for dark theme.")
        plt.rcParams.update({
            'figure.facecolor': 'white', 
            'axes.facecolor': 'white',
            'grid.linewidth': 0.4,
            'grid.alpha': 0.7,            
            })
    logger.debug(f"Updating rcParams for both themes.")
    plt.rcParams.update({
        'axes.grid': True,
        'axes.grid.axis':     'both',
        'axes.grid.which': 'major',
        'axes.titlelocation': 'right',
        'axes.titlesize': 'large',
        'axes.titleweight': 'normal',
        'axes.labelsize': 'medium',
        'axes.labelweight': 'normal',
        'axes.formatter.use_mathtext': True,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
        'axes.spines.right':  False,
        'xtick.bottom': False,
        'ytick.left': False,
    })


def set_theme(name: str = None):
    """Backwards compatibility wrapper for set_phasic_theme."""
    logger.debug(f"Using set_theme (backwards compatibility).")
    if name is not None:
        set_phasic_theme(dark='dark' in name.lower())
    else:
        set_phasic_theme()


def black_white(ax):
    """Returns black for light backgrounds, white for dark backgrounds."""
    if ax is None:
        ax = plt.gca()
    bg_color = ax.get_facecolor()
    # Convert to grayscale to determine brightness
    luminance = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(bg_color))[2]
    return 'black' if luminance > 0.5 else '#FDFDFD'

class phasic_theme:
    def __init__(self, dark:bool = None):
        self.dark = dark if dark is not None else vscode_theme_is_dark()
        logger.debug(f"Auto-detected theme in context manager as {'dark' if self.dark else 'light'}.")
        self.orig_rcParams = matplotlib.rcParams.copy()

    def __enter__(self):
        logger.debug(f"Setting theme in context manager to {'dark' if self.dark else 'light'}.")
        set_phasic_theme(self.dark)

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"Restoring original rcParams in context manager.")
        matplotlib.rcParams.update(self.orig_rcParams)

