"""
Unified logging configuration for phasic.

Provides centralized logging setup with support for environment variables,
file output, and colored console output.

Environment Variables
---------------------
PHASIC_LOG_LEVEL : str
    Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE). Default: WARNING
    Use NONE to completely disable all logging output.
PHASIC_LOG_FILE : str
    Path to log file. If set, logs are written to both console and file.
PHASIC_LOG_FORMAT : str
    Log format string. Default: '[%(levelname)s] %(name)s: %(message)s'
PHASIC_LOG_COLOR : str
    Enable colored output ('1', 'true', 'yes'). Default: auto-detect terminal

Examples
--------
>>> # Enable debug logging via environment variable
>>> import os
>>> os.environ['PHASIC_LOG_LEVEL'] = 'DEBUG'
>>> import phasic
>>>
>>> # Enable debug logging programmatically
>>> import logging
>>> logging.getLogger('phasic').setLevel(logging.DEBUG)
>>>
>>> # Get module-specific logger
>>> from phasic.logging_config import get_logger
>>> logger = get_logger(__name__)
>>> logger.debug("Detailed debug information")
>>> logger.info("General information")
>>> logger.warning("Warning message")
>>> logger.error("Error message")
"""

import logging
import os
import sys
from typing import Optional


# Package-level logger
_PACKAGE_LOGGER_NAME = 'phasic'
_logging_configured = False

# Define NONE level - higher than CRITICAL to disable all logging
NONE = logging.CRITICAL + 10
logging.addLevelName(NONE, 'NONE')


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Save original levelname
        levelname = record.levelname

        # Add color if terminal supports it
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        result = super().format(record)

        # Restore original levelname (for other handlers)
        record.levelname = levelname

        return result


def _should_use_colors() -> bool:
    """
    Determine if colored output should be used.

    Returns
    -------
    bool
        True if colors should be enabled
    """
    # Check environment variable
    color_env = os.environ.get('PHASIC_LOG_COLOR', '').lower()
    if color_env in ('1', 'true', 'yes', 'on'):
        return True
    elif color_env in ('0', 'false', 'no', 'off'):
        return False

    # Default: NO colors (to avoid issues with notebook rendering)
    # Previously auto-detected terminal, but this caused rendering issues
    return False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
    force: bool = False
) -> None:
    """
    Configure package-wide logging.

    This should be called once at package initialization. Subsequent calls
    are ignored unless force=True.

    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Default: from PHASIC_LOG_LEVEL env var, or WARNING
    log_file : str, optional
        Path to log file. If provided, logs are written to both console and file.
        Default: from PHASIC_LOG_FILE env var, or None
    fmt : str, optional
        Log format string.
        Default: from PHASIC_LOG_FORMAT env var, or '[%(levelname)s] %(name)s: %(message)s'
    force : bool, default=False
        If True, reconfigure logging even if already configured

    Examples
    --------
    >>> from phasic.logging_config import setup_logging
    >>> setup_logging(level='DEBUG')
    >>> # Now all phasic loggers will output DEBUG messages
    """
    global _logging_configured

    # Skip if already configured (unless forced)
    if _logging_configured and not force:
        return

    # Get configuration from environment variables or defaults
    if level is None:
        level = os.environ.get('PHASIC_LOG_LEVEL', 'WARNING').upper()

    if log_file is None:
        log_file = os.environ.get('PHASIC_LOG_FILE')

    if fmt is None:
        fmt = os.environ.get('PHASIC_LOG_FORMAT',
                            '[%(levelname)s] %(name)s: %(message)s')

    # Convert level string to logging constant
    if level.upper() == 'NONE':
        numeric_level = NONE
    else:
        numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # Get package logger
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(numeric_level)

    # Remove existing handlers if reconfiguring
    if force:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)

    # Use colored formatter if terminal supports it
    if _should_use_colors():
        console_formatter = ColoredFormatter(fmt)
    else:
        console_formatter = logging.Formatter(fmt)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            # File output should not have colors
            file_formatter = logging.Formatter(fmt)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create log file {log_file}: {e}")

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    # Set up C logging bridge if pybind module is available
    try:
        from . import phasic_pybind

        # Create C logger
        c_logger = logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.c")

        # Define callback that forwards C logs to Python logger
        def c_log_callback(level: int, message: str):
            c_logger.log(level, message)

        # Register callback with C code
        phasic_pybind._c_log_set_callback(c_log_callback)

        # Set C logging level to match Python level
        phasic_pybind._c_log_set_level(numeric_level)

    except (ImportError, AttributeError):
        # C bindings not available or _c_log_* functions not found
        pass

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This function should be used by all phasic modules to create loggers.
    It ensures consistent logger naming and hierarchy.

    Parameters
    ----------
    name : str
        Module name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance

    Examples
    --------
    >>> from phasic.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Module initialized")

    Notes
    -----
    - Loggers use hierarchical names: phasic.module_name
    - Child loggers inherit settings from parent 'phasic' logger
    - This allows fine-grained control: e.g., phasic.svgd can have different level
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()

    # Create hierarchical logger name
    if name.startswith(_PACKAGE_LOGGER_NAME):
        # Already has package prefix
        logger_name = name
    elif name == '__main__':
        # Special case for scripts
        logger_name = _PACKAGE_LOGGER_NAME
    else:
        # Add package prefix
        # Strip 'phasic.' if present to avoid duplication
        if name.startswith('phasic.'):
            name = name[7:]
        logger_name = f"{_PACKAGE_LOGGER_NAME}.{name}"

    return logging.getLogger(logger_name)



class set_log_level:
    def __init__(self, level:str="INFO", module: Optional[str] = None):
        """
        Change logging level at runtime.

        Parameters
        ----------
        level : str
            New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE)
            Use NONE to completely disable all logging.
        module : str, optional
            Specific module to configure (e.g., 'svgd', 'trace_elimination').
            If None, sets level for entire package.

        Examples
        --------
        >>> from phasic.logging_config import set_log_level
        >>> # Enable debug for entire package
        >>> set_log_level('DEBUG')
        >>>
        >>> # Enable debug only for SVGD module
        >>> set_log_level('DEBUG', module='svgd')
        >>>
        >>> # Disable all logging
        >>> set_log_level('NONE')
        """
        self.level = level
        if level.upper() == 'NONE':
            self.numeric_level = NONE
        else:
            self.numeric_level = getattr(logging, level.upper(), logging.WARNING)

        if module is None:
            # Set for entire package
            self.logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
        else:
            # Set for specific module
            self.logger = logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{module}")

        self.original_level = self.logger.level
        self.logger.setLevel(self.numeric_level)

        # Also update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.numeric_level)

        # Update C logging level if setting for entire package
        if module is None:
            try:
                from . import phasic_pybind
                phasic_pybind._c_log_set_level(self.numeric_level)
            except (ImportError, AttributeError):
                pass

    def __enter__(self):
        self.logger.setLevel(self.numeric_level)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.setLevel(self.original_level)




def disable_logging() -> None:
    """
    Disable all phasic logging.

    Useful for testing or when you want complete silence.
    Equivalent to set_log_level('NONE').

    Examples
    --------
    >>> from phasic.logging_config import disable_logging
    >>> disable_logging()
    """
    set_log_level('NONE')


def enable_logging(level: str = 'INFO') -> None:
    """
    Re-enable phasic logging after it was disabled.

    Parameters
    ----------
    level : str, default='INFO'
        Logging level to restore

    Examples
    --------
    >>> from phasic.logging_config import enable_logging
    >>> enable_logging('DEBUG')
    """
    set_log_level(level)

