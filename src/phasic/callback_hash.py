"""
AST-based callback function hashing for graph caching.

This module provides deterministic hashing of Python callback functions
using Abstract Syntax Tree (AST) normalization. This enables graph caching
without requiring the graph to be built first.

Key Features:
- Robust to formatting changes (whitespace, comments, docstrings)
- Rejects closures (captured variables) with helpful errors
- Includes parameter values in hash key
- Version tagged to invalidate cache on logic changes

Author: Claude Code
Date: 2025-11-14
"""

import ast
import hashlib
import inspect
import sys
import textwrap
from typing import Callable, Dict, Any

# Version tag - increment to invalidate all cached graphs
PHASIC_CALLBACK_VERSION = "1.0"


def hash_callback(callback: Callable, **params) -> str:
    """
    Compute stable hash for callback function + parameters.

    Uses AST-based content hashing (robust to formatting changes).
    The hash is computed WITHOUT building the graph, enabling true caching.

    Parameters
    ----------
    callback : Callable
        Callback function to hash
    **params : dict
        Additional parameters (nr_samples, theta, etc.)

    Returns
    -------
    str
        SHA256 hash hex string (32 characters for cache key)

    Raises
    ------
    ValueError
        If callback uses closures (captured variables)
    TypeError
        If callback is not hashable (C extensions, built-ins)

    Examples
    --------
    >>> import phasic
    >>> @phasic.callback([5])
    ... def coalescent(state, theta=1.0):
    ...     n = state[0]
    ...     if n <= 1:
    ...         return []
    ...     return [[[n-1], [n*(n-1)/2 * theta]]]

    >>> hash1 = hash_callback(coalescent, nr_samples=10, theta=1.0)
    >>> hash1
    'a3f2b8c9def12345...'

    >>> # Same callback, different formatting → SAME hash
    >>> @phasic.callback([5])
    ... def coalescent(state, theta=1.0):
    ...     n = state[0]  # Added comment
    ...     if n <= 1:
    ...         return []
    ...     return [[[n-1], [n*(n-1)/2 * theta]]]

    >>> hash2 = hash_callback(coalescent, nr_samples=10, theta=1.0)
    >>> hash1 == hash2
    True
    """
    components = []

    # Component 1: Version tag (invalidate cache if hashing logic changes)
    components.append(f"version:{PHASIC_CALLBACK_VERSION}")

    # Component 2: Python version (AST structure changes between versions)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    components.append(f"python:{py_version}")

    # Component 3: Unwrap decorators to get original function
    func = callback
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__

    # Component 4: Check for closures (reject if found)
    _detect_closures(func)

    # Component 5: Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise TypeError(
            f"Cannot hash callback '{func.__name__}': unable to get source code.\n"
            "This usually means the function is:\n"
            "  - A C extension function\n"
            "  - A built-in function\n"
            "  - Defined in an interactive session without proper module\n"
            "  - A lambda without accessible source\n"
            f"\nOriginal error: {e}"
        ) from e

    # Component 6: Parse to AST and normalize
    try:
        # Dedent source to fix indentation issues (decorator may add indent)
        dedented_source = textwrap.dedent(source)
        tree = ast.parse(dedented_source)
        normalized = _normalize_ast(tree)
        ast_str = ast.dump(normalized, annotate_fields=True)
        components.append(f"ast:{ast_str}")
    except SyntaxError as e:
        raise ValueError(
            f"Failed to parse callback '{func.__name__}' source code.\n"
            f"Syntax error: {e}"
        ) from e

    # Component 7: Add sorted parameters (deterministic order)
    if params:
        # Sort by key for determinism
        param_items = sorted(params.items())
        # Use repr() for consistent representation
        param_str = ",".join(f"{k}={repr(v)}" for k, v in param_items)
        components.append(f"params:{param_str}")

    # Component 8: Compute SHA256 hash
    combined = "||".join(components)
    hash_bytes = hashlib.sha256(combined.encode('utf-8')).digest()

    # Return first 32 hex chars (128 bits - sufficient for collision resistance)
    return hash_bytes.hex()[:32]


def _normalize_ast(node: ast.AST) -> ast.AST:
    """
    Normalize AST for stable hashing.

    Removes:
    - Position info (lineno, col_offset, end_lineno, end_col_offset)
    - Docstrings (first string in function/class/module body)

    Preserves:
    - Logic structure
    - Variable names
    - Control flow
    - Operations

    Parameters
    ----------
    node : ast.AST
        AST node to normalize

    Returns
    -------
    ast.AST
        Normalized AST (modifies in-place and returns)
    """
    # Step 1: Remove all position attributes
    for child in ast.walk(node):
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(child, attr):
                delattr(child, attr)

    # Step 2: Remove docstrings from functions, classes, modules
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            # Check if first statement is a string constant (docstring)
            if (child.body and
                isinstance(child.body[0], ast.Expr) and
                isinstance(child.body[0].value, ast.Constant) and
                isinstance(child.body[0].value.value, str)):
                # Remove docstring
                child.body = child.body[1:]

    return node


def _detect_closures(func: Callable) -> None:
    """
    Check if function uses closures (captured variables).

    Raises ValueError with helpful message if closures detected.

    Parameters
    ----------
    func : Callable
        Function to check (must be unwrapped)

    Raises
    ------
    ValueError
        If function captures variables from outer scope
    """
    # Check __closure__ attribute
    if func.__closure__ is not None:
        # Get names of captured variables
        captured_vars = func.__code__.co_freevars

        raise ValueError(
            f"Callback '{func.__name__}' uses closures (captured variables).\n"
            "\n"
            "For caching to work, pass captured variables as explicit parameters.\n"
            "\n"
            "Example:\n"
            "  # GOOD (explicit parameter):\n"
            f"  def {func.__name__}(state, theta=2.0):\n"
            "      return [[state[0] - 1, [theta]]]\n"
            "\n"
            "  # BAD (closure - captures 'theta' from outer scope):\n"
            "  theta = 2.0\n"
            f"  def {func.__name__}(state):\n"
            "      return [[state[0] - 1, [theta]]]  # ← Captures theta\n"
            "\n"
            f"Captured variables detected: {', '.join(captured_vars)}"
        )
