#!/usr/bin/env python3
"""
Find orphaned functions and methods in the phasic codebase.

This script uses Python AST to analyze the codebase and identify functions/methods
that are defined but never called (orphaned code).
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple


class FunctionDefCollector(ast.NodeVisitor):
    """Collect all function and method definitions."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.functions: Set[str] = set()
        self.methods: Dict[str, Set[str]] = defaultdict(set)
        self.current_class = None

    def visit_ClassDef(self, node):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        if self.current_class:
            # It's a method
            self.methods[self.current_class].add(node.name)
        else:
            # It's a function
            self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)


class FunctionCallCollector(ast.NodeVisitor):
    """Collect all function and method calls."""

    def __init__(self):
        self.calls: Set[str] = set()
        self.method_calls: Set[str] = set()
        self.attribute_accesses: Set[str] = set()

    def visit_Call(self, node):
        """Visit function/method call."""
        # Direct function call: foo()
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)

        # Method call: obj.method()
        elif isinstance(node.func, ast.Attribute):
            self.method_calls.add(node.func.attr)

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attribute access (catches references even without calls)."""
        self.attribute_accesses.add(node.attr)
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Tuple[FunctionDefCollector, FunctionCallCollector]:
    """Analyze a single Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=str(filepath))
        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
            return None, None

    def_collector = FunctionDefCollector(str(filepath))
    def_collector.visit(tree)

    call_collector = FunctionCallCollector()
    call_collector.visit(tree)

    return def_collector, call_collector


def find_orphaned_code(src_dir: str = "src/phasic"):
    """Find orphaned functions and methods in the codebase."""
    src_path = Path(src_dir)

    # Collect all definitions and calls
    all_functions: Dict[str, List[str]] = defaultdict(list)
    all_methods: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    all_function_calls: Set[str] = set()
    all_method_calls: Set[str] = set()
    all_attribute_accesses: Set[str] = set()

    # Analyze all Python files
    python_files = list(src_path.glob("*.py"))
    print(f"Analyzing {len(python_files)} Python files in {src_dir}...\n")

    for filepath in python_files:
        if filepath.name.startswith("_") and filepath.name != "__init__.py":
            continue  # Skip private modules unless it's __init__

        def_collector, call_collector = analyze_file(filepath)
        if def_collector is None:
            continue

        # Collect definitions
        for func_name in def_collector.functions:
            all_functions[func_name].append(str(filepath))

        for class_name, methods in def_collector.methods.items():
            for method_name in methods:
                all_methods[class_name][method_name].append(str(filepath))

        # Collect calls
        all_function_calls.update(call_collector.calls)
        all_method_calls.update(call_collector.method_calls)
        all_attribute_accesses.update(call_collector.attribute_accesses)

    # Find orphaned functions
    print("=" * 80)
    print("ORPHANED FUNCTIONS")
    print("=" * 80)

    # Exclude common special functions and magic methods
    exclude_functions = {
        '__init__', '__str__', '__repr__', '__enter__', '__exit__',
        '__call__', '__len__', '__getitem__', '__setitem__',
        '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
        '__hash__', '__bool__', '__iter__', '__next__',
        'setup', 'teardown', 'setUp', 'tearDown',  # Test methods
        'main',  # Entry points
    }

    orphaned_functions = []
    for func_name, filepaths in all_functions.items():
        if func_name.startswith('_'):
            continue  # Skip private functions
        if func_name in exclude_functions:
            continue
        if func_name not in all_function_calls and func_name not in all_attribute_accesses:
            orphaned_functions.append((func_name, filepaths))

    if orphaned_functions:
        for func_name, filepaths in sorted(orphaned_functions):
            print(f"\n{func_name}:")
            for filepath in filepaths:
                print(f"  - {filepath}")
    else:
        print("\nNo orphaned functions found.")

    # Find orphaned methods
    print("\n" + "=" * 80)
    print("ORPHANED METHODS")
    print("=" * 80)

    orphaned_methods = []
    for class_name, methods in all_methods.items():
        for method_name, filepaths in methods.items():
            if method_name.startswith('_') and method_name not in exclude_functions:
                continue  # Skip private methods
            if method_name in exclude_functions:
                continue
            if method_name not in all_method_calls and method_name not in all_attribute_accesses:
                orphaned_methods.append((class_name, method_name, filepaths))

    if orphaned_methods:
        for class_name, method_name, filepaths in sorted(orphaned_methods):
            print(f"\n{class_name}.{method_name}:")
            for filepath in filepaths:
                print(f"  - {filepath}")
    else:
        print("\nNo orphaned methods found.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total functions defined: {sum(len(v) for v in all_functions.values())}")
    print(f"Total methods defined: {sum(len(methods) for methods in all_methods.values() for _ in methods.values())}")
    print(f"Orphaned functions: {len(orphaned_functions)}")
    print(f"Orphaned methods: {len(orphaned_methods)}")

    # Additional analysis: List all function calls not defined in the analyzed files
    print("\n" + "=" * 80)
    print("EXTERNAL/MISSING FUNCTION CALLS (might be from imports or C extensions)")
    print("=" * 80)
    undefined_calls = all_function_calls - set(all_functions.keys())
    # Filter out common built-ins
    builtins = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
        'open', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
        'super', 'property', 'staticmethod', 'classmethod',
        'min', 'max', 'sum', 'abs', 'round', 'sorted', 'reversed',
    }
    undefined_calls = undefined_calls - builtins

    if undefined_calls:
        print("\nCalls to functions not defined in analyzed files (sample):")
        for call in sorted(list(undefined_calls)[:30]):
            print(f"  - {call}")
        if len(undefined_calls) > 30:
            print(f"  ... and {len(undefined_calls) - 30} more")


if __name__ == "__main__":
    find_orphaned_code()
