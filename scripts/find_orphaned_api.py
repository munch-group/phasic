#!/usr/bin/env python3
"""
Find orphaned functions/methods vs the public API defined in __init__.py.

This script identifies code that is:
1. Not part of the public API (__all__ in __init__.py)
2. Not used internally by other code
3. Potentially dead code that can be removed
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple


class PublicAPICollector(ast.NodeVisitor):
    """Extract __all__ from __init__.py to identify public API."""

    def __init__(self):
        self.public_api: Set[str] = set()

    def visit_Assign(self, node):
        """Find __all__ = [...] assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            self.public_api.add(elt.value)
        self.generic_visit(node)


class FunctionDefCollector(ast.NodeVisitor):
    """Collect all function and method definitions."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.functions: Set[str] = set()
        self.methods: Dict[str, Set[str]] = defaultdict(set)
        self.classes: Set[str] = set()
        self.current_class = None

    def visit_ClassDef(self, node):
        """Visit class definition."""
        self.classes.add(node.name)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        if self.current_class:
            self.methods[self.current_class].add(node.name)
        else:
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
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.method_calls.add(node.func.attr)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attribute access."""
        self.attribute_accesses.add(node.attr)
        self.generic_visit(node)


def get_public_api(init_file: Path) -> Set[str]:
    """Extract public API from __init__.py."""
    with open(init_file, 'r', encoding='utf-8') as f:
        try:
            # Read just the first 50000 chars to find __all__
            content = f.read(50000)
            tree = ast.parse(content, filename=str(init_file))
        except SyntaxError as e:
            print(f"Syntax error in {init_file}: {e}")
            return set()

    collector = PublicAPICollector()
    collector.visit(tree)
    return collector.public_api


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


def find_orphaned_code_vs_api(src_dir: str = "src/phasic"):
    """Find code that's not in public API and not used internally."""
    src_path = Path(src_dir)
    init_file = src_path / "__init__.py"

    # Get public API
    print("Extracting public API from __init__.py...")
    public_api = get_public_api(init_file)
    print(f"Found {len(public_api)} items in public API (__all__)\n")

    # Collect all definitions and calls
    all_functions: Dict[str, List[str]] = defaultdict(list)
    all_methods: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    all_classes: Set[str] = set()
    all_function_calls: Set[str] = set()
    all_method_calls: Set[str] = set()
    all_attribute_accesses: Set[str] = set()

    # Analyze all Python files
    python_files = list(src_path.glob("*.py"))
    print(f"Analyzing {len(python_files)} Python files in {src_dir}...\n")

    for filepath in python_files:
        def_collector, call_collector = analyze_file(filepath)
        if def_collector is None:
            continue

        # Collect definitions
        for func_name in def_collector.functions:
            all_functions[func_name].append(str(filepath.name))

        for class_name, methods in def_collector.methods.items():
            all_classes.add(class_name)
            for method_name in methods:
                all_methods[class_name][method_name].append(str(filepath.name))

        # Collect calls
        all_function_calls.update(call_collector.calls)
        all_method_calls.update(call_collector.method_calls)
        all_attribute_accesses.update(call_collector.attribute_accesses)

    # Categorize functions and methods
    print("=" * 80)
    print("ANALYSIS: Functions and Methods vs Public API")
    print("=" * 80)

    # Magic methods and special cases to exclude
    exclude_names = {
        '__init__', '__str__', '__repr__', '__enter__', '__exit__',
        '__call__', '__len__', '__getitem__', '__setitem__',
        '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
        '__hash__', '__bool__', '__iter__', '__next__',
        '__post_init__', '__del__', '__new__',
        'setup', 'teardown', 'setUp', 'tearDown',
        'main', 'wrapper',  # Common patterns
    }

    # Category 1: Functions in public API but not used internally
    print("\n" + "=" * 80)
    print("PUBLIC API FUNCTIONS (exported in __all__)")
    print("=" * 80)
    public_functions = [name for name in public_api if name in all_functions]
    for func_name in sorted(public_functions):
        used = "✓" if func_name in all_function_calls or func_name in all_attribute_accesses else "✗"
        files = all_functions[func_name]
        print(f"{used} {func_name:40s} {files[0]}")

    # Category 2: Classes in public API
    print("\n" + "=" * 80)
    print("PUBLIC API CLASSES (exported in __all__)")
    print("=" * 80)
    public_classes = [name for name in public_api if name in all_classes]
    for class_name in sorted(public_classes):
        print(f"  {class_name}")
        if class_name in all_methods:
            for method_name in sorted(all_methods[class_name].keys()):
                if method_name in exclude_names:
                    continue
                used = "✓" if method_name in all_method_calls or method_name in all_attribute_accesses else "✗"
                print(f"    {used} {method_name}")

    # Category 3: Orphaned functions (not in API, not called internally)
    print("\n" + "=" * 80)
    print("ORPHANED FUNCTIONS (not in API, not used internally)")
    print("=" * 80)
    orphaned_functions = []
    for func_name, filepaths in all_functions.items():
        if func_name.startswith('_'):
            continue
        if func_name in exclude_names:
            continue
        if func_name in public_api:
            continue
        if func_name not in all_function_calls and func_name not in all_attribute_accesses:
            orphaned_functions.append((func_name, filepaths))

    if orphaned_functions:
        for func_name, filepaths in sorted(orphaned_functions):
            print(f"  {func_name:40s} {filepaths[0]}")
    else:
        print("\n✓ No orphaned functions found.")

    # Category 4: Orphaned methods (class not in API, method not called)
    print("\n" + "=" * 80)
    print("ORPHANED METHODS (class not in API or method not used)")
    print("=" * 80)
    orphaned_methods = []
    for class_name, methods in all_methods.items():
        for method_name, filepaths in methods.items():
            if method_name.startswith('_') and method_name not in exclude_names:
                continue
            if method_name in exclude_names:
                continue
            # Orphaned if: class not in API AND method not called
            if class_name not in public_api or (
                method_name not in all_method_calls and
                method_name not in all_attribute_accesses
            ):
                orphaned_methods.append((class_name, method_name, filepaths))

    # Group by class
    orphaned_by_class = defaultdict(list)
    for class_name, method_name, filepaths in orphaned_methods:
        orphaned_by_class[class_name].append((method_name, filepaths))

    if orphaned_by_class:
        for class_name in sorted(orphaned_by_class.keys()):
            in_api = "✓" if class_name in public_api else "✗"
            print(f"\n  {class_name} (in API: {in_api}):")
            for method_name, filepaths in sorted(orphaned_by_class[class_name]):
                print(f"    {method_name:35s} {filepaths[0]}")
    else:
        print("\n✓ No orphaned methods found.")

    # Category 5: Internal utility functions (used internally but not in API)
    print("\n" + "=" * 80)
    print("INTERNAL UTILITY FUNCTIONS (used internally, not in API)")
    print("=" * 80)
    internal_utils = []
    for func_name in all_functions.keys():
        if func_name.startswith('_'):
            continue
        if func_name in exclude_names:
            continue
        if func_name in public_api:
            continue
        if func_name in all_function_calls or func_name in all_attribute_accesses:
            internal_utils.append(func_name)

    if internal_utils:
        for func_name in sorted(internal_utils):
            files = all_functions[func_name]
            print(f"  {func_name:40s} {files[0]}")
    else:
        print("\n✓ No internal utility functions.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Public API items: {len(public_api)}")
    print(f"  - Functions: {len(public_functions)}")
    print(f"  - Classes: {len(public_classes)}")
    print(f"\nOrphaned code (candidates for removal):")
    print(f"  - Functions: {len(orphaned_functions)}")
    print(f"  - Methods: {len(orphaned_methods)}")
    print(f"\nInternal utilities: {len(internal_utils)}")

    # Potential cleanup suggestions
    if orphaned_functions or orphaned_methods:
        print("\n" + "=" * 80)
        print("CLEANUP SUGGESTIONS")
        print("=" * 80)
        print("\nConsider removing these orphaned functions:")
        for func_name, filepaths in sorted(orphaned_functions)[:10]:
            print(f"  - {func_name} in {filepaths[0]}")
        if len(orphaned_functions) > 10:
            print(f"  ... and {len(orphaned_functions) - 10} more")


if __name__ == "__main__":
    find_orphaned_code_vs_api()
