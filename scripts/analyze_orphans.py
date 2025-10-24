#!/usr/bin/env python3
"""
Simple orphan finder: what's defined but never called.
"""

import ast
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict


def find_all_names(src_dir="src/phasic"):
    """Find all defined names and all used names."""

    src_path = Path(src_dir)

    # What's defined
    defined_functions = set()
    defined_classes = set()
    defined_methods = defaultdict(set)  # class -> {methods}

    # What's referenced
    used_names = set()

    for py_file in src_path.glob("*.py"):
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
        except:
            continue

        # Find definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's inside a class
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            defined_methods[parent.name].add(node.name)
                            break
                else:
                    defined_functions.add(node.name)

            elif isinstance(node, ast.ClassDef):
                defined_classes.add(node.name)

        # Find all name references
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)

    return defined_functions, defined_classes, defined_methods, used_names


def main():
    defined_funcs, defined_classes, defined_methods, used = find_all_names()

    # Exclude special names
    exclude = {
        '__init__', '__str__', '__repr__', '__enter__', '__exit__',
        '__call__', '__len__', '__getitem__', '__setitem__',
        'main', 'wrapper', 'setup', 'teardown', 'setUp', 'tearDown',
    }

    print("=" * 80)
    print("ORPHANED FUNCTIONS (defined but never referenced)")
    print("=" * 80)

    orphan_funcs = sorted([
        f for f in defined_funcs
        if f not in used and not f.startswith('_') and f not in exclude
    ])

    for name in orphan_funcs:
        print(f"  {name}")

    print(f"\nTotal: {len(orphan_funcs)}")

    print("\n" + "=" * 80)
    print("ORPHANED METHODS (defined but never referenced)")
    print("=" * 80)

    orphan_methods = []
    for class_name, methods in defined_methods.items():
        class_orphans = [
            m for m in methods
            if m not in used and not m.startswith('_') and m not in exclude
        ]
        if class_orphans:
            orphan_methods.append((class_name, class_orphans))

    for class_name, methods in sorted(orphan_methods):
        print(f"\n{class_name}:")
        for method in sorted(methods):
            print(f"    {method}")

    total_orphan_methods = sum(len(m) for _, m in orphan_methods)
    print(f"\nTotal: {total_orphan_methods}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Defined functions: {len(defined_funcs)}")
    print(f"Defined classes: {len(defined_classes)}")
    print(f"Defined methods: {sum(len(m) for m in defined_methods.values())}")
    print(f"Orphaned functions: {len(orphan_funcs)}")
    print(f"Orphaned methods: {total_orphan_methods}")


if __name__ == "__main__":
    main()
