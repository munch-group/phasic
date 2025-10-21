#!/usr/bin/env python3
"""
Python Call Tree Analyzer
Builds a hierarchical tree of function/method calls starting from __init__.py exports.
Tracks only calls to source code functions, filtering out built-in, standard library,
and third-party library calls.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import argparse


@dataclass
class FunctionSignature:
    """Represents a function/method signature"""
    name: str
    args: List[str]
    kwargs: List[str]
    is_method: bool
    class_name: Optional[str]
    file_path: str
    line_number: int

    def get_function_label(self, show_params: bool = True) -> str:
        """Get just the function/method name with parameters (no file path)"""
        if show_params:
            args_str = ", ".join(self.args)
            kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs)
            params = ", ".join(filter(None, [args_str, kwargs_str]))
            params_part = f"({params})"
        else:
            params_part = "()"

        if self.class_name:
            return f"{self.class_name}.{self.name}{params_part}"
        else:
            return f"{self.name}{params_part}"

    def get_file_location(self) -> str:
        """Get file path and line number"""
        return f"{self.file_path}:{self.line_number}"


@dataclass
class CallNode:
    """Represents a node in the call tree"""
    signature: FunctionSignature
    children: List['CallNode'] = field(default_factory=list)
    visited: bool = False

    def get_max_line_width(self, prefix: str = "", is_last: bool = True, current_depth: int = 0, max_depth: int = 100) -> int:
        """Calculate maximum line width in the tree INCLUDING tree structure prefixes"""
        if current_depth >= max_depth:
            connector = "└── " if is_last else "├── "
            return len(prefix) + len(connector) + len("... [max depth reached]")

        # Get this node's function label (without file path)
        if self.signature.class_name:
            func_part = f"{self.signature.class_name}.{self.signature.name}"
        else:
            func_part = f"{self.signature.name}"

        # Add parameters
        if self.signature.args or self.signature.kwargs:
            args_str = ", ".join(self.signature.args)
            kwargs_str = ", ".join(f"{k}={v}" for k, v in self.signature.kwargs)
            params = ", ".join(filter(None, [args_str, kwargs_str]))
            func_part += f"({params})"
        else:
            func_part += "()"

        # Calculate this line's width with tree structure
        connector = "└── " if is_last else "├── "
        my_width = len(prefix) + len(connector) + len(func_part)

        # Check circular reference case - no children for circular refs
        if self.signature.file_path == "CIRCULAR":
            return my_width

        # Get max width from children
        max_child_width = my_width
        if self.children and current_depth + 1 < max_depth:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(self.children):
                is_last_child = i == len(self.children) - 1
                child_width = child.get_max_line_width(
                    prefix + extension,
                    is_last_child,
                    current_depth + 1,
                    max_depth
                )
                max_child_width = max(max_child_width, child_width)

        return max_child_width


class ImportResolver:
    """Resolves import statements to actual module paths"""
    
    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.import_map: Dict[str, Dict[str, str]] = {}
        
    def add_imports(self, file_path: str, tree: ast.AST):
        """Extract and store imports from a file"""
        self.import_map[file_path] = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.import_map[file_path][name] = alias.name
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        full_name = f"{node.module}.{alias.name}"
                        self.import_map[file_path][name] = full_name
    
    def resolve_call(self, file_path: str, call_name: str) -> Optional[str]:
        """Resolve a function call to its full module path"""
        if file_path in self.import_map:
            return self.import_map[file_path].get(call_name, call_name)
        return call_name


class CallGraphAnalyzer(ast.NodeVisitor):
    """AST visitor that builds a call graph"""

    # Built-in and common standard library functions to exclude
    BUILTIN_NAMES = set(dir(__builtins__))
    COMMON_STDLIB = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'min', 'max', 'abs', 'all', 'any', 'sorted', 'reversed',
        'open', 'input', 'format', 'isinstance', 'issubclass', 'super',
        'getattr', 'setattr', 'hasattr', 'delattr', 'type', 'int', 'str',
        'float', 'bool', 'list', 'dict', 'set', 'tuple', 'frozenset',
        'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'get',
        'items', 'keys', 'values', 'update', 'add', 'join', 'split',
        'strip', 'replace', 'startswith', 'endswith', 'lower', 'upper'
    }

    def __init__(self, file_path: str, package_root: Path, import_resolver: ImportResolver):
        self.file_path = file_path
        self.relative_path = os.path.relpath(file_path, package_root)
        self.package_root = package_root
        self.import_resolver = import_resolver

        # Current context
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

        # Storage
        self.functions: Dict[str, FunctionSignature] = {}
        self.calls: Dict[str, Set[str]] = defaultdict(set)
        self.exports: Set[str] = set()
        
    def get_function_key(self, name: str, class_name: Optional[str] = None) -> str:
        """Generate unique key for a function"""
        if class_name:
            return f"{self.relative_path}::{class_name}:{name}"
        return f"{self.relative_path}::{name}"
    
    def extract_arguments(self, node: ast.FunctionDef) -> Tuple[List[str], List[str]]:
        """Extract function arguments and keyword arguments"""
        args = []
        kwargs = []
        
        # Regular arguments
        for arg in node.args.args:
            # Skip 'self' and 'cls' for methods
            if self.current_class and arg.arg in ('self', 'cls'):
                continue
            args.append(arg.arg)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            default_idx = len(node.args.kwonlyargs) - len(node.args.kw_defaults)
            arg_idx = node.args.kwonlyargs.index(arg)
            if arg_idx >= default_idx:
                default = node.args.kw_defaults[arg_idx - default_idx]
                if default:
                    kwargs.append((arg.arg, ast.unparse(default)))
                else:
                    kwargs.append((arg.arg, 'None'))
            else:
                args.append(arg.arg)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
            
        return args, kwargs
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition"""
        args, kwargs = self.extract_arguments(node)
        
        signature = FunctionSignature(
            name=node.name,
            args=args,
            kwargs=kwargs,
            is_method=self.current_class is not None,
            class_name=self.current_class,
            file_path=self.relative_path,
            line_number=node.lineno
        )
        
        key = self.get_function_key(node.name, self.current_class)
        self.functions[key] = signature
        
        # Track function context
        old_function = self.current_function
        self.current_function = key
        self.generic_visit(node)
        self.current_function = old_function
    
    visit_AsyncFunctionDef = visit_FunctionDef

    # Note: is_source_code_call method removed - filtering now done in build_call_tree

    def visit_Call(self, node: ast.Call):
        """Visit function call"""
        if not self.current_function:
            self.generic_visit(node)
            return

        call_name = None
        class_name = None

        # Direct function call
        if isinstance(node.func, ast.Name):
            call_name = node.func.id

        # Method call
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Could be module.function or instance.method
                call_name = node.func.attr
                class_name = node.func.value.id
            elif isinstance(node.func.value, ast.Attribute):
                # Nested attribute (e.g., self.obj.method)
                call_name = node.func.attr

        if call_name:
            # Try to resolve the full path
            if class_name:
                resolved = self.import_resolver.resolve_call(self.file_path, class_name)
                if resolved:
                    call_key = f"{resolved}:{call_name}"
                else:
                    call_key = f"{class_name}:{call_name}"
            else:
                resolved = self.import_resolver.resolve_call(self.file_path, call_name)
                call_key = resolved if resolved else call_name

            # Track calls (filter mode is set by analyzer)
            # Note: filter_mode will be set by the analyzer when needed
            self.calls[self.current_function].add(call_key)

        self.generic_visit(node)


class PackageAnalyzer:
    """Main analyzer for Python packages"""

    def __init__(self, package_path: str):
        self.package_path = Path(package_path).resolve()
        self.package_name = self.package_path.name
        self.import_resolver = ImportResolver(self.package_path)

        # Results
        self.all_functions: Dict[str, FunctionSignature] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.exports: Set[str] = set()
        
    def analyze_file(self, file_path: Path) -> CallGraphAnalyzer:
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=str(file_path))
                
                # First pass: collect imports
                self.import_resolver.add_imports(str(file_path), tree)
                
                # Second pass: analyze calls
                analyzer = CallGraphAnalyzer(
                    str(file_path), 
                    self.package_path,
                    self.import_resolver
                )
                analyzer.visit(tree)
                
                return analyzer
                
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
                return None
    
    def analyze_package(self):
        """Analyze entire package"""
        # Find all Python files
        python_files = list(self.package_path.rglob("*.py"))
        
        # First, analyze __init__.py to find exports
        init_file = self.package_path / "__init__.py"
        if init_file.exists():
            analyzer = self.analyze_file(init_file)
            if analyzer:
                # Track exports from __init__.py
                for key in analyzer.functions:
                    self.exports.add(key)
                
                # Also track __all__ exports if present
                with open(init_file, 'r') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == '__all__':
                                    if isinstance(node.value, ast.List):
                                        for elt in node.value.elts:
                                            if isinstance(elt, ast.Constant):
                                                self.exports.add(elt.value)
        
        # Analyze all files
        analyzers = []
        for file_path in python_files:
            analyzer = self.analyze_file(file_path)
            if analyzer:
                analyzers.append(analyzer)
                
                # Merge results
                self.all_functions.update(analyzer.functions)
                for func, calls in analyzer.calls.items():
                    self.call_graph[func].update(calls)
    
    def build_call_tree(self, root_key: str, visited: Optional[Set[str]] = None) -> Optional[CallNode]:
        """Build a tree starting from a root function (only package code)"""
        if visited is None:
            visited = set()

        if root_key in visited:
            # Circular dependency - extract function name from key
            if root_key in self.all_functions:
                sig = self.all_functions[root_key]
                return CallNode(
                    signature=FunctionSignature(
                        name=sig.name,
                        args=[],
                        kwargs=[],
                        is_method=sig.is_method,
                        class_name=sig.class_name,
                        file_path="CIRCULAR",
                        line_number=0
                    )
                )
            else:
                # Fallback for unresolved circular references
                return CallNode(
                    signature=FunctionSignature(
                        name="CIRCULAR",
                        args=[],
                        kwargs=[],
                        is_method=False,
                        class_name=None,
                        file_path="CIRCULAR",
                        line_number=0
                    )
                )

        visited.add(root_key)

        # Get function signature - only include if it's in our package
        if root_key not in self.all_functions:
            # Skip - not from our package
            return None

        node = CallNode(signature=self.all_functions[root_key])

        # Add children (only from package code)
        for call_key in self.call_graph.get(root_key, []):
            # Try to find exact match in package code
            found = False
            for func_key in self.all_functions:
                if func_key.endswith(call_key) or call_key in func_key:
                    child = self.build_call_tree(func_key, visited.copy())
                    if child:  # Only add if it's from our package
                        node.children.append(child)
                    found = True
                    break

        return node
    
    def find_class_methods(self, class_name: str) -> List[str]:
        """Find all methods for a given class"""
        methods = []
        for key, sig in self.all_functions.items():
            if sig.class_name == class_name:
                methods.append(key)
        return methods

    def find_function(self, function_name: str) -> Optional[str]:
        """Find a function by name (returns first match)"""
        for key, sig in self.all_functions.items():
            if sig.name == function_name and sig.class_name is None:
                return key
        return None

    def find_method(self, class_name: str, method_name: str) -> Optional[str]:
        """Find a specific method of a class"""
        for key, sig in self.all_functions.items():
            if sig.class_name == class_name and sig.name == method_name:
                return key
        return None

    def build_full_tree(self) -> List[CallNode]:
        """Build complete call tree starting from exports"""
        roots = []

        # Start from exports
        if self.exports:
            for export in self.exports:
                tree = self.build_call_tree(export)
                if tree:
                    roots.append(tree)
        else:
            # If no exports, use all top-level functions
            for key in self.all_functions:
                if '::' in key and not any('::' + key.split('::')[1] in k for k in self.call_graph.values()):
                    tree = self.build_call_tree(key)
                    if tree:
                        roots.append(tree)

        return roots
    
    def print_ascii_tree(self, node: CallNode, prefix: str = "", is_last: bool = True, max_depth: int = 10, current_depth: int = 0, align_width: Optional[int] = None):
        """Print tree in ASCII format like Linux tree command with aligned file paths"""
        # Calculate alignment width on first call (longest line + 5 spaces)
        if align_width is None:
            align_width = node.get_max_line_width(prefix="", is_last=True, max_depth=max_depth) + 5

        if current_depth >= max_depth:
            print(f"{prefix}{'└── ' if is_last else '├── '}... [max depth reached]")
            return

        # Current node
        connector = "└── " if is_last else "├── "
        func_label = node.signature.get_function_label()
        file_location = node.signature.get_file_location()

        # Calculate current line width (prefix + connector + function label)
        current_line_width = len(prefix) + len(connector) + len(func_label)

        # Pad to alignment width
        padding = max(1, align_width - current_line_width)

        # Print with aligned file path
        print(f"{prefix}{connector}{func_label}{' ' * padding}{file_location}")

        # Children
        if node.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self.print_ascii_tree(child, prefix + extension, is_last_child, max_depth, current_depth + 1, align_width)
    
    def save_to_json(self, output_path: str):
        """Save call tree to JSON file"""
        roots = self.build_full_tree()

        # For JSON, we'll create a simple label combining function and file
        def tree_to_dict(node: CallNode) -> Dict:
            func_label = node.signature.get_function_label()
            file_location = node.signature.get_file_location()
            return {
                'label': f"{func_label} -- {file_location}",
                'file': node.signature.file_path,
                'line': node.signature.line_number,
                'children': [tree_to_dict(child) for child in node.children]
            }

        data = {
            'package': str(self.package_path),
            'exports': list(self.exports),
            'call_trees': [tree_to_dict(root) for root in roots]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Call tree saved to {output_path}")
    
    def print_summary(self):
        """Print analysis summary"""
        print(f"\n{'='*60}")
        print(f"Package Analysis Summary: {self.package_name}")
        print(f"{'='*60}")
        print(f"Total functions/methods found: {len(self.all_functions)}")
        print(f"Functions with calls: {len(self.call_graph)}")
        print(f"Exported functions: {len(self.exports)}")
        
        if self.exports:
            print(f"\nExported functions:")
            for export in sorted(self.exports):
                print(f"  - {export}")
        
        print(f"\n{'='*60}")
        print("Call Tree:")
        print(f"{'='*60}\n")
        
        roots = self.build_full_tree()
        for root in roots:
            self.print_ascii_tree(root)
            print()  # Empty line between trees


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Python package call tree (only package code, no external libraries)',
        epilog='''
Examples:
  # Analyze entire package
  %(prog)s src/phasic

  # Show call tree for a specific function
  %(prog)s src/phasic --callable record_elimination_trace

  # Show call tree for all methods of a class
  %(prog)s src/phasic --callable Graph

  # Show call tree for a specific method
  %(prog)s src/phasic --callable Graph.serialize

Note: Only shows calls to functions/methods implemented in the specified package.
External library calls (numpy, jax, etc.) are excluded.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('package_path', help='Path to Python package')
    parser.add_argument('-o', '--output', default='call_tree.json',
                       help='Output JSON file (default: call_tree.json)')
    parser.add_argument('-d', '--max-depth', type=int, default=10,
                       help='Maximum tree depth to display (default: 10)')
    parser.add_argument('--callable', dest='callable_name',
                       help='Show call tree for function, class, or Class.method')

    args = parser.parse_args()

    if not os.path.exists(args.package_path):
        print(f"Error: Package path '{args.package_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    analyzer = PackageAnalyzer(args.package_path)
    print(f"Analyzing package: {args.package_path}")
    analyzer.analyze_package()

    # Handle specific callable request
    if args.callable_name:
        # Check if it's a Class.method format
        if '.' in args.callable_name:
            parts = args.callable_name.split('.', 1)
            class_name = parts[0]
            method_name = parts[1]

            # Try to find the specific method
            key = analyzer.find_method(class_name, method_name)
            if not key:
                print(f"Error: Method '{class_name}.{method_name}' not found", file=sys.stderr)
                sys.exit(1)
            print(f"\n{'='*60}")
            print(f"Call Tree for {class_name}.{method_name}()")
            print(f"{'='*60}\n")
            tree = analyzer.build_call_tree(key)
            if tree:
                analyzer.print_ascii_tree(tree, max_depth=args.max_depth)
            else:
                print("No call tree found")

        else:
            # Try as a class first (show all methods)
            methods = analyzer.find_class_methods(args.callable_name)
            if methods:
                print(f"\n{'='*60}")
                print(f"Call Trees for class {args.callable_name} ({len(methods)} methods)")
                print(f"{'='*60}\n")
                for method_key in sorted(methods):
                    sig = analyzer.all_functions[method_key]
                    print(f"\n{sig.class_name}.{sig.name}():")
                    print("-" * 60)
                    tree = analyzer.build_call_tree(method_key)
                    if tree:
                        analyzer.print_ascii_tree(tree, max_depth=args.max_depth)
                    print()
            else:
                # Try as a function
                key = analyzer.find_function(args.callable_name)
                if not key:
                    print(f"Error: Callable '{args.callable_name}' not found (tried as function, class, and method)", file=sys.stderr)
                    sys.exit(1)
                print(f"\n{'='*60}")
                print(f"Call Tree for {args.callable_name}()")
                print(f"{'='*60}\n")
                tree = analyzer.build_call_tree(key)
                if tree:
                    analyzer.print_ascii_tree(tree, max_depth=args.max_depth)
                else:
                    print("No call tree found")

    else:
        # Print full summary
        analyzer.print_summary()

        # Save to JSON
        analyzer.save_to_json(args.output)


if __name__ == "__main__":
    main()

# # Basic usage
# python call_tree_analyzer.py /path/to/package

# # With custom output file
# python call_tree_analyzer.py /path/to/package -o my_tree.json

# # Limit tree depth
# python call_tree_analyzer.py /path/to/package -d 5

# # Example with a package
# python call_tree_analyzer.py ./src/phasic
# ```

# ## Example Output
# ```
# Package Analysis Summary: phasic
# ============================================================
# Total functions/methods found: 42
# Functions with calls: 18
# Exported functions: 3

# Call Tree:
# ============================================================

# └── src/phasic/__init__.py::Graph.__init__(nodes, edges)
#     ├── src/cpp/phasic_pybind.py::Graph:__init__()
#     ├── src/phasic/graph.py::validate_edges(edges)
#     │   └── src/phasic/utils.py::check_type(obj, expected_type)
#     └── src/phasic/graph.py::build_adjacency_list(nodes, edges)
#         ├── src/phasic/utils.py::create_dict()
#         └── src/phasic/graph.py::Edge:validate()