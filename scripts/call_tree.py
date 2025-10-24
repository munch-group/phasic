#!/usr/bin/env python3
"""
Runtime Call Tree Analyzer using py-spy or cProfile
Profiles Python code execution including Python->C/C++ calls via pybind11.
Shows nested call hierarchy with Python and native code boundaries.

Falls back to cProfile if py-spy is unavailable or requires elevated permissions.
"""

import subprocess
import json
import sys
import argparse
import os
import tempfile
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CallInfo:
    """Information about a function/method call"""
    name: str
    module: str
    file: str
    line: int
    is_native: bool  # C/C++ function
    call_count: int = 0
    children: Dict[str, 'CallInfo'] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get full qualified name"""
        if self.is_native:
            return f"[C++] {self.module}::{self.name}" if self.module else f"[C] {self.name}"
        return f"{self.module}.{self.name}" if self.module else self.name

    @property
    def display_name(self) -> str:
        """Get display name for tree"""
        if self.is_native:
            prefix = "[C++]" if self.module else "[C]"
            return f"{prefix} {self.name}"
        return self.name

    @property
    def location(self) -> str:
        """Get file location"""
        if self.is_native:
            return f"{self.file}" if self.file else "[native]"
        return f"{self.file}:{self.line}" if self.line > 0 else self.file

    def get_max_line_width(self, prefix: str = "", is_last: bool = True, current_depth: int = 0, max_depth: int = 100, show_counts: bool = True) -> int:
        """Calculate maximum line width in the tree INCLUDING tree structure prefixes"""
        if current_depth >= max_depth:
            connector = "└── " if is_last else "├── "
            return len(prefix) + len(connector) + len("... [max depth reached]")

        # Get this node's display name
        func_part = self.display_name
        if show_counts:
            func_part += f" (×{self.call_count})"

        # Calculate this line's width with tree structure
        connector = "└── " if is_last else "├── "
        my_width = len(prefix) + len(connector) + len(func_part)

        # Get max width from children
        max_child_width = my_width
        if self.children and current_depth + 1 < max_depth:
            extension = "    " if is_last else "│   "
            for child in self.children.values():
                child_width = child.get_max_line_width(
                    prefix + extension,
                    False,  # Not last by default
                    current_depth + 1,
                    max_depth,
                    show_counts
                )
                max_child_width = max(max_child_width, child_width)

        return max_child_width


@dataclass
class Pybind11Mapping:
    """Maps Python API to C++ implementation"""
    python_name: str
    python_module: str
    cpp_class: Optional[str]
    cpp_function: str
    cpp_file: str
    binding_type: str  # "class", "function", "method"


class PySpyCallTreeAnalyzer:
    """Analyzes runtime call traces using py-spy"""

    def __init__(self, package_root: Path):
        self.package_root = package_root.resolve()
        self.package_name = self.package_root.name
        self.call_tree: Optional[CallInfo] = None
        self.pybind11_mappings: List[Pybind11Mapping] = []

    def detect_pybind11_mappings(self) -> List[Pybind11Mapping]:
        """Detect pybind11 bindings by scanning C++ source"""
        mappings = []

        # Look for pybind11 binding files
        cpp_files = list(self.package_root.rglob("*pybind*.cpp"))
        cpp_files.extend(self.package_root.rglob("*bindings*.cpp"))

        for cpp_file in cpp_files:
            try:
                with open(cpp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Look for PYBIND11_MODULE
                if "PYBIND11_MODULE" in content:
                    # Extract module name
                    import re
                    module_match = re.search(r'PYBIND11_MODULE\s*\(\s*(\w+)\s*,', content)
                    if module_match:
                        module_name = module_match.group(1)

                        # Find class bindings: py::class_<CppClass>(m, "PyName")
                        class_bindings = re.finditer(
                            r'py::class_<([^>]+)>\s*\([^,]+,\s*"([^"]+)"',
                            content
                        )
                        for match in class_bindings:
                            cpp_class = match.group(1).strip()
                            py_name = match.group(2)
                            mappings.append(Pybind11Mapping(
                                python_name=py_name,
                                python_module=module_name,
                                cpp_class=cpp_class,
                                cpp_function="__init__",
                                cpp_file=str(cpp_file.relative_to(self.package_root)),
                                binding_type="class"
                            ))

                        # Find method bindings: .def("py_name", &CppClass::method)
                        method_bindings = re.finditer(
                            r'\.def\s*\(\s*"([^"]+)"\s*,\s*&([^:]+)::([^,\)]+)',
                            content
                        )
                        for match in method_bindings:
                            py_name = match.group(1)
                            cpp_class = match.group(2)
                            cpp_method = match.group(3)
                            mappings.append(Pybind11Mapping(
                                python_name=py_name,
                                python_module=module_name,
                                cpp_class=cpp_class,
                                cpp_function=cpp_method,
                                cpp_file=str(cpp_file.relative_to(self.package_root)),
                                binding_type="method"
                            ))

                        # Find function bindings: m.def("py_name", &cpp_function)
                        func_bindings = re.finditer(
                            r'm\.def\s*\(\s*"([^"]+)"\s*,\s*&([^,\)]+)',
                            content
                        )
                        for match in func_bindings:
                            py_name = match.group(1)
                            cpp_func = match.group(2)
                            mappings.append(Pybind11Mapping(
                                python_name=py_name,
                                python_module=module_name,
                                cpp_class=None,
                                cpp_function=cpp_func,
                                cpp_file=str(cpp_file.relative_to(self.package_root)),
                                binding_type="function"
                            ))

            except Exception as e:
                print(f"Warning: Could not parse {cpp_file}: {e}", file=sys.stderr)

        return mappings

    def run_pyspy_record(self,
                        python_script: str,
                        script_args: List[str] = None,
                        duration: int = 10,
                        rate: int = 100,
                        native: bool = True) -> Path:
        """Run py-spy to record execution profile"""

        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)

        # Build py-spy command - try with sudo on macOS
        import platform
        is_macos = platform.system() == 'Darwin'

        cmd = []
        if is_macos:
            cmd.append('sudo')

        cmd.extend([
            'py-spy', 'record',
            '--format', 'speedscope',
            '--rate', str(rate),
            '--duration', str(duration),
            '--output', str(output_file),
        ])

        # Native profiling not supported on macOS
        if native and not is_macos:
            cmd.append('--native')
        elif native and is_macos:
            print("Note: Native profiling (--native) not supported on macOS, using Python-only mode")

        # Add the Python script to profile
        cmd.extend(['--', sys.executable, python_script])

        if script_args:
            cmd.extend(script_args)

        print(f"Running py-spy profiler...")
        if is_macos:
            print("Note: py-spy requires sudo on macOS")
        print(f"Command: {' '.join(cmd)}")
        print(f"Duration: {duration}s")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 30  # Add buffer
            )

            if result.returncode != 0:
                print(f"\npy-spy error: {result.stderr}", file=sys.stderr)

                # Check for sudo/permission issues
                if "requires root" in result.stderr or "Permission denied" in result.stderr:
                    print("\nERROR: py-spy requires elevated permissions.", file=sys.stderr)
                    print("Please run with sudo:", file=sys.stderr)
                    print(f"  sudo python {' '.join(sys.argv)}", file=sys.stderr)

                return None

            print(f"Profile saved to: {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            print("py-spy timed out", file=sys.stderr)
            return None
        except FileNotFoundError:
            print("Error: py-spy not found. Install with: pip install py-spy", file=sys.stderr)
            return None

    def run_cprofile(self,
                    python_script: str,
                    script_args: List[str] = None) -> Path:
        """Run cProfile as fallback profiler"""

        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pstats', delete=False) as f:
            output_file = Path(f.name)

        print(f"Running cProfile (fallback profiler)...")
        print(f"Script: {python_script}")

        # Build command to run script with cProfile
        cmd = [
            sys.executable, '-m', 'cProfile',
            '-o', str(output_file),
            python_script
        ]

        if script_args:
            cmd.extend(script_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"cProfile error: {result.stderr}", file=sys.stderr)
                return None

            print(f"Profile saved to: {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            print("cProfile timed out", file=sys.stderr)
            return None

    def parse_cprofile_stats(self, profile_path: Path, min_calls: int = 1) -> CallInfo:
        """Parse cProfile stats into call tree (hierarchical view by module/file)"""

        # Load pstats
        stats = pstats.Stats(str(profile_path))

        # Build call tree from call relationships
        root = CallInfo(
            name="<root>",
            module="",
            file="",
            line=0,
            is_native=False
        )

        # Organize functions by file/module for better grouping
        by_file = defaultdict(list)

        # Get all function stats
        for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
            if nc < min_calls:
                continue

            file, line, func_name = func_key

            # Detect if native (C/C++) code
            is_native = any([
                file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')),
                '<built-in' in file,
                '{' in func_name,  # C++ mangled names
                file == '~',
                'pybind11' in file.lower()
            ])

            # Extract module from file path
            module = ""
            if self.package_root and file and file != '~':
                try:
                    rel_path = Path(file).relative_to(self.package_root)
                    module = str(rel_path.with_suffix('')).replace('/', '.')
                    file_key = str(rel_path)
                except (ValueError, OSError):
                    module = Path(file).stem
                    file_key = file
            else:
                module = Path(file).stem if file != '~' else ""
                file_key = file

            # Only include package code and native/C++ code
            # Be strict: only our source files or native code we care about
            in_package_root = False
            if self.package_root and file:
                try:
                    # Check if file is in package root and not in dependencies
                    Path(file).relative_to(self.package_root)
                    in_package_root = '/.pixi/' not in file and '/site-packages/' not in file
                except (ValueError, OSError):
                    pass

            include = (
                (is_native and ('{' in func_name or 'phasic' in file.lower())) or  # C++ with our code
                file_key.startswith('src/phasic/') or  # Our Python source relative
                file_key.startswith('test_') or  # Test scripts
                in_package_root  # Files in package root, not dependencies
            )

            if not include:
                continue

            by_file[file_key].append({
                'name': func_name,
                'module': module,
                'file': file,
                'line': line,
                'is_native': is_native,
                'call_count': nc,
                'cumtime': ct
            })

        # Sort files by total cumulative time
        sorted_files = sorted(
            by_file.items(),
            key=lambda x: sum(f['cumtime'] for f in x[1]),
            reverse=True
        )

        # Build tree grouped by file
        for file_path, functions in sorted_files[:50]:  # Limit to top 50 files
            # Create a file-level grouping node
            file_node_key = f"FILE:{file_path}"

            if file_node_key not in root.children:
                # Determine if this is a native file
                is_native_file = any(f['is_native'] for f in functions)

                root.children[file_node_key] = CallInfo(
                    name=f"[{'C++' if is_native_file else 'Python'}] {Path(file_path).name}",
                    module="",
                    file=file_path,
                    line=0,
                    is_native=is_native_file,
                    call_count=sum(f['call_count'] for f in functions)
                )

            # Add functions under the file node
            file_node = root.children[file_node_key]

            # Sort functions by call count
            for func_info in sorted(functions, key=lambda x: x['call_count'], reverse=True)[:20]:
                call_key = f"{func_info['name']}@{func_info['file']}:{func_info['line']}"

                if call_key not in file_node.children:
                    file_node.children[call_key] = CallInfo(
                        name=func_info['name'],
                        module=func_info['module'],
                        file=func_info['file'],
                        line=func_info['line'],
                        is_native=func_info['is_native'],
                        call_count=func_info['call_count']
                    )

        return root

    def parse_speedscope_profile(self, profile_path: Path) -> CallInfo:
        """Parse speedscope JSON format into call tree"""

        with open(profile_path, 'r') as f:
            data = json.load(f)

        # Speedscope format has profiles array
        if 'profiles' not in data or not data['profiles']:
            print("No profiles found in speedscope data", file=sys.stderr)
            return None

        profile = data['profiles'][0]  # Use first profile

        # Build call tree from samples
        root = CallInfo(
            name="<root>",
            module="",
            file="",
            line=0,
            is_native=False
        )

        # Get frames and samples
        frames = data.get('shared', {}).get('frames', [])
        samples = profile.get('samples', [])
        weights = profile.get('weights', [1] * len(samples))

        # Build call tree from samples
        for sample, weight in zip(samples, weights):
            current = root
            current.call_count += weight

            # Each sample is a stack trace (list of frame indices)
            for frame_idx in reversed(sample):  # Bottom-up
                if frame_idx >= len(frames):
                    continue

                frame = frames[frame_idx]
                name = frame.get('name', 'unknown')
                file = frame.get('file', '')
                line = frame.get('line', 0)

                # Detect if native (C/C++) code
                is_native = any([
                    file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')),
                    file.startswith('/lib'),
                    file.startswith('/usr/lib'),
                    'pybind11' in file.lower(),
                    name.startswith('_') and not name.startswith('__')
                ])

                # Extract module from file path
                module = ""
                if self.package_root and file:
                    try:
                        rel_path = Path(file).relative_to(self.package_root)
                        module = str(rel_path.with_suffix('')).replace('/', '.')
                    except ValueError:
                        # Not in package root
                        module = Path(file).stem

                # Create unique key for this call site
                call_key = f"{name}@{file}:{line}"

                if call_key not in current.children:
                    current.children[call_key] = CallInfo(
                        name=name,
                        module=module,
                        file=file,
                        line=line,
                        is_native=is_native,
                        call_count=0
                    )

                current.children[call_key].call_count += weight
                current = current.children[call_key]

        return root

    def annotate_pybind11_boundaries(self, node: CallInfo, depth: int = 0):
        """Annotate tree with Python->C++ boundary crossings"""

        for child_key, child in node.children.items():
            # Check if this is a pybind11 boundary crossing
            if child.is_native and not node.is_native:
                # Python -> C/C++ transition
                child.name = f"→{child.name}"
            elif not child.is_native and node.is_native:
                # C/C++ -> Python callback
                child.name = f"←{child.name}"

            # Recurse
            self.annotate_pybind11_boundaries(child, depth + 1)

    def print_call_tree(self,
                       node: CallInfo,
                       prefix: str = "",
                       is_last: bool = True,
                       min_count: int = 1,
                       max_depth: int = 20,
                       current_depth: int = 0,
                       show_counts: bool = True,
                       align_width: Optional[int] = None):
        """Print call tree in ASCII format with aligned file paths"""

        # Calculate alignment width on first call
        if align_width is None and current_depth == 0:
            align_width = node.get_max_line_width(
                prefix="",
                is_last=True,
                current_depth=0,
                max_depth=max_depth,
                show_counts=show_counts
            ) + 5

        if current_depth >= max_depth:
            print(f"{prefix}{'└── ' if is_last else '├── '}... [max depth reached]")
            return

        # Apply min_count filter, but not to root (depth 0)
        if current_depth > 0 and node.call_count < min_count:
            return

        # Print current node
        if current_depth > 0:  # Skip root
            connector = "└── " if is_last else "├── "

            # Format display name
            display = node.display_name
            if show_counts:
                display += f" (×{node.call_count})"

            # Calculate current line width
            current_line_width = len(prefix) + len(connector) + len(display)

            # Pad to alignment width
            padding = max(1, align_width - current_line_width)

            # Format location
            location = node.location
            if location and location != "[native]":
                # Try to make path relative
                try:
                    rel_path = Path(location).relative_to(self.package_root)
                    location = str(rel_path)
                except (ValueError, OSError):
                    pass

            # Print with aligned file path
            print(f"{prefix}{connector}{display}{' ' * padding}{location}")

        # Print children (sorted by call count, descending)
        children = sorted(
            node.children.values(),
            key=lambda x: x.call_count,
            reverse=True
        )

        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension if current_depth > 0 else ""

        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            self.print_call_tree(
                child,
                new_prefix,
                is_last_child,
                min_count,
                max_depth,
                current_depth + 1,
                show_counts,
                align_width
            )

    def save_to_json(self, output_path: str):
        """Save call tree and pybind11 mappings to JSON file"""

        def tree_to_dict(node: CallInfo) -> dict:
            """Convert CallInfo tree to dict for JSON serialization"""
            return {
                'name': node.name,
                'display_name': node.display_name,
                'module': node.module,
                'file': node.file,
                'line': node.line,
                'is_native': node.is_native,
                'call_count': node.call_count,
                'location': node.location,
                'children': [tree_to_dict(child) for child in sorted(
                    node.children.values(),
                    key=lambda x: x.call_count,
                    reverse=True
                )]
            }

        # Convert pybind11 mappings to dicts
        mappings_data = []
        for mapping in self.pybind11_mappings:
            mappings_data.append({
                'python_name': mapping.python_name,
                'python_module': mapping.python_module,
                'cpp_class': mapping.cpp_class,
                'cpp_function': mapping.cpp_function,
                'cpp_file': mapping.cpp_file,
                'binding_type': mapping.binding_type
            })

        data = {
            'package_root': str(self.package_root),
            'call_tree': tree_to_dict(self.call_tree) if self.call_tree else None,
            'pybind11_mappings': mappings_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\nCall tree and mappings saved to {output_path}")

    def print_pybind11_mappings(self):
        """Print detected pybind11 mappings"""
        if not self.pybind11_mappings:
            return

        print("\n" + "="*80)
        print("Pybind11 Python->C++ Mappings")
        print("="*80)

        # Group by type
        by_type = defaultdict(list)
        for mapping in self.pybind11_mappings:
            by_type[mapping.binding_type].append(mapping)

        for binding_type in ['class', 'method', 'function']:
            if binding_type not in by_type:
                continue

            print(f"\n{binding_type.upper()}ES:")
            print("-" * 80)

            for mapping in sorted(by_type[binding_type], key=lambda x: x.python_name):
                if binding_type == 'class':
                    print(f"  Python: {mapping.python_module}.{mapping.python_name}")
                    print(f"  C++:    {mapping.cpp_class}")
                elif binding_type == 'method':
                    print(f"  Python: {mapping.python_name}()")
                    print(f"  C++:    {mapping.cpp_class}::{mapping.cpp_function}()")
                elif binding_type == 'function':
                    print(f"  Python: {mapping.python_module}.{mapping.python_name}()")
                    print(f"  C++:    {mapping.cpp_function}()")

                print(f"  File:   {mapping.cpp_file}")
                print()

    def analyze(self,
                python_script: str,
                script_args: List[str] = None,
                duration: int = 10,
                rate: int = 100,
                native: bool = True,
                min_count: int = 1,
                max_depth: int = 20,
                show_counts: bool = True,
                use_cprofile: bool = False,
                json_output: Optional[str] = None):
        """Run full analysis"""

        # Detect pybind11 mappings
        print(f"Detecting pybind11 bindings in {self.package_root}...")
        self.pybind11_mappings = self.detect_pybind11_mappings()
        print(f"Found {len(self.pybind11_mappings)} pybind11 bindings\n")

        profile_path = None
        profiler_name = "cProfile" if use_cprofile else "py-spy"

        # Try py-spy first (unless explicitly using cProfile)
        if not use_cprofile:
            profile_path = self.run_pyspy_record(
                python_script,
                script_args,
                duration,
                rate,
                native
            )

        # Fall back to cProfile if py-spy failed or explicitly requested
        if not profile_path or not profile_path.exists():
            if not use_cprofile:
                print("\nFalling back to cProfile...\n")
            profiler_name = "cProfile"
            profile_path = self.run_cprofile(python_script, script_args)

        if not profile_path or not profile_path.exists():
            print("Failed to generate profile", file=sys.stderr)
            return

        # Parse profile based on type
        print("\nParsing profile...")
        if profiler_name == "py-spy":
            self.call_tree = self.parse_speedscope_profile(profile_path)
        else:
            self.call_tree = self.parse_cprofile_stats(profile_path, min_calls=min_count)

        if not self.call_tree:
            print("Failed to parse profile", file=sys.stderr)
            return

        # Annotate boundaries
        print("Annotating Python<->C++ boundaries...")
        self.annotate_pybind11_boundaries(self.call_tree)

        # Print results
        print("\n" + "="*80)
        print(f"Runtime Call Tree")
        print(f"Script: {python_script}")
        print(f"Profiler: {profiler_name}")
        if profiler_name == "py-spy":
            print(f"Duration: {duration}s, Rate: {rate} Hz, Native: {native}")
        print("="*80)
        print("\nLegend:")
        print("  → = Python calling C/C++")
        print("  ← = C/C++ calling Python")
        if show_counts:
            print("  (×N) = call count" if profiler_name == "py-spy" else "  (×N) = number of calls")
        print()

        self.print_call_tree(
            self.call_tree,
            min_count=min_count,
            max_depth=max_depth,
            show_counts=show_counts
        )

        # Print pybind11 mappings
        self.print_pybind11_mappings()

        # Save to JSON if requested
        if json_output:
            self.save_to_json(json_output)

        # Cleanup
        try:
            profile_path.unlink()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Runtime call tree analyzer using py-spy with Python/C++ boundary tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Profile a script with cProfile (no sudo required)
  %(prog)s --script test.py --use-cprofile

  # Save results to JSON file
  %(prog)s --script test.py --use-cprofile -o output.json

  # Profile with script arguments
  %(prog)s --script test.py --script-args "--param value" --use-cprofile

  # Show only frequent calls (minimum 10 samples)
  %(prog)s --script test.py --min-count 10 --use-cprofile

  # Increase depth to see deeper call stacks
  %(prog)s --script test.py --max-depth 30 --use-cprofile

  # With py-spy (requires sudo on macOS) to track C/C++ extension calls
  sudo %(prog)s --script test.py --duration 5

Notes:
  - py-spy provides best results (tracks C/C++ calls) but requires sudo on macOS
  - cProfile works without sudo but only sees Python code
  - Always detects pybind11 Python→C++ mappings from source files
  - Arrow symbols (→←) show Python<->C++ transitions in call tree
  - Use --output/-o to save call tree and mappings as JSON
        '''
    )

    parser.add_argument('--package-root',
                       default='.',
                       help='Root directory of package (default: current directory)')
    parser.add_argument('--script',
                       required=True,
                       help='Python script to profile')
    parser.add_argument('--script-args',
                       default='',
                       help='Arguments to pass to script (space-separated string)')
    parser.add_argument('--duration',
                       type=int,
                       default=10,
                       help='Profiling duration in seconds (default: 10)')
    parser.add_argument('--rate',
                       type=int,
                       default=100,
                       help='Sampling rate in Hz (default: 100)')
    parser.add_argument('--no-native',
                       action='store_true',
                       help='Disable native (C/C++) profiling')
    parser.add_argument('--min-count',
                       type=int,
                       default=1,
                       help='Minimum sample count to display (default: 1)')
    parser.add_argument('--max-depth',
                       type=int,
                       default=20,
                       help='Maximum tree depth (default: 20)')
    parser.add_argument('--no-counts',
                       action='store_true',
                       help='Hide call counts')
    parser.add_argument('--use-cprofile',
                       action='store_true',
                       help='Use cProfile instead of py-spy (no sudo required)')
    parser.add_argument('-o', '--output',
                       dest='json_output',
                       help='Save call tree and pybind11 mappings to JSON file')

    args = parser.parse_args()

    # Validate script exists
    if not os.path.exists(args.script):
        print(f"Error: Script '{args.script}' not found", file=sys.stderr)
        sys.exit(1)

    # Parse script args
    script_args = args.script_args.split() if args.script_args else None

    # Run analyzer
    analyzer = PySpyCallTreeAnalyzer(Path(args.package_root))

    try:
        analyzer.analyze(
            python_script=args.script,
            script_args=script_args,
            duration=args.duration,
            rate=args.rate,
            native=not args.no_native,
            min_count=args.min_count,
            max_depth=args.max_depth,
            show_counts=not args.no_counts,
            use_cprofile=args.use_cprofile,
            json_output=args.json_output
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
