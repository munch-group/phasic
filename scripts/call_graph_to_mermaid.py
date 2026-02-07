#!/usr/bin/env python3
"""
Mermaid Sequence Diagram Generator for Call Graphs

Reads JSON output from call_tree_analyzer.py and generates mermaid
sequenceDiagram syntax showing the temporal flow of function calls
between modules/classes.

Usage:
    # Generate JSON with existing tool
    python scripts/call_tree_analyzer.py src/phasic --callable Graph.svgd -o call_tree.json

    # Convert to mermaid
    python scripts/call_graph_to_mermaid.py call_tree.json
    python scripts/call_graph_to_mermaid.py call_tree.json --max-depth 3
    python scripts/call_graph_to_mermaid.py call_tree.json --by-class
    python scripts/call_graph_to_mermaid.py call_tree.json -o diagram.md
"""

import argparse
import json
import re
import sys
from collections import OrderedDict


def load_call_tree(json_path):
    """Read JSON call tree file produced by call_tree_analyzer.py."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_label(label):
    """Extract function name and class from a call tree label.

    Labels have the form: "ClassName.method(args) -- file.py:123"
    or "function(args) -- file.py:123"
    """
    # Split on " -- " to separate function part from file part
    parts = label.split(" -- ", 1)
    func_part = parts[0].strip()

    # Strip arguments: take everything before the first "("
    paren_idx = func_part.find("(")
    if paren_idx != -1:
        func_name = func_part[:paren_idx]
    else:
        func_name = func_part

    return func_name


def _participant_key(node, by_class):
    """Return the participant key for a node.

    by_class=False: group by file (module)
    by_class=True:  group by class (or file for module-level functions)
    """
    file_path = node.get("file", "unknown")

    if by_class:
        func_name = _parse_label(node.get("label", ""))
        if "." in func_name:
            # Class.method -> use class name as participant
            return func_name.split(".")[0]

    return file_path


def extract_participants(tree_nodes, by_class):
    """Collect unique participants preserving first-seen order.

    Returns an OrderedDict mapping participant key -> short alias.
    """
    participants = OrderedDict()

    def walk(node):
        if node.get("file") == "CIRCULAR":
            return
        key = _participant_key(node, by_class)
        if key not in participants:
            participants[key] = key
        for child in node.get("children", []):
            walk(child)

    for node in tree_nodes:
        walk(node)

    # Create short aliases for participants
    aliases = OrderedDict()
    seen_aliases = set()
    for key in participants:
        # Generate alias: remove extension, replace special chars
        alias = re.sub(r"\.py$", "", key)
        alias = re.sub(r"[^a-zA-Z0-9_]", "_", alias)
        # Deduplicate
        base = alias
        counter = 2
        while alias in seen_aliases:
            alias = f"{base}{counter}"
            counter += 1
        seen_aliases.add(alias)
        aliases[key] = alias

    return aliases


def generate_sequence_diagram(tree_data, max_depth, by_class):
    """Generate mermaid sequenceDiagram from call tree JSON.

    Returns the diagram as a string.
    """
    tree_nodes = tree_data.get("call_trees", [])
    if not tree_nodes:
        return "sequenceDiagram\n    Note over Client: No call trees found\n"

    participants = extract_participants(tree_nodes, by_class)
    lines = ["sequenceDiagram"]

    # Declare participants
    for key, alias in participants.items():
        display = key
        lines.append(f"    participant {alias} as {display}")

    lines.append("")

    # Walk trees and emit call arrows
    def walk(node, caller_key, depth):
        if depth > max_depth:
            return

        file_path = node.get("file", "unknown")
        is_circular = file_path == "CIRCULAR"
        callee_key = _participant_key(node, by_class)
        func_name = _parse_label(node.get("label", ""))

        # For the root node at depth 0, we just process children
        if caller_key is not None:
            caller_alias = participants.get(caller_key, caller_key)
            callee_alias = participants.get(callee_key, callee_key)

            if is_circular:
                lines.append(
                    f"    Note over {caller_alias}: circular: {func_name}"
                )
                return

            # Emit call arrow
            lines.append(
                f"    {caller_alias}->>{callee_alias}: {func_name}()"
            )
            lines.append(f"    activate {callee_alias}")

        children = node.get("children", [])
        for child in children:
            walk(child, callee_key, depth + 1)

        # Emit return arrow (skip for root)
        if caller_key is not None:
            caller_alias = participants.get(caller_key, caller_key)
            callee_alias = participants.get(callee_key, callee_key)
            lines.append(f"    {callee_alias}-->>{caller_alias}: return")
            lines.append(f"    deactivate {callee_alias}")

    for root in tree_nodes:
        root_key = _participant_key(root, by_class)
        children = root.get("children", [])
        # Show the root as a self-activation
        root_alias = participants.get(root_key, root_key)
        root_func = _parse_label(root.get("label", ""))
        lines.append(f"    Note over {root_alias}: {root_func}()")

        for child in children:
            walk(child, root_key, 1)

        lines.append("")

    return "\n".join(lines) + "\n"


def generate_flowchart(tree_data, max_depth, by_class):
    """Generate mermaid flowchart from call tree JSON.

    Returns the diagram as a string.
    """
    tree_nodes = tree_data.get("call_trees", [])
    if not tree_nodes:
        return "flowchart TD\n    empty[No call trees found]\n"

    lines = ["flowchart TD"]
    node_counter = [0]
    # Track edges to deduplicate
    seen_edges = set()
    # Track declared nodes to avoid redeclaration
    declared_nodes = {}

    def _node_id():
        node_counter[0] += 1
        return f"n{node_counter[0]}"

    def _node_label(node):
        func_name = _parse_label(node.get("label", ""))
        file_path = node.get("file", "unknown")
        if by_class and "." in func_name:
            return f"{func_name}()"
        return f"{func_name}()<br/><small>{file_path}</small>"

    def _dedup_key(node, by_class):
        """Key for deduplicating nodes that represent the same function."""
        func_name = _parse_label(node.get("label", ""))
        file_path = node.get("file", "unknown")
        if by_class:
            return func_name
        return f"{file_path}::{func_name}"

    def walk(node, parent_id, depth):
        if depth > max_depth:
            return

        file_path = node.get("file", "unknown")
        is_circular = file_path == "CIRCULAR"
        func_name = _parse_label(node.get("label", ""))
        dk = _dedup_key(node, by_class)

        if is_circular:
            if dk not in declared_nodes:
                nid = _node_id()
                declared_nodes[dk] = nid
                lines.append(f"    {nid}[/\"{func_name}() ↻\"/]")
            nid = declared_nodes[dk]
            edge = (parent_id, nid)
            if parent_id and edge not in seen_edges:
                seen_edges.add(edge)
                lines.append(f"    {parent_id} -.-> {nid}")
            return

        # Reuse existing node or create new one
        if dk in declared_nodes:
            nid = declared_nodes[dk]
        else:
            nid = _node_id()
            declared_nodes[dk] = nid
            label = _node_label(node)
            lines.append(f'    {nid}["{label}"]')

        # Add edge from parent
        if parent_id:
            edge = (parent_id, nid)
            if edge not in seen_edges:
                seen_edges.add(edge)
                lines.append(f"    {parent_id} --> {nid}")

        for child in node.get("children", []):
            walk(child, nid, depth + 1)

    for root in tree_nodes:
        walk(root, None, 0)

    return "\n".join(lines) + "\n"


def generate_diagram(tree_data, diagram_type, max_depth, by_class):
    """Generate a mermaid diagram of the specified type.

    Parameters
    ----------
    tree_data : dict
        Call tree data (as produced by call_tree_analyzer.py JSON output).
    diagram_type : str
        Either ``'sequence'`` or ``'flow'``.
    max_depth : int
        Maximum call depth to include.
    by_class : bool
        Group participants/nodes by class instead of module.

    Returns
    -------
    str
        Mermaid diagram source.
    """
    if diagram_type == "flow":
        return generate_flowchart(tree_data, max_depth, by_class)
    return generate_sequence_diagram(tree_data, max_depth, by_class)


def main():
    parser = argparse.ArgumentParser(
        description="Convert call_tree_analyzer.py JSON output to mermaid diagrams",
        epilog="""
Examples:
  # Generate JSON first
  python scripts/call_tree_analyzer.py src/phasic --callable Graph.serialize -o tree.json

  # Convert to mermaid sequence diagram (stdout)
  %(prog)s tree.json

  # Flowchart instead of sequence diagram
  %(prog)s tree.json --diagram flow

  # Limit depth and write to file
  %(prog)s tree.json --max-depth 3 -o diagram.md

  # Group participants by class instead of module
  %(prog)s tree.json --by-class
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("json_file", help="JSON file from call_tree_analyzer.py")
    parser.add_argument(
        "--diagram",
        choices=["sequence", "flow"],
        default="sequence",
        help="Diagram type: 'sequence' or 'flow' (default: sequence)",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=5,
        help="Maximum call depth to include (default: 5)",
    )
    parser.add_argument(
        "--by-class",
        action="store_true",
        help="Group participants by class instead of module",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: stdout). Wraps in ```mermaid fences if .md",
    )

    args = parser.parse_args()

    data = load_call_tree(args.json_file)
    diagram = generate_diagram(data, args.diagram, args.max_depth, args.by_class)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if args.output.endswith(".md"):
                f.write("```mermaid\n")
                f.write(diagram)
                f.write("```\n")
            else:
                f.write(diagram)
        print(f"Diagram written to {args.output}", file=sys.stderr)
    else:
        print(diagram)


if __name__ == "__main__":
    main()
