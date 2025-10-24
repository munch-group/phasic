#!/usr/bin/env python3
"""
Add graph_hash fields to registry.json for all existing traces.

This computes the SHA-256 hash of the graph structure for each trace
and adds it to the registry metadata.

Usage:
    python scripts/add_graph_hashes_to_registry.py
"""

import json
import sys
from pathlib import Path

# Add source to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import phasic
from phasic import get_trace
from phasic.trace_elimination import instantiate_from_trace


def compute_hash_for_trace(trace_id: str) -> str:
    """Compute graph hash for a trace by reconstructing the graph."""
    print(f"\nProcessing {trace_id}...")

    # Download trace
    trace = get_trace(trace_id)
    print(f"  Vertices: {trace.n_vertices}, Params: {trace.param_length}")

    # Instantiate graph with dummy parameters
    theta = np.ones(trace.param_length)
    graph = instantiate_from_trace(trace, theta)

    # Compute hash
    hash_result = phasic.hash.compute_graph_hash(graph)
    print(f"  Hash: {hash_result.hash_hex}")

    return hash_result.hash_hex


def main():
    """Add graph_hash to all traces in registry."""
    registry_path = Path("/tmp/phasic-traces/registry.json")

    if not registry_path.exists():
        print(f"Error: Registry not found at {registry_path}")
        print("Run this script after setting up the repository")
        return 1

    print(f"Loading registry from {registry_path}")
    registry = json.loads(registry_path.read_text())

    # Process each trace
    updated_count = 0
    for trace_id in list(registry['traces'].keys()):
        try:
            # Compute hash
            graph_hash = compute_hash_for_trace(trace_id)

            # Add to registry
            registry['traces'][trace_id]['graph_hash'] = graph_hash
            updated_count += 1

        except Exception as e:
            print(f"  ✗ Error processing {trace_id}: {e}")
            continue

    # Save updated registry
    registry_path.write_text(json.dumps(registry, indent=2))

    print(f"\n✓ Updated {updated_count} traces with graph hashes")
    print(f"✓ Registry saved to {registry_path}")
    print("\nNext steps:")
    print("  cd /tmp/phasic-traces")
    print("  git add registry.json")
    print("  git commit -m 'Add graph_hash fields to all traces'")
    print("  git push")

    return 0


if __name__ == "__main__":
    sys.exit(main())
