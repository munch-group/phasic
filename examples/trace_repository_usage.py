"""
IPFS Trace Repository - Usage Examples

This file demonstrates how to use the IPFS-based trace repository
for downloading, browsing, and publishing pre-computed traces.

Prerequisites:
    pip install phasic[ipfs]  # For full functionality
    # OR
    pip install phasic        # Basic functionality (HTTP gateways only)

Optional (for best performance):
    brew install ipfs         # macOS
    ipfs init && ipfs daemon &
"""

import numpy as np


# ============================================================================
# Example 1: Download and Use a Trace (Simplest)
# ============================================================================

def example_download_and_use():
    """Download a trace and use it for SVGD inference."""
    from phasic import get_trace
    from phasic.trace_elimination import trace_to_log_likelihood
    from phasic import SVGD

    # Download trace (one-liner)
    # - Checks local cache first
    # - Downloads from IPFS if not cached
    # - Falls back to HTTP gateways if IPFS unavailable
    trace = get_trace("coalescent_n5_theta1")

    # Generate some mock observed data
    observed_times = np.array([1.2, 2.3, 0.8, 1.5, 3.2])

    # Create log-likelihood function from trace
    log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)

    # Run SVGD inference
    svgd = SVGD(
        log_prob=log_lik,
        theta_dim=1,
        n_particles=100,
        n_iterations=1000
    )

    results = svgd.fit()
    print(f"Posterior mean theta: {results['theta_mean']}")
    print(f"Posterior std theta: {results['theta_std']}")


# ============================================================================
# Example 2: Browse Available Traces
# ============================================================================

def example_browse_traces():
    """Browse and filter available traces in the repository."""
    from phasic import TraceRegistry

    # Create registry (auto-updates from GitHub)
    registry = TraceRegistry()

    # List all available traces
    print("\n=== All Available Traces ===")
    all_traces = registry.list_traces()
    for t in all_traces:
        print(f"{t['trace_id']}: {t['description']}")

    # Filter by domain
    print("\n=== Population Genetics Traces ===")
    pop_gen = registry.list_traces(domain="population-genetics")
    for t in pop_gen:
        print(f"{t['trace_id']}: {t['vertices']} vertices, {t['param_length']} params")

    # Filter by model type
    print("\n=== Coalescent Models ===")
    coalescent = registry.list_traces(model_type="coalescent")
    for t in coalescent:
        print(f"{t['trace_id']}: {t['description']}")

    # Filter by tags
    print("\n=== Structured Models ===")
    structured = registry.list_traces(tags=["structured"])
    for t in structured:
        print(f"{t['trace_id']}: {t['description']}")


# ============================================================================
# Example 3: Download Multiple Traces for Offline Use
# ============================================================================

def example_bulk_download():
    """Download a collection of traces for offline usage."""
    from phasic import install_trace_library

    # Download a specific collection
    # This downloads all traces in the collection to local cache
    install_trace_library("coalescent_basic")

    # Now you can use any trace from the collection offline
    # (they're all cached in ~/.phasic_traces/)


# ============================================================================
# Example 4: Manual Registry Operations
# ============================================================================

def example_manual_registry():
    """Detailed registry operations for advanced usage."""
    from phasic import TraceRegistry

    # Create registry with custom cache directory
    from pathlib import Path
    custom_cache = Path("/tmp/my_traces")
    registry = TraceRegistry(cache_dir=custom_cache, auto_update=False)

    # Manually update registry
    registry.update_registry()

    # Get trace with forced re-download
    trace = registry.get_trace("coalescent_n5_theta1", force_download=True)

    # Access trace data
    print(f"Vertex rates: {trace['vertex_rates']}")
    print(f"Edge probs: {trace['edge_probs']}")
    print(f"Vertex targets: {trace['vertex_targets']}")


# ============================================================================
# Example 5: Publish Your Own Trace
# ============================================================================

def example_publish_trace():
    """Build, record, and publish a trace to IPFS."""
    from phasic import Graph, TraceRegistry
    from phasic.trace_elimination import record_elimination_trace

    # Build a simple coalescent model
    def coalescent_callback(state):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [(np.array([n - 1]), 0.0, [rate])]

    graph = Graph(
        state_length=1,
        callback=coalescent_callback,
        parameterized=True,
        nr_samples=5
    )

    # Record elimination trace
    trace = record_elimination_trace(graph, param_length=1)

    # Prepare metadata
    metadata = {
        "model_type": "coalescent",
        "domain": "population-genetics",
        "param_length": 1,
        "vertices": 5,
        "edges": 10,
        "description": "Kingman coalescent for n=5 samples with theta parameter",
        "created": "2025-10-21",
        "author": "Your Name <your.email@domain.com>",
        "citation": {
            "text": "Røikjer, Hobolth & Munch (2022)",
            "doi": "10.1007/s11222-022-10155-6",
            "url": "https://doi.org/10.1007/s11222-022-10155-6"
        },
        "tags": ["coalescent", "kingman", "population-genetics"],
        "license": "MIT"
    }

    # Publish to IPFS
    registry = TraceRegistry()
    cid = registry.publish_trace(
        trace=trace,
        trace_id="my_coalescent_n5",
        metadata=metadata,
        submit_pr=True  # Prints instructions for submitting to public registry
    )

    print(f"\n✓ Published to IPFS with CID: {cid}")
    print(f"✓ Follow the printed instructions to add to public registry")


# ============================================================================
# Example 6: Using IPFS Backend Directly (Advanced)
# ============================================================================

def example_ipfs_backend():
    """Direct usage of IPFSBackend for custom workflows."""
    from phasic import IPFSBackend
    from pathlib import Path

    # Create backend (auto-starts daemon if available)
    backend = IPFSBackend()

    # Download content by CID
    cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    content = backend.get(cid)
    print(f"Downloaded {len(content)} bytes")

    # Download to file
    output_path = Path("/tmp/downloaded_content")
    backend.get(cid, output_path=output_path)
    print(f"Saved to {output_path}")

    # Publish content (requires IPFS daemon)
    try:
        local_file = Path("/tmp/my_file.txt")
        local_file.write_text("Hello IPFS!")
        new_cid = backend.add(local_file)
        print(f"Published with CID: {new_cid}")
    except Exception as e:
        print(f"Publishing requires IPFS daemon: {e}")


# ============================================================================
# Example 7: Progressive Enhancement Demo
# ============================================================================

def example_progressive_enhancement():
    """Demonstrate how the system works at different levels."""
    from phasic import IPFSBackend

    # Level 1: HTTP gateways only (no IPFS, no ipfshttpclient)
    print("\n=== Level 1: HTTP Gateways (Zero Config) ===")
    backend = IPFSBackend(auto_start=False)
    # Will use public gateways: ipfs.io, cloudflare-ipfs.com, etc.

    # Level 2: Auto-start daemon (IPFS installed, ipfshttpclient installed)
    print("\n=== Level 2: Auto-Start Daemon ===")
    backend = IPFSBackend(auto_start=True)
    # Will auto-start daemon if installed but not running
    # Faster downloads via local node

    # Level 3: Existing daemon (user started IPFS service)
    print("\n=== Level 3: Existing Daemon (Optimal) ===")
    backend = IPFSBackend()
    # Connects to existing daemon
    # Lowest latency, contributes to network


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Download and use trace", example_download_and_use),
        "2": ("Browse available traces", example_browse_traces),
        "3": ("Bulk download collection", example_bulk_download),
        "4": ("Manual registry operations", example_manual_registry),
        "5": ("Publish your own trace", example_publish_trace),
        "6": ("Direct IPFSBackend usage", example_ipfs_backend),
        "7": ("Progressive enhancement demo", example_progressive_enhancement),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            print(f"\n{'='*70}")
            print(f"Running Example {choice}: {examples[choice][0]}")
            print(f"{'='*70}\n")
            examples[choice][1]()
        else:
            print(f"Unknown example: {choice}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("\nAvailable examples:")
        for key, (desc, _) in examples.items():
            print(f"  {key}. {desc}")
        print(f"\nUsage: python {sys.argv[0]} <example_number>")
        print(f"Example: python {sys.argv[0]} 1")
