"""
Diagnostic script for vertex bypass bug in reward_transform

This script traces the vertex bypass operations to help understand
where the graph structure breaks down with 3+ consecutive bypasses.
"""
import phasic
import numpy as np
import json

def print_graph_structure(graph, title):
    """Print detailed graph structure"""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"Total vertices: {graph.vertices_length()}")

    for i in range(graph.vertices_length()):
        v = graph.vertex_at(i)
        state = v.state()
        print(f"\nVertex {i}: state={state}")

        # Can't easily get edges from Python API, but we can try
        # to infer from sampling or PDF behavior

def print_reward_info(rewards, graph):
    """Print reward vector and which vertices will be bypassed"""
    print(f"\n{'='*70}")
    print("Reward Vector Analysis")
    print(f"{'='*70}")
    print(f"Rewards: {rewards}")
    print(f"\nVertices to be bypassed (reward=0):")

    bypass_indices = []
    for i, r in enumerate(rewards):
        if r == 0:
            if i < graph.vertices_length():
                state = graph.vertex_at(i).state()
                print(f"  Vertex {i}: state={state}, reward={r}")
                bypass_indices.append(i)

    if len(bypass_indices) > 0:
        print(f"\nConsecutive bypass chain: {bypass_indices}")
        print(f"Chain length: {len(bypass_indices)}")

        if len(bypass_indices) >= 3:
            print("⚠️  WARNING: 3+ consecutive bypasses - known bug!")
    else:
        print("  (none)")

    return bypass_indices

def test_pdf_normalization(graph, times):
    """Test if PDF is properly normalized"""
    pdf = [graph.pdf(t, granularity=0) for t in times]
    integral = np.trapezoid(pdf, times)

    print(f"\nPDF Normalization Check:")
    print(f"  Integral: {integral:.6f} (expected: ~1.0)")

    if abs(integral - 1.0) < 0.1:
        print(f"  ✓ PDF properly normalized")
        return True
    else:
        print(f"  ❌ PDF NOT normalized (error: {abs(integral - 1.0):.6f})")
        return False

def test_sampling(graph, n_samples=100):
    """Test if sampling works"""
    samples = graph.sample(n_samples)
    mean = np.mean(samples)
    std = np.std(samples)

    print(f"\nSampling Test:")
    print(f"  n={n_samples}, mean={mean:.4f}, std={std:.4f}")
    print(f"  First 5 samples: {samples[:5]}")

    if mean == 0 and std == 0:
        print(f"  ❌ All samples are 0 - graph is broken!")
        return False
    else:
        print(f"  ✓ Sampling works")
        return True

def trace_expected_bypass_operations(rewards):
    """
    Trace what SHOULD happen during bypass operations.
    This shows the expected graph transformations.
    """
    print(f"\n{'='*70}")
    print("Expected Bypass Operations (Sequential)")
    print(f"{'='*70}")

    # Original chain structure
    print("\nOriginal chain: 0 → 1 → 2 → 3 → 4")
    print("Rewards:       [1, 0, 0, 0, 1]")
    print("\nExpected transformations:")

    print("\n1. Bypass vertex 1 (reward=0):")
    print("   Before: 0 → 1 → 2 → 3 → 4")
    print("   After:  0 ----→ 2 → 3 → 4")
    print("   Action: Redirect edge 0→1→2 to 0→2")
    print("   Parent update needed: vertex_parents[2] should change from [1] to [0]")

    print("\n2. Bypass vertex 2 (reward=0):")
    print("   Before: 0 ----→ 2 → 3 → 4")
    print("   After:  0 -------→ 3 → 4")
    print("   Action: Redirect edge 0→2→3 to 0→3")
    print("   ⚠️  PROBLEM: vertex_parents[2] still shows [1] from initialization!")
    print("   ⚠️  Algorithm tries to process parent 1, but 1's edges were modified!")
    print("   ⚠️  This creates inconsistent graph state")

    print("\n3. Bypass vertex 3 (reward=0):")
    print("   Before: 0 -------→ 3 → 4")
    print("   After:  0 ----------→ 4")
    print("   Action: Redirect edge 0→3→4 to 0→4")
    print("   ⚠️  By this point, graph structure is corrupted from step 2")

    print("\nExpected final result: 0 → 4 (direct edge)")
    print("Actual result: Broken graph with no valid paths")

def create_chain_graph():
    """Create simple linear chain graph"""
    def chain_callback(state):
        if not state.size:
            return [([1], 1.0)]
        n = state[0]
        if n == 1:
            return [([2], 1.0)]
        elif n == 2:
            return [([3], 1.0)]
        elif n == 3:
            return [([4], 1.0)]
        else:
            return []

    return phasic.Graph(callback=chain_callback, parameterized=False)

def main():
    print("="*70)
    print("DIAGNOSTIC: Vertex Bypass Bug")
    print("="*70)

    # Create test graph
    print("\n📊 Creating test graph...")
    graph = create_chain_graph()

    print_graph_structure(graph, "Original Graph Structure")

    # Test case: 3 consecutive bypasses (known to fail)
    rewards = [1, 0, 0, 0, 1]

    bypass_indices = print_reward_info(rewards, graph)

    # Show expected operations
    trace_expected_bypass_operations(rewards)

    # Apply reward transformation
    print(f"\n{'='*70}")
    print("Applying reward_transform...")
    print(f"{'='*70}")

    try:
        graph_transformed = graph.reward_transform(rewards)

        print_graph_structure(graph_transformed, "Transformed Graph Structure")

        # Test the transformed graph
        times = np.linspace(0.01, 20.0, 200)

        pdf_ok = test_pdf_normalization(graph_transformed, times)
        sample_ok = test_sampling(graph_transformed)

        # Summary
        print(f"\n{'='*70}")
        print("DIAGNOSIS SUMMARY")
        print(f"{'='*70}")

        print(f"\nTest case: {len(bypass_indices)} consecutive bypasses")
        print(f"Bypassed vertices: {bypass_indices}")
        print(f"Original vertices: {graph.vertices_length()}")
        print(f"Final vertices: {graph_transformed.vertices_length()}")
        print(f"Expected final vertices: 2 (vertices 0 and 4)")

        vertex_count_ok = (graph_transformed.vertices_length() == 2)
        print(f"\nVertex count: {'✓' if vertex_count_ok else '❌'}")
        print(f"PDF normalized: {'✓' if pdf_ok else '❌'}")
        print(f"Sampling works: {'✓' if sample_ok else '❌'}")

        if vertex_count_ok and not (pdf_ok and sample_ok):
            print("\n🔍 KEY FINDING:")
            print("   - Correct number of vertices (2)")
            print("   - But PDF and sampling are broken")
            print("   - This means: Graph structure exists but edges are wrong!")
            print("   - Likely cause: Edge probabilities or connections corrupted")

        print(f"\n{'='*70}")
        print("BUG LOCATION ANALYSIS")
        print(f"{'='*70}")

        print("\nThe bug is in: src/c/phasic.c, _ptd_graph_reward_transform()")
        print("Lines: ~2151-2312 (after my attempted fixes)")
        print("Original lines: ~2151-2294")

        print("\n🔧 SPECIFIC ISSUE:")
        print("In the bypass loop (for i = 0; i < vertices_length; ++i):")
        print("  - When processing vertex i with reward=0")
        print("  - The algorithm reads: vertex_parents[i][p].p")
        print("  - This parent vertex may have been bypassed in iteration j < i")
        print("  - But vertex_parents[i] was built BEFORE any bypasses")
        print("  - So it contains STALE parent references")
        print("  - Attempting to redirect edges from these stale parents fails")

        print("\n📋 DATA STRUCTURES:")
        print("  vertex_parents[i][p].p = pointer to parent vertex")
        print("  vertex_parents[i][p].arr_c_index = index in parent's edge array")
        print("  vertex_edges[i][j].to = pointer to child vertex")
        print("  vertex_edges[i][j].prob = edge probability")
        print("  vertex_edges[i][j].arr_p_index = index in child's parent array")

        print("\n🔍 DEBUGGING STRATEGY:")
        print("1. Add printf statements in C code at:")
        print("   - Start of bypass loop: print i, rewards[i], vertex state")
        print("   - Start of parent loop: print parent vertex index, bypassed status")
        print("   - Edge redirect operations: print old/new edges")
        print("   - End of iteration: print updated graph state")

        print("\n2. Key values to track:")
        print("   - rewards[parent_vertex_index] (0=bypassed, >0=active)")
        print("   - vertex_parents_length[i] (how many parents)")
        print("   - vertex_edges_length[i] (how many children)")
        print("   - Whether bypassed parents are being skipped")

        print("\n3. Expected behavior for vertex 2:")
        print("   - Initial parent: vertex 1")
        print("   - After vertex 1 bypassed: should be vertex 0")
        print("   - But parent list not updated → still shows vertex 1")
        print("   - When processing vertex 2, tries to use vertex 1's edges")
        print("   - Vertex 1's edges already modified → corruption")

        print("\n💡 POTENTIAL FIX APPROACHES:")
        print("A. Update parent lists after each bypass (attempted, failed)")
        print("B. Skip bypassed parents (attempted, fails for all-bypassed case)")
        print("C. Process in topological order (not attempted)")
        print("D. Rebuild parent lists from scratch after each bypass (expensive)")
        print("E. Resolve parent chain recursively when encountered (complex)")

        print("\n📄 Documentation files:")
        print("  /tmp/VERTEX_BYPASS_BUG_REPORT.md - Initial analysis")
        print("  /tmp/VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md - Detailed breakdown")
        print("  /tmp/VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md - All fix attempts")

    except Exception as e:
        print(f"\n❌ ERROR during reward_transform: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("Diagnostic complete")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
