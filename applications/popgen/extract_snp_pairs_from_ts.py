#!/usr/bin/env python3
"""
Compute SNP descendant counts across adjacent trees in a TreeSequence.

For each pair of adjacent trees, and for each pair of SNPs (one in each tree),
computes the number of sampled descendants from a specified sample subset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tskit


def load_sample_subset(sample_file: Path, ts: tskit.TreeSequence) -> np.ndarray:
    """
    Load sample identifiers from file and convert to node IDs.
    
    Parameters
    ----------
    sample_file : Path
        File with one sample identifier per line
    ts : TreeSequence
        The tree sequence to match samples against
        
    Returns
    -------
    np.ndarray
        Array of node IDs corresponding to the sample subset
    """
    # Read sample identifiers from file
    with open(sample_file) as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    # Build mapping from individual/sample names to node IDs
    # First try individual names
    individual_name_to_nodes = {}
    for ind in ts.individuals():
        if ind.metadata:
            # Try to get name from metadata
            try:
                if isinstance(ind.metadata, dict):
                    name = ind.metadata.get('name') or ind.metadata.get('id')
                elif hasattr(ind.metadata, 'name'):
                    name = ind.metadata.name
                else:
                    name = None
                if name:
                    individual_name_to_nodes[str(name)] = list(ind.nodes)
            except:
                pass
        # Also map by individual ID
        individual_name_to_nodes[str(ind.id)] = list(ind.nodes)
    
    # Also try sample node IDs directly
    sample_nodes = set(ts.samples())
    
    # Match sample identifiers to node IDs
    matched_nodes = []
    unmatched = []
    
    for sample_id in sample_ids:
        matched = False
        
        # Try as individual name/id
        if sample_id in individual_name_to_nodes:
            matched_nodes.extend(individual_name_to_nodes[sample_id])
            matched = True
        # Try as node ID
        elif sample_id.isdigit():
            node_id = int(sample_id)
            if node_id in sample_nodes:
                matched_nodes.append(node_id)
                matched = True
        
        if not matched:
            unmatched.append(sample_id)
    
    if unmatched:
        print(f"Warning: {len(unmatched)} sample identifiers not found in TreeSequence:", 
              file=sys.stderr)
        for s in unmatched[:10]:
            print(f"  {s}", file=sys.stderr)
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more", file=sys.stderr)
    
    if not matched_nodes:
        raise ValueError("No samples matched. Check sample file format.")
    
    return np.array(sorted(set(matched_nodes)), dtype=np.int32)


def get_mutations_in_tree(ts: tskit.TreeSequence, tree: tskit.Tree) -> list:
    """
    Get all mutations (SNPs) within a tree's genomic interval.
    
    Returns list of (site_id, site_position, node_id) tuples.
    """
    mutations = []
    left, right = tree.interval
    
    for site in ts.sites():
        if left <= site.position < right:
            for mutation in site.mutations:
                mutations.append((site.id, site.position, mutation.node))
    
    return mutations


def count_sample_descendants(tree: tskit.Tree, node: int, 
                             sample_set: set) -> int:
    """
    Count how many samples from sample_set are descendants of node in tree.
    
    Uses tree.samples(node) which efficiently iterates through all sample
    descendants of a node.
    """
    count = 0
    for sample in tree.samples(node):
        if sample in sample_set:
            count += 1
    return count


def count_sample_descendants_direct(ts: tskit.TreeSequence, tree_index: int,
                                    node: int, sample_set: set) -> int:
    """
    Count descendants by getting tree at specific index.
    
    Used as fallback when precomputed counts aren't available.
    """
    tree = ts.at_index(tree_index)
    return count_sample_descendants(tree, node, sample_set)


def precompute_sample_counts(tree: tskit.Tree, sample_subset: np.ndarray) -> dict:
    """
    Precompute sample descendant counts for all nodes in a tree.
    
    This is more efficient when we need counts for many nodes in the same tree.
    Uses a postorder traversal to compute counts bottom-up.
    
    Parameters
    ----------
    tree : tskit.Tree
        The tree to compute counts for
    sample_subset : np.ndarray
        Array of sample node IDs to count
        
    Returns
    -------
    dict
        Mapping from node ID to descendant count
    """
    sample_set = set(sample_subset)
    counts = {}
    
    # Initialize leaf counts
    for node in tree.nodes():
        if tree.is_leaf(node):
            counts[node] = 1 if node in sample_set else 0
    
    # Postorder traversal to sum up counts
    for node in tree.nodes(order="postorder"):
        if node not in counts:
            counts[node] = sum(counts.get(child, 0) for child in tree.children(node))
    
    return counts


def process_adjacent_trees(ts: tskit.TreeSequence, 
                           sample_subset: np.ndarray,
                           output_file: str = None,
                           verbose: bool = False,
                           use_precompute: bool = True,
                           max_offset: int = 1):
    """
    Process pairs of trees within max_offset and compute SNP descendant counts.
    
    For each pair of trees (left_tree, right_tree) where right_tree.index - left_tree.index <= max_offset,
    and for each SNP in left_tree paired with each SNP in right_tree, output:
    - left_snp_position
    - left_snp_node
    - right_snp_position  
    - right_snp_node
    - left_descendant_count
    - right_descendant_count
    
    Parameters
    ----------
    ts : TreeSequence
        Input tree sequence
    sample_subset : np.ndarray
        Sample node IDs to count descendants for
    output_file : str, optional
        Output file path (default: stdout)
    verbose : bool
        Print progress to stderr
    use_precompute : bool
        Use precomputed counts (faster for trees with many mutations)
    max_offset : int
        Maximum tree index difference to consider (default: 1 for adjacent only)
    """
    sample_set = set(sample_subset)
    
    # Open output file or use stdout
    if output_file:
        out = open(output_file, 'w')
    else:
        out = sys.stdout
    
    # Write header
    header = [
        "left_tree_index",
        "right_tree_index",
        "tree_offset",
        "left_tree_start",
        "left_tree_end",
        "right_tree_start", 
        "right_tree_end",
        "left_snp_site_id",
        "left_snp_position",
        "left_snp_node",
        "right_snp_site_id",
        "right_snp_position",
        "right_snp_node",
        "left_snp_descendants",
        "right_snp_descendants"
    ]
    print("\t".join(header), file=out)
    
    # First pass: collect all mutations and precompute counts for each tree
    if verbose:
        print("Collecting mutations from all trees...", file=sys.stderr)
    
    tree_data = []  # List of (tree_index, interval, mutations, counts)
    n_trees = ts.num_trees
    
    for tree in ts.trees():
        mutations = get_mutations_in_tree(ts, tree)
        
        if mutations:
            if use_precompute:
                counts = precompute_sample_counts(tree, sample_subset)
            else:
                counts = None
            tree_data.append((tree.index, tree.interval, mutations, counts))
        
        if verbose and tree.index % 1000 == 0:
            print(f"Scanned {tree.index}/{n_trees} trees...", file=sys.stderr)
    
    if verbose:
        print(f"Found {len(tree_data)} trees with mutations", file=sys.stderr)
        print("Processing tree pairs...", file=sys.stderr)
    
    # Second pass: process all pairs within max_offset
    n_pairs = 0
    n_tree_pairs = 0
    
    for i, (left_idx, left_interval, left_mutations, left_counts) in enumerate(tree_data):
        left_start, left_end = left_interval
        
        # Look at subsequent trees within max_offset
        for j in range(i + 1, len(tree_data)):
            right_idx, right_interval, right_mutations, right_counts = tree_data[j]
            
            offset = right_idx - left_idx
            if offset > max_offset:
                break  # No need to look further
            
            right_start, right_end = right_interval
            n_tree_pairs += 1
            
            for left_site_id, left_pos, left_node in left_mutations:
                # Get descendant count for left SNP
                if use_precompute and left_counts:
                    left_desc = left_counts.get(left_node, 0)
                else:
                    left_desc = count_sample_descendants_direct(ts, left_idx, left_node, sample_set)
                
                for right_site_id, right_pos, right_node in right_mutations:
                    # Get descendant count for right SNP
                    if use_precompute and right_counts:
                        right_desc = right_counts.get(right_node, 0)
                    else:
                        right_desc = count_sample_descendants_direct(ts, right_idx, right_node, sample_set)
                    
                    row = [
                        left_idx,
                        right_idx,
                        offset,
                        f"{left_start:.0f}",
                        f"{left_end:.0f}",
                        f"{right_start:.0f}",
                        f"{right_end:.0f}",
                        left_site_id,
                        f"{left_pos:.0f}",
                        left_node,
                        right_site_id,
                        f"{right_pos:.0f}",
                        right_node,
                        left_desc,
                        right_desc
                    ]
                    print("\t".join(map(str, row)), file=out)
                    n_pairs += 1
        
        if verbose and i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(tree_data)} trees with mutations...", file=sys.stderr)
    
    if output_file:
        out.close()
    
    if verbose:
        print(f"Total tree pairs processed: {n_tree_pairs}", file=sys.stderr)
        print(f"Total SNP pairs processed: {n_pairs}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SNP descendant counts across tree pairs in a TreeSequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python snp_descendants.py input.trees samples.txt -o output.tsv
    python snp_descendants.py input.trees samples.txt --max-offset 5 -o output.tsv
    
Sample file format:
    One sample identifier per line. Identifiers can be:
    - Node IDs (integers)
    - Individual IDs (integers)
    - Individual names (from metadata)
    
Output format (TSV):
    For each pair of SNPs in tree pairs within max-offset, outputs:
    - Tree indices, offset, and genomic intervals
    - SNP positions and nodes
    - Descendant counts in the sample subset
        """
    )
    
    parser.add_argument("treefile", type=Path,
                        help="Input TreeSequence file (.trees)")
    parser.add_argument("samplefile", type=Path,
                        help="File with sample identifiers (one per line)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--max-offset", type=int, default=1,
                        help="Maximum tree index offset to consider (default: 1 for adjacent only)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress to stderr")
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.treefile.exists():
        sys.exit(f"Error: TreeSequence file not found: {args.treefile}")
    if not args.samplefile.exists():
        sys.exit(f"Error: Sample file not found: {args.samplefile}")
    
    # Load tree sequence
    if args.verbose:
        print(f"Loading TreeSequence from {args.treefile}...", file=sys.stderr)
    ts = tskit.load(str(args.treefile))
    
    if args.verbose:
        print(f"  {ts.num_samples} samples, {ts.num_trees} trees, "
              f"{ts.num_sites} sites, {ts.num_mutations} mutations", file=sys.stderr)
    
    # Load sample subset
    if args.verbose:
        print(f"Loading sample subset from {args.samplefile}...", file=sys.stderr)
    sample_subset = load_sample_subset(args.samplefile, ts)
    
    if args.verbose:
        print(f"  {len(sample_subset)} samples matched", file=sys.stderr)
    
    # Process trees
    if args.verbose:
        print(f"Processing tree pairs (max offset: {args.max_offset})...", file=sys.stderr)
    
    process_adjacent_trees(ts, sample_subset, args.output, args.verbose, 
                           max_offset=args.max_offset)
    
    if args.verbose:
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
