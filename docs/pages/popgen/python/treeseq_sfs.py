"""
treeseq_sfs.py - Efficient extraction of per-tree SFS statistics from TreeSequences

This module provides efficient extraction of branch-length and mutation Site Frequency 
Spectrum (SFS) statistics from tskit TreeSequence files. It leverages tskit's built-in 
optimized C algorithms that exploit the incremental nature of tree sequences, making it
orders of magnitude faster than manual tree iteration.

Author: Generated for Kasper Munch
"""

import numpy as np
import pandas as pd
import tskit
from typing import Optional, Union
from pathlib import Path


def extract_per_tree_sfs(
    ts: Union[tskit.TreeSequence, str, Path],
    chrom_name: str = "chr",
    sample_size: Optional[int] = None,
    polarised: bool = True,
) -> pd.DataFrame:
    """
    Extract per-tree branch-length SFS and mutation counts from a TreeSequence.
    
    Uses tskit's highly optimized allele_frequency_spectrum() method with windows
    set to tree breakpoints, which exploits the incremental algorithm for maximum
    efficiency.
    
    Parameters
    ----------
    ts : tskit.TreeSequence or str or Path
        Either a TreeSequence object or path to a .trees file
    chrom_name : str
        Name to use for the chromosome column (default: "chr")
    sample_size : int, optional
        Expected sample size. If provided, validates against ts.num_samples.
        If None, uses ts.num_samples.
    polarised : bool
        Whether to compute polarised (derived allele) spectrum. Default True.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - chrom: chromosome name
        - start: tree start position (genome coordinates)
        - end: tree end position (genome coordinates)
        - branch_len_1 ... branch_len_{n-1}: sum of branch lengths (in time units)
          with 1 to n-1 descendants
        - mutations_1 ... mutations_{n-1}: count of mutations with 1 to n-1 descendants
        
    Notes
    -----
    The SFS indices 1 to n-1 represent branches/mutations with that many descendants.
    Index 0 (monomorphic/fixed ancestral) and index n (fixed derived) are excluded
    as they represent non-polymorphic sites.
    
    Branch lengths are in time units (coalescent time or generations depending on
    the tree sequence). The tskit AFS returns branch_length × span; we divide by
    span to get pure time units per tree.
    
    Examples
    --------
    >>> ts = tskit.load("my_simulation.trees")
    >>> df = extract_per_tree_sfs(ts, chrom_name="chr1", sample_size=10)
    >>> df.to_csv("sfs_per_tree.tsv", sep="\\t", index=False)
    """
    # Load tree sequence if path provided
    if isinstance(ts, (str, Path)):
        ts = tskit.load(ts)
    
    # Validate/set sample size
    n = ts.num_samples
    if sample_size is not None and sample_size != n:
        raise ValueError(
            f"sample_size={sample_size} does not match ts.num_samples={n}"
        )
    
    # Get tree breakpoints as windows
    # breakpoints() returns array of length num_trees + 1
    windows = np.array(list(ts.breakpoints()))
    
    # Compute branch-length SFS per tree
    # Shape: (num_trees, n+1) where index i = count of descendants
    # Note: This returns branch_length × span (area). We'll divide by span below.
    branch_afs = ts.allele_frequency_spectrum(
        mode="branch",
        windows=windows,
        span_normalise=False,  # Returns area (branch_length × span)
        polarised=polarised,
    )
    
    # Compute site (mutation) SFS per tree
    # Shape: (num_trees, n+1)
    site_afs = ts.allele_frequency_spectrum(
        mode="site", 
        windows=windows,
        span_normalise=False,  # We want raw counts
        polarised=polarised,
    )
    
    # Calculate spans for each tree to convert area back to branch length
    spans = windows[1:] - windows[:-1]
    
    # Build the output DataFrame
    num_trees = ts.num_trees
    
    # Start with coordinates
    data = {
        "chrom": [chrom_name] * num_trees,
        "start": windows[:-1],
        "end": windows[1:],
    }
    
    # Add branch length columns for 1 to n-1 descendants
    # Divide by span to get branch length in time units (not area)
    for i in range(1, n):
        data[f"branch_len_{i}"] = branch_afs[:, i] / spans
    
    # Add mutation count columns for 1 to n-1 descendants
    for i in range(1, n):
        data[f"mutations_{i}"] = site_afs[:, i]
    
    return pd.DataFrame(data)


def extract_per_tree_sfs_detailed(
    ts: Union[tskit.TreeSequence, str, Path],
    chrom_name: str = "chr",
) -> pd.DataFrame:
    """
    Extract detailed per-tree statistics including tree index and span.
    
    Similar to extract_per_tree_sfs but includes additional metadata columns.
    
    Parameters
    ----------
    ts : tskit.TreeSequence or str or Path
        Either a TreeSequence object or path to a .trees file
    chrom_name : str
        Name to use for the chromosome column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - tree_index: index of the tree (0-based)
        - chrom: chromosome name
        - start: tree start position
        - end: tree end position
        - span: tree span (end - start)
        - total_branch_length: total branch length in the tree (in time units)
        - num_mutations: total mutations in this tree interval
        - branch_len_1 ... branch_len_{n-1}: branch lengths by descendant count
        - mutations_1 ... mutations_{n-1}: mutations by descendant count
    """
    if isinstance(ts, (str, Path)):
        ts = tskit.load(ts)
    
    n = ts.num_samples
    windows = np.array(list(ts.breakpoints()))
    
    branch_afs = ts.allele_frequency_spectrum(
        mode="branch",
        windows=windows,
        span_normalise=False,
        polarised=True,
    )
    
    site_afs = ts.allele_frequency_spectrum(
        mode="site",
        windows=windows, 
        span_normalise=False,
        polarised=True,
    )
    
    num_trees = ts.num_trees
    starts = windows[:-1]
    ends = windows[1:]
    spans = ends - starts
    
    # Convert branch_afs from area to time by dividing by span
    # branch_afs has shape (num_trees, n+1)
    branch_lens_per_tree = branch_afs / spans[:, np.newaxis]
    
    data = {
        "tree_index": np.arange(num_trees),
        "chrom": [chrom_name] * num_trees,
        "start": starts,
        "end": ends,
        "span": spans,
        "total_branch_length": branch_lens_per_tree.sum(axis=1),
        "num_mutations": site_afs.sum(axis=1),
    }
    
    for i in range(1, n):
        data[f"branch_len_{i}"] = branch_lens_per_tree[:, i]
    
    for i in range(1, n):
        data[f"mutations_{i}"] = site_afs[:, i]
    
    return pd.DataFrame(data)


def write_per_tree_sfs(
    ts: Union[tskit.TreeSequence, str, Path],
    output_path: Union[str, Path],
    chrom_name: str = "chr",
    sample_size: Optional[int] = None,
    sep: str = "\t",
) -> None:
    """
    Extract per-tree SFS and write directly to file.
    
    Convenience function that combines extraction and file output.
    
    Parameters
    ----------
    ts : tskit.TreeSequence or str or Path
        TreeSequence or path to .trees file
    output_path : str or Path
        Output file path
    chrom_name : str
        Chromosome name for output
    sample_size : int, optional
        Expected sample size (for validation)
    sep : str
        Field separator (default: tab)
    """
    df = extract_per_tree_sfs(ts, chrom_name=chrom_name, sample_size=sample_size)
    df.to_csv(output_path, sep=sep, index=False)


# =============================================================================
# Alternative: Manual iteration approach (for comparison/validation)
# This is MUCH slower but shows what's happening under the hood
# =============================================================================

def extract_per_tree_sfs_manual(
    ts: Union[tskit.TreeSequence, str, Path],
    chrom_name: str = "chr",
) -> pd.DataFrame:
    """
    Extract per-tree SFS using manual tree iteration.
    
    WARNING: This is much slower than extract_per_tree_sfs() and is provided
    only for validation/understanding. Use the optimized version for real work.
    
    This manually iterates through each tree and computes branch lengths and
    mutation counts by descendant count.
    """
    if isinstance(ts, (str, Path)):
        ts = tskit.load(ts)
    
    n = ts.num_samples
    records = []
    
    for tree in ts.trees():
        record = {
            "chrom": chrom_name,
            "start": tree.interval.left,
            "end": tree.interval.right,
        }
        
        # Initialize arrays for this tree
        branch_lens = np.zeros(n - 1)  # indices 0 to n-2 represent 1 to n-1 descendants
        
        # Calculate branch lengths by descendant count (in time units)
        for node in tree.nodes():
            if tree.parent(node) != tskit.NULL:  # Skip roots
                num_samples_below = tree.num_samples(node)
                if 1 <= num_samples_below <= n - 1:
                    branch_lens[num_samples_below - 1] += tree.branch_length(node)
        
        # Add branch length columns
        for i in range(1, n):
            record[f"branch_len_{i}"] = branch_lens[i - 1]
        
        # Count mutations by descendant count
        mut_counts = np.zeros(n - 1)
        for site in tree.sites():
            for mutation in site.mutations:
                num_samples_below = tree.num_samples(mutation.node)
                if 1 <= num_samples_below <= n - 1:
                    mut_counts[num_samples_below - 1] += 1
        
        for i in range(1, n):
            record[f"mutations_{i}"] = mut_counts[i - 1]
        
        records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# CLI interface
# =============================================================================

def main():
    """Command-line interface for extracting per-tree SFS."""
    import argparse
    import sys
    import time
    
    parser = argparse.ArgumentParser(
        description="Extract per-tree SFS statistics from a TreeSequence file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format:
  Tab-separated file with one row per tree containing:
  - chrom, start, end: genomic coordinates
  - branch_len_1 to branch_len_{n-1}: branch lengths by descendant count  
  - mutations_1 to mutations_{n-1}: mutation counts by descendant count

Example:
  python treeseq_sfs.py simulation.trees -o sfs_output.tsv -c chr1 -n 10
        """
    )
    
    parser.add_argument(
        "treefile",
        help="Path to TreeSequence file (.trees)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "-c", "--chrom",
        default="chr",
        help="Chromosome name (default: chr)"
    )
    parser.add_argument(
        "-n", "--sample-size",
        type=int,
        default=None,
        help="Expected sample size (optional, for validation)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include additional columns (tree_index, span, totals)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading TreeSequence from {args.treefile}...", file=sys.stderr)
        start_time = time.time()
    
    ts = tskit.load(args.treefile)
    
    if args.verbose:
        load_time = time.time() - start_time
        print(f"  Loaded: {ts.num_trees} trees, {ts.num_samples} samples, "
              f"{ts.num_sites} sites, {ts.num_mutations} mutations", file=sys.stderr)
        print(f"  Load time: {load_time:.2f}s", file=sys.stderr)
        print(f"Extracting per-tree SFS...", file=sys.stderr)
        start_time = time.time()
    
    if args.detailed:
        df = extract_per_tree_sfs_detailed(ts, chrom_name=args.chrom)
    else:
        df = extract_per_tree_sfs(
            ts, 
            chrom_name=args.chrom, 
            sample_size=args.sample_size
        )
    
    if args.verbose:
        extract_time = time.time() - start_time
        print(f"  Extraction time: {extract_time:.2f}s", file=sys.stderr)
        print(f"Writing output to {args.output}...", file=sys.stderr)
    
    df.to_csv(args.output, sep="\t", index=False)
    
    if args.verbose:
        print(f"Done! Output has {len(df)} rows.", file=sys.stderr)


if __name__ == "__main__":
    main()
