"""
Test and benchmark the treeseq_sfs module.

This script:
1. Creates a simulated TreeSequence using msprime
2. Extracts per-tree SFS using both optimized and manual methods
3. Validates they give the same results
4. Benchmarks the performance difference
"""

import numpy as np
import time
import sys

# Install msprime if needed
try:
    import msprime
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "msprime", "-q"])
    import msprime

try:
    import tskit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tskit", "-q"])
    import tskit

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

from treeseq_sfs import (
    extract_per_tree_sfs,
    extract_per_tree_sfs_detailed,
    extract_per_tree_sfs_manual,
)


def simulate_tree_sequence(
    sample_size: int = 10,
    sequence_length: float = 1e6,
    recombination_rate: float = 1e-8,
    mutation_rate: float = 1e-8,
    random_seed: int = 42,
) -> tskit.TreeSequence:
    """Simulate a tree sequence using msprime.
    
    Note: sample_size is the number of haploid genomes (not diploid individuals).
    """
    # Use ploidy=1 for haploid to get exact sample_size haploid genomes
    # Use higher recombination rate to generate more trees
    ts = msprime.sim_ancestry(
        samples=sample_size,
        ploidy=1,  # Haploid - so sample_size = num_samples
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=random_seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)
    return ts


def test_basic_extraction():
    """Test basic SFS extraction with small example."""
    print("=" * 60)
    print("TEST 1: Basic extraction with n=10 samples")
    print("=" * 60)
    
    # Higher recombination rate to generate multiple trees
    ts = simulate_tree_sequence(
        sample_size=10, 
        sequence_length=1e5,
        recombination_rate=1e-7,  # Higher rate for more trees
        mutation_rate=1e-7,
    )
    
    print(f"\nSimulated TreeSequence:")
    print(f"  Samples: {ts.num_samples}")
    print(f"  Trees: {ts.num_trees}")
    print(f"  Sites: {ts.num_sites}")
    print(f"  Mutations: {ts.num_mutations}")
    print(f"  Sequence length: {ts.sequence_length:,.0f}")
    
    # Extract using optimized method
    df = extract_per_tree_sfs(ts, chrom_name="chr1", sample_size=10)
    
    print(f"\nOutput DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Check that branch_len columns sum correctly
    branch_cols = [c for c in df.columns if c.startswith("branch_len_")]
    mut_cols = [c for c in df.columns if c.startswith("mutations_")]
    
    print(f"\nValidation:")
    print(f"  Number of branch_len columns: {len(branch_cols)} (expected {ts.num_samples - 1})")
    print(f"  Number of mutations columns: {len(mut_cols)} (expected {ts.num_samples - 1})")
    print(f"  Total mutations in SFS: {df[mut_cols].sum().sum():.0f} (expected ~{ts.num_mutations})")
    
    return df


def test_detailed_extraction():
    """Test detailed extraction with additional columns."""
    print("\n" + "=" * 60)
    print("TEST 2: Detailed extraction with extra metadata")
    print("=" * 60)
    
    ts = simulate_tree_sequence(
        sample_size=10, 
        sequence_length=1e5,
        recombination_rate=1e-7,
        mutation_rate=1e-7,
    )
    
    df = extract_per_tree_sfs_detailed(ts, chrom_name="chrX")
    
    print(f"\nDetailed output columns: {list(df.columns)}")
    print(f"\nFirst 3 rows (selected columns):")
    display_cols = ["tree_index", "chrom", "start", "end", "span", 
                    "total_branch_length", "num_mutations"]
    print(df[display_cols].head(3).to_string())
    
    return df


def test_validation_against_manual():
    """Validate that optimized method matches manual iteration."""
    print("\n" + "=" * 60)
    print("TEST 3: Validation - optimized vs manual")
    print("=" * 60)
    
    # Use smaller example for manual method
    ts = simulate_tree_sequence(
        sample_size=10, 
        sequence_length=5e4,
        recombination_rate=1e-7,
        mutation_rate=1e-7,
    )
    
    print(f"\nSmall test case: {ts.num_trees} trees")
    
    # Extract both ways
    df_opt = extract_per_tree_sfs(ts)
    df_manual = extract_per_tree_sfs_manual(ts)
    
    # Compare
    branch_cols = [c for c in df_opt.columns if c.startswith("branch_len_")]
    mut_cols = [c for c in df_opt.columns if c.startswith("mutations_")]
    
    # Check coordinates match
    coords_match = np.allclose(df_opt["start"], df_manual["start"]) and \
                   np.allclose(df_opt["end"], df_manual["end"])
    
    # Check branch lengths match
    branch_match = np.allclose(
        df_opt[branch_cols].values, 
        df_manual[branch_cols].values,
        rtol=1e-10
    )
    
    # Check mutations match
    mut_match = np.allclose(
        df_opt[mut_cols].values,
        df_manual[mut_cols].values
    )
    
    print(f"\nValidation results:")
    print(f"  Coordinates match: {coords_match}")
    print(f"  Branch lengths match: {branch_match}")
    print(f"  Mutation counts match: {mut_match}")
    
    if all([coords_match, branch_match, mut_match]):
        print("\n  ✓ All validations PASSED!")
    else:
        print("\n  ✗ Some validations FAILED!")
        # Show differences if any
        if not branch_match:
            diff = np.abs(df_opt[branch_cols].values - df_manual[branch_cols].values)
            print(f"    Max branch length diff: {diff.max()}")
        if not mut_match:
            diff = np.abs(df_opt[mut_cols].values - df_manual[mut_cols].values)
            print(f"    Max mutation diff: {diff.max()}")


def benchmark_performance():
    """Benchmark optimized vs manual extraction."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Optimized vs Manual extraction")
    print("=" * 60)
    
    # Test with increasing sizes
    sizes = [1e5, 5e5, 1e6]
    
    print(f"\n{'Seq Length':>12} {'Trees':>8} {'Optimized':>12} {'Manual':>12} {'Speedup':>10}")
    print("-" * 60)
    
    for seq_len in sizes:
        ts = simulate_tree_sequence(
            sample_size=10, 
            sequence_length=seq_len,
            recombination_rate=1e-7,
            mutation_rate=1e-7,
        )
        
        # Time optimized
        start = time.time()
        df_opt = extract_per_tree_sfs(ts)
        time_opt = time.time() - start
        
        # Time manual (skip if too many trees)
        if ts.num_trees < 2000:
            start = time.time()
            df_manual = extract_per_tree_sfs_manual(ts)
            time_manual = time.time() - start
            speedup = time_manual / time_opt
        else:
            time_manual = float('nan')
            speedup = float('nan')
            
        print(f"{seq_len:>12,.0f} {ts.num_trees:>8,d} {time_opt:>11.4f}s "
              f"{time_manual:>11.4f}s {speedup:>9.1f}x")
    
    print("\nNote: Manual method becomes impractical for large tree sequences.")
    print("The optimized method uses tskit's incremental C algorithm.")


def test_file_output():
    """Test writing output to file."""
    print("\n" + "=" * 60)
    print("TEST 4: File output")
    print("=" * 60)
    
    ts = simulate_tree_sequence(
        sample_size=10, 
        sequence_length=1e5,
        recombination_rate=1e-7,
        mutation_rate=1e-7,
    )
    
    output_file = "/home/claude/test_sfs_output.tsv"
    
    df = extract_per_tree_sfs(ts, chrom_name="chr1")
    df.to_csv(output_file, sep="\t", index=False)
    
    print(f"\nWrote output to: {output_file}")
    
    # Show first few lines
    print("\nFirst 5 lines of output:")
    with open(output_file) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            # Truncate long lines for display
            if len(line) > 100:
                print(line[:100] + "...")
            else:
                print(line.rstrip())
    
    return output_file


if __name__ == "__main__":
    print("TreeSequence SFS Module - Test Suite")
    print("=" * 60)
    
    test_basic_extraction()
    test_detailed_extraction()
    test_validation_against_manual()
    benchmark_performance()
    output_file = test_file_output()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
