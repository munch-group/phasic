#!/usr/bin/env python3
"""
Compute the joint distribution of homozygous run lengths flanking heterozygous sites.

For each diploid individual, at each heterozygous site, measures:
- Left run: number of consecutive homozygous sites to the left
- Right run: number of consecutive homozygous sites to the right

Outputs a joint distribution (2D histogram) of (left_run_length, right_run_length).
"""

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def open_vcf(vcf_path: Path):
    """Open VCF file, handling gzip compression."""
    if str(vcf_path).endswith('.gz'):
        return gzip.open(vcf_path, 'rt')
    return open(vcf_path, 'r')


def parse_gt(gt_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse genotype string to tuple of alleles.
    
    Returns None for missing data.
    Returns (a1, a2) tuple for valid genotypes.
    """
    # Handle missing data
    if gt_str in ('.', './.', '.|.'):
        return None
    
    # Split on / or |
    if '|' in gt_str:
        parts = gt_str.split('|')
    elif '/' in gt_str:
        parts = gt_str.split('/')
    else:
        return None
    
    if len(parts) != 2:
        return None
    
    try:
        a1 = int(parts[0]) if parts[0] != '.' else None
        a2 = int(parts[1]) if parts[1] != '.' else None
        if a1 is None or a2 is None:
            return None
        return (a1, a2)
    except ValueError:
        return None


def is_heterozygous(gt: Optional[Tuple[int, int]]) -> Optional[bool]:
    """
    Check if genotype is heterozygous.
    
    Returns None for missing data, True for het, False for hom.
    """
    if gt is None:
        return None
    return gt[0] != gt[1]


def load_individuals(ind_file: Path) -> List[str]:
    """Load list of individual IDs from file."""
    individuals = []
    with open(ind_file) as f:
        for line in f:
            ind = line.strip()
            if ind:
                individuals.append(ind)
    return individuals


def process_vcf(vcf_path: Path, 
                individuals: List[str],
                verbose: bool = False) -> Dict[str, Dict[Tuple[int, int], int]]:
    """
    Process VCF and compute joint run length distributions.
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
    individuals : List[str]
        List of individual IDs to process
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict[str, Dict[Tuple[int, int], int]]
        For each individual, a dict mapping (left_run, right_run) to count
    """
    # First pass: collect genotype data per chromosome
    # We need to process each chromosome separately to handle run lengths correctly
    
    # Storage: chrom -> list of (pos, {ind: is_het}) 
    # where is_het is True/False/None
    chrom_data = defaultdict(list)
    sample_indices = {}
    
    if verbose:
        print("Reading VCF file...", file=sys.stderr)
    
    n_variants = 0
    with open_vcf(vcf_path) as f:
        for line in f:
            if line.startswith('##'):
                continue
            
            if line.startswith('#CHROM'):
                # Parse header
                fields = line.strip().split('\t')
                samples = fields[9:]
                
                # Find indices for requested individuals
                for ind in individuals:
                    if ind in samples:
                        sample_indices[ind] = samples.index(ind)
                    else:
                        print(f"Warning: Individual '{ind}' not found in VCF", 
                              file=sys.stderr)
                
                if not sample_indices:
                    raise ValueError("No requested individuals found in VCF")
                
                if verbose:
                    print(f"Found {len(sample_indices)}/{len(individuals)} "
                          f"individuals in VCF", file=sys.stderr)
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            
            # Get format field to find GT index
            fmt = fields[8].split(':')
            try:
                gt_idx = fmt.index('GT')
            except ValueError:
                continue  # No GT field
            
            # Extract genotypes for our individuals
            site_data = {}
            for ind, idx in sample_indices.items():
                sample_field = fields[9 + idx]
                gt_parts = sample_field.split(':')
                if gt_idx < len(gt_parts):
                    gt = parse_gt(gt_parts[gt_idx])
                    site_data[ind] = is_heterozygous(gt)
                else:
                    site_data[ind] = None
            
            chrom_data[chrom].append((pos, site_data))
            n_variants += 1
            
            if verbose and n_variants % 100000 == 0:
                print(f"Read {n_variants} variants...", file=sys.stderr)
    
    if verbose:
        print(f"Total variants read: {n_variants}", file=sys.stderr)
        print(f"Chromosomes: {list(chrom_data.keys())}", file=sys.stderr)
    
    # Second pass: compute run lengths for each individual
    joint_distributions = {ind: defaultdict(int) for ind in sample_indices}
    
    if verbose:
        print("Computing run length distributions...", file=sys.stderr)
    
    for chrom, sites in chrom_data.items():
        if verbose:
            print(f"Processing chromosome {chrom} ({len(sites)} sites)...", 
                  file=sys.stderr)
        
        # For each individual, compute runs
        for ind in sample_indices:
            # Extract het/hom status for this individual
            # True = het, False = hom, None = missing
            statuses = [(pos, site_data.get(ind)) for pos, site_data in sites]
            
            # Find heterozygous positions and compute flanking hom runs
            n_sites = len(statuses)
            
            for i, (pos, is_het) in enumerate(statuses):
                if is_het is not True:
                    continue  # Skip non-het sites
                
                # Count homozygous run to the left
                left_run = 0
                j = i - 1
                while j >= 0:
                    left_status = statuses[j][1]
                    if left_status is False:  # Homozygous
                        left_run += 1
                        j -= 1
                    elif left_status is True:  # Hit another het
                        break
                    else:  # Missing data - skip
                        j -= 1
                
                # Count homozygous run to the right
                right_run = 0
                j = i + 1
                while j < n_sites:
                    right_status = statuses[j][1]
                    if right_status is False:  # Homozygous
                        right_run += 1
                        j += 1
                    elif right_status is True:  # Hit another het
                        break
                    else:  # Missing data - skip
                        j += 1
                
                joint_distributions[ind][(left_run, right_run)] += 1
    
    return joint_distributions


def write_output(distributions: Dict[str, Dict[Tuple[int, int], int]],
                 output_prefix: str,
                 output_format: str = 'tsv',
                 max_run: Optional[int] = None):
    """
    Write output files.
    
    Parameters
    ----------
    distributions : dict
        Joint distributions per individual
    output_prefix : str
        Output file prefix
    output_format : str
        Output format: 'tsv', 'matrix', or 'json'
    max_run : int, optional
        Maximum run length to include (for binning)
    """
    if output_format == 'tsv':
        # Long format: individual, left_run, right_run, count
        outfile = f"{output_prefix}.tsv" if output_prefix else None
        out = open(outfile, 'w') if outfile else sys.stdout
        
        print("individual\tleft_run\tright_run\tcount", file=out)
        
        for ind, dist in sorted(distributions.items()):
            for (left, right), count in sorted(dist.items()):
                if max_run is not None:
                    left = min(left, max_run)
                    right = min(right, max_run)
                print(f"{ind}\t{left}\t{right}\t{count}", file=out)
        
        if outfile:
            out.close()
            
    elif output_format == 'matrix':
        # One matrix file per individual
        for ind, dist in distributions.items():
            # Determine matrix size
            if max_run is not None:
                size = max_run + 1
            else:
                max_left = max((k[0] for k in dist.keys()), default=0)
                max_right = max((k[1] for k in dist.keys()), default=0)
                size = max(max_left, max_right) + 1
            
            outfile = f"{output_prefix}_{ind}.matrix.tsv"
            with open(outfile, 'w') as out:
                # Header row
                print("left\\right\t" + "\t".join(str(i) for i in range(size)), 
                      file=out)
                
                # Aggregate counts if max_run is set
                if max_run is not None:
                    agg_dist = defaultdict(int)
                    for (left, right), count in dist.items():
                        agg_left = min(left, max_run)
                        agg_right = min(right, max_run)
                        agg_dist[(agg_left, agg_right)] += count
                    dist = agg_dist
                
                # Matrix rows
                for i in range(size):
                    row = [str(i)]
                    for j in range(size):
                        row.append(str(dist.get((i, j), 0)))
                    print("\t".join(row), file=out)
                    
    elif output_format == 'json':
        # JSON output
        outfile = f"{output_prefix}.json" if output_prefix else None
        
        # Convert tuple keys to strings for JSON
        json_data = {}
        for ind, dist in distributions.items():
            json_data[ind] = {f"{k[0]},{k[1]}": v for k, v in dist.items()}
        
        out = open(outfile, 'w') if outfile else sys.stdout
        json.dump(json_data, out, indent=2)
        if outfile:
            out.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute joint distribution of homozygous run lengths "
                    "flanking heterozygous sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python hom_run_distribution.py input.vcf.gz individuals.txt -o output
    python hom_run_distribution.py input.vcf individuals.txt --format matrix --max-run 50
    
Individual file format:
    One individual ID per line (must match VCF sample names)
    
Output formats:
    tsv    - Long format: individual, left_run, right_run, count (default)
    matrix - One matrix file per individual with counts
    json   - JSON format with all distributions

The script counts, for each heterozygous site in each individual:
    - left_run: consecutive homozygous sites to the left (until het or chrom start)
    - right_run: consecutive homozygous sites to the right (until het or chrom end)

Missing genotypes are skipped (not counted in runs).
        """
    )
    
    parser.add_argument("vcf", type=Path,
                        help="Input VCF file (.vcf or .vcf.gz)")
    parser.add_argument("individuals", type=Path,
                        help="File with individual IDs (one per line)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file prefix (default: stdout for tsv)")
    parser.add_argument("--format", type=str, default="tsv",
                        choices=["tsv", "matrix", "json"],
                        help="Output format (default: tsv)")
    parser.add_argument("--max-run", type=int, default=None,
                        help="Maximum run length (values above are binned together)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress to stderr")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.vcf.exists():
        sys.exit(f"Error: VCF file not found: {args.vcf}")
    if not args.individuals.exists():
        sys.exit(f"Error: Individuals file not found: {args.individuals}")
    
    # Load individuals
    individuals = load_individuals(args.individuals)
    if not individuals:
        sys.exit("Error: No individuals found in file")
    
    if args.verbose:
        print(f"Loaded {len(individuals)} individuals", file=sys.stderr)
    
    # Process VCF
    distributions = process_vcf(args.vcf, individuals, args.verbose)
    
    # Summary stats
    if args.verbose:
        for ind, dist in distributions.items():
            total_hets = sum(dist.values())
            print(f"Individual {ind}: {total_hets} heterozygous sites", 
                  file=sys.stderr)
    
    # Write output
    write_output(distributions, args.output, args.format, args.max_run)
    
    if args.verbose:
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
