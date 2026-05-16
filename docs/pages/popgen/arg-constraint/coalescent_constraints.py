"""
Coalescent Constraint System Using Bipartitions

This module provides tools for constraining coalescent simulations using
bipartitions extracted from Newick trees. Each bipartition represents an
ancestral state where lineages can only coalesce if they share the same partition.
"""

from __future__ import annotations

from bitarray import bitarray, frozenbitarray
from bitarray.util import any_and
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class Bipartition:
    """
    Immutable representation of a bipartition using frozenbitarray.

    A bipartition divides n samples into two groups. Stored as a bitarray
    where 1 = partition A, 0 = partition B. Uses canonical form where the
    first sample (index 0) is always in partition A for consistent hashing.
    """

    __slots__ = ('_bits', '_n')

    def __init__(self, bits: frozenbitarray):
        """Initialize with a frozenbitarray in canonical form."""
        self._bits = bits
        self._n = len(bits)

    @classmethod
    def from_partition(cls, indices: set[int], n: int) -> Bipartition:
        """
        Create a bipartition where the given indices are in partition A.

        Args:
            indices: Sample indices to place in partition A
            n: Total number of samples

        Returns:
            Bipartition in canonical form
        """
        bits = bitarray(n)
        bits.setall(0)
        for i in indices:
            bits[i] = 1

        # Canonical form: first sample always in partition A
        if not bits[0]:
            bits.invert()

        return cls(frozenbitarray(bits))

    @classmethod
    def from_split(cls, left: set[int], right: set[int], n: int) -> Bipartition:
        """
        Create a bipartition from explicit left and right partitions.

        Args:
            left: Indices for one side of the split
            right: Indices for the other side
            n: Total number of samples

        Returns:
            Bipartition in canonical form
        """
        return cls.from_partition(left, n)

    @property
    def bits(self) -> frozenbitarray:
        """The underlying bit representation."""
        return self._bits

    @property
    def n_samples(self) -> int:
        """Number of samples this bipartition covers."""
        return self._n

    def same_partition(self, i: int, j: int) -> bool:
        """
        Check if two sample indices are in the same partition.

        Args:
            i: First sample index
            j: Second sample index

        Returns:
            True if both samples are in the same partition
        """
        return self._bits[i] == self._bits[j]

    def partition_of(self, i: int) -> bool:
        """Return which partition sample i is in (True=A, False=B)."""
        return self._bits[i]

    def __hash__(self) -> int:
        return hash(self._bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bipartition):
            return NotImplemented
        return self._bits == other._bits

    def is_satisfied_by(self, lineages: set[Lineage]) -> bool:
        """
        Check if this bipartition is satisfied (each partition fully coalesced).

        A bipartition is satisfied when all samples in partition A are covered
        by exactly one lineage, and all samples in partition B are covered by
        exactly one lineage.

        Args:
            lineages: Current set of active lineages

        Returns:
            True if the bipartition constraint is satisfied
        """
        partition_a_lineages = set()
        partition_b_lineages = set()

        for lin in lineages:
            covers_a = any_and(lin.bits, self._bits)
            covers_b = any_and(lin.bits, ~self._bits)

            if covers_a and covers_b:
                # Lineage spans both partitions - not satisfied
                return False
            if covers_a:
                partition_a_lineages.add(lin)
            if covers_b:
                partition_b_lineages.add(lin)

        # Satisfied if exactly one lineage covers each partition
        return len(partition_a_lineages) == 1 and len(partition_b_lineages) == 1

    def __repr__(self) -> str:
        one = bitarray('1')
        a_indices = list(self._bits.search(one))
        b_indices = list((~self._bits).search(one))
        return f"Bipartition({{{', '.join(map(str, a_indices))}}} | {{{', '.join(map(str, b_indices))}}})"


class TreeNode:
    """
    Simple recursive tree structure for Newick parsing.
    """

    __slots__ = ('children', 'leaf_id', '_leaf_cache')

    def __init__(self, leaf_id: int | None = None):
        """
        Initialize a tree node.

        Args:
            leaf_id: Integer sample index for leaf nodes, None for internal nodes
        """
        self.children: list[TreeNode] = []
        self.leaf_id = leaf_id
        self._leaf_cache: set[int] | None = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf."""
        return self.leaf_id is not None

    def get_leaf_set(self) -> set[int]:
        """
        Get the set of all descendant leaf indices.

        Returns:
            Set of leaf indices descended from this node
        """
        if self._leaf_cache is not None:
            return self._leaf_cache

        if self.is_leaf():
            self._leaf_cache = {self.leaf_id}
        else:
            self._leaf_cache = set()
            for child in self.children:
                self._leaf_cache |= child.get_leaf_set()

        return self._leaf_cache

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"TreeNode(leaf={self.leaf_id})"
        return f"TreeNode(internal, {len(self.children)} children)"


def parse_newick(newick_str: str) -> TreeNode:
    """
    Parse a Newick format string into a tree structure.

    Handles format like: ((0,1),(2,3)); where leaves are integer sample indices.
    Also supports branch lengths which are ignored: ((0:0.1,1:0.2):0.3,(2,3));

    Args:
        newick_str: Newick format string

    Returns:
        Root TreeNode of the parsed tree

    Raises:
        ValueError: If the Newick string is malformed
    """
    newick_str = newick_str.strip()
    if newick_str.endswith(';'):
        newick_str = newick_str[:-1]

    pos = 0

    def parse_node() -> TreeNode:
        nonlocal pos

        if pos >= len(newick_str):
            raise ValueError("Unexpected end of Newick string")

        if newick_str[pos] == '(':
            # Internal node
            pos += 1  # consume '('
            node = TreeNode()

            while True:
                child = parse_node()
                node.children.append(child)

                if pos >= len(newick_str):
                    raise ValueError("Unexpected end while parsing children")

                if newick_str[pos] == ',':
                    pos += 1  # consume ','
                elif newick_str[pos] == ')':
                    pos += 1  # consume ')'
                    break
                else:
                    raise ValueError(f"Expected ',' or ')' at position {pos}")

            # Skip any branch length after )
            _skip_branch_length()
            return node
        else:
            # Leaf node - parse the label (integer)
            start = pos
            while pos < len(newick_str) and newick_str[pos] not in ',:();':
                pos += 1

            label = newick_str[start:pos]

            try:
                leaf_id = int(label)
            except ValueError:
                raise ValueError(f"Leaf label must be an integer, got: {label}")

            # Skip any branch length after label
            _skip_branch_length()

            return TreeNode(leaf_id=leaf_id)

    def _skip_branch_length():
        nonlocal pos
        if pos < len(newick_str) and newick_str[pos] == ':':
            pos += 1
            while pos < len(newick_str) and newick_str[pos] not in ',();':
                pos += 1

    root = parse_node()
    return root


def bipartitions_from_tree(root: TreeNode, n_samples: int) -> list[Bipartition]:
    """
    Extract all bipartitions from a tree.

    Each internal node produces one bipartition: {descendants} | {rest}.
    Excludes trivial bipartitions (root node containing all samples).

    Args:
        root: Root TreeNode of the tree
        n_samples: Total number of samples

    Returns:
        List of Bipartition objects
    """
    all_samples = set(range(n_samples))
    bipartitions = []

    def traverse(node: TreeNode):
        if node.is_leaf():
            return

        descendants = node.get_leaf_set()

        # Skip trivial bipartition (root with all samples)
        if descendants != all_samples and len(descendants) > 0:
            bipartitions.append(Bipartition.from_partition(descendants, n_samples))

        for child in node.children:
            traverse(child)

    traverse(root)
    return bipartitions


class Lineage:
    """
    Represents a lineage tracking which samples are its descendants.

    Uses frozenbitarray for immutable, hashable descendant masks.
    """

    __slots__ = ('_bits',)

    def __init__(self, bits: frozenbitarray):
        """Initialize with a frozenbitarray descendant mask."""
        self._bits = bits

    @classmethod
    def from_sample(cls, idx: int, n: int) -> Lineage:
        """
        Create a lineage for a single sample.

        Args:
            idx: Sample index
            n: Total number of samples

        Returns:
            Lineage representing single sample
        """
        bits = bitarray(n)
        bits.setall(0)
        bits[idx] = 1
        return cls(frozenbitarray(bits))

    @classmethod
    def coalesce(cls, lin_a: Lineage, lin_b: Lineage) -> Lineage:
        """
        Create a merged lineage from two parent lineages.

        Args:
            lin_a: First parent lineage
            lin_b: Second parent lineage

        Returns:
            New lineage with combined descendants
        """
        merged = lin_a._bits | lin_b._bits
        return cls(frozenbitarray(merged))

    @property
    def bits(self) -> frozenbitarray:
        """The underlying bit representation of descendants."""
        return self._bits

    @property
    def n_samples(self) -> int:
        """Total number of samples in the system."""
        return len(self._bits)

    def descendant_indices(self) -> list[int]:
        """Get list of descendant sample indices."""
        return list(self._bits.search(bitarray('1')))

    def __hash__(self) -> int:
        return hash(self._bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lineage):
            return NotImplemented
        return self._bits == other._bits

    def __repr__(self) -> str:
        indices = self.descendant_indices()
        return f"Lineage({{{', '.join(map(str, indices))}}})"


class ConstraintSet:
    """
    Holds multiple Bipartition objects and checks coalescence compatibility.
    """

    __slots__ = ('_bipartitions',)

    def __init__(self, bipartitions: set[Bipartition] | None = None):
        """
        Initialize with optional set of bipartitions.

        Args:
            bipartitions: Initial set of bipartitions (copied)
        """
        self._bipartitions: set[Bipartition] = set(bipartitions) if bipartitions else set()

    def add(self, bp: Bipartition) -> None:
        """Add a bipartition to the constraint set."""
        self._bipartitions.add(bp)

    def remove(self, bp: Bipartition) -> None:
        """Remove a bipartition from the constraint set."""
        self._bipartitions.discard(bp)

    def clear(self) -> None:
        """Remove all bipartitions."""
        self._bipartitions.clear()

    def update_after_coalescence(self, lineages: set[Lineage]) -> None:
        """
        Remove any bipartitions that are now satisfied.

        A bipartition is satisfied when each partition has been fully
        coalesced into a single lineage.

        Args:
            lineages: Current set of active lineages
        """
        satisfied = {bp for bp in self._bipartitions if bp.is_satisfied_by(lineages)}
        self._bipartitions -= satisfied

    def __len__(self) -> int:
        return len(self._bipartitions)

    def __iter__(self) -> Iterator[Bipartition]:
        return iter(self._bipartitions)

    def can_coalesce(self, lin_a: Lineage, lin_b: Lineage) -> bool:
        """
        Check if two lineages can coalesce given all constraints.

        Two lineages can coalesce only if ALL their descendants are in the
        same partition for ALL bipartitions.

        Args:
            lin_a: First lineage
            lin_b: Second lineage

        Returns:
            True if coalescence is allowed
        """
        for bp in self._bipartitions:
            if not self._compatible_with_bipartition(bp, lin_a, lin_b):
                return False
        return True

    def _compatible_with_bipartition(
        self, bp: Bipartition, lin_a: Lineage, lin_b: Lineage
    ) -> bool:
        """
        Check if two lineages can coalesce with respect to a single bipartition.

        A lineage is compatible with a bipartition if:
        - It has no overlap with the bipartition (unconstrained), OR
        - Both lineages are entirely in the same partition (A or B)

        Unconstrained lineages (those with no samples in either partition)
        can coalesce freely with any other lineage for this bipartition.
        """
        bp_bits = bp.bits
        not_bp = ~bp_bits

        # Check if lineage A has any descendants in partition A or B
        a_in_A = any_and(lin_a.bits, bp_bits)
        a_in_B = any_and(lin_a.bits, not_bp)

        # Check if lineage B has any descendants in partition A or B
        b_in_A = any_and(lin_b.bits, bp_bits)
        b_in_B = any_and(lin_b.bits, not_bp)

        # If lineage A has no overlap with this bipartition, constraint doesn't apply
        a_unconstrained = not a_in_A and not a_in_B
        # If lineage B has no overlap with this bipartition, constraint doesn't apply
        b_unconstrained = not b_in_A and not b_in_B

        if a_unconstrained or b_unconstrained:
            return True

        # Both must be entirely in A, or both entirely in B
        both_in_A = a_in_A and b_in_A and not a_in_B and not b_in_B
        both_in_B = a_in_B and b_in_B and not a_in_A and not b_in_A

        return both_in_A or both_in_B

    def __repr__(self) -> str:
        return f"ConstraintSet({len(self._bipartitions)} bipartitions)"


class PartitionRefinement:
    """
    Computes the finest partition compatible with all bipartitions.

    Groups samples by their "signature" (tuple of partition memberships
    across all bipartitions), enabling O(k*n) grouping instead of O(n^2*k)
    pairwise checks.
    """

    __slots__ = ('_groups',)

    def __init__(self, bipartitions: list[Bipartition], n_samples: int):
        """
        Compute the refined partition.

        Args:
            bipartitions: List of bipartitions to consider
            n_samples: Total number of samples
        """
        self._groups = self._compute_groups(bipartitions, n_samples)

    def _compute_groups(
        self, bipartitions: list[Bipartition], n_samples: int
    ) -> list[set[int]]:
        """Group samples by their membership signature across all bipartitions."""
        if not bipartitions:
            # No constraints: all samples can coalesce
            return [set(range(n_samples))]

        signatures: dict[tuple[bool, ...], set[int]] = {}
        for i in range(n_samples):
            sig = tuple(bp.bits[i] for bp in bipartitions)
            if sig not in signatures:
                signatures[sig] = set()
            signatures[sig].add(i)

        return list(signatures.values())

    @property
    def groups(self) -> list[set[int]]:
        """Get the list of groups (sets of sample indices that can coalesce)."""
        return self._groups

    def find_group(self, sample_idx: int) -> set[int]:
        """Find which group a sample belongs to."""
        for group in self._groups:
            if sample_idx in group:
                return group
        raise ValueError(f"Sample {sample_idx} not found in any group")

    def __repr__(self) -> str:
        return f"PartitionRefinement({len(self._groups)} groups)"


class CoalescentState:
    """
    Tracks active lineages and constraints for coalescent simulation.
    """

    __slots__ = ('_n_samples', '_lineages', '_constraints')

    def __init__(self, n_samples: int):
        """
        Initialize coalescent state with n samples.

        Creates initial lineages where each sample is its own lineage.

        Args:
            n_samples: Number of samples
        """
        self._n_samples = n_samples
        self._lineages: set[Lineage] = {
            Lineage.from_sample(i, n_samples) for i in range(n_samples)
        }
        self._constraints = ConstraintSet()

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self._n_samples

    @property
    def lineages(self) -> set[Lineage]:
        """Current set of active lineages."""
        return self._lineages

    @property
    def constraints(self) -> ConstraintSet:
        """Current constraint set."""
        return self._constraints

    def add_constraint(self, bp: Bipartition) -> None:
        """Add a bipartition constraint."""
        self._constraints.add(bp)

    def remove_constraint(self, bp: Bipartition) -> None:
        """Remove a bipartition constraint."""
        self._constraints.remove(bp)

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self._constraints.clear()

    def get_compatible_pairs(self) -> list[tuple[Lineage, Lineage]]:
        """
        Get all pairs of lineages that can coalesce.

        Returns:
            List of (lineage_a, lineage_b) tuples
        """
        pairs = []
        lineage_list = list(self._lineages)

        for i in range(len(lineage_list)):
            for j in range(i + 1, len(lineage_list)):
                lin_a, lin_b = lineage_list[i], lineage_list[j]
                if self._constraints.can_coalesce(lin_a, lin_b):
                    pairs.append((lin_a, lin_b))

        return pairs

    def get_coalescence_groups(self) -> list[set[Lineage]]:
        """
        Get groups of lineages that can coalesce with each other.

        Uses partition refinement for efficient grouping when lineages
        are still single-sample. Falls back to pairwise checking for
        merged lineages.

        Returns:
            List of sets, where lineages in each set can coalesce together
        """
        # Check if all lineages are single-sample (can use fast refinement)
        all_single = all(
            lin.bits.count() == 1 for lin in self._lineages
        )

        if all_single and len(self._constraints) > 0:
            # Use partition refinement for efficiency
            bipartitions = list(self._constraints)
            refinement = PartitionRefinement(bipartitions, self._n_samples)

            # Map sample groups to lineage groups
            sample_to_lineage = {}
            for lin in self._lineages:
                idx = lin.descendant_indices()[0]
                sample_to_lineage[idx] = lin

            groups = []
            for sample_group in refinement.groups:
                lineage_group = {
                    sample_to_lineage[idx]
                    for idx in sample_group
                    if idx in sample_to_lineage
                }
                if len(lineage_group) > 1:
                    groups.append(lineage_group)

            return groups

        # Fall back to pairwise checking
        return self._compute_groups_pairwise()

    def _compute_groups_pairwise(self) -> list[set[Lineage]]:
        """Compute groups using pairwise compatibility checking."""
        # Build adjacency using compatible pairs
        compatible = {lin: set() for lin in self._lineages}

        lineage_list = list(self._lineages)
        for i in range(len(lineage_list)):
            for j in range(i + 1, len(lineage_list)):
                lin_a, lin_b = lineage_list[i], lineage_list[j]
                if self._constraints.can_coalesce(lin_a, lin_b):
                    compatible[lin_a].add(lin_b)
                    compatible[lin_b].add(lin_a)

        # Find connected components where all pairs are compatible
        # (cliques in the compatibility graph)
        visited = set()
        groups = []

        for lin in self._lineages:
            if lin in visited:
                continue

            # Build group: all lineages compatible with each other
            group = {lin}
            candidates = compatible[lin].copy()

            for candidate in candidates:
                if candidate in visited:
                    continue
                # Check if candidate is compatible with all current group members
                if all(candidate in compatible[member] for member in group):
                    group.add(candidate)

            if len(group) > 1:
                groups.append(group)
                visited.update(group)
            else:
                visited.add(lin)

        return groups

    def coalesce(self, lin_a: Lineage, lin_b: Lineage) -> Lineage:
        """
        Coalesce two lineages into one.

        Args:
            lin_a: First lineage (must be active)
            lin_b: Second lineage (must be active)

        Returns:
            The newly created merged lineage

        Raises:
            ValueError: If lineages are not active or cannot coalesce
        """
        if lin_a not in self._lineages:
            raise ValueError(f"Lineage {lin_a} is not active")
        if lin_b not in self._lineages:
            raise ValueError(f"Lineage {lin_b} is not active")
        if not self._constraints.can_coalesce(lin_a, lin_b):
            raise ValueError(f"Lineages {lin_a} and {lin_b} cannot coalesce")

        # Create merged lineage
        merged = Lineage.coalesce(lin_a, lin_b)

        # Update active lineages
        self._lineages.remove(lin_a)
        self._lineages.remove(lin_b)
        self._lineages.add(merged)

        # Remove any bipartitions that are now satisfied
        self._constraints.update_after_coalescence(self._lineages)

        return merged

    def is_mrca(self) -> bool:
        """Check if coalescence is complete (single lineage remaining)."""
        return len(self._lineages) == 1

    def __repr__(self) -> str:
        return f"CoalescentState({len(self._lineages)} lineages, {len(self._constraints)} constraints)"
