"""
Clade Constraint Module

Provides constraints that enforce samples to form exclusive clades -
groups that must fully coalesce before any member can coalesce with
samples outside the group.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from bitarray import bitarray, frozenbitarray
from bitarray.util import any_and, subset

from coalescent_constraints import Lineage


class CladeConstraint:
    """
    Constraint requiring samples to form an exclusive clade.

    Samples marked with 1 in the numpy mask must coalesce together
    before any can coalesce with samples outside the group.
    """

    __slots__ = ('_bits', '_n_samples')

    def __init__(self, mask: NDArray[np.int_]):
        """
        Initialize with a numpy array mask.

        Args:
            mask: Array of 0s and 1s where 1 indicates membership in the clade
        """
        self._n_samples = len(mask)
        bits = bitarray(self._n_samples)
        bits.setall(0)
        for i in np.nonzero(mask)[0]:
            bits[i] = 1
        self._bits = frozenbitarray(bits)

    @classmethod
    def from_indices(cls, indices: set[int], n: int) -> CladeConstraint:
        """
        Create a clade constraint from sample indices.

        Args:
            indices: Sample indices that must form a clade
            n: Total number of samples

        Returns:
            CladeConstraint for the specified samples
        """
        mask = np.zeros(n, dtype=np.int8)
        for i in indices:
            mask[i] = 1
        return cls(mask)

    @property
    def bits(self) -> frozenbitarray:
        """The underlying bitarray representation."""
        return self._bits

    @property
    def indices(self) -> frozenset[int]:
        """Frozenset of sample indices in this clade."""
        return frozenset(self._bits.search(bitarray('1')))

    def n_samples(self) -> int:
        """Total number of samples in the system."""
        return self._n_samples

    def clade_size(self) -> int:
        """Number of samples in this clade."""
        return self._bits.count()

    def contains_sample(self, idx: int) -> bool:
        """Check if a sample index is in this clade."""
        return self._bits[idx]

    def overlaps_with(self, other: CladeConstraint) -> bool:
        """Check if this constraint shares any samples with another."""
        return any_and(self._bits, other._bits)

    def is_satisfied_by(self, lineages: set[Lineage]) -> bool:
        """
        Check if this constraint is satisfied (clade fully coalesced).

        The constraint is satisfied when all samples in the clade are
        descendants of a single lineage.
        """
        for lin in lineages:
            if subset(self._bits, lin.bits):
                return True
        return False

    def can_coalesce(self, lin_a: Lineage, lin_b: Lineage) -> bool:
        """
        Check if two lineages can coalesce under this constraint.

        Coalescence is allowed if:
        - Neither lineage contains any samples from this clade, OR
        - Both lineages contain ONLY samples from this clade (no outsiders), OR
        - One lineage already covers the entire clade (constraint satisfied)
        """
        a_bits = lin_a.bits
        b_bits = lin_b.bits

        # If either lineage already covers the entire clade,
        # the constraint is satisfied and doesn't restrict further
        if subset(self._bits, a_bits) or subset(self._bits, b_bits):
            return True

        a_in_clade = any_and(a_bits, self._bits)
        b_in_clade = any_and(b_bits, self._bits)

        # If neither is in the clade, constraint doesn't apply
        if not a_in_clade and not b_in_clade:
            return True

        # If both are entirely within the clade, allow coalescence
        if a_in_clade and b_in_clade:
            a_only_in_clade = subset(a_bits, self._bits)
            b_only_in_clade = subset(b_bits, self._bits)
            if a_only_in_clade and b_only_in_clade:
                return True

        # One is in clade, other is not (or one spans clade boundary)
        return False

    def __hash__(self) -> int:
        return hash(self._bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CladeConstraint):
            return NotImplemented
        return self._bits == other._bits

    def __repr__(self) -> str:
        indices = list(self._bits.search(bitarray('1')))
        indices_str = ', '.join(map(str, indices))
        return f"CladeConstraint({{{indices_str}}})"


class CladeConstraintSet:
    """
    Manages multiple CladeConstraint objects.

    Validates that no sample appears in multiple active constraints.
    Uses bitarrays for efficient overlap detection.
    """

    __slots__ = ('_constraints', '_occupied_bits', '_n_samples')

    def __init__(self, n_samples: int = 0):
        """
        Initialize empty constraint set.

        Args:
            n_samples: Number of samples (set automatically on first add if 0)
        """
        self._n_samples = n_samples
        self._constraints: set[CladeConstraint] = set()
        if n_samples > 0:
            self._occupied_bits = bitarray(n_samples)
            self._occupied_bits.setall(0)
        else:
            self._occupied_bits: bitarray | None = None

    def add(self, constraint: CladeConstraint) -> None:
        """
        Add a clade constraint.

        Args:
            constraint: The constraint to add

        Raises:
            ValueError: If any sample in the constraint is already in another active constraint
        """
        # Initialize occupied bits on first add
        if self._occupied_bits is None:
            self._n_samples = constraint.n_samples()
            self._occupied_bits = bitarray(self._n_samples)
            self._occupied_bits.setall(0)

        # Check for overlaps using bitarray operations
        if any_and(constraint.bits, self._occupied_bits):
            overlap_bits = constraint.bits & self._occupied_bits
            overlapping = set(overlap_bits.search(bitarray('1')))
            raise ValueError(
                f"Sample(s) {overlapping} already in another constraint"
            )

        self._constraints.add(constraint)
        self._occupied_bits |= constraint.bits

    def remove(self, constraint: CladeConstraint) -> None:
        """Remove a clade constraint."""
        self._constraints.discard(constraint)
        if self._occupied_bits is not None:
            self._occupied_bits &= ~constraint.bits

    def clear(self) -> None:
        """Remove all constraints."""
        self._constraints.clear()
        if self._occupied_bits is not None:
            self._occupied_bits.setall(0)

    def update_after_coalescence(self, lineages: set[Lineage]) -> None:
        """
        Remove any constraints that are now satisfied.

        A constraint is satisfied when its clade samples are fully
        coalesced into a single lineage.

        Args:
            lineages: Current set of active lineages
        """
        satisfied = {c for c in self._constraints if c.is_satisfied_by(lineages)}
        for constraint in satisfied:
            self.remove(constraint)

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self) -> Iterator[CladeConstraint]:
        return iter(self._constraints)

    def can_coalesce(self, lin_a: Lineage, lin_b: Lineage) -> bool:
        """
        Check if two lineages can coalesce given all constraints.

        Args:
            lin_a: First lineage
            lin_b: Second lineage

        Returns:
            True if coalescence is allowed by all constraints
        """
        for constraint in self._constraints:
            if not constraint.can_coalesce(lin_a, lin_b):
                return False
        return True

    def get_compatible_groups(self, lineages: set[Lineage]) -> list[set[Lineage]]:
        """
        Efficiently compute groups of lineages that can coalesce with each other.

        Groups lineages by their constraint membership signature. Lineages in the
        same group can coalesce with each other.

        Args:
            lineages: Current set of active lineages

        Returns:
            List of sets, where lineages in each set can coalesce together.
            Only groups with 2+ lineages are returned.
        """
        if not self._constraints:
            # No constraints - all lineages can coalesce
            if len(lineages) >= 2:
                return [lineages.copy()]
            return []

        # For each lineage, compute its "signature" across all constraints
        # Signature value for each constraint:
        #   0 = not in clade
        #   1 = partially in clade (some samples, not all)
        #   2 = covers entire clade (constraint satisfied)
        #  -1 = spans boundary (has samples both in and out of clade)
        def compute_signature(lin: Lineage) -> tuple:
            sig = []
            lin_bits = lin.bits
            for constraint in self._constraints:
                clade_bits = constraint.bits
                # Check if lineage covers entire clade
                if subset(clade_bits, lin_bits):
                    sig.append(2)  # Covers clade - effectively "free"
                elif any_and(lin_bits, clade_bits):
                    # Has some overlap - check if entirely within clade
                    if subset(lin_bits, clade_bits):
                        sig.append(1)  # Entirely within clade
                    else:
                        sig.append(-1)  # Spans boundary
                else:
                    sig.append(0)  # No overlap with clade
            return tuple(sig)

        # Group lineages by signature
        signature_to_lineages: dict[tuple, set[Lineage]] = {}
        for lin in lineages:
            sig = compute_signature(lin)
            if sig not in signature_to_lineages:
                signature_to_lineages[sig] = set()
            signature_to_lineages[sig].add(lin)

        # Determine which signatures are compatible
        def signatures_compatible(sig_a: tuple, sig_b: tuple) -> bool:
            for a, b in zip(sig_a, sig_b):
                if a == -1 or b == -1:
                    return False  # Boundary-spanning lineage
                if a == 2 or b == 2:
                    continue  # Clade satisfied, no restriction
                if a == 0 and b == 0:
                    continue  # Neither in clade
                if a == 1 and b == 1:
                    continue  # Both in same clade
                if (a == 0 and b == 1) or (a == 1 and b == 0):
                    return False  # One in clade, one out
            return True

        # Build groups of compatible signatures
        signatures = list(signature_to_lineages.keys())
        visited = set()
        groups = []

        for i, sig_a in enumerate(signatures):
            if sig_a in visited:
                continue

            group_sigs = {sig_a}
            for sig_b in signatures[i+1:]:
                if sig_b in visited:
                    continue
                if all(signatures_compatible(sig_b, gs) for gs in group_sigs):
                    group_sigs.add(sig_b)

            group_lineages: set[Lineage] = set()
            for sig in group_sigs:
                group_lineages.update(signature_to_lineages[sig])
                visited.add(sig)

            if len(group_lineages) >= 2:
                groups.append(group_lineages)

        return groups

    def get_compatible_pairs(self, lineages: set[Lineage]) -> list[tuple[Lineage, Lineage]]:
        """
        Get all pairs of lineages that can coalesce.

        Uses efficient grouping to avoid O(n²) pairwise checks when possible.

        Args:
            lineages: Current set of active lineages

        Returns:
            List of (lineage_a, lineage_b) tuples that can coalesce
        """
        groups = self.get_compatible_groups(lineages)
        pairs = []
        for group in groups:
            group_list = list(group)
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    pairs.append((group_list[i], group_list[j]))
        return pairs

    def __repr__(self) -> str:
        return f"CladeConstraintSet({len(self._constraints)} constraints)"
