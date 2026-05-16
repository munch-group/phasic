"""
ARG Constraint Module

Provides constraints for ancestral recombination graph (ARG) simulations.
Constraints enforce that specific ancestral haplotypes (ARG nodes) must exist,
where each haplotype has potentially different descendant samples at each locus.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from bitarray import bitarray, frozenbitarray
from bitarray.util import any_and, subset


class ARGLineage:
    """
    Represents a lineage in an ARG, tracking descendants at each locus.

    Uses a tuple of frozenbitarrays, one per locus, for efficient
    per-locus ancestry tracking.
    """

    __slots__ = ('_bits_per_locus', '_n_samples', '_n_loci')

    def __init__(self, bits_per_locus: tuple[frozenbitarray, ...]):
        """
        Initialize with tuple of frozenbitarrays, one per locus.

        Args:
            bits_per_locus: Tuple of frozenbitarrays, each of length n_samples
        """
        self._bits_per_locus = bits_per_locus
        self._n_samples = len(bits_per_locus[0]) if bits_per_locus else 0
        self._n_loci = len(bits_per_locus)

    @classmethod
    def from_sample(cls, idx: int, n_samples: int, n_loci: int) -> ARGLineage:
        """
        Create a lineage for a single sample (ancestral to itself at all loci).

        Args:
            idx: Sample index
            n_samples: Total number of samples
            n_loci: Number of loci

        Returns:
            ARGLineage representing single sample at all loci
        """
        bits = bitarray(n_samples)
        bits.setall(0)
        bits[idx] = 1
        frozen = frozenbitarray(bits)
        # Same bits for all loci (sample is its own ancestor everywhere)
        return cls(tuple(frozen for _ in range(n_loci)))

    @classmethod
    def coalesce(cls, lin_a: ARGLineage, lin_b: ARGLineage,
                 loci: set[int] | None = None) -> ARGLineage:
        """
        Create a merged lineage from two parent lineages.

        Args:
            lin_a: First parent lineage
            lin_b: Second parent lineage
            loci: Which loci to coalesce at (None = all loci)

        Returns:
            New lineage with combined descendants at specified loci
        """
        if loci is None:
            loci = set(range(lin_a._n_loci))

        merged_bits = []
        for loc in range(lin_a._n_loci):
            if loc in loci:
                merged = lin_a._bits_per_locus[loc] | lin_b._bits_per_locus[loc]
                merged_bits.append(frozenbitarray(merged))
            else:
                # Keep lin_a's bits at non-coalescing loci (arbitrary choice)
                merged_bits.append(lin_a._bits_per_locus[loc])

        return cls(tuple(merged_bits))

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self._n_samples

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return self._n_loci

    def bits_at_locus(self, locus: int) -> frozenbitarray:
        """Get the bitarray for a specific locus."""
        return self._bits_per_locus[locus]

    def descendant_indices_at_locus(self, locus: int) -> list[int]:
        """Get descendant sample indices at a specific locus."""
        return list(self._bits_per_locus[locus].search(bitarray('1')))

    def descendants_at_locus(self, locus: int) -> frozenset[int]:
        """Get frozenset of descendants at a locus."""
        return frozenset(self.descendant_indices_at_locus(locus))

    def __hash__(self) -> int:
        return hash(self._bits_per_locus)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ARGLineage):
            return NotImplemented
        return self._bits_per_locus == other._bits_per_locus

    def __repr__(self) -> str:
        loci_strs = []
        for loc in range(self._n_loci):
            indices = self.descendant_indices_at_locus(loc)
            loci_strs.append(f"{{{', '.join(map(str, indices))}}}")
        return f"ARGLineage([{', '.join(loci_strs)}])"


class ARGConstraint:
    """
    Constraint requiring an ancestral haplotype to exist in the ARG.

    Specified as a 2D numpy array (n_samples, n_loci) where 1s mark
    samples that must be descendants of this haplotype at each locus.
    """

    __slots__ = ('_bits_per_locus', '_n_samples', '_n_loci')

    def __init__(self, mask: NDArray[np.int_]):
        """
        Initialize with 2D numpy array.

        Args:
            mask: Shape (n_samples, n_loci), 1 = in clade at that locus
        """
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {mask.ndim}D")

        self._n_samples, self._n_loci = mask.shape

        # Store as frozenbitarrays for efficient bitwise operations
        bits_list = []
        for loc in range(self._n_loci):
            bits = bitarray(self._n_samples)
            bits.setall(0)
            for i in np.nonzero(mask[:, loc])[0]:
                bits[i] = 1
            bits_list.append(frozenbitarray(bits))
        self._bits_per_locus = tuple(bits_list)

    @classmethod
    def from_indices(cls, indices_per_locus: list[set[int]], n_samples: int) -> ARGConstraint:
        """
        Create from list of index sets, one per locus.

        Args:
            indices_per_locus: List of sets, indices_per_locus[loc] = samples in clade at locus
            n_samples: Total number of samples

        Returns:
            ARGConstraint for the specified samples
        """
        n_loci = len(indices_per_locus)
        mask = np.zeros((n_samples, n_loci), dtype=np.int8)
        for loc, indices in enumerate(indices_per_locus):
            for i in indices:
                mask[i, loc] = 1
        return cls(mask)

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self._n_samples

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return self._n_loci

    def bits_at_locus(self, locus: int) -> frozenbitarray:
        """Get the bitarray for a specific locus."""
        return self._bits_per_locus[locus]

    def indices_at_locus(self, locus: int) -> frozenset[int]:
        """Get sample indices in clade at given locus."""
        return frozenset(self._bits_per_locus[locus].search(bitarray('1')))

    def clade_size_at_locus(self, locus: int) -> int:
        """Number of samples in clade at given locus."""
        return self._bits_per_locus[locus].count()

    def is_satisfied_at_locus(self, locus: int, lineages: set[ARGLineage]) -> bool:
        """
        Check if constraint is satisfied at a specific locus.

        Satisfied when all required samples at this locus are descendants
        of a single lineage.
        """
        clade_bits = self._bits_per_locus[locus]
        if not clade_bits.any():
            return True  # Empty constraint is trivially satisfied

        for lin in lineages:
            # Check if clade_bits is a subset of lineage bits (all clade samples are descendants)
            if subset(clade_bits, lin.bits_at_locus(locus)):
                return True
        return False

    def is_satisfied(self, lineages: set[ARGLineage]) -> bool:
        """
        Check if constraint is satisfied at ALL loci by a single lineage.

        The ancestral haplotype exists when one lineage covers all required
        samples at all loci simultaneously.
        """
        for lin in lineages:
            covers_all_loci = True
            for locus in range(self._n_loci):
                clade_bits = self._bits_per_locus[locus]
                if clade_bits.any():  # Skip empty loci
                    if not subset(clade_bits, lin.bits_at_locus(locus)):
                        covers_all_loci = False
                        break
            if covers_all_loci:
                return True
        return False

    def can_coalesce_at_locus(self, locus: int, lin_a: ARGLineage, lin_b: ARGLineage) -> bool:
        """
        Check if two lineages can coalesce at a specific locus.

        Coalescence is allowed if:
        - Neither lineage contains any samples from this clade at this locus, OR
        - Both lineages contain ONLY samples from this clade at this locus, OR
        - One lineage already covers the entire clade (constraint satisfied at this locus)
        """
        clade_bits = self._bits_per_locus[locus]
        if not clade_bits.any():
            return True  # Empty constraint doesn't restrict

        a_bits = lin_a.bits_at_locus(locus)
        b_bits = lin_b.bits_at_locus(locus)

        # If either lineage already covers the entire clade at this locus,
        # the constraint is satisfied here and doesn't restrict further
        if subset(clade_bits, a_bits) or subset(clade_bits, b_bits):
            return True

        # Check if lineages have any overlap with clade
        a_in_clade = any_and(a_bits, clade_bits)
        b_in_clade = any_and(b_bits, clade_bits)

        # If neither is in the clade, constraint doesn't apply
        if not a_in_clade and not b_in_clade:
            return True

        # If both are entirely within the clade, allow coalescence
        if a_in_clade and b_in_clade:
            # Check if lineage bits are subset of clade bits (only clade samples)
            a_only_in_clade = subset(a_bits, clade_bits)
            b_only_in_clade = subset(b_bits, clade_bits)
            if a_only_in_clade and b_only_in_clade:
                return True

        # One is in clade, other is not (or one spans clade boundary)
        return False

    def can_coalesce(self, lin_a: ARGLineage, lin_b: ARGLineage,
                     loci: set[int] | None = None) -> bool:
        """
        Check if coalescence is allowed at specified loci.

        Args:
            lin_a: First lineage
            lin_b: Second lineage
            loci: Which loci to check (None = all loci)

        Returns:
            True if coalescence is allowed at all specified loci
        """
        if loci is None:
            loci = set(range(self._n_loci))

        for loc in loci:
            if not self.can_coalesce_at_locus(loc, lin_a, lin_b):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self._bits_per_locus)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ARGConstraint):
            return NotImplemented
        return self._bits_per_locus == other._bits_per_locus

    def __repr__(self) -> str:
        loci_strs = []
        one = bitarray('1')
        for loc in range(self._n_loci):
            indices = list(self._bits_per_locus[loc].search(one))
            if indices:
                loci_strs.append(f"locus{loc}={{{', '.join(map(str, indices))}}}")
        return f"ARGConstraint({', '.join(loci_strs)})"


class ARGConstraintSet:
    """
    Manages multiple ARGConstraint objects.

    Validates that no sample appears in multiple constraints at the same locus.
    Uses bitarrays for efficient overlap detection.
    """

    __slots__ = ('_constraints', '_occupied_bits_per_locus', '_n_loci', '_n_samples')

    def __init__(self, n_loci: int, n_samples: int = 0):
        """
        Initialize empty constraint set for given number of loci.

        Args:
            n_loci: Number of loci in the ARG
            n_samples: Number of samples (set automatically on first add if 0)
        """
        self._n_loci = n_loci
        self._n_samples = n_samples
        self._constraints: set[ARGConstraint] = set()
        # Track which samples are already in a constraint at each locus
        if n_samples > 0:
            self._occupied_bits_per_locus: list[bitarray] = [
                bitarray(n_samples) for _ in range(n_loci)
            ]
            for bits in self._occupied_bits_per_locus:
                bits.setall(0)
        else:
            self._occupied_bits_per_locus = []

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return self._n_loci

    def add(self, constraint: ARGConstraint) -> None:
        """
        Add a constraint, validates no per-locus overlap with existing.

        Args:
            constraint: The constraint to add

        Raises:
            ValueError: If any sample at any locus is already in another constraint
        """
        if constraint.n_loci != self._n_loci:
            raise ValueError(
                f"Constraint has {constraint.n_loci} loci, expected {self._n_loci}"
            )

        # Initialize occupied bits on first add
        if not self._occupied_bits_per_locus:
            self._n_samples = constraint.n_samples
            self._occupied_bits_per_locus = [
                bitarray(self._n_samples) for _ in range(self._n_loci)
            ]
            for bits in self._occupied_bits_per_locus:
                bits.setall(0)

        # Check for overlaps at each locus using bitarray operations
        for locus in range(self._n_loci):
            clade_bits = constraint.bits_at_locus(locus)
            occupied_bits = self._occupied_bits_per_locus[locus]
            if any_and(clade_bits, occupied_bits):
                # Find overlapping indices for error message
                overlap_bits = clade_bits & occupied_bits
                overlapping = set(overlap_bits.search(bitarray('1')))
                raise ValueError(
                    f"Sample(s) {overlapping} at locus {locus} already in another constraint"
                )

        # Add constraint and mark samples as occupied
        self._constraints.add(constraint)
        for locus in range(self._n_loci):
            self._occupied_bits_per_locus[locus] |= constraint.bits_at_locus(locus)

    def remove(self, constraint: ARGConstraint) -> None:
        """Remove a constraint."""
        self._constraints.discard(constraint)
        # Clear the occupied bits for this constraint
        for locus in range(self._n_loci):
            clade_bits = constraint.bits_at_locus(locus)
            # XOR to clear only the bits that were set by this constraint
            # (assumes no overlap, which is enforced by add())
            self._occupied_bits_per_locus[locus] &= ~clade_bits

    def clear(self) -> None:
        """Remove all constraints."""
        self._constraints.clear()
        for bits in self._occupied_bits_per_locus:
            bits.setall(0)

    def update_after_coalescence(self, lineages: set[ARGLineage]) -> None:
        """
        Remove constraints where the ancestral haplotype now exists.

        A constraint is satisfied when a single lineage covers all required
        samples at all loci.

        Args:
            lineages: Current set of active lineages
        """
        satisfied = {c for c in self._constraints if c.is_satisfied(lineages)}
        for constraint in satisfied:
            self.remove(constraint)

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self) -> Iterator[ARGConstraint]:
        return iter(self._constraints)

    def can_coalesce(self, lin_a: ARGLineage, lin_b: ARGLineage,
                     loci: set[int] | None = None) -> bool:
        """
        Check if coalescence is allowed at specified loci.

        Args:
            lin_a: First lineage
            lin_b: Second lineage
            loci: Which loci to check (None = all loci)

        Returns:
            True if coalescence is allowed by all constraints at all specified loci
        """
        for constraint in self._constraints:
            if not constraint.can_coalesce(lin_a, lin_b, loci):
                return False
        return True

    def get_compatible_groups(self, lineages: set[ARGLineage]) -> list[set[ARGLineage]]:
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

        # For each lineage, compute its "signature" across all constraints at all loci
        # Signature tuple: for each constraint, for each locus:
        #   0 = not in clade
        #   1 = partially in clade (some samples, not all)
        #   2 = covers entire clade (constraint satisfied at this locus)
        def compute_signature(lin: ARGLineage) -> tuple:
            sig = []
            for constraint in self._constraints:
                for locus in range(self._n_loci):
                    clade_bits = constraint.bits_at_locus(locus)
                    if not clade_bits.any():
                        sig.append(0)  # Empty clade at this locus
                        continue

                    lin_bits = lin.bits_at_locus(locus)

                    # Check if lineage covers entire clade
                    if subset(clade_bits, lin_bits):
                        sig.append(2)  # Covers clade - effectively "free"
                    elif any_and(lin_bits, clade_bits):
                        # Has some overlap - check if entirely within clade
                        if subset(lin_bits, clade_bits):
                            sig.append(1)  # Entirely within clade
                        else:
                            sig.append(-1)  # Spans boundary - incompatible with most
                    else:
                        sig.append(0)  # No overlap with clade
            return tuple(sig)

        # Group lineages by signature
        signature_to_lineages: dict[tuple, set[ARGLineage]] = {}
        for lin in lineages:
            sig = compute_signature(lin)
            if sig not in signature_to_lineages:
                signature_to_lineages[sig] = set()
            signature_to_lineages[sig].add(lin)

        # Now determine which signatures are compatible with each other
        # Two signatures are compatible if for each constraint/locus:
        #   - Both are 0 (neither in clade), OR
        #   - Both are 1 (both entirely in same clade), OR
        #   - At least one is 2 (clade satisfied, free to coalesce), OR
        #   - Both are 0 or 2 (free lineages)
        def signatures_compatible(sig_a: tuple, sig_b: tuple) -> bool:
            for a, b in zip(sig_a, sig_b):
                if a == -1 or b == -1:
                    # Boundary-spanning lineage - check pairwise
                    return False
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

            # Find all signatures compatible with sig_a
            group_sigs = {sig_a}
            for sig_b in signatures[i+1:]:
                if sig_b in visited:
                    continue
                # Check if sig_b is compatible with ALL signatures in group
                if all(signatures_compatible(sig_b, gs) for gs in group_sigs):
                    group_sigs.add(sig_b)

            # Collect lineages from all compatible signatures
            group_lineages: set[ARGLineage] = set()
            for sig in group_sigs:
                group_lineages.update(signature_to_lineages[sig])
                visited.add(sig)

            if len(group_lineages) >= 2:
                groups.append(group_lineages)

        return groups

    def get_compatible_pairs(self, lineages: set[ARGLineage]) -> list[tuple[ARGLineage, ARGLineage]]:
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
        return f"ARGConstraintSet({len(self._constraints)} constraints, {self._n_loci} loci)"
