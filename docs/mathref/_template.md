# [NN] Title

<!-- 
  SOM File Template — Mandatory structure for every file in docs/mathref/ (except 00_index.md).
  Replace [NN] with the file number. Replace "Title" with the descriptive title.
  Delete this comment block before finalizing.
  
  All notation must follow docs/notation_standard.md.
  All cross-references must follow docs/mathref/PRODUCTION_PLAN.md Section 5.
  Algorithm numbers use the global counter registered in 00_index.md.
-->

## Introduction

<!-- 
  Required subsections:
  - What problem this file addresses (1-2 paragraphs)
  - Where this fits in the phasic pipeline (which stage of computation uses this algorithm)
  - Prerequisites: list earlier SOM files the reader must have read
  - Source files: which C/C++/Python files implement the algorithms documented here
-->

**Prerequisites:** [01], [02], ...

**Source files:**
- `src/c/phasic.c` (functions: ...)
- ...

## Definitions

<!--
  Definition NN.1, NN.2, etc. in narrative format.
  Each definition must be followed by:
  1. An intuitive explanation of what the definition captures
  2. The relationship to prior definitions (in this file or prerequisites)
  3. A concrete example when the concept is non-trivial
  
  Use the notation standard symbols (Section 12). If a new symbol is needed,
  flag it — do not invent notation silently.
-->

**Definition NN.1** (Name). *Let ...*

> **Intuition.** ...

> **Example.** ...

## Theorems and Proofs

<!--
  Theorem NN.1, Lemma NN.1, Proposition NN.1, Corollary NN.1.
  Every theorem must have an inline proof. "Proof omitted" is not acceptable.
  Proofs must be self-contained.
  
  Correctness results for algorithms go here (proved before the algorithm is stated,
  so the algorithm section can reference the theorem).
-->

**Theorem NN.1** (Name). *Statement.*

*Proof.* ... $\square$

## Algorithms

<!--
  Algorithm K: Title (K is the next available global number from 00_index.md).
  
  For each algorithm:
  1. Prose description of purpose and strategy (before the pseudocode)
  2. Formal pseudocode following notation_standard.md Section 8
  3. Correspondence table mapping pseudocode variables to math symbols and code variables
  4. Complexity analysis (time and space) as a formal statement
  5. Correctness argument referencing theorems from the section above
-->

### Algorithm K: Title

*Description.* ...

```
Algorithm K: Title
1: Let ... describe preconditions
2:
3: function FunctionName(parameters)
4:   ...                                ▷ Comment
5: end function
```

**Correspondence table:**

| Pseudocode variable | Math symbol | Code variable (file:function) |
|---------------------|-------------|-------------------------------|
| ... | ... | ... |

**Complexity.** Time: $O(...)$. Space: $O(...)$.

**Correctness.** Follows from Theorem NN.M. ...

## Numerical Considerations

<!--
  Only include this section if the algorithm has non-trivial numerical concerns.
  Topics: stability analysis, guard conditions, overflow/underflow handling,
  compensated summation, log-space computation.
  Delete this section entirely if not applicable.
-->

## Implementation Notes

<!--
  Bridge between the SOM formalization and the actual codebase.
  Do NOT duplicate API documentation from docs/pages/.
  
  Required content:
  - Source code mapping: file, function name, line ranges
  - Any deviations between pseudocode and implementation, with justification
  - Performance-relevant implementation choices not captured in the algorithm
-->

**Source code mapping:**

| Algorithm | File | Function | Lines |
|-----------|------|----------|-------|
| Algorithm K | `src/c/phasic.c` | `ptd_...` | L100–L200 |

**Deviations from pseudocode:**
- ...

## Symbol Index

<!--
  Alphabetical listing of all symbols introduced in THIS file.
  Symbols imported from prerequisite files are not listed here.
  Format: symbol | name | first appearance
-->

| Symbol | Name | First appearance |
|--------|------|-----------------|
| ... | ... | Definition NN.1 |
