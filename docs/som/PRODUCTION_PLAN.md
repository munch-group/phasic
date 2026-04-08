# SOM Production Plan

**Version:** 1.0  
**Date:** 2026-04-08  
**Status:** Draft  

---

## 1. Purpose

This document defines the structure, production schedule, and quality assurance process for the Phasic Supplementary Online Material (SOM). The SOM contains exhaustive mathematical and algorithmic documentation with proofs for every algorithm in phasic. It is produced by the 5-agent pipeline defined in `docs/notation_standard.md` Section 14.

The SOM is:
- **Self-contained**: A mathematician or CS researcher can read it without the codebase
- **Non-duplicating**: It does not repeat API/howto docs in `docs/pages/`; it formalizes and proves
- **Notation-compliant**: Every symbol follows `docs/notation_standard.md` exactly

---

## 2. File Structure

### 2.1 Naming Convention

All files: `NN_descriptive_snake_case.md` where `NN` is zero-padded. A file may only reference files with a lower number. No forward references, no circular references.

### 2.2 Complete File Listing

```
docs/som/
  00_index.md                         # TOC, dependency DAG, global registries
  01_preliminaries.md                 # PH definitions (CPH, DPH, MPH), notation preamble
  02_graph_representation.md          # Weighted directed graph, AVL tree, validation, cloning
  03_numerical_primitives.md          # Kahan summation, log-space arithmetic
  04_scc_and_topological_sort.md      # Tarjan's SCC, topological sort
  05_graph_normalization.md           # Continuous and discrete normalization
  06_graph_elimination.md             # Gaussian elimination on graph (Alg 3, Røikjer 2022)
  07_reward_transformation.md         # Reward transformation (Alg 2, Røikjer 2022)
  08_state_space_construction.md      # State space construction (Alg 1, Røikjer 2022)
  09_symbolic_expressions.md          # Expression tree ADT, evaluation, differentiation
  10_symbolic_graph_elimination.md    # Symbolic elimination, instantiation
  11_trace_system.md                  # Trace recording, evaluation, caching
  12_graph_hashing.md                 # Modified Weisfeiler-Lehman + SHA-256
  13_scc_trace_stitching.md           # SCC-based trace decomposition and stitching
  14_distribution_computation.md      # PDF/PMF forward algorithm (Alg 4), CDF
  15_moment_computation.md            # All-order moments, joint/cross moments (MPH)
  16_pdf_gradients.md                 # Uniformization-based PDF gradients
  17_sampling.md                      # Unconditioned/conditioned sampling, reward decomposition
  18_svgd.md                          # Stein Variational Gradient Descent
  19_mcmc.md                          # Metropolis-Hastings with adaptive proposals
  20_bffg.md                          # Backward Filtering Forward Guiding
  21_method_of_moments.md             # Method of moments estimation
  22_probability_matching.md          # Probability matching estimation
  23_state_indexing.md                # Mixed-radix StateIndexer, Property, PropertySet
  24_hex_grid.md                      # Hexagonal grid tessellation
  25_jax_integration.md               # JAX FFI, pure_callback, custom VJP
  26_graph_builder.md                 # C++ parameterized graph construction
  27_van_loan_equivalence.md          # Van Loan block matrix equivalence (formalized)
```

### 2.3 Granularity Rationale

- **One file per algorithm or tightly coupled cluster** (2–3 algorithms sharing definitions)
- Large layers split to keep files processable by one pipeline run
- Moments split from PDF/CDF: different preconditions (acyclic graph vs. original), different complexity
- Numerical primitives separate from preliminaries: different character (numerical analysis vs. mathematical setup)
- Index separate from preliminaries: navigation doc with no definitions

---

## 3. Within-File Template

Every file (except `00_index.md`) follows this mandatory section sequence:

```markdown
# [NN] Title

## Introduction
- What problem this file addresses
- Where this fits in the phasic pipeline
- Prerequisites: list of earlier SOM files assumed read
- Source files: which C/C++/Python files implement this

## Definitions
- Definition NN.1, NN.2, ... in narrative format
- Each followed by: intuitive explanation, relation to prior definitions, example

## Theorems and Proofs
- Theorem NN.1, Lemma NN.1, etc.
- Each with inline proof (not deferred)
- Correctness results for algorithms

## Algorithms
- Algorithm K: Title (global counter, not per-file)
- For each:
  - Prose description of purpose and strategy
  - Formal pseudocode (notation_standard.md Section 8)
  - Correspondence table: pseudocode var | math symbol | code variable
  - Complexity analysis (time and space)
  - Correctness argument referencing theorems above

## Numerical Considerations
- Stability analysis, guard conditions, overflow/underflow
- Only present when algorithm has non-trivial numerical concerns

## Implementation Notes
- Source code mapping: file, function, line ranges
- Deviations between pseudocode and implementation (with justification)
- Bridges SOM to codebase without duplicating API docs

## Symbol Index
- Alphabetical listing per notation_standard.md Section 10.2
- Format: symbol | name | first appearance (Definition/Equation number)
```

The `00_index.md` file has a different structure:

```markdown
# Phasic Supplementary Online Material — Index

## Reading Guide
## Dependency DAG
## Global Algorithm Registry    (Algorithm K -> file, title)
## Global Definition Registry   (Definition NN.M -> file, name)
## Coverage Matrix              (source file -> SOM file(s))
```

---

## 4. Numbering System

| Entity | Pattern | Example | Scope |
|--------|---------|---------|-------|
| Definitions, Theorems, Lemmas | `NN.M` | Definition 6.3, Theorem 4.1 | Per-file |
| Algorithms | Global sequential | Algorithm 1, Algorithm 2, ... | Entire SOM |
| Equations | Sequential within file | (1), (2), ... | Per-file |
| Cross-file equation refs | File-qualified | Equation (3) in [01] | — |

---

## 5. Cross-File Reference System

### Format

```
(see Definition 6.3 in [06])
(by Theorem 4.1 in [04])
(Algorithm 5 in [06])
```

`[NN]` is the file number. Full filename resolved via `00_index.md`.

### Rules

1. References only point to lower-numbered files
2. Always cite specific definition/theorem/algorithm number
3. When a concept from another file is used >3 times in a section, add an **Extended Recall**:

> **Recall** (Definition 6.3, [06]). *Vertex elimination* of $v \in V$ removes $v$ from $G$ and adds, for each parent $u$ and child $z$, a bypass edge $(u \to z)$ with weight $w(u \to v) \cdot w(v \to z) / \lambda_v$.

Extended recalls are italic, carry no Definition number, and must match the source definition verbatim.

4. Each file's Introduction lists prerequisite files (reading-order guide + dependency declaration)

---

## 6. Deduplication Strategy

### 6.1 Canonical Ownership

Each concept has exactly one owning file: the earliest file (by number) that needs it as a non-trivial component.

| Concept | Owner | Referenced by |
|---------|-------|---------------|
| Weighted directed graph, vertex, edge | 02 | All subsequent |
| Vertex elimination (bypass edges) | 06 | 07, 10, 11, 13 |
| Normalized graph | 05 | 06, 07, 14 |
| SCC decomposition | 04 | 06, 13 |
| Topological ordering | 04 | 06, 07, 14, 15 |
| Expression tree | 09 | 10, 11, 16, 25 |
| Trace operation | 11 | 12, 13 |
| Acyclic graph (as output of elimination) | 06 | 14, 15, 17 |
| Reward vector and reward matrix | 01 | 07, 08, 15, 17 |
| Forward probability vector | 14 | 16, 17 |
| Backward probability vector | 17 | (self-contained) |
| Parameter vector and parameterized edges | 01 | 09, 10, 25, 26 |

### 6.2 Rules

- Later files use the Recall mechanism, never redefine
- Agent 5 checks that every Extended Recall matches its source verbatim
- If two files need the same concept, the lower-numbered file owns it

### 6.3 Specific Deduplication Decisions

**Files 06 and 07** (elimination vs. reward transformation): `06` owns vertex elimination in full. `07` opens with an Extended Recall from `06`, then defines only the reward-augmented delta.

**Files 10 and 11** (symbolic elimination vs. trace system): `10` owns deferred evaluation at the expression level. `11` owns deferred evaluation at the operation level, referencing `10` for expression semantics.

---

## 7. Production Batches

Each batch = one invocation of the 5-agent pipeline (Agent 1 → Agents 2∥3 → Agent 4 → Agent 5).

| Batch | Files | Source files for Agent 1 | Prerequisites | Notes |
|-------|-------|------------------------|---------------|-------|
| 1 | 00 (skeleton), 01 | `api/c/phasic.h` (structs), notation standard §3–5 | None | Foundation; every later file depends on 01 |
| 2 | 02, 03 | `src/c/phasic.c` (graph creation, AVL, validate, clone, Kahan) | Batch 1 | Pure data structures |
| 3 | 04, 05 | `src/c/phasic.c` (SCC, topo sort, normalize) | Batch 2 | Structural algorithms |
| 4 | 06, 07 | `src/c/phasic.c` (elimination, reward transform) | Batches 1–3 | Central algorithms of phasic |
| 5 | 08 | `src/c/phasic.c` (state space), `src/phasic/state_indexing.py` | Batches 1–2 | Can parallel with Batch 3+ |
| 6 | 09, 10 | `src/c/phasic_symbolic.c`, `api/c/phasic.h` (expr structs) | Batches 1–4 | Symbolic system |
| 7 | 11, 12, 13 | `src/c/trace/trace_internal.h`, `trace_cache.c`, `phasic_hash.c` | Batches 1–4, 6 | Split to 7a(11,12)+7b(13) if too large |
| 8 | 14, 15, 16 | `src/c/phasic.c` (PDF, moments, gradients) | Batches 1–5 | Split to 8a(14,15)+8b(16) if needed |
| 9 | 17 | `src/c/phasic.c` (sampling, backward probs) | Batches 1–5, 8 | |
| 10 | 18 | `src/phasic/svgd.py` | Batches 1–8 | Largest single file; can parallel with 11, 13 |
| 11 | 19 | `src/phasic/mcmc.py` | Batches 1–8 | Can parallel with 10, 13 |
| 12 | 20, 21, 22 | `src/phasic/bffg.py`, `method_of_moments.py`, `probability_matching.py` | Batches 1–8 | Can parallel with 13 |
| 13 | 23, 24 | `src/phasic/state_indexing.py`, `src/phasic/hex_grid.py` | Batches 1–2 | Can parallel with Batches 3–12 |
| 14 | 25, 26, 27 | `src/cpp/phasic_pybind.cpp`, `graph_builder.cpp`, `ffi_wrappers.py`, `van_loen_graph_equivalence.md` | Batches 1–8 | |
| 15 | 00 (finalize) | All SOM files | All batches | Populate registries, verify DAG |

### Parallelization Opportunities

```
Batch 1 ──→ Batch 2 ──→ Batch 3 ──→ Batch 4 ──→ Batch 6 ──→ Batch 7
                │                                                  │
                └──→ Batch 5 (parallel with 3+)                    │
                │                                                  ▼
                └──→ Batch 13 (parallel with 3–12)          Batch 8 ──→ Batch 9
                                                               │
                                                    ┌──────────┼──────────┐
                                                    ▼          ▼          ▼
                                                 Batch 10   Batch 11   Batch 12
                                                    │          │          │
                                                    └──────────┼──────────┘
                                                               ▼
                                                           Batch 14 ──→ Batch 15
```

---

## 8. Final Homogeneity Pass

After all 15 batches, five structured checks. Each is a separate pipeline invocation.

### H1: Internal Consistency (Agent 5, extended scope)

1. Build global symbol table from all Definition NN.M across all files
2. Verify each symbol has exactly one authoritative definition (recalls OK, duplicate definitions not)
3. Verify all cross-file references resolve (`[NN]` targets exist, Definition/Theorem/Algorithm numbers exist in target)
4. Verify global algorithm counter is contiguous (no gaps, no duplicates)
5. Verify every file's prerequisites list is accurate (file actually references all listed prerequisites and no unlisted ones)

### H2: Exhaustiveness (Agent 1, extended scope)

1. For each source file in notation_standard.md Section 14.1, extract every public function
2. Verify each appears in at least one SOM file's Implementation Notes
3. Output coverage matrix, flag uncovered functions
4. Cross-check against `00_index.md` coverage matrix
5. Verify every algorithm in the SOM maps to at least one source function

### H3: Non-Duplication (Agent 5)

1. For each Definition NN.M, search all other files for definitions of the same concept (by name)
2. Flag any concept defined authoritatively in two places
3. Flag any two algorithms computing the same function (unless one generalizes the other, stated explicitly)
4. Verify Extended Recalls match their source definitions verbatim

### H4: Notation Compliance Sweep (Agent 5)

1. Every symbol against Section 12 of notation_standard.md
2. Typographic conventions: matrices bold uppercase, vectors bold lowercase, scalars italic
3. Pseudocode: Section 8 conventions (line numbers, bold keywords, PascalCase, arrow assignment)
4. No orphan numbered equations (every numbered equation referenced at least once)

### H5: Existing-Doc Coherence

For files formalizing existing informal docs:
- `27` vs `docs/pages/topics/van_loen_graph_equivalence.md`
- `11` vs `docs/pages/internals/trace_elimination_retracing.md`
- `18` vs `docs/pages/internals/preconditioning.md`

Verify:
1. Every claim in informal doc is proved in SOM or noted as out-of-scope
2. No contradictions
3. SOM uses notation standard; informal doc may use ad-hoc notation (acceptable)

---

## 9. Handling Plan Updates

This plan may be updated as batches are produced and new information emerges. When updating:

1. Record the change in the revision history below
2. If a file is added or removed, update the file listing (Section 2.2), the batch table (Section 7), and the dependency DAG
3. If a concept ownership changes, update the deduplication table (Section 6.1)

---

## 10. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-08 | Initial plan |
