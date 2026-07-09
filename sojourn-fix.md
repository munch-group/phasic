# Fixing the O(n·k) sojourn-time allocation

**Component:** `ptd_expected_sojourn_time_subset` (`src/c/phasic.c`)
**Trigger:** `Graph.svgd(...)` / `Graph.joint_prob_table()` on a large joint-probability graph
**Change:** replace the forward `n × k` dense replay with an equivalent **adjoint
(reverse-mode) pass** — `O(n·k)` memory and `O(len(trace)·k)` time become `O(n)`
memory and `O(len(trace))` time.

---

## TL;DR

Computing expected sojourn (residence) times at `k` target vertices used to build
a dense `n × k` matrix (`n` = number of vertices). On a joint-probability graph
the targets are the *t-vertices* — one per joint outcome — so `k ≈ n`, and the
allocation is effectively `O(n²)`. A coalescent joint-prob model over 8 samples
(`n = 684,226`, `k = 279,936 = 6⁷`) asked for a single

```
684,226 × 279,936 × 8 bytes ≈ 1.5 TB
```

block and crashed with

```
[ERROR] phasic.c: Failed to allocate results matrix (684226 x 279936 doubles)
```

The quantity being computed — the expected sojourn time at each vertex — is
exactly the **gradient** of a single scalar (the read-out at the start vertex)
with respect to the reward seed. Reverse-mode differentiation produces that
gradient for *all* vertices in **one pass over the elimination trace** using
**`O(n)` memory**. The result is identical to the old path to floating-point
rounding; the 1.5 TB allocation becomes ~7.7 MB.

---

## 1. The problem

### 1.1 The symptom

```
RuntimeError
  ...
  File .../phasic/__init__.py:11559, in Graph._get_joint_probs(self)
      sojourn_times = self.expected_sojourn_time(t_indices)
  File .../phasic/__init__.py:2194, in Graph.expected_sojourn_time(...)
      return super().expected_sojourn_time(*args, **kwargs)
RuntimeError

[ERROR] phasic.c: Failed to allocate results matrix (684226 x 279936 doubles)
```

(The traceback also shows `svgd` at the `finally` block around line 7076 — that is
just the exception unwinding through the weight-state restore; the originating
call is `joint_prob_table = self.joint_prob_table()` at `__init__.py:6808`.)

### 1.2 The call chain

`svgd` on a joint-probability graph maps each observed count-vector to a
`t_vertex_index` by building the full joint-probability table:

```
Graph.svgd                         __init__.py:6808   joint_prob_table = self.joint_prob_table()
  └─ Graph.joint_prob_table        __init__.py:11598  outcomes, probs, t = self._get_joint_probs()
       └─ Graph._get_joint_probs   __init__.py:11559  sojourn_times = self.expected_sojourn_time(t_indices)
            └─ (pybind) Graph::expected_sojourn_time   api/cpp/phasiccpp.h:475
                 └─ ptd_expected_sojourn_time_subset   src/c/phasic.c:10181   ← the allocation
```

`t_indices` is the list of every t-vertex (a state whose only child is the
absorbing vertex — i.e. one distinct joint outcome). For the model above there
are `k = 279,936` of them.

### 1.3 Why the matrix is astronomically large

The old routine computed the sojourn time at each of the `k` targets by seeding
**one one-hot reward column per target** and replaying the elimination trace on
all columns at once, stored row-major as a single flat `n·k` block:

```c
double *results_flat = (double *) calloc(n * k, sizeof(double));   // n·k doubles
```

For a joint-prob graph, `k` (number of outcomes) grows with `n` (number of
states), so `n·k ≈ n²`. The state space here is a mixed-radix product — capping
each of the 7 non-trivial mutation-count categories at `reward_limit = 5` gives
`6⁷ = 279,936` outcomes — and the augmented graph carrying them has 684,226
vertices. `n·k·8 ≈ 1.5 TB`.

### 1.4 It is not only a memory problem

Even with unlimited RAM the forward subset is `O(len(trace) · k)` in time: the
inner loop runs over all `k` columns for every trace command. With
`k = 279,936` that never finishes. The fix removes both costs at once — the
memory *and* the time — because it does not iterate over the targets at all.

---

## 2. Background: the elimination trace and what "sojourn time" means

### 2.1 The trace

`phasic` eliminates the graph **once** with unit weights while recording a linear
list of arithmetic operations (`trace_elimination.py`; the concrete-`double`
version lives on the graph as `reward_compute_graph`). Each operation is a
`ptd_reward_increase`:

```c
struct ptd_reward_increase { size_t from; size_t to; double multiplier; };

struct ptd_desc_reward_compute { size_t length; struct ptd_reward_increase *commands; };
```

and means, over a working vector `results[]`:

```
results[from] += results[to] * multiplier
```

Replaying the whole list is `O(len(trace))` per reward vector and is what makes
the method JAX-friendly and scalable.

### 2.2 Forward replay = expected accumulated reward

Seed `results[v]` with a reward vector `s[v]`, replay every command in order, and
read `results[0]` (vertex `0` is the starting vertex). This is exactly what
`ptd_expected_waiting_time` does (`src/c/phasic.c`, returns the full length-`n`
vector). The read-out at the start vertex is

```
results[0] = Σ_v sojourn(v) · s[v]
```

where `sojourn(v)` is the expected time spent in vertex `v` before absorption,
starting from the initial distribution at vertex `0`. In words: forward replay
turns a *reward vector* into the *expected accumulated reward from the start*.

### 2.3 The subset function's job

`ptd_expected_sojourn_time_subset(graph, indices, k)` must return, for each
requested target `indices[r]`, the scalar `sojourn(indices[r])`. Feeding the
**one-hot** reward `s = e_{indices[r]}` into the identity above gives

```
results[0] = sojourn(indices[r])
```

so the old code used one one-hot column per target and read `results[0]` out of
each column — hence the `n × k` matrix.

The sibling `ptd_expected_sojourn_time` (no subset) does the same with the full
`n × n` identity and returns `results[0][:]`, i.e. the entire residence-time
vector — an independent `O(n²)` reference we rely on for validation below.

---

## 3. The key insight: sojourn time is a gradient

### 3.1 The forward read is linear in the seed

From §2.2, the scalar `R(s) := results[0]` is a **linear** function of the seed:

```
R(s) = Σ_v sojourn(v) · s[v]      ⇒      sojourn(v) = ∂R/∂s[v]
```

The one-hot forward replay recovers one component of this gradient per column —
`R(e_w) = sojourn(w)`. Doing that `k` times (or `n` times, for the full version)
is just evaluating the same linear map against `k` basis vectors, which is a very
expensive way to read off the coefficients of a linear form.

### 3.2 Reverse-mode gives every partial derivative in one pass

Reverse-mode differentiation computes the gradient of one scalar output with
respect to **all** inputs in a single reverse sweep. Maintain an adjoint
`adjoint[v] = ∂R / ∂results[v]`. The forward operation

```
results[from] += results[to] * multiplier
```

reads `results[to]` with coefficient `multiplier` and leaves `results[from]`'s
own dependence at coefficient `1`, so its transpose (processed in **reverse**
command order) is

```
adjoint[to] += adjoint[from] * multiplier
```

Seed `adjoint[0] = 1` (because `R = results[0]`, so `∂R/∂results[0] = 1`). After
the reverse sweep, `adjoint[v] = ∂R/∂s[v] = sojourn(v)` for every `v`. The
requested targets are then a plain gather out of that length-`n` vector.

This is `O(len(trace))` time and `O(n)` memory, **independent of `k`**.

### 3.3 Worked example

Three vertices; vertex `0` is the start/read vertex. Trace:

```
cmd1:  results[1] += results[2] * m1
cmd2:  results[0] += results[1] * m2      (uses the updated results[1])
```

Forward, from seed `s`:

```
results[0] = s[0] + (s[1] + s[2]·m1)·m2
           = 1·s[0] + m2·s[1] + (m1·m2)·s[2]
```

so the true sojourn vector is `sojourn = (1, m2, m1·m2)`.

Adjoint — seed `adjoint = (1, 0, 0)`, process in reverse:

```
cmd2:  adjoint[1] += adjoint[0]·m2  →  adjoint[1] = m2
cmd1:  adjoint[2] += adjoint[1]·m1  →  adjoint[2] = m1·m2
```

Result `adjoint = (1, m2, m1·m2)` — identical, in one pass, without ever forming
a matrix.

---

## 4. The fix

`src/c/phasic.c`, `ptd_expected_sojourn_time_subset`. The default path is now the
adjoint; the old forward path is retained behind an environment flag as a
bit-for-bit reference / escape hatch.

### 4.1 The adjoint algorithm

```c
double *adjoint = (double *) calloc(n, sizeof(double));   // O(n), not O(n·k)
adjoint[0] = 1.0;                                         // start vertex index = 0

for (size_t ci = compute->length; ci-- > 0; ) {          // reverse command order
    struct ptd_reward_increase cmd = compute->commands[ci];

    if (cmd.multiplier == 0.0) {                         // 0 × ∞ = 0
        continue;
    }
    if (isinf(cmd.multiplier) && adjoint[cmd.from] == 0.0) {   // ∞ × 0 = 0
        continue;
    }
    adjoint[cmd.to] += adjoint[cmd.from] * cmd.multiplier;     // transpose update
}

// gather the requested targets out of the length-n adjoint vector
for (size_t r = 0; r < k; r++) {
    size_t v = indices[r];
    if (v >= n) { /* error + free + return NULL */ }
    sojourn_times[r] = adjoint[v];
}
```

Total allocation: `n` doubles for `adjoint` plus `k` doubles for the result —
`O(n + k)`.

### 4.2 The limit-convention guards, transposed

The forward path carries two limit conventions so degenerate multipliers do not
produce `NaN`:

| convention | forward guard | adjoint guard |
|---|---|---|
| `0 × ∞ = 0` | `if (mult == 0) continue;` | `if (mult == 0) continue;` |
| `∞ × 0 = 0` | `if (isinf(mult) && results[to] == 0) continue;` | `if (isinf(mult) && adjoint[from] == 0) continue;` |

The `0 × ∞` guard is direction-agnostic — a zero multiplier contributes nothing
either way. The `∞ × 0` guard skips an infinite multiplier when the operand it
multiplies is zero; in the forward path that operand is `results[to]`, and in the
transpose it is `adjoint[from]`, so the guard moves with it. Infinite multipliers
arise from the elimination itself (`1/x` with `x = 0` when a vertex has zero net
outgoing weight — e.g. the deficit sink described next).

### 4.3 Deficit-sink NaNs and why the targets stay clean

A joint-probability graph routes overflow events (more than `reward_limit`
occurrences of a category) into a **deficit sink**: an absorption-avoiding trap
that carries the distribution's missing mass and contributes to no joint outcome.
Eliminating that trap produces infinite multipliers, and the *full* residence
vector is `NaN` at those 1–2 trap vertices (confirmed directly:
`expected_sojourn_time()` returns exactly two `NaN` entries on these graphs).

Those `NaN`s never reach the answer, for a structural reason. Forward
accumulation `results[from] += results[to]·mult` flows **absorption → start**
(`from` is nearer the start, `to` nearer absorption). The transpose therefore
flows **start → absorption**, and the deficit sink sits at the absorption end —
it is reached *last* and feeds nothing upstream. The t-vertices that carry joint
probability are all upstream of the trap and have finite sojourn, so the reverse
sweep fills them in before the trap's `NaN` is ever formed. Empirically, across
every tested configuration, **all t-vertices come back finite** and equal to the
forward reference.

### 4.4 The escape hatch

Setting `PHASIC_SOJOURN_FORWARD=1` selects the original forward `n × k` path
verbatim. It exists purely as a bit-for-bit correctness reference and a fallback;
it still allocates the `n·k` matrix and will OOM on large `k` by design. This
mirrors the other opt-in numerical alternates in the file
(`PHASIC_HIERAR_ELIMINATION`, `PHASIC_FORCE_MPFR`, `PHASIC_USE_MPFR_LEGACY`,
`PHASIC_CONDITION_THRESHOLD`). Because `getenv` is read on each call, the two
paths can be A/B-compared within one process (the trace is cached on the graph;
only the read-out differs).

---

## 5. Correctness & validation

### 5.1 Equivalence to the forward reference

The adjoint was compared against **two independent forward implementations** — the
old subset path (via `PHASIC_SOJOURN_FORWARD=1`) and the no-arg `n × n`
Kahan-summed `ptd_expected_sojourn_time` — across acyclic, cyclic, discrete, and
continuous joint-prob graphs:

- Adjoint vs. forward at every finite vertex: **max relative difference ≈ 1e-15**
  (pure summation-order rounding; the two paths sum the same terms in transposed
  order, so bit-identity is not expected, but agreement to a few ULP is).
- End-to-end `joint_prob_table()` probabilities: match to ≈ 1e-15 relative,
  discrete and continuous.
- Acyclic (Erlang) and cyclic (back-edge, triangle) graphs: bit-identical to the
  forward path.

### 5.2 Non-contamination

For every configuration the requested t-vertex sojourns are finite and equal the
forward reference (`test_tvertex_targets_finite_and_correct`), and the subset
gather is exactly a slice of the full adjoint vector
(`test_subset_gather_is_a_slice_of_full`).

### 5.3 Test suite

- New gate `tests/pytest/test_sojourn_subset_adjoint.py` — 20 tests: adjoint vs
  forward-full equivalence, target finiteness, gather consistency, the
  `PHASIC_SOJOURN_FORWARD` escape hatch, and a `joint_prob_table()` over a
  >3,000-target graph.
- Existing suites re-run green, including the sojourn / daisy-chain / joint-index
  paths (`test_epoch_sojourn_finalread.py`, `test_daisy_chain_c_path.py`,
  `test_joint_index_callback.py`, `test_optimized_joint_index.py`) and the
  elimination **bit-identity** and moment gates
  (`test_gate_elimination_bit_identity.py`, `test_gate_moments_3way.py`,
  `test_gate_ffi_vs_pybind.py`). Total: **277 passed**, no regressions.
- End-to-end: `svgd` with the reported call signature (continuous joint graph +
  `epoch_starts` + `exposure` + `exposure_param_index` + `fixed` + prior) runs to
  completion through the adjoint sojourn path.

---

## 6. Performance

`L` = trace length. `n` = vertices, `k` = requested targets.

| | Forward (old / `PHASIC_SOJOURN_FORWARD=1`) | Adjoint (new default) |
|---|---|---|
| Memory | `O(n·k)` | `O(n + k)` |
| Time | `O(L·k)` | `O(L)` |
| Model `n=684,226, k=279,936` | ~1.5 TB → **crash** | ~7.7 MB |
| Model `n=39,603, k=16,807` | 5.3 GB, slow | ~0.3 MB, **123 ms** |

The adjoint's working set is one `double` per vertex regardless of how many
targets are requested, which is what turns the joint-probability read-out from
infeasible into instant.

---

## 7. Files changed

- `src/c/phasic.c` — `ptd_expected_sojourn_time_subset` rewritten: adjoint fast
  path as default, forward path retained behind `PHASIC_SOJOURN_FORWARD=1`
  (+85 / −30). Public signature unchanged, so the C++/pybind/FFI callers
  (`api/cpp/phasiccpp.h`, `src/cpp/parameterized/graph_builder_ffi.cpp`) are
  untouched.
- `tests/pytest/test_sojourn_subset_adjoint.py` — new equivalence/finiteness/scale
  gate.

Rebuild with `pixi run install-dev` (the package is a real copy in
`site-packages`, not an editable install — native edits are not live until
reinstalled).

---

## 8. Caveats & follow-ups

- **Graph construction is a separate bottleneck.** Building the augmented
  joint-prob graph at `nr_samples=8, reward_limit=5` (684 k vertices) is slow in
  the Python builder (`joint_prob_graph`, `__init__.py:10677`) because it grows
  child-state arrays with `np.append` in a loop — `O(n²)` reallocation. This fix
  does not touch that path; the reported *sojourn allocation* crash is resolved,
  but a model that large will still spend significant time in construction.
- **The no-arg full `ptd_expected_sojourn_time` is still `O(n²)`.** It is only
  called on small graphs today and was left as the independent validation
  reference. If a large graph ever needs the *entire* residence vector, the same
  adjoint pass supplies it directly (it already computes all `n` components) and
  could replace the `n × n` forward there too.
- **Escape hatch retained deliberately.** If a future graph class ever exercises
  an infinite-multiplier pattern the adjoint guards do not cover, `PHASIC_SOJOURN_FORWARD=1`
  reproduces the exact legacy arithmetic for comparison.
