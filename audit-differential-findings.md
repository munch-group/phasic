# Differential audit — findings log

Running log of the secure differential process (see `audit-situation-map.md` for the atlas,
baseline, and gap list). Each finding is **verified by execution against an independent
reference** (closed form / cross-backend / the `3082ebc6` baseline), scoped via the atlas, and
dated (pre-existing vs refactoring-introduced) before any fix decision.

Baseline built and verified: `3082ebc6` (parent of first refactor commit `9b8bc3f6`) builds
under the current pixi env and computes `moments(2) = [0.325, 0.16125]` correctly.

---

## F-001 — discrete-PMF `normalize()` divergence in `pmf_from_graph_parameterized`

**Verdict: REAL bug, but LATENT and PRE-EXISTING. Not refactoring-introduced. Do not fix in
isolation — record to fix together with the upstream break that masks it.**

### What the atlas flagged
Three discrete-PMF compute paths, one of which normalizes:
- `pmf_from_graph(discrete=True)` → FFI `ComputePmfFfiImpl`: `dph_pmf`, no normalize.
- `pmf_from_cpp(discrete=True)` → ctypes-JIT: `dph_pmf`, no normalize (the fix, `__init__.py`
  "NO normalize() here").
- `pmf_from_graph_parameterized(discrete=True)` → ctypes-JIT: **`g.normalize(); g.dph_pmf(...)`**
  (`__init__.py`, wrapper string, comment "Normalize for discrete mode (required for DPH)").

In a DPH the edge weights ARE per-step transition probabilities and the deficit (1 − row sum)
is the implicit stay-in-place probability, so the continuous `normalize()` (which rescales each
vertex's out-edges to sum to 1) **collapses the chain to a deterministic walk**. This is the
exact class of bug already fixed in `pmf_from_cpp`.

### Verified by execution (closed-form oracle)
Forced 2-phase DPH `s→v3-(p)->v2-(p)->v1`, each transient vertex one out-edge of weight
`p = 0.3` (row sum 0.3 < 1). Correct DPH: jump count ~ NegBinomial(2, p),
`P(T=n) = (n-1)p²(1-p)^(n-2)`. Normalize would rescale the single out-edge to 1.0 → `P(T=2)=1`.

```
closed form NegBinomial(2,0.3) = [0.09, 0.126, 0.12348, 0.07412]
pmf_from_graph (FFI)                -> [0.09, 0.126, 0.12348, 0.07412]   CORRECT
pmf_from_cpp (ctypes JIT)           -> [0.09, 0.126, 0.12348, 0.07412]   CORRECT
pmf_from_graph_parameterized (JIT)  -> RAISED AttributeError
```

### Why it's LATENT (verified, not assumed)
`pmf_from_graph_parameterized` raises **before** the ctypes wrapper (with the `normalize()`)
ever runs:
```
result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
AttributeError: 'NoneType' object has no attribute 'ShapeDtypeStruct'
```
Module-level `jax` is `None` (lazily imported) and the code requests `jnp.float32` where the FFI
requires F64. This is the original hand-off's "bug 5". So the buggy `normalize()` is **dead code
today** — no user can reach it — and the divergence is not observable by any current caller.

### Dated: PRE-EXISTING (not refactoring-introduced)
The `normalize(); dph_pmf` block is **byte-identical** on the baseline `3082ebc6` (which predates
the entire refactoring) and the current tree. The refactoring neither introduced nor touched it.

### Scope & fix
- **Scope:** one path (`pmf_from_graph_parameterized`), one wrapper string. The other two
  discrete-PMF paths are correct. No cross-path divergence is *observable* while bug 5 stands.
- **Fix (deferred):** remove the `g.normalize()` call, mirroring the `pmf_from_cpp` fix — **but
  only once bug 5 (the `jax`-is-None / `float32` break) is fixed**, since it is otherwise dead.
  Fixing dead code in isolation adds risk (a rebuild, a diff) with no observable benefit. Bundle
  it with any `pmf_from_graph_parameterized` revival.
- **Guard needed at that time:** a discrete cross-path gate (`pmf_from_graph_parameterized` ==
  `pmf_from_cpp` == FFI, against NegBinomial) on a row-sum≠1 graph. No existing gate covers this
  (gap §4.2 / §4.5: ctypes-JIT and discrete cross-path are ungated).

### Process note
Reading the code alone would have yielded a false "cross-path divergence bug (HIGH)". Executing
the differential downgraded it correctly to **latent + pre-existing** — the calibrated verdict.
This is the method working: reproduce → find it doesn't run → find *why* → scope → date against
baseline → defer the fix.
