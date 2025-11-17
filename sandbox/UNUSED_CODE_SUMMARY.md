# Unused Code Analysis - Quick Summary

**Date**: 2025-11-07
**Full Report**: See `UNUSED_CODE_AUDIT.md` for complete details

---

## Key Findings

### ✅ SAFE TO DELETE NOW (~2,000 lines)

1. **`src/c/phasic_symbolic.c`** (1,015 lines)
   - Obsolete symbolic elimination system
   - Explicitly disabled in CMakeLists.txt (line 16-17, 128)
   - Replaced by trace-based elimination (Phases 1-4)
   - **Action**: Delete entire file

2. **`src/cpp/phasic_pybind.cpp.backup`** (161 KB)
   - Backup file created during development
   - Should not be in version control
   - **Action**: Delete and add `*.backup` to .gitignore

3. **`src/phasic/cloud_cache.py`** (561/685 lines = 82% commented)
   - Nearly entire file commented out
   - Planned cloud storage integration never implemented
   - **Action**: Delete entire file (restore from git if needed later)

4. **Commented code in `src/phasic/svgd.py`** (~200 lines)
   - Old bandwidth scheduling classes (lines 460-527)
   - Experimental utility functions (lines 578-807)
   - Advanced SVGD experiments (lines 847-945)
   - **Action**: Delete commented sections

5. **Commented test executables in `CMakeLists.txt`** (lines 205-215)
   - Phase 5 development tests
   - Superseded by Python tests
   - **Action**: Delete commented cmake code

**Total immediate cleanup**: ~2,000 lines

---

## ⚠️ REVIEW BEFORE DELETING (~400 lines)

1. **`src/phasic/cpu_monitor.py`** (full file)
   - CPU monitoring utility for SLURM
   - Not imported anywhere in main codebase
   - **Consider**: Move to `tools/` or `scripts/` if useful for development

2. **`src/phasic/decoders.py`** (full file)
   - Neural network decoders for variable-dimension PTD
   - Only reference is commented out in svgd.py (line 38-39)
   - **Consider**: Delete if research complete, else keep

3. **`src/phasic/auto_parallel.py`** (full file)
   - Automatic parallelization feature
   - Only imported in commented code
   - **Consider**: Delete if feature abandoned

4. **Commented functions in active files**:
   - `trace_elimination.py` - serialization functions (lines 861-954, ~150 lines)
   - `parallel_utils.py` - auto_parallel_batch decorator (lines 284-394, ~120 lines)
   - `distributed_utils.py` - initialize_distributed (lines 280-319, ~50 lines)
   - `__init__.py` - old Graph.__init__ (lines 1324-1369, ~40 lines)
   - `plot.py` - random_color, plot_graph wrapper (~10 lines)

**Total**: ~400 lines

---

## 🤔 DECISION NEEDED: R Bindings (~2,900 lines)

The codebase has extensive R bindings that are no longer actively maintained:

1. **`src/RcppExports.cpp`** (754 lines)
2. **`src/phasic.cpp`** (2,164 lines)
3. **`src/phasic_types.h`** (small header)
4. **`src/phasic.h`** (small header)

**Question**: Do you have any users still using the R interface?
- **If NO**: Delete all R bindings (~2,900 lines)
- **If YES**: Keep for backward compatibility

**Evidence suggesting no R users**:
- CMakeLists.txt line 172-173 has commented R bindings section
- Current documentation (CLAUDE.md) focuses on Python/JAX only
- No R examples in docs/pages/ (all are Python now)

---

## ℹ️ KEEP (Backward Compatibility)

These should NOT be deleted yet:

1. **Deprecated API functions** (keep until v1.0):
   - `update_parameterized_weights()` → use `update_weights()`
   - `add_edge_parameterized()` → use `add_edge(to, [coefficients])`
   - `precompile` parameter → use `jit`

2. **Active modules** (verified in use):
   - `cache_manager.py` - Used by model_export.py ✓
   - `exceptions.py` - Used in test_trace_repository.py ✓
   - `model_export.py` - Used in __init__.py ✓
   - `profiling.py` - Performance analysis utility ✓
   - `trace_repository.py` - Used in tests ✓

3. **Important comments** (documentation value):
   - `ffi_wrappers.py` comments - Document memory corruption fix
   - `svgd.py` deprecated fit_regularized() - Migration documentation

---

## Quick Cleanup Commands

### Safe Immediate Cleanup (HIGH PRIORITY)

```bash
#!/bin/bash
# Remove obsolete code - safe to run immediately

# Delete obsolete files
git rm src/c/phasic_symbolic.c
git rm src/cpp/phasic_pybind.cpp.backup
git rm src/phasic/cloud_cache.py

# Update .gitignore
cat >> .gitignore << 'EOF'
# Backup files
*.backup
*.bak
*~
EOF

# Commit
git add .gitignore
git commit -m "Remove obsolete code: symbolic elimination and unused modules

- Remove phasic_symbolic.c (1,015 lines) - obsolete, replaced by trace system
- Remove phasic_pybind.cpp.backup (161KB) - backup file
- Remove cloud_cache.py (561/685 lines commented) - no functionality
- Add backup file patterns to .gitignore

Total cleanup: ~1,600 lines of dead code
"
```

### Manual Cleanup (MEDIUM PRIORITY)

Edit these files to remove commented code:

1. **src/phasic/svgd.py**: Remove lines 460-527, 578-807, 847-945
2. **CMakeLists.txt**: Remove lines 205-215 (commented test executables)
3. **src/phasic/__init__.py**: Remove lines 1324-1369 (old Graph.__init__)
4. **src/phasic/trace_elimination.py**: Remove lines 861-954 (old serialization)
5. **src/phasic/parallel_utils.py**: Remove lines 284-394 (auto_parallel)
6. **src/phasic/plot.py**: Remove lines 18, 188 (commented functions)

---

## Statistics Summary

| Priority | Files | Lines | Status |
|----------|-------|-------|--------|
| HIGH - Safe to delete now | 3-5 files | ~2,000 | Immediate action |
| MEDIUM - Review first | 3 files + sections | ~400 | Decision needed |
| R bindings | 4 files | ~2,900 | User check needed |
| Deprecated APIs | ~4 functions | ~50 | Keep until v1.0 |
| **TOTAL REMOVABLE** | **~15 items** | **~5,300+** | **Various priorities** |

---

## Recommendations

### Immediate Actions

1. ✅ Run the safe cleanup script above
2. ✅ Decide on R bindings (survey users?)
3. ✅ Manual cleanup of commented code in active files

### For v1.0 Release

1. Remove all deprecated APIs (with migration guide)
2. Remove R bindings (if confirmed no users)
3. Remove remaining commented "research code"
4. Final review of unused utilities

**Potential total cleanup**: 5,000-6,500 lines

---

## Full Details

See `UNUSED_CODE_AUDIT.md` for:
- Complete file-by-file analysis
- Line-by-line breakdown
- Detailed justifications
- Evidence for each finding
- Safety assessments

---

**Generated**: 2025-11-07
