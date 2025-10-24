# Orphaned Code Analysis Report

**Date:** 2025-10-22
**Project:** phasic v0.21.3

## Summary

Using Python AST analysis, identified code defined in `src/phasic/` but never referenced anywhere in the codebase:

- **51 orphaned functions**
- **39 orphaned methods** (across 13 classes)

These are candidates for removal to reduce maintenance burden and improve code clarity.

---

## Orphaned Functions (51)

### Visualization/Plotting (7)
- `animate_parameter_pairs` - svgd_plots.py
- `animate_svgd_2d` - svgd_plots.py
- `check_convergence` - svgd_plots.py
- `map_estimate_with_optimization` - svgd_plots.py
- `plot_parameter_matrix` - svgd_plots.py
- `plot_svgd_posterior_2d` - svgd_plots.py
- `visualize_hdr_2d` - svgd_plots.py

**Recommendation:** These seem like legacy visualization code. If not used in documentation/examples, consider removing or moving to examples/.

### Cache Management (7)
- `clear_trace_cache` - __init__.py, trace_cache.py
- `cleanup_old_traces` - trace_cache.py
- `get_trace_cache_stats` - trace_cache.py
- `list_cached_traces` - trace_cache.py
- `remove_cached_trace` - trace_cache.py
- `clear_cache` - model_export.py
- `print_cache_info` - model_export.py

**Recommendation:** If cache management is not part of the public API, these can be removed. If needed, expose via a CacheManager class method instead.

### Trace Serialization (5)
- `load_trace_json` - trace_elimination.py
- `load_trace_pickle` - trace_elimination.py
- `save_trace_pickle` - trace_elimination.py
- `trace_from_graph` - trace_elimination.py
- `trace_to_jax_fn` - trace_elimination.py

**Recommendation:** Legacy serialization? If pickle/JSON formats are deprecated in favor of the C++ serialization, remove.

### FFI/JAX Integration (3)
- `compute_pmf_fallback` - ffi_wrappers.py
- `compute_pmf_and_moments_fallback` - ffi_wrappers.py
- `labelled` - __init__.py

**Recommendation:** Check if these are truly unused or called via `pure_callback` (AST can't detect dynamic calls).

### Configuration (6)
- `configure` - config.py
- `reset_config` - config.py
- `load_config` - cluster_configs.py
- `suggest_config` - cluster_configs.py
- `validate_config` - cluster_configs.py
- `set_default_config` - jax_config.py

**Recommendation:** If configuration is only done via class methods, these standalone functions can be removed.

### SVGD Utilities (9)
- `calculate_param_dim` - svgd.py
- `decayed_kl_target` - svgd.py
- `example_ptd_spec` - svgd.py
- `logp_z` - svgd.py
- `median_heuristic` - svgd.py
- `simulate_example_data` - svgd.py
- `step_size_schedule` - svgd.py
- `string_to_class` - svgd.py
- `svgd_update_z` - svgd.py
- `update_local_bw_kl_step` - svgd.py
- `update_median_bw_kl_step` - svgd.py

**Recommendation:** Legacy SVGD helpers? If SVGD class doesn't use them, remove or move to examples.

### Parallel/Distributed (4)
- `auto_parallel_batch` - parallel_utils.py
- `execute_batch` - parallel_utils.py
- `init_parallel` - __init__.py
- `initialize_distributed` - distributed_utils.py

**Recommendation:** Check if parallel features are still experimental. If not ready for release, remove or mark as experimental.

### Miscellaneous (10)
- `configure_layered_cache` - cache_manager.py
- `create_svgd_model_from_trace` - trace_elimination.py
- `export_model_package` - model_export.py
- `install_model_library` - cloud_cache.py
- `install_trace_library` - trace_repository.py
- `monitor_cpu` - cpu_monitor.py ⚠️
- `print_jax_cache_info` - cache_manager.py
- `set_theme` - plot.py

**⚠️ Warning:** `monitor_cpu` is a decorator that IS used (via `@monitor_cpu`). AST analysis misses decorator usage - **do not remove**.

---

## Orphaned Methods (39)

### Cloud Storage Backends (5 methods, likely entire classes orphaned)

**AzureBlobBackend:**
- `delete_file`

**GCSBackend:**
- `delete_file`

**S3Backend:**
- `delete_file`

**CloudBackend (base class):**
- `delete_file`
- `download_cache`
- `upload_cache`

**Recommendation:** If cloud storage is not implemented/tested, remove entire cloud_cache.py module or mark as experimental.

### Cache Management (5)

**CacheManager:**
- `export_cache`
- `import_cache`
- `prewarm_model`
- `sync_from_remote`
- `vacuum`

**Recommendation:** Feature creep? Remove if not planned for near-term release.

### Configuration Classes (5)

**ClusterConfig:**
- `to_yaml`
- `total_cpus`

**CompilationConfig:**
- `fast_compile`
- `max_performance`

**PTDAlgorithmsConfig:**
- `cpp_only`
- `jax_only`
- `permissive`

**Recommendation:** Factory methods not used? Could be for future programmatic config.

### Graph Methods (9)

**Graph:**
- `discretize`
- `dph_pmf_batch`
- `eliminate_to_dag`
- `moments_batch`
- `moments_from_graph`
- `pdf_batch`
- `pmf_and_moments_from_graph`
- `pmf_from_graph_parameterized`

**Recommendation:** Legacy batch/parameterized methods? If trace-based workflow is preferred, remove.

### SVGD Visualization (8)

**SVGD:**
- `analyze_trace`
- `animate`
- `animate_pairwise`
- `fit_regularized`
- `plot_convergence`
- `plot_pairwise`
- `plot_posterior`
- `plot_trace`

**Recommendation:** If plotting moved to separate module (svgd_plots.py), remove from SVGD class.

### Trace Infrastructure (4)

**SymbolicDAG:**
- `instantiate`

**TraceBuilder:**
- `add_param`

**TraceRegistry:**
- `list_traces`
- `publish_trace`

**IPFSBackend:**
- `get_directory`

**Recommendation:** IPFS/trace repository features incomplete? Remove if not shipping.

---

## Recommendations by Priority

### High Priority - Safe to Remove

1. **Cloud storage backends** (cloud_cache.py) - appears incomplete
2. **IPFS trace repository** (trace_repository.py) - appears experimental
3. **Legacy visualization functions** (svgd_plots.py unused functions)
4. **Pickle/JSON trace serialization** - if C++ serialization is standard
5. **Orphaned cache management** functions

### Medium Priority - Investigate Before Removing

1. **Configuration helper functions** - may be called dynamically
2. **Parallel/distributed utilities** - check if experimental features are used
3. **Graph batch methods** - may be used in examples/docs
4. **SVGD regularized fitting** - future feature?

### Low Priority - Keep for Now

1. **FFI fallback functions** - may be called via `pure_callback`
2. **Decorator functions** - AST can't detect `@decorator` usage
3. **Factory methods** (fast_compile, permissive, etc.) - API convenience

---

## Next Steps

1. **Grep for usage in examples/docs:**
   ```bash
   grep -r "monitor_cpu\|animate_svgd_2d\|install_trace_library" docs/ examples/
   ```

2. **Check git history:**
   ```bash
   git log --all --full-history -- '**/cloud_cache.py'
   ```

3. **Mark experimental features:**
   Add docstring warnings for incomplete features instead of deleting.

4. **Create deprecation plan:**
   If removing from public API, deprecate first with warnings.

---

## False Positives (Do NOT Remove)

These appear orphaned but are actually used:

- **Decorators:** `monitor_cpu` - used as `@monitor_cpu`
- **Callbacks:** Functions passed to `pure_callback`, `partial`, etc.
- **Dynamic dispatch:** Methods called via `getattr()` or string lookup

**⚠️ Always validate with grep before deleting!**
