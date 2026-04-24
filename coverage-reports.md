# Coverage — src/phasic

_Generated 20260424-173029 by /coverage. Threshold: 80%._

Overall coverage: 46% (threshold: 80%). ~800 uncovered regions across 33 files (34 including fully-covered `exceptions.py`). Capping at ~60 findings below; remainder summarized in the per-file table.

Tests were run per-file to survive several C-level segfaults/aborts that killed a single `pytest --cov=src/phasic` run. Six test files were excluded from measurement — see Cross-cutting themes.

---

[🔴] [8534da5e] `svgd.py::animate` — src/phasic/svgd.py:7623-7736
Signature: `def animate(self, param_idx, true_theta, param_name, figsize, skip, thin, interval, duration, bins, show_particles, max_particles, save_as_gif, save_as_mp4, unconstrained)`
Kind: function
Why uncovered: test exists, branch unexercised (no SVGD animation tests in suite)
Suggested test: instantiate an SVGD run, call `animate` with a small trajectory and `save_as_gif=False`, assert a matplotlib `FuncAnimation` is returned (or call `to_html5_video()`).

[🔴] [9c14b5f2] `svgd.py::plot_ci` — src/phasic/svgd.py:6044-6131
Signature: `def plot_ci(self, figsize, save_path, skip, unconstrained, true_theta, ci, alpha, target, median, return_fig, ci_method)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: after a short SVGD fit, call `plot_ci(return_fig=True)` and assert returned figure has the expected number of axes matching `theta_dim`.

[🔴] [9309af75] `svgd.py::plot_hdr` — src/phasic/svgd.py:6498-6570, 6619-6692
Signature: `def plot_hdr(self, alphas, idx, figsize, hexgrid, trim, n, margin, xlim, ylim, palette, pad, unconstrained, return_fig, show_hpd, hpd_alpha)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: 2D parameter SVGD fit, call `plot_hdr(idx=(0,1), return_fig=True)`; assert HPD contours are drawn.

[🔴] [7ac99e3a] `svgd.py::animate_parameter_pairs` — src/phasic/svgd.py:6787-6864
Signature: `def animate_parameter_pairs(self, param_pairs, true_params, figsize, save_as_gif)`
Kind: function
Why uncovered: no test exists
Suggested test: small SVGD result, call with `param_pairs=[(0,1)]`, `save_as_gif=False`; assert animation object type.

[🔴] [03ef134d] `svgd.py::plot_pairwise` — src/phasic/svgd.py:7429-7503
Signature: `def plot_pairwise(self, true_theta, param_names, figsize, save_path, unconstrained, return_fig)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: call with a 3-parameter result and `return_fig=True`; assert the corner-plot grid shape is 3×3.

[🔴] [2c03e171] `svgd.py::plot_convergence` — src/phasic/svgd.py:5914-5967
Signature: `def plot_convergence(self, figsize, save_path, skip, unconstrained, return_fig)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: after `optimize()`, call with `return_fig=True`; assert at least one line per parameter present.

[🔴] [c40f91c7] `svgd.hex_grid` — src/phasic/svgd.py:6409-6475
Signature: `def hex_grid(x_min, x_max, y_min, y_max, n, aspect, flat_topped, pad)`
Kind: function
Why uncovered: internal helper used by `plot_hdr`
Suggested test: call with simple bounds, assert produced arrays have the expected shape and spacing.

[🔴] [3bef3265] `__init__.py::_compute_moments_pure` — src/phasic/__init__.py:5589-5629
Signature: `def _compute_moments_pure(theta_flat)`
Kind: function
Why uncovered: test exists, branch unexercised (likely only used in certain JAX gradient paths)
Suggested test: build a parameterized graph, call `_compute_moments_pure` through its public entrypoint on a known-exponential graph, assert first moment = 1/rate.

[🔴] [9997bd1a] `__init__.py::_wrap_trace_log_likelihood_for_jax` — src/phasic/__init__.py:1272-1303
Signature: `def _wrap_trace_log_likelihood_for_jax(lib_path, param_length)`
Kind: function
Why uncovered: no test exists (requires loaded C library trace path)
Suggested test: record a trace, call wrapper, evaluate gradient with `jax.grad` at known theta.

[🔴] [2583049e] `Graph.pmf_from_cpp` — src/phasic/__init__.py:4628-4658
Signature: `def pmf_from_cpp(cls, cpp_file, discrete)`
Kind: function
Why uncovered: no test exists
Suggested test: compile a small C++ model file, call `pmf_from_cpp`, verify PMF output against analytic exponential.

[🔴] [ccb865af] `Graph.pmf_from_graph` — src/phasic/__init__.py:4168-4191
Signature: `def pmf_from_graph(cls, graph, discrete, use_cache, theta_dim)`
Kind: function
Why uncovered: test exists, branch unexercised (the 24-statement block is likely an error/caching path)
Suggested test: call with `use_cache=True` on a parameterized graph, then again to exercise the hit branch; assert identical results and that a cache file exists.

[🔴] [0136b722] `bffg.py::likelihood_correction` — src/phasic/bffg.py:439-532
Signature: `def likelihood_correction(theta_mcmc)`
Kind: function
Why uncovered: test exists, branch unexercised (`test_bffg.py` covers path construction but not this correction term)
Suggested test: with a proposal graph, compute `path_to_rewards` + `path_exit_rates`, call `likelihood_correction(theta)` for two thetas, assert log-ratio difference matches expected form.

[🔴] [cff41e57] `bffg.py::bffg_log_prob` — src/phasic/bffg.py:325-396
Signature: `def bffg_log_prob(jg_disc, jg_continuous, theta_proposal, theta_target_fn, observed_data, n_paths, zero_mut_idx, theta_proposal_fn, return_model)`
Kind: function
Why uncovered: test exists, branch unexercised — this is the main BFFG entry point
Suggested test: build a 2-component coalescent model, set `n_paths=10`, call with known data, assert return is a finite scalar and differentiable under `jax.grad`.

[🔴] [4f070d34] `bffg.likelihood_correction_jit` — src/phasic/bffg.py:561-612
Signature: `def likelihood_correction_jit(theta_mcmc)`
Kind: function
Why uncovered: no test exists for the JIT-wrapped variant
Suggested test: wrap the same scenario as above with `jax.jit`, assert output equals the non-jit variant within 1e-10.

[🔴] [b6bd6a95] `state_indexing.StateIndexer.index_to_props` — src/phasic/state_indexing.py:1292-1346
Signature: `def index_to_props(self, index, as_dict, as_values, flatten)`
Kind: function (public API, `StateIndexer` is top-level exported)
Why uncovered: test exists, branch unexercised — `as_dict`/`as_values`/`flatten` variants
Suggested test: build a 2-property indexer, call `index_to_props(0, as_dict=True)` and `(0, as_values=True)`; assert correct property names/values returned.

[🔴] [aeac0ee2] `state_indexing.StateIndexer.props_to_index` — src/phasic/state_indexing.py:532-581
Signature: `def props_to_index(self, props)`
Kind: function (public API)
Why uncovered: test exists, branch unexercised (likely 2D batch path)
Suggested test: build indexer, pass a list of dicts, assert returned indices match individual `props_to_index` calls.

[🔴] [5e432bf9] `state_indexing._detect_property_set` — src/phasic/state_indexing.py:1027-1065
Signature: `def _detect_property_set(self, props, raise_on_ambiguous)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: construct an ambiguous `props` matching two property sets; assert it raises when `raise_on_ambiguous=True` and picks first otherwise.

[🔴] [6c4b7ac0] `state_indexing.StateVector.__getattr__` — src/phasic/state_indexing.py:1975-2005
Signature: `def __getattr__(self, name)`
Kind: function
Why uncovered: test exists, branch unexercised (fallback path)
Suggested test: request a non-existent attribute, assert `AttributeError`; request a valid property name, assert it returns slice of vector.

[🔴] [911612b0] `state_indexing.PropertyDict.__eq__` — src/phasic/state_indexing.py:1557-1584
Signature: `def __eq__(self, other)`
Kind: function
Why uncovered: no equality tests
Suggested test: build two `PropertyDict` with same/different contents, assert `==`/`!=` behaves correctly including with a non-`PropertyDict`.

[🔴] [07ad93a2] `mcmc.MCMC._run_chains_vmap` — src/phasic/mcmc.py:611-652, 657-673, 679-727
Signature: `def _run_chains_vmap(self, log_prob_fn)`
Kind: function
Why uncovered: `test_mcmc.py`/`test_mcmc_accuracy.py` were excluded (hung); this is the vmap multi-chain inner loop
Suggested test: small toy exponential model, 2 chains, 20 samples, `parallel=False`; assert traces shape is `(2, 20, theta_dim)`.

[🔴] [75188093] `MCMC.__init__` — src/phasic/mcmc.py:318-352
Signature: `def __init__(self, model, observed_data, prior, n_samples, n_chains, burn_in, thin, proposal_scale, theta_init, theta_dim, seed, verbose, progress, jit, positive_params, param_transform, rewards, fixed, log_prob_fn, adaptive, adapt_after, target_acceptance, parallel, likelihood_correction)`
Kind: function (public API top-level class)
Why uncovered: tests skipped (hung); covers parameter-transform wiring and validation
Suggested test: construct with `positive_params=[0]` + `param_transform='exp'`, assert `self._transform_fn(0.0) == 1.0`; invalid combinations raise.

[🔴] [d112a2f1] `MCMC.run` — src/phasic/mcmc.py:768-784
Signature: `def run(self)`
Kind: function (public API)
Why uncovered: test skipped (hung)
Suggested test: short chain (5 samples), assert return `dict` has `'samples'`, `'log_prob'`, `'accept_rate'` keys with correct shapes.

[🔴] [cdeb689b] `MCMC.summary` — src/phasic/mcmc.py:964-978, 984-1013
Signature: `def summary(self)`
Kind: function (public API)
Why uncovered: test skipped (hung)
Suggested test: after a short run, assert summary dataframe has columns `mean`, `sd`, `r_hat`, `ess` and expected index length.

[🔴] [4c1e5828] `method_of_moments.method_of_moments` — src/phasic/method_of_moments.py:401-410, 500-501, 543-547
Signature: `def method_of_moments(graph, observed_data, nr_moments, theta_dim, theta_init, rewards, fixed, std_multiplier, discrete, verbose, weighted)`
Kind: function (public API top-level)
Why uncovered: test exists (`test_method_of_moments.py`) but doesn't exercise the `weighted=True` path, error branches, and summary formatter
Suggested test: call with `weighted=True` and 2+ moments on exponential data; assert estimated theta within 3 std of truth.

[🔴] [aba780d5] `hex_grid.HexGrid.build_graph` — src/phasic/hex_grid.py:492-533, 477-484
Signature: `def build_graph(self, transition_fn, property_set, start_cell, parameterized)`
Kind: function (public API)
Why uncovered: no test exists for the convenience builder path
Suggested test: make a 3×3 HexGrid, define a trivial transition_fn returning single neighbor, call `build_graph` and assert graph.vertices_length > 0.

[🔴] [29de0b32] `hex_grid.HexGrid._compute_valid_mask` — src/phasic/hex_grid.py:604-637
Signature: `def _compute_valid_mask(self)`
Kind: function
Why uncovered: test exists, branch unexercised (boundary-clipping path)
Suggested test: construct from a simple rectangular boundary, assert mask shape matches grid and edge cells are excluded.

[🔴] [554c0260] `hex_grid.HexGrid.map_to_grid` — src/phasic/hex_grid.py:566-598
Signature: `def map_to_grid(self, graph, property_set, values, prop_filter)`
Kind: function (public API)
Why uncovered: test exists, branch unexercised
Suggested test: build grid + graph, pass `values` vector, assert returned 2D grid has non-zero entries only where property matched.

[🔴] [023f6475] `hex_grid.HexGrid.__init__` — src/phasic/hex_grid.py:147-173
Signature: `def __init__(self, boundary, hex_size, orientation)`
Kind: function (public API)
Why uncovered: test exists, branch unexercised (orientation='flat' vs 'pointy', boundary='shapefile' vs polygon)
Suggested test: construct from numpy polygon vs shapefile; assert `grid.properties()` returns same length.

[🔴] [adbc5a59] `trace_elimination.record_elimination_trace` — src/phasic/trace_elimination.py:702-717, 793-812
Signature: `def record_elimination_trace(graph, theta_dim, reward_length, enable_rewards)`
Kind: function (public API, exported)
Why uncovered: test exists, branch unexercised (the `enable_rewards=True` and reward_length>0 branches)
Suggested test: build a small parameterized graph, call with `enable_rewards=True`, `reward_length=2`, assert trace contains reward info.

[🔴] [978dd7e7] `trace_elimination.trace_to_log_likelihood` — src/phasic/trace_elimination.py:1676-1688, 1698-1713
Signature: `def trace_to_log_likelihood(trace, observed_data, reward_vector, granularity, use_cpp, use_log)`
Kind: function (public API, exported — central for SVGD)
Why uncovered: test exists, branch unexercised (`reward_vector is not None`, `use_cpp=False`)
Suggested test: call with `reward_vector=np.ones(n_vertices)`, `use_cpp=False`, and compare to `use_cpp=True` within 1e-6.

[🔴] [a0e24aa7] `trace_elimination.trace_to_c_arrays` — src/phasic/trace_elimination.py:1203-1232
Signature: `def trace_to_c_arrays(trace)`
Kind: function
Why uncovered: no test exists — it's the conversion shim to the C forward algorithm
Suggested test: record a trace, call and assert returned dtype is float64 and shapes match `trace.vertex_count`.

[🔴] [8f81fe6f] `trace_repository.IPNSClient.get_trace_by_hash` — src/phasic/trace_repository.py:1339-1371
Signature: `def get_trace_by_hash(self, graph_hash, force_download)`
Kind: function (public API)
Why uncovered: fixture gap (requires IPFS daemon or mock)
Suggested test: mock `self._request_with_retry` to return a trace blob; assert deserialized trace matches input.

[🔴] [e112c769] `trace_serialization._c_trace_to_python` — src/phasic/trace_serialization.py:97-143
Signature: `def _c_trace_to_python(trace_ptr)`
Kind: function
Why uncovered: test exists, branch unexercised (pointer-walking path)
Suggested test: record a trace via C backend, round-trip through `_c_trace_to_python` then back, assert equality.

[🔴] [4f864336] `trace_serialization.clear_cache` / `get_cache_info` — src/phasic/trace_serialization.py:335-355, 382-402
Signature: `def clear_cache()`, `def get_cache_info()`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: save a trace, call `get_cache_info()`, assert size > 0; then `clear_cache()` and assert it returns to 0.

[🔴] [0ceb6877] `hierarchical_trace_cache._build_scc_subgraph` — src/phasic/hierarchical_trace_cache.py:1149-1203
Signature: `def _build_scc_subgraph(original_graph, scc, scc_graph)`
Kind: function
Why uncovered: test exists (`test_hierarchical_cache.py`) but doesn't exercise this code path
Suggested test: construct a graph with a known 3-node SCC, call, assert returned subgraph has 3 vertices and equivalent transition rates.

[🔴] [67b5ab81] `hierarchical_trace_cache.get_scc_graphs` — src/phasic/hierarchical_trace_cache.py:118-154
Signature: `def get_scc_graphs(graph, min_size)`
Kind: function (public API entrypoint to SCC pipeline)
Why uncovered: test exists, branch unexercised (the `min_size > 1` filter)
Suggested test: graph with trivial and nontrivial SCCs, call with `min_size=2`, assert only non-trivial returned.

[🔴] [53667867] `cache_manager.CacheManager.sync_from_remote` — src/phasic/cache_manager.py:365-407
Signature: `def sync_from_remote(self, remote_cache_dir, dry_run)`
Kind: function (public API)
Why uncovered: fixture gap (needs a second cache directory)
Suggested test: create two tmpdirs, populate one, call `sync_from_remote(dry_run=False)`, assert files are copied.

[🔴] [190dcbc1] `cache_manager.CacheManager.prewarm_model` — src/phasic/cache_manager.py:189-227
Signature: `def prewarm_model(self, model_fn, theta_samples, time_grids, show_progress)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: small exponential model_fn, 5 theta samples, assert `cache_info()` size grows.

[🟡] [4d643d78] `__init__.py module top-level` — src/phasic/__init__.py:265-290
Kind: module-top-level
Why uncovered: test exists, branch unexercised (conditional imports / env-driven setup)
Suggested test: cover via `PHASIC_JAX=0` / `PHASIC_FORCE_MPFR=1` env-controlled branches.

[🟡] [57f0308b] `Graph._apply_weight_callback` — src/phasic/__init__.py:653-678
Signature: `def _apply_weight_callback(serialized, theta, callback)`
Kind: function
Why uncovered: `test_callback_weights.py` failures prevent exercising this path
Suggested test: create graph with `weight_callback`, call `_apply_weight_callback` directly with a known callback and theta, assert weights == callback output.

[🟡] [f654ba86] `Graph._variance_from_trace` — src/phasic/__init__.py:2205-2233
Signature: `def _variance_from_trace(self, rewards, discrete)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: for an Erlang(2) graph with unit rewards, assert variance equals 2/λ² analytically.

[🟡] [79091ec1] `Graph.joint_prob_graph` — src/phasic/__init__.py:7368-7395
Signature: `def joint_prob_graph(self, base_graph_indexer, reward_only, reward_rates_callback, mutation_rate, reward_limit, tot_reward_limit)`
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: small coalescent graph + mutation indexer, call, assert resulting graph has product state space of correct size.

[🟡] [78b91a04] `srun_magic.srun` — src/phasic/srun_magic.py:290-391, 422-508
Signature: `def srun(self, line, cell)` (IPython magic)
Kind: function (public API, user-visible magic)
Why uncovered: no test exists; requires IPython kernel
Suggested test: mock IPython shell, invoke magic with a simple cell, assert subprocess script contains the cell body.

[🟡] [672a0aa7] `srun_magic.serialize_globals` — src/phasic/srun_magic.py:127-168
Signature: `def serialize_globals(globals_dict)`
Kind: function
Why uncovered: no test exists
Suggested test: pass a dict with picklable/unpicklable values, assert unpicklable ones are skipped with a warning.

[🟡] [2e0cd7e7] `profiling.analyze_svgd_profile` — src/phasic/profiling.py:106-166, 189-232, 281-362, 369-412
Signature: `def analyze_svgd_profile(stats, top_n, print_report)`
Kind: function (public API)
Why uncovered: no test exists (profiling harness)
Suggested test: feed a minimal `cProfile.Stats` object with known functions; assert report lists them in expected order.

[🟡] [b556ca7c] `profiling.profile_svgd` — src/phasic/profiling.py:464-478
Signature: `def profile_svgd(model, observed_data)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: call on a trivial model; assert it returns a `Stats`-like object.

[🟡] [bd48d5f4] `plot.plot_posterior` — src/phasic/plot.py:83-149
Signature: `def plot_posterior(results, true_theta, param_names, bins, figsize, save_path, ci_method, ci_level, return_fig)`
Kind: function (public API)
Why uncovered: no test exists (module has zero coverage)
Suggested test: pass a dict with `samples` array of shape `(n, 2)`, call `return_fig=True`, assert axes count == 2.

[🟡] [71bb2199] `plot.plot_chains` — src/phasic/plot.py:178-223
Signature: `def plot_chains(results, true_theta, param_names, figsize, save_path, return_fig, sharey)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: pass a `(4, 100, 2)` samples array (4 chains), assert subplot count matches theta_dim.

[🟡] [5cc10645] `plot.plot_autocorrelation` — src/phasic/plot.py:251-304
Signature: `def plot_autocorrelation(results, max_lag, param_names, figsize, save_path, return_fig)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: pass well-mixed iid samples, assert rho[0]=1 and rho[1] ≈ 0.

[🟡] [51756ec8] `plot.plot_pairwise` — src/phasic/plot.py:332-380
Signature: `def plot_pairwise(results, true_theta, param_names, figsize, save_path, return_fig)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: 3-parameter samples; assert returned figure axes form a 3×3 grid.

[🟡] [f434eabd] `graph_cache._serialize_value` / `_deserialize_value` — src/phasic/graph_cache.py:322-345, 391-424
Signature: `def _serialize_value(value)`, `def _deserialize_value(value)`
Kind: function
Why uncovered: test exists, branch unexercised (handles numpy, bytes, tuples)
Suggested test: round-trip dicts containing np.array, bytes, tuple, None, int; assert equality after deserialize.

[🟡] [f7dee7e4] `graph_cache.GraphCache.load_graph` / `save_graph` — src/phasic/graph_cache.py:105-120, 130-139, 158-178, 184-195
Signature: `def load_graph(self, callback)`, `def save_graph(self, graph, callback)`
Kind: function (public API, exported as `GraphCache`)
Why uncovered: test exists, branch unexercised
Suggested test: save a graph, load via the same `callback`, assert edges equal; then save with a changed callback and assert cache miss.

[🟡] [96f5eec1] `trace_cache.verify_cache_working` — src/phasic/trace_cache.py:306-343
Signature: `def verify_cache_working()`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: call in a fresh tmpdir, assert it reports "cache healthy".

[🟡] [606abccd] `trace_cache.cleanup_old_traces` — src/phasic/trace_cache.py:214-226, 231-265
Signature: `def cleanup_old_traces(max_size_mb, max_age_days)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: write 3 old trace files, call with `max_age_days=0`; assert they are removed.

[🟡] [6aef49d1] `model_export.print_model_cache_info` — src/phasic/model_export.py:277-297
Signature: `def print_model_cache_info(cache_dir, max_files)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: populate cache_dir with 2 model files, call, capture stdout, assert model names present.

[🟡] [4ab15c11] `model_export.export_model_package` — src/phasic/model_export.py:510-524, 541-551, 594-607
Signature: `def export_model_package(output_dir, model_code, theta_dim, compilation_config, n_particles, data_shape, metadata)`
Kind: function (public API)
Why uncovered: no test exists
Suggested test: export to tmpdir, assert `model.py`, `metadata.json`, `README.md` exist with correct content.

[🟡] [083079f3] `distributed_utils.initialize_jax_distributed` — src/phasic/distributed_utils.py:276-296, 301-309
Signature: `def initialize_jax_distributed(coordinator_address, num_processes, process_id)`
Kind: function (public API)
Why uncovered: fixture gap (multi-process setup)
Suggested test: monkeypatch `jax.distributed.initialize`, call with fake args, assert it's forwarded correctly.

[🟡] [7d9fcd78] `distributed_utils.detect_slurm_environment` — src/phasic/distributed_utils.py:128-140, 146-155
Signature: `def detect_slurm_environment()`
Kind: function
Why uncovered: test exists, branch unexercised (SLURM env vars not set in CI)
Suggested test: `monkeypatch.setenv('SLURM_JOB_ID','123')`, call; assert returned dict contains `job_id=123`.

[🟡] [d581035a] `callback_hash.hash_callback` — src/phasic/callback_hash.py:83-116, 127-135, 141-151, 159-168
Signature: `def hash_callback(callback)`
Kind: function
Why uncovered: test exists but only hits the simple path; the closure/source/AST branches are unexercised
Suggested test: hash a closure that captures a variable, assert changing the captured value changes the hash; hash a builtin callable, assert it doesn't raise.

[🟡] [23e0b7b9] `config.PTDAlgorithmsConfig.validate` error branches — src/phasic/config.py:206-216, 242-246, 253-259, 266-272, 324-332
Kind: error-path
Why uncovered: test exists (`test_input_validation.py`) but doesn't hit every branch (OpenMP-without-FFI, MPFR-unavailable, platform-unavailable)
Suggested test: call `configure(openmp=True, ffi=False)`, assert `PTDConfigError` with the "requires ffi=True" message.

[🟡] [55f6a4c3] `config._get_available_platforms` — src/phasic/config.py:69-94
Signature: `def _get_available_platforms()`
Kind: function
Why uncovered: test exists, branch unexercised (no GPU/TPU on CI)
Suggested test: monkeypatch `jax.devices` to raise RuntimeError; assert returns `['cpu']`.

[🟡] [b020d1d1] `logging_config.set_log_level.__init__` — src/phasic/logging_config.py:322-353
Signature: `def __init__(self, level, module)`
Kind: function
Why uncovered: test exists (used across suite) but `level='NONE'` and `module=<specific>` branches unexercised
Suggested test: call `set_log_level('NONE', module='phasic.svgd')`, emit a log, assert no output.

[🟡] [8e944665] `ffi_wrappers.backward_probabilities_ffi` — src/phasic/ffi_wrappers.py:1079-1086
Signature: `def backward_probabilities_ffi(structure_json, theta, target_vertices, n_vertices)`
Kind: function
Why uncovered: no test exists
Suggested test: 2-state absorbing chain, compute backward probabilities via FFI, compare to Python reference.

[🟡] [5383aede] `ffi_wrappers.sample_path_conditioned_ffi` — src/phasic/ffi_wrappers.py:1029-1035
Signature: `def sample_path_conditioned_ffi(structure_json, theta, target_vertex, seed, max_length)`
Kind: function
Why uncovered: `test_sample_with_rewards.py` was excluded (segfault)
Suggested test: seed=0, sample 10 paths, assert each ends at target_vertex.

[🟡] [5ca0288a] `parallel_utils.apply_pmap` — src/phasic/parallel_utils.py:224-249
Signature: `def apply_pmap(func, args, n_devices)`
Kind: function
Why uncovered: no test exists
Suggested test: call with `func=lambda x: x**2`, `args=np.arange(8)`, `n_devices=2`, assert result equals `np.arange(8)**2`.

[🟡] [0f5523b1] `auto_parallel.detect_environment` — src/phasic/auto_parallel.py:159-168, 174-181
Signature: `def detect_environment()`
Kind: function
Why uncovered: test exists, branch unexercised (SLURM-detection branch)
Suggested test: monkeypatch env, assert returned `env_info.platform == 'slurm'`.

[🟡] [9c54c27e] `method_of_moments._select_nr_moments` / `_estimate_moment_covariance` — src/phasic/method_of_moments.py:115-119, 141-145
Kind: function
Why uncovered: test exists, branch unexercised
Suggested test: pass fixed data matrix, assert returned `nr_moments` respects `max_nr_moments`.

[🟢] [53609608] `utils.py` module-level `matplotlib`, `Counter` imports and helpers — src/phasic/utils.py:60-212
Kind: module-top-level
Why uncovered: possibly dead (see Dead-code candidates below).

[🟢] [b6c5ebd0] `probability_matching.probability_matching` — src/phasic/probability_matching.py:197-209, 254-256, 315-332
Kind: branch / error-path
Why uncovered: test exists (module at 89%); remaining lines are warning/log paths
Suggested test: pass degenerate observed_indices that triggers warning, assert warning emitted.

[🟢] [c634857a] `optax_wrapper.*` — src/phasic/optax_wrapper.py:92, 117, 213, 244, 268, 294, 318, 363, 398
Kind: branch
Why uncovered: trivial one-line properties / constructor param checks (module at 82%)
Suggested test: instantiate each `optax_*` wrapper and assert `.lr == learning_rate`.

---

### Per-file summary

| File | Stmts | Miss | Cover% |
|---|---:|---:|---:|
| src/phasic/svgd.py | 2697 | 1626 | 39.71% |
| src/phasic/__init__.py | 2279 | 835 | 63.36% |
| src/phasic/hierarchical_trace_cache.py | 841 | 471 | 44.00% |
| src/phasic/state_indexing.py | 532 | 314 | 40.98% |
| src/phasic/mcmc.py | 497 | 246 | 50.50% |
| src/phasic/srun_magic.py | 237 | 237 | 0.00% |
| src/phasic/bffg.py | 246 | 198 | 19.51% |
| src/phasic/cache_manager.py | 223 | 198 | 11.21% |
| src/phasic/plot.py | 175 | 175 | 0.00% |
| src/phasic/hex_grid.py | 228 | 173 | 24.12% |
| src/phasic/profiling.py | 177 | 172 | 2.82% |
| src/phasic/utils.py | 170 | 170 | 0.00% |
| src/phasic/trace_elimination.py | 558 | 166 | 70.25% |
| src/phasic/trace_cache.py | 160 | 146 | 8.75% |
| src/phasic/trace_repository.py | 560 | 123 | 78.04% |
| src/phasic/graph_cache.py | 142 | 117 | 17.61% |
| src/phasic/model_export.py | 124 | 106 | 14.52% |
| src/phasic/trace_serialization.py | 150 | 98 | 34.67% |
| src/phasic/config.py | 165 | 84 | 49.09% |
| src/phasic/parallel_utils.py | 76 | 76 | 0.00% |
| src/phasic/distributed_utils.py | 107 | 74 | 30.84% |
| src/phasic/cluster_configs.py | 100 | 58 | 42.00% |
| src/phasic/ffi_wrappers.py | 162 | 53 | 67.28% |
| src/phasic/callback_hash.py | 63 | 52 | 17.46% |
| src/phasic/logging_config.py | 107 | 49 | 54.21% |
| src/phasic/decoders.py | 43 | 43 | 0.00% |
| src/phasic/auto_parallel.py | 121 | 41 | 66.12% |
| src/phasic/method_of_moments.py | 239 | 37 | 84.52% |
| src/phasic/jax_config.py | 85 | 20 | 76.47% |
| src/phasic/probability_matching.py | 134 | 14 | 89.55% |
| src/phasic/optax_wrapper.py | 68 | 12 | 82.35% |
| src/phasic/cloud_cache.py | 1 | 1 | 0.00% |
| src/phasic/vscode_theme.py | 1 | 1 | 0.00% |
| src/phasic/exceptions.py | 12 | 0 | 100.00% |

Tail: the ~160 small low-severity regions across the `svgd.py` and `__init__.py` (mostly one-line error/logging branches and plot option paths) are elided. See "Cross-cutting themes" below.

### Dead-code candidates (do not test)

- `src/phasic/vscode_theme.py:1-1` — `<module>`: all three `from .vscode_theme import …` lines in `__init__.py:55-57` are commented out; no other references. **Possibly dead.**
- `src/phasic/cloud_cache.py:1-1` — `<module>`: the `from .cloud_cache import …` block in `__init__.py:371` is commented out; no other references. **Possibly dead.**
- `src/phasic/utils.py:60-212, 219-280` — `_Node`, `_build_balanced`, `_merge`, `_layout`, `_find_parent_y`, `_data_units_per_point`, `draw_coalescent_tree`: `draw_coalescent_tree` is defined but never imported by `__init__.py` or referenced anywhere outside `utils.py` itself (Grep hits are only the definition site). **Possibly dead.**
- `src/phasic/utils.py:15-33` — `hand_off` decorator: defined but never called outside the file. **Possibly dead.**
- `src/phasic/utils.py:35-57` — `download_link`: defined but never called outside the file. **Possibly dead.**
- `src/phasic/decoders.py:7-189` — `VariableDimPTDDecoder`, `LessThanOneDecoder`, `SumToOneDecoder`, `IndependentProbDecoder`: the only external reference is a commented-out import in `svgd.py`; not exported from `__init__.py`. **Possibly dead.**

### Cross-cutting themes

1. **Plotting / animation coverage is near-zero across `plot.py` (0%), `svgd.py::plot_*/animate_*` (several 100-stmt blocks), and `profiling.py` (3%).** These are large public APIs but have no tests at all. Priority for test investment.
2. **Six test files had to be excluded due to C-level crashes** (`test_nan_observations_correctness.py`, `test_notebook_multivar_reproduction.py`, `test_sample_with_rewards.py`, `test_scc_api.py`, `test_mcmc.py`, `test_mcmc_accuracy.py`). The underlying crash pattern is a segfault/abort during pytest's `saferepr` on failed-test argument rendering — likely a pybind11-held object whose `__repr__` dereferences freed memory. Root-causing this would restore MCMC and NaN-handling coverage in one shot and is the single highest-leverage fix.
3. **Cache/export/distributed-infrastructure modules are under-tested**: `cache_manager.py` (11%), `trace_cache.py` (9%), `graph_cache.py` (18%), `model_export.py` (15%), `callback_hash.py` (17%), `parallel_utils.py` (0%), `srun_magic.py` (0%), `distributed_utils.py` (31%). These are infrastructure rather than numerics, so tests are straightforward (tmpdir + fixture), and coverage here would materially lift the whole-package number above 60%.

---

_To add tests for these findings in test-verified batches, run:
`/coverage-apply .claude/coverage-reports/src-phasic-20260424-173029.md`._
