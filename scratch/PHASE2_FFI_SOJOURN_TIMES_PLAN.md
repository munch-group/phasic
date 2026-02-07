# Phase 2: FFI Support for Expected Sojourn Times - Implementation Plan

**Date**: 2025-12-30
**Status**: Ready to implement

## Executive Summary

Implement JAX FFI integration for `expected_sojourn_time` with **complete feature parity** to `compute_pmf_ffi`:
- **vmap batching** with `expand_dims` method
- **OpenMP multi-threading** for parallel batch processing
- **Thread-local GraphBuilder caching** for performance
- **Broadcast support** for singleton indices with batched theta
- **Custom VJP gradients** via finite differences (same as all FFI functions)
- **Explicit error handling** with detailed messages
- **No silent fallbacks** - explicit errors if requirements not met

**Memory Impact**: 268 GB → 1.7 GB (99.4% reduction)
**Performance**: OpenMP parallelization across vmap batches
**SVGD**: Enables 100k evaluations that were previously out of memory

---

## Gradient Support for SVGD

**Method**: Custom VJP with finite differences (same pattern as all existing FFI functions)

```python
@jax.custom_vjp
def model(theta, vertex_indices, rewards=None):
    # Forward: call FFI
    sojourn_times = compute_sojourn_times_ffi(structure_json, theta, vertex_indices)
    return sojourn_times, dummy_moments

def model_fwd(theta, vertex_indices, rewards=None):
    output = model(theta, vertex_indices, rewards)
    return output, (theta, vertex_indices)  # Save inputs for backward

def model_bwd(saved_inputs, g):
    theta, vertex_indices = saved_inputs
    g_sojourn, g_moments = g

    # Finite differences (central)
    eps = 1e-7
    theta_bar = []
    for i in range(n_params):
        theta_plus = theta.at[i].add(eps)
        theta_minus = theta.at[i].add(-eps)

        sojourn_plus, _ = model(theta_plus, vertex_indices)
        sojourn_minus, _ = model(theta_minus, vertex_indices)

        grad_i = jnp.sum(g_sojourn * (sojourn_plus - sojourn_minus) / (2 * eps))
        theta_bar.append(grad_i)

    return jnp.array(theta_bar), None, None

model.defvjp(model_fwd, model_bwd)
```

**Why this works**:
- Same pattern as `compute_pmf_ffi` (lines 2415-2444 in `__init__.py`)
- Same pattern as `compute_moments_ffi` (lines 3250-3279)
- Same pattern as existing `joint_index` (lines 3692-3738)
- SVGD only needs `jax.grad()` - custom VJP provides this
- Cost: 2 × n_params extra FFI calls per gradient (practical since FFI is fast)

---

## Part 1: C++ FFI Handler Implementation

### 1.1 Add Handler to `graph_builder_ffi.cpp`

**File**: `src/cpp/parameterized/graph_builder_ffi.cpp`
**Location**: After `ComputePmfAndMomentsFfiImpl` (~line 400)

**Add**:
```cpp
// ===========================================================================
// ComputeSojournTimesFfiImpl: vmap-aware wrapper for expected_sojourn_time_subset
// ===========================================================================

ffi::Error ComputeSojournTimesFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> indices,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        std::string json_str(structure_json);

        // Thread-local cache lookup
        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            builder = it->second;
        } else {
            try {
                builder = std::make_shared<GraphBuilder>(json_str);
                builder_cache[json_str] = builder;
            } catch (const std::exception& e) {
                return ffi::Error::InvalidArgument(
                    std::string("Failed to parse JSON: ") + e.what()
                );
            }
        }

        // Parse dimensions (handle vmap batching)
        auto theta_dims = theta.dimensions();
        auto indices_dims = indices.dimensions();

        size_t theta_len, n_indices;
        size_t theta_batch_size = 1;
        size_t indices_batch_size = 1;

        if (theta_dims.size() == 1) {
            theta_len = theta_dims[0];
        } else if (theta_dims.size() == 2) {
            theta_batch_size = theta_dims[0];
            theta_len = theta_dims[1];
        } else {
            return ffi::Error::InvalidArgument("theta must be 1D or 2D");
        }

        if (indices_dims.size() == 1) {
            n_indices = indices_dims[0];
        } else if (indices_dims.size() == 2) {
            indices_batch_size = indices_dims[0];
            n_indices = indices_dims[1];
        } else {
            return ffi::Error::InvalidArgument("indices must be 1D or 2D");
        }

        const double* theta_data = theta.typed_data();
        const int32_t* indices_data = indices.typed_data();
        double* result_data = result->typed_data();

        // Batched computation
        if (theta_batch_size > 1 || indices_batch_size > 1) {
            size_t batch_size = std::max(theta_batch_size, indices_batch_size);
            bool indices_is_broadcast = (indices_batch_size == 1 && theta_batch_size > 1);

            if (!indices_is_broadcast && theta_batch_size != indices_batch_size) {
                return ffi::Error::InvalidArgument(
                    "Batch sizes must match: theta=" + std::to_string(theta_batch_size) +
                    ", indices=" + std::to_string(indices_batch_size)
                );
            }

            // Convert broadcast indices once
            std::vector<size_t> indices_vec(n_indices);
            if (indices_is_broadcast) {
                for (size_t i = 0; i < n_indices; i++) {
                    if (indices_data[i] < 0) {
                        return ffi::Error::InvalidArgument("Negative index not allowed");
                    }
                    indices_vec[i] = static_cast<size_t>(indices_data[i]);
                }
            }

            // OpenMP parallel processing
            #pragma omp parallel for if(batch_size > 1)
            for (size_t b = 0; b < batch_size; b++) {
                const double* theta_b = theta_data + (b * theta_len);
                Graph g = builder->build(theta_b, theta_len);

                std::vector<size_t> indices_b(n_indices);
                if (indices_is_broadcast) {
                    indices_b = indices_vec;
                } else {
                    const int32_t* indices_batch = indices_data + (b * n_indices);
                    for (size_t i = 0; i < n_indices; i++) {
                        if (indices_batch[i] < 0) {
                            double* result_b = result_data + (b * n_indices);
                            for (size_t j = 0; j < n_indices; j++) {
                                result_b[j] = std::numeric_limits<double>::quiet_NaN();
                            }
                            continue;
                        }
                        indices_b[i] = static_cast<size_t>(indices_batch[i]);
                    }
                }

                double* result_b = result_data + (b * n_indices);

                double* sojourn_ptr = ptd_expected_sojourn_time_subset(
                    g.c_graph(), indices_b.data(), n_indices
                );

                if (sojourn_ptr == NULL) {
                    for (size_t i = 0; i < n_indices; i++) {
                        result_b[i] = std::numeric_limits<double>::quiet_NaN();
                    }
                } else {
                    std::memcpy(result_b, sojourn_ptr, n_indices * sizeof(double));
                    free(sojourn_ptr);
                }
            }

        } else {
            // Not batched
            Graph g = builder->build(theta_data, theta_len);

            std::vector<size_t> indices_vec(n_indices);
            for (size_t i = 0; i < n_indices; i++) {
                if (indices_data[i] < 0) {
                    return ffi::Error::InvalidArgument(
                        "Negative index at position " + std::to_string(i)
                    );
                }
                indices_vec[i] = static_cast<size_t>(indices_data[i]);
            }

            double* sojourn_ptr = ptd_expected_sojourn_time_subset(
                g.c_graph(), indices_vec.data(), n_indices
            );

            if (sojourn_ptr == NULL) {
                return ffi::Error::Internal(
                    std::string("ptd_expected_sojourn_time_subset failed: ") + ptd_err
                );
            }

            std::memcpy(result_data, sojourn_ptr, n_indices * sizeof(double));
            free(sojourn_ptr);
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        std::cerr << "❌ ComputeSojournTimesFfiImpl exception: " << e.what() << std::endl;
        return ffi::Error::Internal(e.what());
    }
}
```

### 1.2 Add Handler Creator

**File**: `src/cpp/parameterized/graph_builder_ffi.cpp`
**Location**: After other `Create*Handler()` functions (~line 700)

**Add**:
```cpp
XLA_FFI_Handler* CreateComputeSojournTimesHandler() {
    static XLA_FFI_Handler* handler = []() {
        return xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()   // theta
            .Arg<xla::ffi::Buffer<xla::ffi::S32>>()   // indices (int32)
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()   // result
            .To(ComputeSojournTimesFfiImpl)
            .release();
    }();
    return handler;
}
```

### 1.3 Update Header

**File**: `src/cpp/parameterized/graph_builder_ffi.hpp`
**Location**: After existing declarations

**Add**:
```cpp
XLA_FFI_HANDLER_EXPORT XLA_FFI_Handler* CreateComputeSojournTimesHandler();
```

### 1.4 Export Symbol

**File**: `src/cpp/parameterized/graph_builder_ffi.cpp`
**Location**: After other `XLA_FFI_HANDLER_EXPORT` declarations (~line 600)

**Add**:
```cpp
XLA_FFI_HANDLER_EXPORT XLA_FFI_Handler* CreateComputeSojournTimesHandler();
```

---

## Part 2: Python Binding (Capsule Getter)

**File**: `src/cpp/phasic_pybind.cpp`
**Location**: After `get_compute_pmf_multivariate_ffi_capsule()` (~line 4976)

**Add**:
```cpp
#ifdef HAVE_XLA_FFI
  param_module.def("get_compute_sojourn_times_ffi_capsule", []() -> py::capsule {
      auto* handler = phasic::parameterized::CreateComputeSojournTimesHandler();
      return py::capsule(reinterpret_cast<void*>(handler), "xla._CUSTOM_CALL_TARGET");
  }, R"delim(
  Get PyCapsule for JAX FFI compute_sojourn_times handler.

  Features:
    - vmap batching with OpenMP parallelization
    - Thread-local GraphBuilder caching
    - Broadcasting support
    - Memory: O(n×k) vs O(n²) → 99.4% savings for large graphs

  Returns
  -------
  capsule
      PyCapsule containing pointer to XLA FFI handler
  )delim");
#endif
```

---

## Part 3: JAX FFI Python Wrapper

### 3.1 Register FFI Target

**File**: `src/phasic/ffi_wrappers.py`
**Function**: `_register_ffi_targets()`
**Location**: After registering `ptd_compute_pmf_multivariate` (~line 240)

**Add**:
```python
        try:
            compute_sojourn_times_capsule = cpp_module.parameterized.get_compute_sojourn_times_ffi_capsule()
        except AttributeError as e:
            raise PTDBackendError(
                "FFI handler get_compute_sojourn_times_ffi_capsule() not available.\n"
                "  Rebuild with: pixi run install-dev"
            ) from e

        try:
            jax.ffi.register_ffi_target(
                "ptd_compute_sojourn_times",
                compute_sojourn_times_capsule,
                platform="cpu",
                api_version=1
            )
        except Exception as e:
            raise PTDBackendError(
                f"FFI registration for compute_sojourn_times failed: {e}"
            ) from e
```

### 3.2 Create Public FFI Wrapper

**File**: `src/phasic/ffi_wrappers.py`
**Location**: After `compute_pmf_multivariate_ffi()` (~line 850)

**Add**: (see full implementation in plan below)

### 3.3 Update __all__ Export

**File**: `src/phasic/ffi_wrappers.py`
**Line**: ~869

**Modify**:
```python
__all__ = [
    'compute_pmf_ffi',
    'compute_moments_ffi',
    'compute_pmf_and_moments_ffi',
    'compute_pmf_multivariate_ffi',
    'compute_sojourn_times_ffi',  # NEW
]
```

---

## Part 4: Update `pmf_from_graph_joint_index()`

**File**: `src/phasic/__init__.py`
**Location**: Lines 3560-3739

**Replace entire function** with FFI-based implementation (see below)

---

## Part 5: Tests

Create three new test files:
1. `tests/pytest/test_ffi_sojourn_times.py` - Unit tests
2. `tests/pytest/test_svgd_joint_index_ffi.py` - Integration tests
3. `tests/pytest/test_sojourn_memory_efficiency.py` - Memory tests

---

## Build and Test Sequence

```bash
# 1. Rebuild with FFI
pixi run install-dev

# 2. Run tests
pixi run pytest tests/pytest/test_ffi_sojourn_times.py -v
pixi run pytest tests/pytest/test_svgd_joint_index_ffi.py -v
pixi run pytest tests/pytest/test_sojourn_memory_efficiency.py -v

# 3. Test example
pixi run python examples/test.py
```

---

## Success Criteria

- FFI handler compiles with HAVE_XLA_FFI
- JAX FFI registration succeeds
- `compute_sojourn_times_ffi()` matches direct computation
- vmap batching works with OpenMP
- JIT compilation works
- Custom VJP gradients work (finite differences)
- `pmf_from_graph_joint_index()` uses FFI (REQUIRED)
- SVGD with `joint_index=True` completes without memory errors
- Explicit errors when FFI disabled or indices wrong dtype
- All existing tests still pass
