/**
 * JAX FFI Handlers for Parameterized Graph Computations
 *
 * This file implements XLA FFI handlers that enable JAX to call C++ GraphBuilder
 * methods with proper GIL management, batching support, and gradient computation.
 *
 * Key Features:
 * - Zero-copy data transfer via XLA buffers
 * - Automatic batching for vmap/pmap
 * - Thread-safe (GIL released during C++ computation)
 * - Compatible with JAX JIT compilation
 */

#include "graph_builder.hpp"
#include "xla/ffi/api/ffi.h"
#include <memory>
#include <stdexcept>
#include <sstream>

namespace ffi = xla::ffi;

namespace ptdalgorithms {
namespace parameterized {
namespace ffi_handlers {

// ============================================================================
// FFI Handler: Compute PMF/PDF
// ============================================================================

/**
 * Compute PMF (discrete) or PDF (continuous) for given parameters and times.
 *
 * Inputs:
 *   - structure_json: Graph structure (string attribute)
 *   - theta: Parameter array, shape (n_params,)
 *   - times: Time points or jump counts, shape (n_times,)
 *   - discrete: Boolean flag (0=PDF, 1=PMF)
 *   - granularity: Discretization granularity for PDF
 *
 * Output:
 *   - pmf: PMF/PDF values, shape (n_times,)
 */
ffi::Error ComputePmfHandler(
    std::string_view structure_json,
    ffi::Buffer<ffi::DataType::F64> theta,
    ffi::Buffer<ffi::DataType::F64> times,
    int32_t discrete,
    int32_t granularity,
    ffi::ResultBuffer<ffi::DataType::F64> pmf
) {

    try {
        // Create GraphBuilder
        GraphBuilder builder(structure_json);

        // Validate inputs
        if (theta.dimensions.size() != 1) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                "theta must be 1-dimensional");
        }
        if (times.dimensions.size() != 1) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                "times must be 1-dimensional");
        }

        const double* theta_data = theta.typed_data();
        const double* times_data = times.typed_data();
        double* pmf_data = pmf.typed_data();

        size_t n_params = theta.dimensions[0];
        size_t n_times = times.dimensions[0];

        // Build graph with theta
        Graph g = builder.build(theta_data, n_params);

        // Compute PMF/PDF
        if (discrete) {
            for (size_t i = 0; i < n_times; i++) {
                int jump_count = static_cast<int>(times_data[i]);
                pmf_data[i] = g.dph_pmf(jump_count);
            }
        } else {
            for (size_t i = 0; i < n_times; i++) {
                pmf_data[i] = g.pdf(times_data[i], granularity);
            }
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ComputePmfHandler failed: " << e.what();
        return ffi::Error(ffi::ErrorCode::kInternal, oss.str());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ComputePmf, ComputePmfHandler,
    ffi::Ffi::Bind()
        .Attr<std::string_view>("structure_json")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // theta
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // times
        .Attr<int32_t>("discrete")
        .Attr<int32_t>("granularity")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // pmf
);

// ============================================================================
// FFI Handler: Compute Moments
// ============================================================================

/**
 * Compute distribution moments: E[T^k] for k=1,2,...,nr_moments.
 *
 * Inputs:
 *   - structure_json: Graph structure (string attribute)
 *   - theta: Parameter array, shape (n_params,)
 *   - nr_moments: Number of moments to compute
 *
 * Output:
 *   - moments: Moments array, shape (nr_moments,)
 */
ffi::Error ComputeMomentsHandler(
    std::string_view structure_json,
    ffi::Buffer<ffi::DataType::F64> theta,
    int32_t nr_moments,
    ffi::ResultBuffer<ffi::DataType::F64> moments
) {

    try {
        // Create GraphBuilder
        GraphBuilder builder(structure_json);

        // Validate inputs
        if (theta.dimensions.size() != 1) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                "theta must be 1-dimensional");
        }

        const double* theta_data = theta.typed_data();
        double* moments_data = moments.typed_data();
        size_t n_params = theta.dimensions[0];

        // Build graph
        Graph g = builder.build(theta_data, n_params);

        // Compute moments using internal implementation
        std::vector<double> result_vec = builder.compute_moments_impl(g, nr_moments);

        // Copy to output buffer
        for (int i = 0; i < nr_moments; i++) {
            moments_data[i] = result_vec[i];
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ComputeMomentsHandler failed: " << e.what();
        return ffi::Error(ffi::ErrorCode::kInternal, oss.str());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ComputeMoments, ComputeMomentsHandler,
    ffi::Ffi::Bind()
        .Attr<std::string_view>("structure_json")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // theta
        .Attr<int32_t>("nr_moments")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // moments
);

// ============================================================================
// FFI Handler: Compute PMF and Moments (Combined)
// ============================================================================

/**
 * Compute both PMF and moments efficiently in a single pass.
 *
 * More efficient than separate calls because the graph is built only once.
 * Primary use case: SVGD with moment-based regularization.
 *
 * Inputs:
 *   - structure_json: Graph structure (string attribute)
 *   - theta: Parameter array, shape (n_params,)
 *   - times: Time points or jump counts, shape (n_times,)
 *   - nr_moments: Number of moments to compute
 *   - discrete: Boolean flag (0=PDF, 1=PMF)
 *   - granularity: Discretization granularity for PDF
 *
 * Outputs:
 *   - pmf: PMF/PDF values, shape (n_times,)
 *   - moments: Moments array, shape (nr_moments,)
 */
ffi::Error ComputePmfAndMomentsHandler(
    std::string_view structure_json,
    ffi::Buffer<ffi::DataType::F64> theta,
    ffi::Buffer<ffi::DataType::F64> times,
    int32_t nr_moments,
    int32_t discrete,
    int32_t granularity,
    ffi::ResultBuffer<ffi::DataType::F64> pmf,
    ffi::ResultBuffer<ffi::DataType::F64> moments
) {

    try {
        // Create GraphBuilder
        GraphBuilder builder(structure_json);

        // Validate inputs
        if (theta.dimensions.size() != 1) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                "theta must be 1-dimensional");
        }
        if (times.dimensions.size() != 1) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                "times must be 1-dimensional");
        }

        const double* theta_data = theta.typed_data();
        const double* times_data = times.typed_data();
        double* pmf_data = pmf.typed_data();
        double* moments_data = moments.typed_data();

        size_t n_params = theta.dimensions[0];
        size_t n_times = times.dimensions[0];

        // Build graph ONCE (more efficient than separate calls)
        Graph g = builder.build(theta_data, n_params);

        // Compute PMF/PDF
        if (discrete) {
            for (size_t i = 0; i < n_times; i++) {
                int jump_count = static_cast<int>(times_data[i]);
                pmf_data[i] = g.dph_pmf(jump_count);
            }
        } else {
            for (size_t i = 0; i < n_times; i++) {
                pmf_data[i] = g.pdf(times_data[i], granularity);
            }
        }

        // Compute moments using same graph
        std::vector<double> result_vec = builder.compute_moments_impl(g, nr_moments);
        for (int i = 0; i < nr_moments; i++) {
            moments_data[i] = result_vec[i];
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ComputePmfAndMomentsHandler failed: " << e.what();
        return ffi::Error(ffi::ErrorCode::kInternal, oss.str());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ComputePmfAndMoments, ComputePmfAndMomentsHandler,
    ffi::Ffi::Bind()
        .Attr<std::string_view>("structure_json")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // theta
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // times
        .Attr<int32_t>("nr_moments")
        .Attr<int32_t>("discrete")
        .Attr<int32_t>("granularity")
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // pmf
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // moments
);

} // namespace ffi_handlers
} // namespace parameterized
} // namespace ptdalgorithms
