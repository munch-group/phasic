#ifndef PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP
#define PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "graph_builder.hpp"
#include <memory>

namespace ptdalgorithms {
namespace parameterized {
namespace ffi_handlers {

namespace ffi = xla::ffi;

/**
 * @brief JAX FFI handler for computing PMF/PDF using GraphBuilder
 *
 * This handler enables zero-copy, XLA-optimized computation of phase-type
 * distribution PDF values. The GraphBuilder is created from JSON and cached
 * in thread-local storage to avoid repeated parsing.
 *
 * @param structure_json JSON structure buffer (U8, shape: [json_length])
 * @param granularity PDF computation granularity (int32_t attribute)
 * @param discrete Whether to use discrete phase-type (bool attribute)
 * @param theta Parameter array buffer (F64, shape: [n_params])
 * @param times Time/jump points buffer (F64, shape: [n_times])
 * @param result Output buffer (F64, shape: [n_times])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputePmfFfiImpl(
    ffi::Buffer<ffi::U8> structure_json,
    int32_t granularity,
    bool discrete,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> result
);

/**
 * @brief JAX FFI handler for computing both PMF and moments
 *
 * More efficient than separate calls as graph is built only once.
 *
 * @param structure_json JSON structure buffer (U8, shape: [json_length])
 * @param granularity PDF computation granularity
 * @param discrete Whether to use discrete phase-type
 * @param nr_moments Number of moments to compute
 * @param theta Parameter array buffer
 * @param times Time/jump points buffer
 * @param pmf_result Output PMF buffer (shape: [n_times])
 * @param moments_result Output moments buffer (shape: [nr_moments])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputePmfAndMomentsFfiImpl(
    ffi::Buffer<ffi::U8> structure_json,
    int32_t granularity,
    bool discrete,
    int32_t nr_moments,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> pmf_result,
    ffi::ResultBuffer<ffi::F64> moments_result
);

} // namespace ffi_handlers
} // namespace parameterized
} // namespace ptdalgorithms

#endif // PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP
