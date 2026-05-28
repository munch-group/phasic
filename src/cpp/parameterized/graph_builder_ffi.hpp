#ifndef PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP
#define PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "graph_builder.hpp"
#include <memory>

namespace phasic {
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
 * @param structure_json JSON structure as string_view (STATIC attribute, not batched)
 * @param granularity PDF computation granularity (int32_t attribute)
 * @param discrete Whether to use discrete phase-type (bool attribute)
 * @param theta Parameter array buffer (F64, shape: [n_params]) - BATCHED by vmap
 * @param times Time/jump points buffer (F64, shape: [n_times]) - BATCHED by vmap
 * @param result Output buffer (F64, shape: [n_times])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputePmfFfiImpl(
    std::string_view structure_json,
    int32_t granularity,
    bool discrete,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> result
);

/**
 * @brief JAX FFI handler for computing distribution moments
 *
 * This handler computes E[T^k] for k=1,2,...,nr_moments using the GraphBuilder.
 *
 * @param structure_json JSON structure as string_view (STATIC attribute, not batched)
 * @param theta Parameter array buffer (F64, shape: [n_params]) - BATCHED by vmap
 * @param nr_moments Number of moments to compute (int32_t attribute)
 * @param result Output buffer (F64, shape: [nr_moments])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputeMomentsFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    int32_t nr_moments,
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
    std::string_view structure_json,
    int32_t nr_moments,
    int32_t granularity,
    bool discrete,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::Buffer<ffi::F64> rewards,
    ffi::ResultBuffer<ffi::F64> pmf_result,
    ffi::ResultBuffer<ffi::F64> moments_result
);

/**
 * @brief JAX FFI handler for computing multivariate PMF/PDF
 *
 * Computes PDF/PMF for multivariate observations where each observation
 * is a vector of features. Supports two modes:
 * - Sparse mode (compute_joint=false): Independent PDF per feature
 * - Joint mode (compute_joint=true): Joint PDF across features [NOT IMPLEMENTED]
 *
 * @param structure_json JSON structure as string_view (STATIC attribute, not batched)
 * @param granularity PDF computation granularity (int32_t attribute)
 * @param discrete Whether to use discrete phase-type (bool attribute)
 * @param compute_joint Whether to compute joint PDF (bool attribute, must be false for now)
 * @param theta Parameter array buffer (F64, shape: [n_params]) - BATCHED by vmap
 * @param times Time/jump points buffer (F64, shape: [n_times, n_features]) - BATCHED by vmap
 * @param rewards Reward matrix buffer (F64, shape: [n_vertices, n_features]) - BATCHED by vmap
 * @param result Output buffer (F64, shape: [n_times, n_features])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputePmfMultivariateFfiImpl(
    std::string_view structure_json,
    int32_t granularity,
    bool discrete,
    bool compute_joint,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::Buffer<ffi::F64> rewards,
    ffi::ResultBuffer<ffi::F64> result
);

/**
 * @brief JAX FFI handler for computing expected sojourn times for subset of vertices
 *
 * Memory-efficient subset computation using n×k matrix instead of n×n.
 * Supports vmap batching with OpenMP parallelization.
 *
 * @param structure_json JSON structure as string_view (STATIC attribute, not batched)
 * @param theta Parameter array buffer (F64, shape: [n_params]) - BATCHED by vmap
 * @param indices Vertex indices buffer (S32, shape: [k]) - BATCHED by vmap, int32
 * @param result Output buffer (F64, shape: [k])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error ComputeSojournTimesFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> indices,
    ffi::ResultBuffer<ffi::F64> result
);

/**
 * FFI handler for backward probabilities computation.
 *
 * Inputs:
 *   - structure_json: Graph structure (string_view attribute, STATIC)
 *   - theta: Parameters (F64, shape: [n_params]) - BATCHED by vmap
 *   - target_vertices: Target terminal indices (S32, shape: [n_targets])
 *
 * Output:
 *   - backward_probs: P(reach target | start at v) for each vertex (F64, shape: [n_vertices])
 */
ffi::Error BackwardProbabilitiesFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> target_vertices,
    ffi::ResultBuffer<ffi::F64> result
);

/**
 * FFI handler for conditioned path sampling with fixed-size output.
 *
 * Inputs:
 *   - structure_json: Graph structure (string_view attribute, STATIC)
 *   - max_length: Maximum path length (int32 attribute, STATIC)
 *   - theta: Parameters (F64, shape: [n_params]) - BATCHED by vmap
 *   - target_vertex: Target terminal index (S32, shape: [1])
 *   - seed: Random seed (S32, shape: [1])
 *
 * Outputs:
 *   - vertex_indices: Path vertex indices, -1 padded (S32, shape: [max_length])
 *   - entry_times: Cumulative entry times, 0.0 padded (F64, shape: [max_length])
 */
ffi::Error SamplePathConditionedFfiImpl(
    std::string_view structure_json,
    int32_t max_length,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> target_vertex,
    ffi::Buffer<ffi::S32> seed,
    ffi::ResultBuffer<ffi::S32> out_indices,
    ffi::ResultBuffer<ffi::F64> out_times
);

/**
 * @brief JAX FFI handler for daisy-chained joint-prob computation.
 *
 * Performs the full daisy chain end-to-end in C: for each of (n_epochs - 1)
 * epoch transitions, calls ptd_graph_update_ipv (with the propagated IPV)
 * and ptd_graph_update_weights (with that epoch's theta), runs
 * stop_probability(epoch_dt), collapses t-aux pairs, projects into IPV
 * coordinates for the next epoch. Final epoch reads stop_probability(t_eval),
 * collapses t-aux, slices to t-vertex positions and returns.
 *
 * Daisy-chain metadata (epoch_dts, t_eval, n_epochs, ipv_target_indices,
 * t_aux_keys, t_aux_values, t_vertex_indices) is encoded into the
 * structure_json under a top-level "_daisy_chain" object. GraphBuilder
 * silently ignores unknown JSON fields, so the same JSON can be reused
 * for both graph construction and daisy-chain metadata extraction.
 *
 * Mirrors ComputeSojournTimesFfiImpl in structure: thread-local
 * GraphBuilder cache, OpenMP-parallel vmap loop, per-batch
 * get_or_init_persistent_graph (Stage A1) so the symbolic compute graph
 * (Stage A0) is built once per (thread, GraphBuilder) and reused.
 *
 * @param structure_json JSON with graph + "_daisy_chain" metadata (string_view, STATIC)
 * @param theta Per-epoch parameter buffer (F64, shape: [n_epochs * param_length]) - BATCHED
 * @param initial_ipv IPV for epoch 0 (F64, shape: [n_ipv]) - BATCHED
 * @param result Output buffer (F64, shape: [n_t_vertices])
 *
 * @return ffi::Error Success or error status
 */
ffi::Error DaisyChainJointProbsFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> initial_ipv,
    ffi::ResultBuffer<ffi::F64> result
);

} // namespace ffi_handlers

// Functions to create FFI handlers for Python-side registration
// These must be called AFTER JAX is fully initialized
XLA_FFI_Handler* CreateComputePmfHandler();
XLA_FFI_Handler* CreateComputeMomentsHandler();
XLA_FFI_Handler* CreateComputePmfAndMomentsHandler();
XLA_FFI_Handler* CreateComputePmfMultivariateHandler();
XLA_FFI_Handler* CreateComputeSojournTimesHandler();
XLA_FFI_Handler* CreateBackwardProbabilitiesHandler();
XLA_FFI_Handler* CreateSamplePathConditionedHandler();
XLA_FFI_Handler* CreateDaisyChainJointProbsHandler();
XLA_FFI_Handler* CreateDaisyChainSojournHandler();

} // namespace parameterized
} // namespace phasic

#endif // PTDALGORITHMS_PARAMETERIZED_GRAPH_BUILDER_FFI_HPP
