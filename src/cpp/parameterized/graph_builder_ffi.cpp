#include "graph_builder_ffi.hpp"
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>

#ifdef PHASIC_HAVE_OPENMP
#include <omp.h>
#endif

extern "C" {
#include "../../c/phasic_log.h"
}

namespace phasic {
namespace parameterized {
namespace ffi_handlers {

namespace ffi = xla::ffi;

// Thread-local cache for GraphBuilder instances
// Key: JSON string, Value: GraphBuilder instance
thread_local std::unordered_map<std::string, std::shared_ptr<GraphBuilder>> builder_cache;

// Parsed daisy-chain metadata + topology-derived lookup tables.
// Computed once per builder (topology-only — no theta dependence) and
// reused across every FFI call for that model. Eliminates the ~few-ms
// JSON re-parse + the per-batch collapsed_pos rebuild.
struct DaisyChainMeta {
    int n_epochs;
    int param_length;
    double t_eval;
    int granularity;
    std::vector<double> epoch_dts;
    std::vector<int> ipv_target_indices;
    std::vector<int> t_aux_keys;
    std::vector<int> t_aux_values;
    std::vector<int> t_vertex_indices;

    // Derived (topology-only).
    size_t n_vertices = 0;
    size_t n_collapsed = 0;
    std::unordered_set<int> aux_set;
    std::unordered_map<int, int> t_to_aux;
    std::vector<int> collapsed_pos;  // length n_vertices, -1 for aux entries
};

// Cache parsed daisy-chain metadata by structure-json string.
// Process-wide (not thread-local) because the metadata is read-only
// after the initial population — multiple OMP threads can share it
// safely. The initial population happens under
// daisy_chain_meta_init_mutex. Keyed by json_str (rather than
// GraphBuilder*) so the entry is shared across the per-thread
// builder_cache copies of the same model.
static std::unordered_map<std::string, std::shared_ptr<const DaisyChainMeta>>
    daisy_chain_meta_cache;
static std::mutex daisy_chain_meta_init_mutex;

// Per-thread persistent Graph, keyed by GraphBuilder*. Each OMP thread
// builds its Graph once on first use, then reuses via update_weights
// (zero alloc) on every subsequent batch element. Eliminates the
// dominant allocator-contention loss in the per-obs path.
thread_local std::unordered_map<const GraphBuilder*, std::unique_ptr<phasic::Graph>>
    per_thread_graph_cache;

// Disables libomp dynamic-thread-count adjustment so OMP regions get
// the full width configured via OMP_NUM_THREADS (i.e. phasic.config
// cpu_threads). Logs the resulting omp_get_max_threads() once so we
// can verify fan-out at runtime via PHASIC_LOG_LEVEL=DEBUG.
static void ensure_omp_full_width_once() {
#ifdef PHASIC_HAVE_OPENMP
    static std::once_flag flag;
    std::call_once(flag, []() {
        omp_set_dynamic(0);
        const int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        PTD_LOG_DEBUG(
            "phasic FFI OMP: dynamic disabled, max_threads=%d "
            "(from OMP_NUM_THREADS / phasic.config cpu_threads)",
            max_threads
        );
    });
#endif
}

ffi::Error ComputePmfFfiImpl(
    std::string_view structure_json,
    int32_t granularity,
    bool discrete,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        // JSON is now passed as string_view attribute (static, not batched by vmap)
        std::string json_str(structure_json);

        // Look up or create GraphBuilder in thread-local cache
        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            builder = it->second;
        } else {
            // Create new GraphBuilder and cache it
            try {
                builder = std::make_shared<GraphBuilder>(json_str);
                builder_cache[json_str] = builder;
            } catch (const std::exception& e) {
                return ffi::Error::InvalidArgument(
                    std::string("Failed to parse JSON structure: ") + e.what()
                );
            }
        }

        // Extract buffer dimensions
        // NOTE: With vmap, buffers may have batch dimension added
        // theta: 1D (n_params,) OR 2D (batch, n_params)
        // times: 1D (n_times,) OR 2D (1, n_times) when not mapped OR (batch, n_times) when mapped
        auto theta_dims = theta.dimensions();
        auto times_dims = times.dimensions();

        size_t theta_len, n_times;
        size_t theta_batch_size = 1;
        size_t times_batch_size = 1;

        if (theta_dims.size() == 1) {
            // No batch dimension
            theta_len = theta_dims[0];
        } else if (theta_dims.size() == 2) {
            // Batched (from vmap): shape is (batch, n_params)
            theta_batch_size = theta_dims[0];
            theta_len = theta_dims[1];
        } else {
            return ffi::Error::InvalidArgument("theta must be 1D or 2D array");
        }

        if (times_dims.size() == 1) {
            // No batch dimension
            n_times = times_dims[0];
        } else if (times_dims.size() == 2) {
            // Batched OR singleton batch: shape is (batch, n_times)
            times_batch_size = times_dims[0];
            n_times = times_dims[1];
        } else {
            return ffi::Error::InvalidArgument("times must be 1D or 2D array");
        }

        // Get raw data pointers
        const double* theta_data = theta.typed_data();
        const double* times_data = times.typed_data();
        double* result_data = result->typed_data();

        // Check if batched (from vmap)
        if (theta_batch_size > 1 || times_batch_size > 1) {
            // BATCHED: Process multiple theta/times combinations
            size_t batch_size = std::max(theta_batch_size, times_batch_size);

            // Times can be either batched (same size as theta) or singleton (broadcast to all theta)
            bool times_is_broadcast = (times_batch_size == 1 && theta_batch_size > 1);

            // Process each batch element in parallel using OpenMP
            #pragma omp parallel for if(batch_size > 1)
            for (size_t b = 0; b < batch_size; b++) {
                // Build graph for this batch element
                const double* theta_b = theta_data + (b * theta_len);
                Graph g = builder->build(theta_b, theta_len);

                // Get times for this batch (either indexed or broadcast)
                const double* times_b = times_is_broadcast ? times_data : (times_data + (b * n_times));

                // Get result pointer for this batch
                double* result_b = result_data + (b * n_times);

                // Compute PMF/PDF
                if (discrete) {
                    for (size_t i = 0; i < n_times; i++) {
                        int jump_count = static_cast<int>(times_b[i]);
                        result_b[i] = g.dph_pmf(jump_count);
                    }
                } else {
                    for (size_t i = 0; i < n_times; i++) {
                        result_b[i] = g.pdf(times_b[i], granularity);
                    }
                }
            }
        } else {
            // NOT BATCHED: theta shape (n_params,), times shape (n_times,)
            Graph g = builder->build(theta_data, theta_len);

            if (discrete) {
                for (size_t i = 0; i < n_times; i++) {
                    int jump_count = static_cast<int>(times_data[i]);
                    result_data[i] = g.dph_pmf(jump_count);
                }
            } else {
                for (size_t i = 0; i < n_times; i++) {
                    result_data[i] = g.pdf(times_data[i], granularity);
                }
            }
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        // Capture C++ exceptions and return as FFI error
        PTD_LOG_ERROR("ComputePmfFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

// ComputeMomentsFfiImpl: vmap-aware wrapper for moments computation
ffi::Error ComputeMomentsFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    int32_t nr_moments,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        // JSON is passed as string_view attribute (static, not batched by vmap)
        std::string json_str(structure_json);

        // Look up or create GraphBuilder in thread-local cache
        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            builder = it->second;
        } else {
            // Create new GraphBuilder and cache it
            try {
                builder = std::make_shared<GraphBuilder>(json_str);
                builder_cache[json_str] = builder;
            } catch (const std::exception& e) {
                return ffi::Error::InvalidArgument(
                    std::string("Failed to parse JSON structure: ") + e.what()
                );
            }
        }

        // Extract buffer dimensions
        // NOTE: With vmap, theta may have batch dimension added
        // theta: 1D (n_params,) OR 2D (batch, n_params)
        auto theta_dims = theta.dimensions();

        size_t theta_len;
        size_t theta_batch_size = 1;

        if (theta_dims.size() == 1) {
            // No batch dimension
            theta_len = theta_dims[0];
        } else if (theta_dims.size() == 2) {
            // Batched (from vmap): shape is (batch, n_params)
            theta_batch_size = theta_dims[0];
            theta_len = theta_dims[1];
        } else {
            return ffi::Error::InvalidArgument("theta must be 1D or 2D array");
        }

        // Get raw data pointers
        const double* theta_data = theta.typed_data();
        double* result_data = result->typed_data();

        // Empty rewards vector (standard moments)
        std::vector<double> rewards_vec;

        // Check if batched (from vmap)
        if (theta_batch_size > 1) {
            // BATCHED: Process multiple theta values
            // Process each batch element in parallel using OpenMP
            #pragma omp parallel for if(theta_batch_size > 1)
            for (size_t b = 0; b < theta_batch_size; b++) {
                // Build graph for this batch element
                const double* theta_b = theta_data + (b * theta_len);
                Graph g = builder->build(theta_b, theta_len);

                // Get result pointer for this batch
                double* result_b = result_data + (b * nr_moments);

                // Compute moments
                std::vector<double> moments_vec = builder->compute_moments_impl(g, nr_moments, rewards_vec);

                // Copy to output buffer
                for (int i = 0; i < nr_moments; i++) {
                    result_b[i] = moments_vec[i];
                }
            }
        } else {
            // NOT BATCHED: theta shape (n_params,)
            Graph g = builder->build(theta_data, theta_len);

            // Compute moments
            std::vector<double> moments_vec = builder->compute_moments_impl(g, nr_moments, rewards_vec);

            // Copy to output buffer
            for (int i = 0; i < nr_moments; i++) {
                result_data[i] = moments_vec[i];
            }
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        // Capture C++ exceptions and return as FFI error
        PTD_LOG_ERROR("ComputeMomentsFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

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
) {
    // Convert string_view to string (direct conversion, no buffer extraction needed)
    std::string json_str(structure_json);

    // Look up or create GraphBuilder in thread-local cache
    std::shared_ptr<GraphBuilder> builder;
    auto it = builder_cache.find(json_str);
    if (it != builder_cache.end()) {
        builder = it->second;
    } else {
        // Create new GraphBuilder and cache it
        try {
            builder = std::make_shared<GraphBuilder>(json_str);
            builder_cache[json_str] = builder;
        } catch (const std::exception& e) {
            return ffi::Error::InvalidArgument(
                std::string("Failed to parse JSON structure: ") + e.what()
            );
        }
    }

    // Extract buffer dimensions
    // With vmap, buffers may have batch dimension added
    // theta: 1D (n_params,) OR 2D (batch, n_params)
    // times: 1D (n_times,) OR 2D (1, n_times) when not mapped OR (batch, n_times) when mapped
    auto theta_dims = theta.dimensions();
    auto times_dims = times.dimensions();

    size_t theta_len, n_times;
    size_t theta_batch_size = 1;
    size_t times_batch_size = 1;

    if (theta_dims.size() == 1) {
        // No batch dimension
        theta_len = theta_dims[0];
    } else if (theta_dims.size() == 2) {
        // Batched (from vmap): shape is (batch, n_params)
        theta_batch_size = theta_dims[0];
        theta_len = theta_dims[1];
    } else {
        return ffi::Error::InvalidArgument("theta must be 1D or 2D array");
    }

    if (times_dims.size() == 1) {
        // No batch dimension
        n_times = times_dims[0];
    } else if (times_dims.size() == 2) {
        // Batched OR singleton batch: shape is (batch, n_times)
        times_batch_size = times_dims[0];
        n_times = times_dims[1];
    } else {
        return ffi::Error::InvalidArgument("times must be 1D or 2D array");
    }

    // Extract rewards dimensions (always n_vertices, or 0 for standard moments)
    auto rewards_dims = rewards.dimensions();
    size_t n_rewards = 0;
    size_t rewards_batch_size = 1;

    if (rewards_dims.size() == 1) {
        // No batch dimension: shape is (n_rewards,)
        n_rewards = rewards_dims[0];
    } else if (rewards_dims.size() == 2) {
        // Batched: shape is (batch, n_rewards)
        rewards_batch_size = rewards_dims[0];
        n_rewards = rewards_dims[1];
    } else if (rewards_dims.size() != 0) {
        return ffi::Error::InvalidArgument("rewards must be 0D, 1D, or 2D array");
    }

    // Get raw data pointers
    const double* theta_data = theta.typed_data();
    const double* times_data = times.typed_data();
    const double* rewards_data = rewards.typed_data();
    double* pmf_data = pmf_result->typed_data();
    double* moments_data = moments_result->typed_data();

    // Check if batched (from vmap)
    if (theta_batch_size > 1 || times_batch_size > 1) {
        // BATCHED: Process multiple theta/times combinations
        size_t batch_size = std::max(theta_batch_size, times_batch_size);

        // Times can be either batched (same size as theta) or singleton (broadcast to all theta)
        bool times_is_broadcast = (times_batch_size == 1 && theta_batch_size > 1);

        // Process each batch element in parallel using OpenMP
        #pragma omp parallel for if(batch_size > 1)
        for (size_t b = 0; b < batch_size; b++) {
            try {
                // Build graph for this batch element
                const double* theta_b = theta_data + (b * theta_len);
                Graph g = builder->build(theta_b, theta_len);

                // Get times for this batch (either indexed or broadcast)
                const double* times_b = times_is_broadcast ? times_data : (times_data + (b * n_times));

                // Get result pointers for this batch
                double* pmf_b = pmf_data + (b * n_times);
                double* moments_b = moments_data + (b * nr_moments);

                // Compute PMF/PDF
                if (discrete) {
                    for (size_t i = 0; i < n_times; i++) {
                        int jump_count = static_cast<int>(times_b[i]);
                        pmf_b[i] = g.dph_pmf(jump_count);
                    }
                } else {
                    for (size_t i = 0; i < n_times; i++) {
                        pmf_b[i] = g.pdf(times_b[i], granularity);
                    }
                }

                // Compute moments using same graph
                std::vector<double> rewards_vec;
                if (n_rewards > 0) {
                    const double* rewards_b = (rewards_batch_size > 1)
                        ? (rewards_data + (b * n_rewards))
                        : rewards_data;
                    rewards_vec.assign(rewards_b, rewards_b + n_rewards);
                }
                std::vector<double> moments_vec = builder->compute_moments_impl(g, nr_moments, rewards_vec);

                // Copy moments to output buffer
                for (int i = 0; i < nr_moments; i++) {
                    moments_b[i] = moments_vec[i];
                }
            } catch (const std::exception& e) {
                // In parallel region, we can't return error directly
                // Set error in moments output as NaN to signal failure
                double* moments_b = moments_data + (b * nr_moments);
                for (int i = 0; i < nr_moments; i++) {
                    moments_b[i] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    } else {
        // NOT BATCHED: theta shape (n_params,), times shape (n_times,)
        try {
            Graph g = builder->build(theta_data, theta_len);

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
            std::vector<double> rewards_vec;
            if (n_rewards > 0) {
                rewards_vec.assign(rewards_data, rewards_data + n_rewards);
            }
            std::vector<double> moments_vec = builder->compute_moments_impl(g, nr_moments, rewards_vec);

            // Copy moments to output buffer
            for (int i = 0; i < nr_moments; i++) {
                moments_data[i] = moments_vec[i];
            }

            return ffi::Error::Success();

        } catch (const std::exception& e) {
            return ffi::Error::Internal(e.what());
        }
    }

    return ffi::Error::Success();
}

ffi::Error ComputePmfMultivariateFfiImpl(
    std::string_view structure_json,
    int32_t granularity,
    bool discrete,
    bool compute_joint,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::Buffer<ffi::F64> rewards,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        // Validate compute_joint mode
        if (compute_joint) {
            return ffi::Error::InvalidArgument(
                "Joint PDF computation not yet implemented. "
                "Use compute_joint=false for independent feature PDFs (sparse mode)."
            );
        }

        // JSON is passed as string_view attribute (static, not batched by vmap)
        std::string json_str(structure_json);

        // Look up or create GraphBuilder in thread-local cache
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
                    std::string("Failed to parse JSON structure: ") + e.what()
                );
            }
        }

        // Extract buffer dimensions
        // With vmap, buffers may have batch dimension added
        // theta: 1D (n_params,) OR 2D (batch, n_params)
        // times: 2D (n_times, n_features) OR 3D (batch, n_times, n_features)
        // rewards: 2D (n_vertices, n_features) OR 3D (batch, n_vertices, n_features)
        auto theta_dims = theta.dimensions();
        auto times_dims = times.dimensions();
        auto rewards_dims = rewards.dimensions();

        size_t theta_len;
        size_t theta_batch_size = 1;
        size_t n_times, n_features;
        size_t times_batch_size = 1;
        size_t n_vertices, n_features_rewards;
        size_t rewards_batch_size = 1;

        // Parse theta dimensions
        if (theta_dims.size() == 1) {
            theta_len = theta_dims[0];
        } else if (theta_dims.size() == 2) {
            theta_batch_size = theta_dims[0];
            theta_len = theta_dims[1];
        } else {
            return ffi::Error::InvalidArgument("theta must be 1D or 2D array");
        }

        // Parse times dimensions
        if (times_dims.size() == 2) {
            n_times = times_dims[0];
            n_features = times_dims[1];
        } else if (times_dims.size() == 3) {
            times_batch_size = times_dims[0];
            n_times = times_dims[1];
            n_features = times_dims[2];
        } else {
            return ffi::Error::InvalidArgument("times must be 2D or 3D array");
        }

        // Parse rewards dimensions
        if (rewards_dims.size() == 2) {
            n_vertices = rewards_dims[0];
            n_features_rewards = rewards_dims[1];
        } else if (rewards_dims.size() == 3) {
            rewards_batch_size = rewards_dims[0];
            n_vertices = rewards_dims[1];
            n_features_rewards = rewards_dims[2];
        } else {
            return ffi::Error::InvalidArgument("rewards must be 2D or 3D array");
        }

        // Validate dimensions
        if (n_features != n_features_rewards) {
            return ffi::Error::InvalidArgument(
                "times and rewards must have same number of features (dimension 1/2)"
            );
        }

        if (n_vertices != static_cast<size_t>(builder->vertices_length())) {
            return ffi::Error::InvalidArgument(
                "rewards must have n_vertices rows"
            );
        }

        // Get raw data pointers
        const double* theta_data = theta.typed_data();
        const double* times_data = times.typed_data();
        const double* rewards_data = rewards.typed_data();
        double* result_data = result->typed_data();

        // Check if batched (from vmap)
        if (theta_batch_size > 1 || times_batch_size > 1 || rewards_batch_size > 1) {
            // BATCHED: Process multiple parameter combinations
            size_t batch_size = std::max({theta_batch_size, times_batch_size, rewards_batch_size});

            bool times_is_broadcast = (times_batch_size == 1 && batch_size > 1);
            bool rewards_is_broadcast = (rewards_batch_size == 1 && batch_size > 1);

            // Process each batch element in parallel using OpenMP
            #pragma omp parallel for if(batch_size > 1)
            for (size_t b = 0; b < batch_size; b++) {
                try {
                    // Build graph for this batch element
                    const double* theta_b = theta_data + (b * theta_len);
                    Graph g = builder->build(theta_b, theta_len);

                    // Get times/rewards for this batch (either indexed or broadcast)
                    const double* times_b = times_is_broadcast
                        ? times_data
                        : (times_data + (b * n_times * n_features));
                    const double* rewards_b = rewards_is_broadcast
                        ? rewards_data
                        : (rewards_data + (b * n_vertices * n_features));

                    // Get result pointer for this batch
                    double* result_b = result_data + (b * n_times * n_features);

                    // Process each feature dimension independently (sparse mode)
                    for (size_t j = 0; j < n_features; j++) {
                        // Extract reward vector for feature j
                        std::vector<double> rewards_vec(n_vertices);
                        for (size_t v = 0; v < n_vertices; v++) {
                            rewards_vec[v] = rewards_b[v * n_features + j];
                        }

                        // Transform graph with these rewards
                        Graph g_transformed = g.reward_transform(rewards_vec);

                        // Compute PDF for all time points in this feature
                        for (size_t i = 0; i < n_times; i++) {
                            double time_ij = times_b[i * n_features + j];

                            // Sparse mode: zero observation → zero PDF
                            if (time_ij == 0.0) {
                                result_b[i * n_features + j] = 0.0;
                                continue;
                            }

                            // Compute PDF/PMF
                            if (discrete) {
                                int jump_count = static_cast<int>(time_ij);
                                result_b[i * n_features + j] = g_transformed.dph_pmf(jump_count);
                            } else {
                                result_b[i * n_features + j] = g_transformed.pdf(time_ij, granularity);
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    // In parallel region, set error as NaN
                    double* result_b = result_data + (b * n_times * n_features);
                    for (size_t i = 0; i < n_times * n_features; i++) {
                        result_b[i] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        } else {
            // NOT BATCHED: theta shape (n_params,), times shape (n_times, n_features)
            Graph g = builder->build(theta_data, theta_len);

            // Process each feature dimension independently (sparse mode)
            for (size_t j = 0; j < n_features; j++) {
                // Extract reward vector for feature j (column j of rewards)
                std::vector<double> rewards_vec(n_vertices);
                for (size_t v = 0; v < n_vertices; v++) {
                    rewards_vec[v] = rewards_data[v * n_features + j];
                }

                // Transform graph with these rewards
                Graph g_transformed = g.reward_transform(rewards_vec);

                // Compute PDF for all time points in this feature
                for (size_t i = 0; i < n_times; i++) {
                    double time_ij = times_data[i * n_features + j];

                    // Sparse mode: zero observation → zero PDF
                    if (time_ij == 0.0) {
                        result_data[i * n_features + j] = 0.0;
                        continue;
                    }

                    // Compute PDF/PMF
                    if (discrete) {
                        int jump_count = static_cast<int>(time_ij);
                        result_data[i * n_features + j] = g_transformed.dph_pmf(jump_count);
                    } else {
                        result_data[i * n_features + j] = g_transformed.pdf(time_ij, granularity);
                    }
                }
            }
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        PTD_LOG_ERROR("ComputePmfMultivariateFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

// ===========================================================================
// ComputeSojournTimesFfiImpl: vmap-aware wrapper for expected_sojourn_time_subset
// ===========================================================================

/**
 * FFI handler for computing expected sojourn times for subset of vertices.
 *
 * Supports vmap batching with OpenMP parallelization and thread-local caching.
 *
 * Inputs:
 *   - structure_json: Graph structure (string_view attribute, STATIC)
 *   - theta: Parameters, shape (n_params,) OR (batch, n_params) with vmap
 *   - indices: Vertex indices (int32), shape (k,) OR (batch, k) OR (1, k) with vmap
 *
 * Output:
 *   - sojourn_times: Expected sojourn times, shape (k,) OR (batch, k)
 */
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

            // OpenMP parallel processing. Cap the thread count at the
            // batch size — there's no point spawning more workers than
            // elements to process. Also cap at 4 threads for memory-
            // bound work: each batch element holds a per-thread Graph
            // + symbolic compute graph (~30 MB for a 5k-vertex graph)
            // and runs a tight memory-bound consumer loop. Empirically
            // on Apple Silicon, fanning out beyond ~4 threads triggers
            // L2/L3 contention and saturates the LPDDR bus, so
            // additional threads add no parallelism but pay
            // synchronization barrier cost. The 4-thread cap matches
            // the perf-core sweet spot measured on M1 Pro.
            int sojourn_num_threads = 1;
#ifdef PHASIC_HAVE_OPENMP
            sojourn_num_threads = omp_get_max_threads();
            if (sojourn_num_threads > 4) sojourn_num_threads = 4;
            if (static_cast<int>(batch_size) < sojourn_num_threads)
                sojourn_num_threads = static_cast<int>(batch_size);
            if (sojourn_num_threads < 1) sojourn_num_threads = 1;
#endif
            #pragma omp parallel for if(batch_size > 1) num_threads(sojourn_num_threads)
            for (size_t b = 0; b < batch_size; b++) {
                const double* theta_b = theta_data + (b * theta_len);

                // Per-thread Graph reuse: build once on first use, then
                // mutate edge weights via update_weights on every later
                // batch element. The symbolic reward-compute graph
                // (parameterized_reward_compute_graph) is topology- and
                // coefficient-keyed, not theta-keyed, so it survives
                // update_weights, and the O(commands) replay path in
                // ptd_expected_sojourn_time_subset runs without an
                // O(n^3) symbolic rebuild on every batch element.
                // Mirrors the daisy-chain handler's reuse pattern.
                phasic::Graph* g_ptr = nullptr;
                {
                    auto& slot = per_thread_graph_cache[builder.get()];
                    if (!slot) {
                        std::vector<double> dummy_theta(theta_len, 1.0);
                        slot = std::make_unique<phasic::Graph>(
                            builder->build(dummy_theta.data(), theta_len)
                        );
                    }
                    g_ptr = slot.get();
                }
                phasic::Graph& g = *g_ptr;
                ptd_graph_update_weights(
                    g.c_graph(),
                    const_cast<double*>(theta_b),
                    theta_len,
                    /*use_log=*/false
                );

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
            // Not batched — same per-thread reuse as the batched branch.
            phasic::Graph* g_ptr = nullptr;
            {
                auto& slot = per_thread_graph_cache[builder.get()];
                if (!slot) {
                    std::vector<double> dummy_theta(theta_len, 1.0);
                    slot = std::make_unique<phasic::Graph>(
                        builder->build(dummy_theta.data(), theta_len)
                    );
                }
                g_ptr = slot.get();
            }
            phasic::Graph& g = *g_ptr;
            ptd_graph_update_weights(
                g.c_graph(),
                const_cast<double*>(theta_data),
                theta_len,
                /*use_log=*/false
            );

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
                    std::string("ptd_expected_sojourn_time_subset failed: ") + std::string((const char*)ptd_err)
                );
            }

            std::memcpy(result_data, sojourn_ptr, n_indices * sizeof(double));
            free(sojourn_ptr);
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        PTD_LOG_ERROR("ComputeSojournTimesFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

// ===========================================================================
// BackwardProbabilitiesFfiImpl
// ===========================================================================

ffi::Error BackwardProbabilitiesFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> target_vertices,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        std::string json_str(structure_json);

        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            builder = it->second;
        } else {
            builder = std::make_shared<GraphBuilder>(json_str);
            builder_cache[json_str] = builder;
        }

        auto theta_dims = theta.dimensions();
        size_t theta_len = theta_dims[theta_dims.size() - 1];
        auto target_dims = target_vertices.dimensions();
        size_t n_targets = target_dims[target_dims.size() - 1];

        const double* theta_data = theta.typed_data();
        const int32_t* target_data = target_vertices.typed_data();
        double* result_data = result->typed_data();

        // Build concrete graph with these theta values
        Graph g = builder->build(theta_data, theta_len);

        std::vector<size_t> targets(n_targets);
        for (size_t i = 0; i < n_targets; i++) {
            targets[i] = static_cast<size_t>(target_data[i]);
        }

        double* h = ptd_backward_probabilities(g.c_graph(), targets.data(), n_targets);
        size_t n_verts = g.c_graph()->vertices_length;
        for (size_t i = 0; i < n_verts; i++) {
            result_data[i] = h[i];
        }
        free(h);

        return ffi::Error::Success();
    } catch (const std::exception& e) {
        return ffi::Error::Internal(
            std::string("BackwardProbabilitiesFfiImpl error: ") + e.what()
        );
    }
}

// ===========================================================================
// SamplePathConditionedFfiImpl
// ===========================================================================

ffi::Error SamplePathConditionedFfiImpl(
    std::string_view structure_json,
    int32_t max_length,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::S32> target_vertex,
    ffi::Buffer<ffi::S32> seed,
    ffi::ResultBuffer<ffi::S32> out_indices,
    ffi::ResultBuffer<ffi::F64> out_times
) {
    try {
        std::string json_str(structure_json);

        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            builder = it->second;
        } else {
            builder = std::make_shared<GraphBuilder>(json_str);
            builder_cache[json_str] = builder;
        }

        // Parse dimensions for vmap batching
        auto theta_dims = theta.dimensions();
        auto target_dims = target_vertex.dimensions();
        auto seed_dims = seed.dimensions();

        size_t theta_len = theta_dims[theta_dims.size() - 1];
        size_t ml = static_cast<size_t>(max_length);

        const double* theta_data = theta.typed_data();
        const int32_t* target_data = target_vertex.typed_data();
        const int32_t* seed_data = seed.typed_data();
        int32_t* indices_out = out_indices->typed_data();
        double* times_out = out_times->typed_data();

        // Determine batch sizes per buffer (handle broadcasting)
        size_t theta_batch = (theta_dims.size() == 2) ? theta_dims[0] : 1;
        size_t target_batch = (target_dims.size() == 2) ? target_dims[0] : 1;
        size_t seed_batch = (seed_dims.size() == 2) ? seed_dims[0] : 1;
        size_t batch_size = std::max({theta_batch, target_batch, seed_batch});

        if (batch_size > 1) {
            bool theta_is_broadcast = (theta_batch <= 1);

            if (theta_is_broadcast) {
                // Optimized: build graph ONCE, cache backward probs per unique target
                Graph g = builder->build(theta_data, theta_len);

                // Precompute backward probs for each unique target vertex
                std::unordered_map<int32_t, double*> bp_cache;
                for (size_t b = 0; b < batch_size; b++) {
                    int32_t tv = target_data[target_batch > 1 ? b : 0];
                    if (bp_cache.find(tv) == bp_cache.end()) {
                        size_t target_sz = static_cast<size_t>(tv);
                        bp_cache[tv] = ptd_backward_probabilities(
                            g.c_graph(), &target_sz, 1
                        );
                    }
                }

                // Sample all paths in parallel (graph and bp_cache are read-only)
                #pragma omp parallel for if(batch_size > 4)
                for (size_t b = 0; b < batch_size; b++) {
                    int32_t tv = target_data[target_batch > 1 ? b : 0];
                    unsigned int rng_seed = static_cast<unsigned int>(
                        seed_data[seed_batch > 1 ? b : 0]
                    );
                    ptd_random_sample_path_conditioned_fixed(
                        g.c_graph(), bp_cache[tv], ml, rng_seed,
                        indices_out + b * ml, times_out + b * ml
                    );
                }

                // Free cached backward probs
                for (auto& [key, ptr] : bp_cache) {
                    free(ptr);
                }
            } else {
                // Theta varies per batch element: build graph per element
                #pragma omp parallel for if(batch_size > 4)
                for (size_t b = 0; b < batch_size; b++) {
                    std::shared_ptr<GraphBuilder> local_builder;
                    auto local_it = builder_cache.find(json_str);
                    if (local_it != builder_cache.end()) {
                        local_builder = local_it->second;
                    } else {
                        local_builder = std::make_shared<GraphBuilder>(json_str);
                        builder_cache[json_str] = local_builder;
                    }

                    const double* theta_b = theta_data + b * theta_len;
                    int32_t tv = target_data[target_batch > 1 ? b : 0];
                    unsigned int rng_seed = static_cast<unsigned int>(
                        seed_data[seed_batch > 1 ? b : 0]
                    );

                    Graph g = local_builder->build(theta_b, theta_len);
                    size_t target_sz = static_cast<size_t>(tv);
                    double* h = ptd_backward_probabilities(
                        g.c_graph(), &target_sz, 1
                    );

                    ptd_random_sample_path_conditioned_fixed(
                        g.c_graph(), h, ml, rng_seed,
                        indices_out + b * ml, times_out + b * ml
                    );
                    free(h);
                }
            }
        } else {
            // Single sample (no batch)
            int32_t target_v = target_data[0];
            unsigned int rng_seed = static_cast<unsigned int>(seed_data[0]);

            Graph g = builder->build(theta_data, theta_len);
            size_t target_sz = static_cast<size_t>(target_v);
            double* h = ptd_backward_probabilities(g.c_graph(), &target_sz, 1);

            ptd_random_sample_path_conditioned_fixed(
                g.c_graph(), h, ml, rng_seed, indices_out, times_out
            );
            free(h);
        }

        return ffi::Error::Success();
    } catch (const std::exception& e) {
        return ffi::Error::Internal(
            std::string("SamplePathConditionedFfiImpl error: ") + e.what()
        );
    }
}

// ===========================================================================
// DaisyChainJointProbsFfiImpl: full daisy-chain joint-probs in C
// ===========================================================================
//
// Performs the full daisy chain end-to-end inside the FFI handler so the
// SVGD loop crosses the Python↔C boundary exactly once per forward (matching
// the vanilla joint-prob path's perf characteristics). Mirrors the structural
// pattern of ComputeSojournTimesFfiImpl: thread-local builder cache, OpenMP-
// parallel vmap loop, fresh Graph per batch element (each batch element
// receives different theta/IPV so a per-epoch persistent graph would buy
// nothing here).
//
// Daisy-chain metadata (epoch_dts, t_eval, n_epochs, ipv_target_indices,
// t_aux_keys, t_aux_values, t_vertex_indices) is encoded into the
// structure_json under a top-level "_daisy_chain" object. GraphBuilder
// silently ignores unknown JSON fields, so the same JSON is reused for both
// graph construction and metadata extraction.

ffi::Error DaisyChainJointProbsFfiImpl(
    std::string_view structure_json,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> initial_ipv,
    ffi::ResultBuffer<ffi::F64> result
) {
    try {
        ensure_omp_full_width_once();

        std::string json_str(structure_json);

        // Thread-local GraphBuilder cache lookup (mirrors
        // ComputeSojournTimesFfiImpl).
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

        // Look up cached daisy-chain metadata (parsed JSON + topology-
        // only lookup tables: aux_set, t_to_aux, collapsed_pos). On
        // first call for this builder, parse the JSON, build a
        // throwaway Graph to read n_vertices, then populate the cache.
        std::shared_ptr<const DaisyChainMeta> meta;
        {
            std::lock_guard<std::mutex> lock(daisy_chain_meta_init_mutex);
            auto meta_it = daisy_chain_meta_cache.find(json_str);
            if (meta_it != daisy_chain_meta_cache.end()) {
                meta = meta_it->second;
            } else {
                auto m = std::make_shared<DaisyChainMeta>();
                nlohmann::json j;
                try {
                    j = nlohmann::json::parse(json_str);
                } catch (const std::exception& e) {
                    return ffi::Error::InvalidArgument(
                        std::string("Failed to re-parse JSON for daisy-chain metadata: ") + e.what()
                    );
                }
                if (!j.contains("_daisy_chain")) {
                    return ffi::Error::InvalidArgument(
                        "structure_json must contain a top-level \"_daisy_chain\" "
                        "object with daisy-chain metadata."
                    );
                }
                const auto& dc = j["_daisy_chain"];
                m->n_epochs            = dc.at("n_epochs").get<int>();
                m->param_length        = dc.at("param_length").get<int>();
                m->t_eval              = dc.at("t_eval").get<double>();
                // Granularity is optional for backwards compatibility — older
                // callers built JSON without this field. Default 0 = auto.
                m->granularity         = dc.contains("granularity")
                                         ? dc.at("granularity").get<int>() : 0;
                m->epoch_dts           = dc.at("epoch_dts").get<std::vector<double>>();
                m->ipv_target_indices  = dc.at("ipv_target_indices").get<std::vector<int>>();
                m->t_aux_keys          = dc.at("t_aux_keys").get<std::vector<int>>();
                m->t_aux_values        = dc.at("t_aux_values").get<std::vector<int>>();
                m->t_vertex_indices    = dc.at("t_vertex_indices").get<std::vector<int>>();

                if (m->n_epochs < 1) {
                    return ffi::Error::InvalidArgument("n_epochs must be >= 1");
                }
                if (static_cast<int>(m->epoch_dts.size()) != m->n_epochs - 1) {
                    return ffi::Error::InvalidArgument(
                        "epoch_dts must have length n_epochs - 1"
                    );
                }
                if (m->t_aux_keys.size() != m->t_aux_values.size()) {
                    return ffi::Error::InvalidArgument(
                        "_t_aux_map keys and values must have the same length"
                    );
                }

                // aux_set + t_to_aux are pure topology — independent of
                // graph instance. Compute now from the parsed JSON.
                m->aux_set.reserve(m->t_aux_values.size());
                for (int v : m->t_aux_values) m->aux_set.insert(v);
                m->t_to_aux.reserve(m->t_aux_keys.size());
                for (size_t k = 0; k < m->t_aux_keys.size(); ++k) {
                    m->t_to_aux[m->t_aux_keys[k]] = m->t_aux_values[k];
                }

                // collapsed_pos requires n_vertices, which is only
                // available from a built Graph. Build a single
                // throwaway here (small one-time cost; subsequent
                // calls reuse the cached metadata).
                std::vector<double> dummy_theta(m->param_length, 1.0);
                phasic::Graph probe = builder->build(
                    dummy_theta.data(), static_cast<size_t>(m->param_length)
                );
                m->n_vertices = probe.vertices_length();
                m->collapsed_pos.assign(m->n_vertices, -1);
                int rank = 0;
                for (size_t i = 0; i < m->n_vertices; ++i) {
                    if (m->aux_set.count(static_cast<int>(i))) continue;
                    m->collapsed_pos[i] = rank++;
                }
                m->n_collapsed = static_cast<size_t>(rank);

                daisy_chain_meta_cache[json_str] = m;
                meta = m;
            }
        }

        const int n_epochs       = meta->n_epochs;
        const int param_length   = meta->param_length;
        const double t_eval      = meta->t_eval;
        const int granularity    = meta->granularity;
        const auto& epoch_dts          = meta->epoch_dts;
        const auto& ipv_target_indices = meta->ipv_target_indices;
        const auto& t_vertex_indices   = meta->t_vertex_indices;
        const auto& aux_set            = meta->aux_set;
        const auto& t_to_aux           = meta->t_to_aux;
        const auto& collapsed_pos      = meta->collapsed_pos;
        const size_t n_vertices  = meta->n_vertices;
        const size_t n_collapsed = meta->n_collapsed;
        const size_t n_ipv       = ipv_target_indices.size();
        const size_t n_t         = t_vertex_indices.size();

        // Parse dimensions (handle vmap batching; mirrors
        // ComputeSojournTimesFfiImpl).
        auto theta_dims = theta.dimensions();
        auto ipv_dims = initial_ipv.dimensions();

        size_t theta_len, ipv_len;
        size_t theta_batch_size = 1;
        size_t ipv_batch_size = 1;

        if (theta_dims.size() == 1) {
            theta_len = theta_dims[0];
        } else if (theta_dims.size() == 2) {
            theta_batch_size = theta_dims[0];
            theta_len = theta_dims[1];
        } else {
            return ffi::Error::InvalidArgument("theta must be 1D or 2D");
        }
        if (ipv_dims.size() == 1) {
            ipv_len = ipv_dims[0];
        } else if (ipv_dims.size() == 2) {
            ipv_batch_size = ipv_dims[0];
            ipv_len = ipv_dims[1];
        } else {
            return ffi::Error::InvalidArgument("initial_ipv must be 1D or 2D");
        }

        if (theta_len != static_cast<size_t>(n_epochs * param_length)) {
            return ffi::Error::InvalidArgument(
                "theta length must equal n_epochs * param_length"
            );
        }
        if (ipv_len != n_ipv) {
            return ffi::Error::InvalidArgument(
                "initial_ipv length must equal len(ipv_target_indices)"
            );
        }

        const double* theta_data = theta.typed_data();
        const double* ipv_data = initial_ipv.typed_data();
        double* result_data = result->typed_data();

        const size_t batch_size = std::max(theta_batch_size, ipv_batch_size);
        if (theta_batch_size > 1 && ipv_batch_size > 1
            && theta_batch_size != ipv_batch_size) {
            return ffi::Error::InvalidArgument(
                "theta and initial_ipv batch sizes must match (or one must be 1)"
            );
        }

#ifdef PHASIC_HAVE_OPENMP
        PTD_LOG_DEBUG(
            "DaisyChainJointProbsFfiImpl: batch_size=%zu, "
            "omp_get_max_threads()=%d",
            batch_size, omp_get_max_threads()
        );
#else
        PTD_LOG_DEBUG(
            "DaisyChainJointProbsFfiImpl: batch_size=%zu, OpenMP disabled",
            batch_size
        );
#endif

        // Per-batch daisy chain. OpenMP parallelises over batch
        // elements; each thread reuses a persistent Graph across
        // batch elements via update_weights (no per-iter alloc).
        #pragma omp parallel for if(batch_size > 1)
        for (size_t b = 0; b < batch_size; ++b) {
            const double* theta_b =
                (theta_batch_size > 1) ? theta_data + b * theta_len : theta_data;
            const double* ipv_b_in =
                (ipv_batch_size > 1) ? ipv_data + b * ipv_len : ipv_data;
            double* result_b = result_data + b * n_t;

            // Per-thread Graph reuse: each OMP thread builds its
            // Graph once on first use, then reuses via update_weights
            // (in-place edge-weight mutation, zero alloc) and
            // update_ipv on every subsequent batch element. The
            // initial build uses dummy theta — every batch element
            // overwrites via update_weights inside the per-epoch
            // loop, so initial values are irrelevant beyond
            // satisfying the length check.
            phasic::Graph* g_ptr = nullptr;
            {
                auto& slot = per_thread_graph_cache[builder.get()];
                if (!slot) {
                    std::vector<double> dummy_theta(param_length, 1.0);
                    slot = std::make_unique<phasic::Graph>(
                        builder->build(
                            dummy_theta.data(),
                            static_cast<size_t>(param_length)
                        )
                    );
                }
                g_ptr = slot.get();
            }
            phasic::Graph& g = *g_ptr;

            // Working IPV (length n_ipv) — initialised to user's
            // initial_ipv, then overwritten between epochs.
            std::vector<double> ipv_work(ipv_b_in, ipv_b_in + n_ipv);

            // Iterate epochs 0 .. n_epochs - 1.
            for (int epoch = 0; epoch < n_epochs; ++epoch) {
                const double* theta_epoch = theta_b + epoch * param_length;

                // Set IPV and theta on the graph. Both update_ipv and
                // update_weights bump weight_version, so the cached
                // ph_context_markov gets rebuilt on the next
                // stop_probability call.
                ptd_graph_update_ipv(
                    g.c_graph(), ipv_work.data(), n_ipv
                );
                if (ptd_err[0] != '\0') {
                    PTD_LOG_ERROR(
                        "DaisyChainJointProbsFfiImpl: ptd_graph_update_ipv "
                        "failed at epoch %d: %s", epoch, (const char*)ptd_err
                    );
                    ptd_err[0] = '\0';
                    for (size_t k = 0; k < n_t; ++k) {
                        result_b[k] = std::numeric_limits<double>::quiet_NaN();
                    }
                    goto next_batch;
                }

                ptd_graph_update_weights(
                    g.c_graph(),
                    const_cast<double*>(theta_epoch),
                    static_cast<size_t>(param_length),
                    /*use_log=*/false
                );
                if (ptd_err[0] != '\0') {
                    PTD_LOG_ERROR(
                        "DaisyChainJointProbsFfiImpl: ptd_graph_update_weights "
                        "failed at epoch %d: %s", epoch, (const char*)ptd_err
                    );
                    ptd_err[0] = '\0';
                    for (size_t k = 0; k < n_t; ++k) {
                        result_b[k] = std::numeric_limits<double>::quiet_NaN();
                    }
                    goto next_batch;
                }

                // Decide which time to evaluate stop_probability at.
                const double t_step =
                    (epoch < n_epochs - 1) ? epoch_dts[epoch] : t_eval;

                std::vector<double> raw;
                try {
                    raw = g.stop_probability(t_step, granularity);
                } catch (const std::exception& e) {
                    PTD_LOG_ERROR(
                        "DaisyChainJointProbsFfiImpl: stop_probability "
                        "failed at epoch %d (t=%g): %s",
                        epoch, t_step, e.what()
                    );
                    for (size_t k = 0; k < n_t; ++k) {
                        result_b[k] = std::numeric_limits<double>::quiet_NaN();
                    }
                    goto next_batch;
                }
                if (raw.size() != n_vertices) {
                    PTD_LOG_ERROR(
                        "DaisyChainJointProbsFfiImpl: stop_probability "
                        "returned size %zu, expected %zu",
                        raw.size(), n_vertices
                    );
                    for (size_t k = 0; k < n_t; ++k) {
                        result_b[k] = std::numeric_limits<double>::quiet_NaN();
                    }
                    goto next_batch;
                }

                // Collapse t-aux pairs into (n_collapsed,) vector.
                // collapsed[collapsed_pos[v]] = raw[v] +
                //     raw[t_to_aux[v]] (if v is a t-vertex).
                std::vector<double> collapsed(n_collapsed, 0.0);
                for (size_t v = 0; v < n_vertices; ++v) {
                    if (aux_set.count(static_cast<int>(v))) continue;
                    double p = raw[v];
                    auto it_aux = t_to_aux.find(static_cast<int>(v));
                    if (it_aux != t_to_aux.end()) {
                        p += raw[static_cast<size_t>(it_aux->second)];
                    }
                    collapsed[collapsed_pos[v]] = p;
                }

                if (epoch < n_epochs - 1) {
                    // Project collapsed → next-epoch IPV by indexing at
                    // ipv_target_indices' collapsed positions.
                    for (size_t k = 0; k < n_ipv; ++k) {
                        int v_idx = ipv_target_indices[k];
                        ipv_work[k] = collapsed[collapsed_pos[v_idx]];
                    }
                } else {
                    // Final epoch: write joint-probs at t-vertex
                    // collapsed positions to the result buffer.
                    for (size_t k = 0; k < n_t; ++k) {
                        int v_idx = t_vertex_indices[k];
                        result_b[k] = collapsed[collapsed_pos[v_idx]];
                    }
                }
            }
            continue;
            next_batch:;  // jump target on per-batch error
        }

        return ffi::Error::Success();
    } catch (const std::exception& e) {
        PTD_LOG_ERROR("DaisyChainJointProbsFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

// ===========================================================================
// DaisyChainSojournFfiImpl: daisy chain with a granularity-free FINAL-epoch read.
//
// Intermediate epochs (0 .. n_epochs-2) use the JSP stop_probability handoff
// (identical to DaisyChainJointProbsFfiImpl). The FINAL epoch runs to
// absorption (t -> inf), where the joint probability is exactly
//   joint_prob[t-state v] = r_v * expected_sojourn(v) * handoff_mass
// read off the NO-TRAPPING "sojourn" graph via graph elimination -- exact,
// granularity-free, and with no finite-t_eval truncation. Avoids the dominant
// long-time stop_probability(t_eval) forward solve while keeping OpenMP
// particle-batching. (r_v = the t-vertex exit rate = the mutation slot
// theta_final[param_length-1]; handoff_mass = sum of the final-epoch handoff
// IPV, which expected_sojourn normalises to unit mass.)
// ===========================================================================
ffi::Error DaisyChainSojournFfiImpl(
    std::string_view structure_json,          // JSP graph + _daisy_chain (+ sojourn fields)
    std::string_view sojourn_structure_json,  // no-trapping sojourn graph
    ffi::Buffer<ffi::F64> theta,              // (n_epochs*param_length,) or (B, ...)
    ffi::Buffer<ffi::F64> initial_ipv,        // (n_ipv,) or (B, n_ipv)
    ffi::ResultBuffer<ffi::F64> result        // (n_t,) or (B, n_t)
) {
    try {
        ensure_omp_full_width_once();
        std::string jsp_json(structure_json);
        std::string soj_json(sojourn_structure_json);

        // Builders (thread-local cache, keyed by JSON string).
        std::shared_ptr<GraphBuilder> jsp_builder, soj_builder;
        {
            auto it = builder_cache.find(jsp_json);
            if (it != builder_cache.end()) jsp_builder = it->second;
            else {
                try { jsp_builder = std::make_shared<GraphBuilder>(jsp_json); builder_cache[jsp_json] = jsp_builder; }
                catch (const std::exception& e) { return ffi::Error::InvalidArgument(std::string("Failed to parse JSP JSON: ") + e.what()); }
            }
            auto it2 = builder_cache.find(soj_json);
            if (it2 != builder_cache.end()) soj_builder = it2->second;
            else {
                try { soj_builder = std::make_shared<GraphBuilder>(soj_json); builder_cache[soj_json] = soj_builder; }
                catch (const std::exception& e) { return ffi::Error::InvalidArgument(std::string("Failed to parse sojourn JSON: ") + e.what()); }
            }
        }

        // Parse metadata from the JSP JSON's _daisy_chain block.
        nlohmann::json j;
        try { j = nlohmann::json::parse(jsp_json); }
        catch (const std::exception& e) { return ffi::Error::InvalidArgument(std::string("re-parse JSP JSON: ") + e.what()); }
        if (!j.contains("_daisy_chain"))
            return ffi::Error::InvalidArgument("structure_json must contain a _daisy_chain object");
        const auto& dc = j["_daisy_chain"];
        const int n_epochs     = dc.at("n_epochs").get<int>();
        const int param_length = dc.at("param_length").get<int>();
        const int granularity  = dc.contains("granularity") ? dc.at("granularity").get<int>() : 0;
        const auto epoch_dts          = dc.at("epoch_dts").get<std::vector<double>>();
        const auto ipv_target_indices = dc.at("ipv_target_indices").get<std::vector<int>>();
        const auto t_aux_keys         = dc.at("t_aux_keys").get<std::vector<int>>();
        const auto t_aux_values       = dc.at("t_aux_values").get<std::vector<int>>();
        const auto sojourn_jsp_gather = dc.at("sojourn_jsp_gather").get<std::vector<int>>();
        const auto sojourn_t_indices  = dc.at("sojourn_t_indices").get<std::vector<int>>();

        if (n_epochs < 1) return ffi::Error::InvalidArgument("n_epochs must be >= 1");
        if (static_cast<int>(epoch_dts.size()) != n_epochs - 1)
            return ffi::Error::InvalidArgument("epoch_dts must have length n_epochs - 1");

        const size_t n_ipv         = ipv_target_indices.size();
        const size_t sojourn_n_ipv = sojourn_jsp_gather.size();
        const size_t n_t           = sojourn_t_indices.size();

        std::unordered_set<int> aux_set(t_aux_values.begin(), t_aux_values.end());
        std::unordered_map<int,int> t_to_aux;
        for (size_t k = 0; k < t_aux_keys.size(); ++k) t_to_aux[t_aux_keys[k]] = t_aux_values[k];

        // JSP n_vertices + collapsed_pos (one probe build for topology).
        size_t jsp_n_vertices;
        {
            std::vector<double> dummy(param_length, 1.0);
            phasic::Graph probe = jsp_builder->build(dummy.data(), static_cast<size_t>(param_length));
            jsp_n_vertices = probe.vertices_length();
        }
        std::vector<int> collapsed_pos(jsp_n_vertices, -1);
        { int rank = 0; for (size_t i = 0; i < jsp_n_vertices; ++i) { if (aux_set.count(static_cast<int>(i))) continue; collapsed_pos[i] = rank++; } }

        // Dimensions (vmap batching).
        auto theta_dims = theta.dimensions();
        auto ipv_dims = initial_ipv.dimensions();
        size_t theta_len, ipv_len, theta_bs = 1, ipv_bs = 1;
        if (theta_dims.size() == 1) theta_len = theta_dims[0];
        else if (theta_dims.size() == 2) { theta_bs = theta_dims[0]; theta_len = theta_dims[1]; }
        else return ffi::Error::InvalidArgument("theta must be 1D or 2D");
        if (ipv_dims.size() == 1) ipv_len = ipv_dims[0];
        else if (ipv_dims.size() == 2) { ipv_bs = ipv_dims[0]; ipv_len = ipv_dims[1]; }
        else return ffi::Error::InvalidArgument("initial_ipv must be 1D or 2D");
        if (theta_len != static_cast<size_t>(n_epochs * param_length))
            return ffi::Error::InvalidArgument("theta length must equal n_epochs * param_length");
        if (ipv_len != n_ipv)
            return ffi::Error::InvalidArgument("initial_ipv length must equal len(ipv_target_indices)");

        const double* theta_data = theta.typed_data();
        const double* ipv_data = initial_ipv.typed_data();
        double* result_data = result->typed_data();
        const size_t batch_size = std::max(theta_bs, ipv_bs);

        #pragma omp parallel for if(batch_size > 1)
        for (size_t b = 0; b < batch_size; ++b) {
            const double* theta_b = (theta_bs > 1) ? theta_data + b * theta_len : theta_data;
            const double* ipv_b_in = (ipv_bs > 1) ? ipv_data + b * ipv_len : ipv_data;
            double* result_b = result_data + b * n_t;

            phasic::Graph* gj = nullptr;
            phasic::Graph* gs = nullptr;
            {
                auto& slot = per_thread_graph_cache[jsp_builder.get()];
                if (!slot) { std::vector<double> dt(param_length, 1.0); slot = std::make_unique<phasic::Graph>(jsp_builder->build(dt.data(), static_cast<size_t>(param_length))); }
                gj = slot.get();
            }
            {
                auto& slot = per_thread_graph_cache[soj_builder.get()];
                if (!slot) { std::vector<double> dt(param_length, 1.0); slot = std::make_unique<phasic::Graph>(soj_builder->build(dt.data(), static_cast<size_t>(param_length))); }
                gs = slot.get();
            }

            std::vector<double> ipv_work(ipv_b_in, ipv_b_in + n_ipv);
            bool failed = false;

            // Intermediate epochs on the JSP graph (stop_probability handoff).
            for (int epoch = 0; epoch < n_epochs - 1; ++epoch) {
                const double* theta_e = theta_b + epoch * param_length;
                ptd_graph_update_ipv(gj->c_graph(), ipv_work.data(), n_ipv);
                ptd_graph_update_weights(gj->c_graph(), const_cast<double*>(theta_e), static_cast<size_t>(param_length), false);
                std::vector<double> raw;
                try { raw = gj->stop_probability(epoch_dts[epoch], granularity); }
                catch (const std::exception& e) { failed = true; break; }
                if (raw.size() != jsp_n_vertices) { failed = true; break; }
                std::vector<double> collapsed(jsp_n_vertices, 0.0);
                for (size_t v = 0; v < jsp_n_vertices; ++v) {
                    if (aux_set.count(static_cast<int>(v))) continue;
                    double p = raw[v];
                    auto it = t_to_aux.find(static_cast<int>(v));
                    if (it != t_to_aux.end()) p += raw[static_cast<size_t>(it->second)];
                    collapsed[collapsed_pos[v]] = p;
                }
                for (size_t k = 0; k < n_ipv; ++k)
                    ipv_work[k] = collapsed[collapsed_pos[ipv_target_indices[k]]];
            }

            if (failed) {
                for (size_t k = 0; k < n_t; ++k) result_b[k] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // Final epoch: granularity-free sojourn read on the no-trapping graph.
            const double* theta_final = theta_b + (n_epochs - 1) * param_length;
            std::vector<double> sojourn_ipv(sojourn_n_ipv, 0.0);
            double handoff_mass = 0.0;
            for (size_t k = 0; k < sojourn_n_ipv; ++k) {
                double m = ipv_work[static_cast<size_t>(sojourn_jsp_gather[k])];
                sojourn_ipv[k] = m;
                handoff_mass += m;
            }
            ptd_graph_update_ipv(gs->c_graph(), sojourn_ipv.data(), sojourn_n_ipv);
            ptd_graph_update_weights(gs->c_graph(), const_cast<double*>(theta_final), static_cast<size_t>(param_length), false);
            std::vector<size_t> idx(n_t);
            for (size_t k = 0; k < n_t; ++k) idx[k] = static_cast<size_t>(sojourn_t_indices[k]);
            double* soj = ptd_expected_sojourn_time_subset(gs->c_graph(), idx.data(), n_t);
            if (soj == NULL) {
                for (size_t k = 0; k < n_t; ++k) result_b[k] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            // r_v = the t-vertex's total exit rate to absorbing vertices, read
            // per-vertex from the live (updated) graph. No assumption that the
            // exit rate equals any particular theta slot, so this is correct for
            // any joint_prob_graph construction, not just the standard one where
            // the exit edge is the mutation-slot coefficient.
            struct ptd_graph* cg = gs->c_graph();
            for (size_t k = 0; k < n_t; ++k) {
                struct ptd_vertex* tv = cg->vertices[idx[k]];
                double r_v = 0.0;
                for (size_t e = 0; e < tv->edges_length; ++e) {
                    struct ptd_edge* ed = tv->edges[e];
                    if (ed->to->edges_length == 0) r_v += ed->weight;
                }
                result_b[k] = r_v * soj[k] * handoff_mass;
            }
            free(soj);
        }

        return ffi::Error::Success();
    } catch (const std::exception& e) {
        PTD_LOG_ERROR("DaisyChainSojournFfiImpl exception: %s", e.what());
        return ffi::Error::Internal(e.what());
    }
}

} // namespace ffi_handlers

// Export binding creation functions for Python-side FFI registration
// These create the handler functions on-demand when called from Python
// Following the pattern from XLA_FFI_DEFINE_HANDLER in api.h
XLA_FFI_Handler* CreateComputePmfHandler() {
    // Create a static function pointer using the pattern from XLA_FFI_DEFINE_HANDLER
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // JSON as STATIC attribute (not batched)
            .Attr<int32_t>("granularity")
            .Attr<bool>("discrete")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta (batched by vmap)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // times (batched by vmap)
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // result
            .To(ffi_handlers::ComputePmfFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateComputeMomentsHandler() {
    // Create a static function pointer using the pattern from XLA_FFI_DEFINE_HANDLER
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // JSON as STATIC attribute (not batched)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta (batched by vmap)
            .Attr<int32_t>("nr_moments")
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // moments
            .To(ffi_handlers::ComputeMomentsFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateComputePmfAndMomentsHandler() {
    // Create a static function pointer using the pattern from XLA_FFI_DEFINE_HANDLER
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // JSON as STATIC attribute (not batched)
            .Attr<int32_t>("nr_moments")
            .Attr<int32_t>("granularity")
            .Attr<bool>("discrete")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // theta (batched by vmap)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // times (batched by vmap)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // rewards (batched by vmap)
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // pmf_result
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // moments_result
            .To(ffi_handlers::ComputePmfAndMomentsFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateComputePmfMultivariateHandler() {
    // Create a static function pointer using the pattern from XLA_FFI_DEFINE_HANDLER
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // JSON as STATIC attribute (not batched)
            .Attr<int32_t>("granularity")
            .Attr<bool>("discrete")
            .Attr<bool>("compute_joint")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // theta (batched by vmap)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // times (batched by vmap, 2D or 3D)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // rewards (batched by vmap, 2D or 3D)
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // result (2D or 3D)
            .To(ffi_handlers::ComputePmfMultivariateFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateComputeSojournTimesHandler() {
    // Create a static function pointer using the pattern from XLA_FFI_DEFINE_HANDLER
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // JSON as STATIC attribute (not batched)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta (batched by vmap)
            .Arg<xla::ffi::Buffer<xla::ffi::S32>>()    // indices (int32, batched by vmap)
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // result (sojourn times)
            .To(ffi_handlers::ComputeSojournTimesFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateBackwardProbabilitiesHandler() {
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta
            .Arg<xla::ffi::Buffer<xla::ffi::S32>>()    // target_vertices
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // backward_probs
            .To(ffi_handlers::BackwardProbabilitiesFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateSamplePathConditionedHandler() {
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")
            .Attr<int32_t>("max_length")
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta
            .Arg<xla::ffi::Buffer<xla::ffi::S32>>()    // target_vertex
            .Arg<xla::ffi::Buffer<xla::ffi::S32>>()    // seed
            .Ret<xla::ffi::Buffer<xla::ffi::S32>>()    // vertex_indices
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // entry_times
            .To(ffi_handlers::SamplePathConditionedFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateDaisyChainJointProbsHandler() {
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")  // graph + _daisy_chain metadata (STATIC)
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // theta (n_epochs * param_length,) - BATCHED
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()    // initial_ipv (n_ipv,) - BATCHED
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()    // result (n_t_vertices,)
            .To(ffi_handlers::DaisyChainJointProbsFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

XLA_FFI_Handler* CreateDaisyChainSojournHandler() {
    static constexpr XLA_FFI_Handler* handler = +[](XLA_FFI_CallFrame* call_frame) {
        static auto* bound_handler = xla::ffi::Ffi::Bind()
            .Attr<std::string_view>("structure_json")          // JSP graph + _daisy_chain (+ sojourn fields) STATIC
            .Attr<std::string_view>("sojourn_structure_json")  // no-trapping sojourn graph STATIC
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()            // theta (n_epochs * param_length,) - BATCHED
            .Arg<xla::ffi::Buffer<xla::ffi::F64>>()            // initial_ipv (n_ipv,) - BATCHED
            .Ret<xla::ffi::Buffer<xla::ffi::F64>>()            // result (n_t_vertices,)
            .To(ffi_handlers::DaisyChainSojournFfiImpl)
            .release();
        return bound_handler->Call(call_frame);
    };
    return handler;
}

} // namespace parameterized
} // namespace phasic

// IMPORTANT: Do NOT register FFI handlers as global symbols!
// The XLA_FFI_DEFINE_HANDLER_SYMBOL macro creates static global objects that get
// constructed during library load, BEFORE JAX is initialized. This corrupts XLA's
// FFI registry and causes bus errors when JAX tries to allocate memory.
//
// Instead, FFI handlers should be registered explicitly from Python after JAX is
// fully initialized, using jax.extend.ffi.register_ffi_target().
//
// If you need auto-registration, move these to a separate shared library that's
// only loaded after JAX initialization.

// // DISABLED - causes memory corruption when library loads before JAX init
// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     PtdComputePmf, phasic::parameterized::ffi_handlers::ComputePmfFfiImpl,
//     xla::ffi::Ffi::Bind()
//         .Arg<xla::ffi::Buffer<xla::ffi::U8>>()   // structure_json
//         .Attr<int32_t>("granularity")
//         .Attr<bool>("discrete")
//         .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // theta
//         .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // times
//         .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // result
// );
//
// XLA_FFI_DEFINE_HANDLER_SYMBOL(
//     PtdComputePmfAndMoments, phasic::parameterized::ffi_handlers::ComputePmfAndMomentsFfiImpl,
//     xla::ffi::Ffi::Bind()
//         .Arg<xla::ffi::Buffer<xla::ffi::U8>>()   // structure_json
//         .Attr<int32_t>("granularity")
//         .Attr<bool>("discrete")
//         .Attr<int32_t>("nr_moments")
//         .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // theta
//         .Arg<xla::ffi::Buffer<xla::ffi::F64>>()  // times
//         .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // pmf_result
//         .Ret<xla::ffi::Buffer<xla::ffi::F64>>()  // moments_result
// );
