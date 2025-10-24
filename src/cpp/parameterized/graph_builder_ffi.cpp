#include "graph_builder_ffi.hpp"
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <memory>
#include <iostream>
#include <limits>

namespace phasic {
namespace parameterized {
namespace ffi_handlers {

namespace ffi = xla::ffi;

// Thread-local cache for GraphBuilder instances
// Key: JSON string, Value: GraphBuilder instance
thread_local std::unordered_map<std::string, std::shared_ptr<GraphBuilder>> builder_cache;

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
        std::cerr << "❌ Exception caught: " << e.what() << std::endl;
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
