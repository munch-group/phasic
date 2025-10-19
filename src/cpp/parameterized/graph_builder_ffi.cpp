#include "graph_builder_ffi.hpp"
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <memory>
#include <iostream>

namespace ptdalgorithms {
namespace parameterized {
namespace ffi_handlers {

namespace ffi = xla::ffi;

// Thread-local cache for GraphBuilder instances
// Key: JSON string, Value: GraphBuilder instance
thread_local std::unordered_map<std::string, std::shared_ptr<GraphBuilder>> builder_cache;

ffi::Error ComputePmfFfiImpl(
    ffi::Buffer<ffi::U8> structure_json,
    int32_t granularity,
    bool discrete,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> result
) {
    std::cerr << "\n=== FFI Handler Called ===" << std::endl;
    std::cerr << "Granularity: " << granularity << std::endl;
    std::cerr << "Discrete: " << discrete << std::endl;

    try {
        // Extract JSON string from buffer
        std::cerr << "Checking JSON buffer..." << std::endl;
        auto json_dims = structure_json.dimensions();
        std::cerr << "JSON dims size: " << json_dims.size() << std::endl;

        if (json_dims.size() != 1) {
            return ffi::Error::InvalidArgument("structure_json must be 1D array");
        }

        size_t json_length = json_dims[0];
        std::cerr << "JSON length: " << json_length << std::endl;

        std::cerr << "Getting JSON data pointer..." << std::endl;
        const uint8_t* json_data = structure_json.typed_data();
        std::cerr << "JSON data ptr: " << (void*)json_data << std::endl;

        if (json_data == nullptr) {
            return ffi::Error::InvalidArgument("JSON data pointer is null");
        }

        std::cerr << "Creating JSON string..." << std::endl;
        std::string json_str(reinterpret_cast<const char*>(json_data), json_length);
        std::cerr << "✅ JSON string created, length: " << json_str.length() << std::endl;

        // Look up or create GraphBuilder in thread-local cache
        std::cerr << "Looking up GraphBuilder in cache..." << std::endl;
        std::shared_ptr<GraphBuilder> builder;
        auto it = builder_cache.find(json_str);
        if (it != builder_cache.end()) {
            std::cerr << "Found in cache" << std::endl;
            builder = it->second;
        } else {
            std::cerr << "Not in cache, creating new GraphBuilder..." << std::endl;
            // Create new GraphBuilder and cache it
            try {
                builder = std::make_shared<GraphBuilder>(json_str);
                builder_cache[json_str] = builder;
                std::cerr << "✅ GraphBuilder created and cached" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ Failed to create GraphBuilder: " << e.what() << std::endl;
                return ffi::Error::InvalidArgument(
                    std::string("Failed to parse JSON structure: ") + e.what()
                );
            }
        }

        // Extract buffer dimensions
        std::cerr << "Extracting buffer dimensions..." << std::endl;
        auto theta_dims = theta.dimensions();
        auto times_dims = times.dimensions();
        std::cerr << "Theta dims size: " << theta_dims.size() << std::endl;
        std::cerr << "Times dims size: " << times_dims.size() << std::endl;

        if (theta_dims.size() != 1) {
            std::cerr << "❌ Invalid theta dims" << std::endl;
            return ffi::Error::InvalidArgument("theta must be 1D array");
        }
        if (times_dims.size() != 1) {
            std::cerr << "❌ Invalid times dims" << std::endl;
            return ffi::Error::InvalidArgument("times must be 1D array");
        }

        size_t theta_len = theta_dims[0];
        size_t n_times = times_dims[0];
        std::cerr << "Theta length: " << theta_len << std::endl;
        std::cerr << "Times length: " << n_times << std::endl;

        // Get raw data pointers
        std::cerr << "Getting raw data pointers..." << std::endl;
        const double* theta_data = theta.typed_data();
        const double* times_data = times.typed_data();
        double* result_data = result->typed_data();
        std::cerr << "Theta ptr: " << (void*)theta_data << std::endl;
        std::cerr << "Times ptr: " << (void*)times_data << std::endl;
        std::cerr << "Result ptr: " << (void*)result_data << std::endl;

        // Build graph with concrete parameters
        std::cerr << "Building graph with parameters..." << std::endl;
        Graph g = builder->build(theta_data, theta_len);
        std::cerr << "✅ Graph built successfully" << std::endl;

        // Compute PMF/PDF for all time points
        std::cerr << "Computing " << (discrete ? "PMF" : "PDF") << "..." << std::endl;
        if (discrete) {
            // Discrete phase-type (DPH)
            for (size_t i = 0; i < n_times; i++) {
                int jump_count = static_cast<int>(times_data[i]);
                result_data[i] = g.dph_pmf(jump_count);
            }
        } else {
            // Continuous phase-type (PDF)
            for (size_t i = 0; i < n_times; i++) {
                result_data[i] = g.pdf(times_data[i], granularity);
            }
        }
        std::cerr << "✅ Computation complete" << std::endl;

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        // Capture C++ exceptions and return as FFI error
        std::cerr << "❌ Exception caught: " << e.what() << std::endl;
        return ffi::Error::Internal(e.what());
    }
}

ffi::Error ComputePmfAndMomentsFfiImpl(
    ffi::Buffer<ffi::U8> structure_json,
    int32_t granularity,
    bool discrete,
    int32_t nr_moments,
    ffi::Buffer<ffi::F64> theta,
    ffi::Buffer<ffi::F64> times,
    ffi::ResultBuffer<ffi::F64> pmf_result,
    ffi::ResultBuffer<ffi::F64> moments_result
) {
    // Extract JSON string from buffer
    auto json_dims = structure_json.dimensions();
    if (json_dims.size() != 1) {
        return ffi::Error::InvalidArgument("structure_json must be 1D array");
    }

    size_t json_length = json_dims[0];
    const uint8_t* json_data = structure_json.typed_data();
    std::string json_str(reinterpret_cast<const char*>(json_data), json_length);

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
    auto theta_dims = theta.dimensions();
    auto times_dims = times.dimensions();

    if (theta_dims.size() != 1) {
        return ffi::Error::InvalidArgument("theta must be 1D array");
    }
    if (times_dims.size() != 1) {
        return ffi::Error::InvalidArgument("times must be 1D array");
    }

    size_t theta_len = theta_dims[0];
    size_t n_times = times_dims[0];

    // Get raw data pointers
    const double* theta_data = theta.typed_data();
    const double* times_data = times.typed_data();
    double* pmf_data = pmf_result->typed_data();
    double* moments_data = moments_result->typed_data();

    try {
        // Build graph ONCE (efficient!)
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
        std::vector<double> moments = builder->compute_moments_impl(g, nr_moments);

        // Copy moments to output buffer
        for (int i = 0; i < nr_moments; i++) {
            moments_data[i] = moments[i];
        }

        return ffi::Error::Success();

    } catch (const std::exception& e) {
        return ffi::Error::Internal(e.what());
    }
}

} // namespace ffi_handlers
} // namespace parameterized
} // namespace ptdalgorithms

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
//     PtdComputePmf, ptdalgorithms::parameterized::ffi_handlers::ComputePmfFfiImpl,
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
//     PtdComputePmfAndMoments, ptdalgorithms::parameterized::ffi_handlers::ComputePmfAndMomentsFfiImpl,
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
