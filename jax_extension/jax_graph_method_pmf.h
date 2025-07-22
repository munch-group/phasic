#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

// JAX custom call signature with scalar operands
__attribute__((visibility("default")))
void jax_graph_method_pmf(void* out_ptr, void** in_ptrs);

#ifdef __cplusplus
}
#endif

// /////////////


// hdf5_model_store.hpp
#pragma once
#include <H5Cpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <iostream>
#include <sstream>
#include <cstring>
#include <typeindex>
#include <memory>
#include <chrono>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/optional.hpp>


std::string hash_key_from_input(const std::vector<double>& vec) {
    std::ostringstream oss;
    oss << std::setprecision(4) << std::scientific;  // full double precision
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) oss << "_";
        oss << vec[i];
    }

    return oss.str();
}

bool key_exists(const std::string& filename, const std::string& key) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        return file.nameExists(key);
    } catch (...) {
        return false;
    }
}


class BinaryCacheStore {
public:
    // Structure to hold each cache entry: weak pointer, type info, and last access time
    struct CacheEntry {
        std::weak_ptr<void> data;
        std::type_index type;
        std::chrono::steady_clock::time_point last_access;
    };

    static constexpr size_t MAX_CACHE_SIZE = 32; // Max number of entries before eviction

    // Save object to both memory and HDF5
    template <typename T>
    static void save(const std::string& filename, const std::string& key, const T& obj) {
        std::lock_guard<std::mutex> lock(store_mutex);

        // Serialize object to binary using Boost
        std::ostringstream oss(std::ios::binary);
        boost::archive::binary_oarchive oa(oss);
        oa << obj;
        std::string serialized = oss.str();
        std::vector<uint8_t> blob(serialized.begin(), serialized.end());

        // Open or create HDF5 file
        H5::H5File file;
        try {
            file = H5::H5File(filename, H5F_ACC_RDWR);
        } catch (...) {
            file = H5::H5File(filename, H5F_ACC_TRUNC);
        }

        // Create group and dataset
        if (!file.nameExists(key)) file.createGroup(key);
        std::string path = key + "/blob";
        if (file.nameExists(path)) file.unlink(path);
        hsize_t dims[1] = {blob.size()};
        H5::DataSpace dspace(1, dims);
        H5::DataSet dset = file.createDataSet(path, H5::PredType::NATIVE_UINT8, dspace);
        dset.write(blob.data(), H5::PredType::NATIVE_UINT8);

        // Store in memory as a shared pointer
        auto sptr = std::make_shared<T>(obj);
        addToCache<T>(key, sptr);
    }

    // Load object from memory cache if present, else from HDF5 file
    template <typename T>
    static std::shared_ptr<T> load(const std::string& filename, const std::string& key) {
        {
            std::lock_guard<std::mutex> lock(store_mutex);
            auto it = memory_cache.find(key);
            if (it != memory_cache.end() && it->second.type == std::type_index(typeid(T))) {
                auto ptr = std::static_pointer_cast<T>(it->second.data.lock());
                if (ptr) {
                    it->second.last_access = std::chrono::steady_clock::now();
                    return ptr;
                }
            }
        }

        std::lock_guard<std::mutex> lock(store_mutex);

        // Open or create HDF5 file
        H5::H5File file;
        try {
            file = H5::H5File(filename, H5F_ACC_RDWR);
        } catch (...) {
            file = H5::H5File(filename, H5F_ACC_TRUNC);
        }

        // Read binary blob from HDF5
        std::string path = key + "/blob";
        if (!file.nameExists(path)) {
            throw std::runtime_error("Key not found in HDF5: " + path);
        }
        H5::DataSet dset = file.openDataSet(path);
        H5::DataSpace dspace = dset.getSpace();
        hsize_t dims[1];
        dspace.getSimpleExtentDims(dims);
        std::vector<uint8_t> blob(dims[0]);
        dset.read(blob.data(), H5::PredType::NATIVE_UINT8);

        // Deserialize and cache
        std::shared_ptr<T> obj = deserialize<T>(blob);
        addToCache<T>(key, obj);
        return obj;
    }

    // Check if key exists in HDF5 cache
    static bool exists(const std::string& filename, const std::string& key) {
        std::lock_guard<std::mutex> lock(store_mutex);
        try {
            H5::H5File file(filename, H5F_ACC_RDONLY);
            return file.nameExists(key + "/blob");
        } catch (...) {
            return false;
        }
    }

private:
    static std::unordered_map<std::string, CacheEntry> memory_cache; // LRU cache store
    static std::list<std::string> cache_keys; // Order of keys for LRU
    static std::mutex store_mutex; // Thread safety

    // Add new entry to cache and evict if needed
    template <typename T>
    static void addToCache(const std::string& key, const std::shared_ptr<T>& obj) {
        if (memory_cache.size() >= MAX_CACHE_SIZE) {
            auto oldest = cache_keys.front();
            cache_keys.pop_front();
            memory_cache.erase(oldest);
        }
        memory_cache[key] = CacheEntry{obj, std::type_index(typeid(T)), std::chrono::steady_clock::now()};
        cache_keys.push_back(key);
    }

    // Deserialize from binary blob
    template <typename T>
    static std::shared_ptr<T> deserialize(const std::vector<uint8_t>& blob) {
        std::istringstream iss(std::string(blob.begin(), blob.end()), std::ios::binary);
        boost::archive::binary_iarchive ia(iss);
        std::shared_ptr<T> obj = std::make_shared<T>();
        ia >> *obj;
        return obj;
    }
};

// Static member definitions
std::unordered_map<std::string, BinaryCacheStore::CacheEntry> BinaryCacheStore::memory_cache;
std::list<std::string> BinaryCacheStore::cache_keys;
std::mutex BinaryCacheStore::store_mutex;

// Serializable example struct
#include <boost/serialization/access.hpp>
struct MySerializable {
    int id;
    std::string name;
    std::vector<double> values;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar & id;
        ar & name;
        ar & values;
    }
};







// // hdf5_model_store.hpp
// #pragma once
// #include <H5Cpp.h>
// #include <vector>
// #include <string>
// #include <iostream>
// #include <sstream>
// #include <iomanip>
// #include <openssl/sha.h>
// #include <mutex>
// struct MyModel {
//     int id;
//     std::string name;
//     std::vector<double> weights;
// };

// // std::string hash_key_from_input(const std::vector<double>& inputs) {
// //     std::ostringstream oss;
// //     for (double x : inputs) {
// //         oss << std::setprecision(17) << x << ",";
// //     }

// //     std::string str = oss.str();
// //     unsigned char hash[SHA256_DIGEST_LENGTH];
// //     SHA256((const unsigned char*)str.data(), str.size(), hash);

// //     std::ostringstream key;
// //     key << "key_";
// //     for (int i = 0; i < 8; ++i) {  // short 64-bit prefix
// //         key << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
// //     }
// //     return key.str();
// // }

// std::string hash_key_from_input(const std::vector<double>& vec) {
//     std::ostringstream oss;
//     oss << std::setprecision(4) << std::scientific;  // full double precision
//     for (size_t i = 0; i < vec.size(); ++i) {
//         if (i != 0) oss << "_";
//         oss << vec[i];
//     }

//     return oss.str();
// }

// bool key_exists(const std::string& filename, const std::string& key) {
//     try {
//         H5::H5File file(filename, H5F_ACC_RDONLY);
//         return file.nameExists(key);
//     } catch (...) {
//         return false;
//     }
// }


// // hdf5_model_store.hpp
// #pragma once
// #include <H5Cpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <mutex>
// #include <iostream>
// #include <sstream>
// #include <cstring>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>
// #include <boost/serialization/vector.hpp>
// #include <boost/serialization/string.hpp>
// #include <boost/serialization/utility.hpp>
// #include <boost/serialization/serialization.hpp>
// #include <boost/serialization/optional.hpp>


// class BinaryCacheStore {
// public:
//     template <typename T>
//     static void save(const std::string& filename, const std::string& key, const T& obj) {
//         std::lock_guard<std::mutex> lock(store_mutex);

//         // Serialize to memory
//         std::ostringstream oss(std::ios::binary);
//         boost::archive::binary_oarchive oa(oss);
//         oa << obj;
//         std::string serialized = oss.str();
//         std::vector<uint8_t> blob(serialized.begin(), serialized.end());

//         // Write to HDF5
//         H5::H5File file;
//         try {
//             file = H5::H5File(filename, H5F_ACC_RDWR);
//         } catch (...) {
//             file = H5::H5File(filename, H5F_ACC_TRUNC);
//         }

//         if (!file.nameExists(key)) {
//             file.createGroup(key);
//         }

//         std::string path = key + "/blob";
//         if (file.nameExists(path)) file.unlink(path);

//         hsize_t dims[1] = {blob.size()};
//         H5::DataSpace dspace(1, dims);
//         H5::DataSet dset = file.createDataSet(path, H5::PredType::NATIVE_UINT8, dspace);
//         dset.write(blob.data(), H5::PredType::NATIVE_UINT8);

//         memory_cache[key] = blob;
//     }

//     template <typename T>
//     static T load(const std::string& filename, const std::string& key) {
//         {
//             std::lock_guard<std::mutex> lock(store_mutex);
//             auto it = memory_cache.find(key);
//             if (it != memory_cache.end()) {
//                 return deserialize<T>(it->second);
//             }
//         }

//         std::lock_guard<std::mutex> lock(store_mutex);

//         H5::H5File file;
//         try {
//             file = H5::H5File(filename, H5F_ACC_RDWR);
//         } catch (...) {
//             file = H5::H5File(filename, H5F_ACC_TRUNC);
//         }

//         std::string path = key + "/blob";
//         if (!file.nameExists(path)) {
//             throw std::runtime_error("Key not found in HDF5: " + path);
//         }

//         H5::DataSet dset = file.openDataSet(path);
//         H5::DataSpace dspace = dset.getSpace();
//         hsize_t dims[1];
//         dspace.getSimpleExtentDims(dims);
//         std::vector<uint8_t> blob(dims[0]);
//         dset.read(blob.data(), H5::PredType::NATIVE_UINT8);

//         memory_cache[key] = blob;
//         return deserialize<T>(blob);
//     }

//     static bool exists(const std::string& filename, const std::string& key) {
//         std::lock_guard<std::mutex> lock(store_mutex);
//         try {
//             H5::H5File file(filename, H5F_ACC_RDONLY);
//             return file.nameExists(key + "/blob");
//         } catch (...) {
//             return false;
//         }
//     }

// private:
//     static std::unordered_map<std::string, std::vector<uint8_t>> memory_cache;
//     static std::mutex store_mutex;

//     template <typename T>
//     static T deserialize(const std::vector<uint8_t>& blob) {
//         std::istringstream iss(std::string(blob.begin(), blob.end()), std::ios::binary);
//         boost::archive::binary_iarchive ia(iss);
//         T obj;
//         ia >> obj;
//         return obj;
//     }
// };

// std::unordered_map<std::string, std::vector<uint8_t>> BinaryCacheStore::memory_cache;
// std::mutex BinaryCacheStore::store_mutex;

// // Example serializable struct
// #include <boost/serialization/access.hpp>
// struct MySerializable {
//     int id;
//     std::string name;
//     std::vector<double> values;

//     template<class Archive>
//     void serialize(Archive& ar, const unsigned int) {
//         ar & id;
//         ar & name;
//         ar & values;
//     }
// };

// int main() {
//     const std::string filename = "boost_cache.h5";
//     const std::string key = "example";

//     MySerializable obj{42, "test", {1.0, 2.0, 3.0}};
//     BinaryCacheStore::save(filename, key, obj);

//     auto loaded = BinaryCacheStore::load<MySerializable>(filename, key);
//     std::cout << "ID: " << loaded.id << ", Name: " << loaded.name << ", Values: ";
//     for (auto v : loaded.values) std::cout << v << " ";
//     std::cout << std::endl;
//     return 0;
// }









// #pragma once
// #include <H5Cpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <mutex>
// #include <iostream>
// #include <sstream>
// #include <cstring>

// class BinaryCacheStore {
// public:
//     // Save binary blob under a key in HDF5 and memory cache
//     static void save(const std::string& filename, const std::string& key, const std::vector<uint8_t>& blob) {
//         std::lock_guard<std::mutex> lock(store_mutex);

//         H5::H5File file;
//         try {
//             file = H5::H5File(filename, H5F_ACC_RDWR);
//         } catch (...) {
//             file = H5::H5File(filename, H5F_ACC_TRUNC);
//         }

//         if (!file.nameExists(key)) {
//             file.createGroup(key);
//         }

//         std::string path = key + "/blob";
//         if (file.nameExists(path)) file.unlink(path);

//         hsize_t dims[1] = {blob.size()};
//         H5::DataSpace dspace(1, dims);
//         H5::DataSet dset = file.createDataSet(path, H5::PredType::NATIVE_UINT8, dspace);
//         dset.write(blob.data(), H5::PredType::NATIVE_UINT8);

//         memory_cache[key] = blob;
//     }

//     // Load from memory or HDF5, create disk file if missing
//     static std::vector<uint8_t> load(const std::string& filename, const std::string& key) {
//         {
//             std::lock_guard<std::mutex> lock(store_mutex);
//             auto it = memory_cache.find(key);
//             if (it != memory_cache.end()) {
//                 return it->second;
//             }
//         }

//         std::lock_guard<std::mutex> lock(store_mutex);

//         H5::H5File file;
//         try {
//             file = H5::H5File(filename, H5F_ACC_RDWR);
//         } catch (...) {
//             // Create the file if it doesn't exist yet
//             file = H5::H5File(filename, H5F_ACC_TRUNC);
//         }

//         std::string path = key + "/blob";
//         if (!file.nameExists(path)) {
//             throw std::runtime_error("Key not found in HDF5: " + path);
//         }

//         H5::DataSet dset = file.openDataSet(path);
//         H5::DataSpace dspace = dset.getSpace();
//         hsize_t dims[1];
//         dspace.getSimpleExtentDims(dims);
//         std::vector<uint8_t> blob(dims[0]);
//         dset.read(blob.data(), H5::PredType::NATIVE_UINT8);

//         memory_cache[key] = blob;
//         return blob;
//     }

//     static bool exists(const std::string& filename, const std::string& key) {
//         std::lock_guard<std::mutex> lock(store_mutex);
//         try {
//             H5::H5File file(filename, H5F_ACC_RDONLY);
//             return file.nameExists(key + "/blob");
//         } catch (...) {
//             return false;
//         }
//     }

// private:
//     static std::unordered_map<std::string, std::vector<uint8_t>> memory_cache;
//     static std::mutex store_mutex;
// };

// std::unordered_map<std::string, std::vector<uint8_t>> BinaryCacheStore::memory_cache;
// std::mutex BinaryCacheStore::store_mutex;







// class HDF5ModelStore {
//     public:
//         static void save(const std::string& filename, const std::string& key, const MyModel& model) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             H5::H5File file;
//             try {
//                 file = H5::H5File(filename, H5F_ACC_RDWR);
//             } catch (...) {
//                 file = H5::H5File(filename, H5F_ACC_TRUNC);
//             }

//             if (!file.nameExists(key)) {
//                 file.createGroup(key);
//             }

//             hsize_t dims[1] = {model.weights.size()};
//             H5::DataSpace wspace(1, dims);
//             H5::DataSet wset = file.createDataSet(key + "/weights", H5::PredType::NATIVE_DOUBLE, wspace);
//             wset.write(model.weights.data(), H5::PredType::NATIVE_DOUBLE);

//             hsize_t id_dims[1] = {1};
//             H5::DataSpace id_space(1, id_dims);
//             H5::DataSet idset = file.createDataSet(key + "/id", H5::PredType::NATIVE_INT, id_space);
//             idset.write(&model.id, H5::PredType::NATIVE_INT);

//             hsize_t str_dims[1] = {model.name.size()};
//             H5::StrType str_type(H5::PredType::C_S1, model.name.size());
//             H5::DataSpace str_space(1, str_dims);
//             H5::DataSet strset = file.createDataSet(key + "/name", str_type, str_space);
//             strset.write(model.name, str_type);
//         }

//         static MyModel load(const std::string& filename, const std::string& key) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             H5::H5File file(filename, H5F_ACC_RDONLY);

//             H5::DataSet idset = file.openDataSet(key + "/id");
//             int id;
//             idset.read(&id, H5::PredType::NATIVE_INT);

//             H5::DataSet strset = file.openDataSet(key + "/name");
//             H5::StrType str_type = strset.getStrType();
//             std::string name;
//             strset.read(name, str_type);

//             H5::DataSet wset = file.openDataSet(key + "/weights");
//             H5::DataSpace wspace = wset.getSpace();
//             hsize_t dims[1];
//             wspace.getSimpleExtentDims(dims);
//             std::vector<double> weights(dims[0]);
//             wset.read(weights.data(), H5::PredType::NATIVE_DOUBLE);

//             return {id, name, weights};
//         }

//         static bool key_exists(const std::string& filename, const std::string& key) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             try {
//                 H5::H5File file(filename, H5F_ACC_RDONLY);
//                 return file.nameExists(key);
//             } catch (...) {
//                 return false;
//             }
//         }

//     private:
//         static std::mutex hdf5_mutex;
// };

// std::mutex HDF5ModelStore::hdf5_mutex;
