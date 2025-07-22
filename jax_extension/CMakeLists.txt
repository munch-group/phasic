cmake_minimum_required(VERSION 3.16)
project(binary_cache_example LANGUAGES CXX)

# === Configuration ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === Dependencies ===
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# === Include Directories ===
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

# === Source Files ===
add_executable(cache_example
    main.cpp
    # replace with your actual file(s), e.g., src/hdf5_model_store.hpp or .cpp
)

# === Linking ===
target_link_libraries(cache_example
    ${Boost_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
)

# === Optional: Show paths ===
message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libs: ${Boost_LIBRARIES}")
message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 C++ libs: ${HDF5_CXX_LIBRARIES}")