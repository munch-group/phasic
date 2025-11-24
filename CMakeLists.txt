cmake_minimum_required(VERSION 3.30)
project(PtDAlgorithms)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BINARY_DIR .)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_CXX_FLAGS_DEBUG "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(phasic VERSION 0.20.0 DESCRIPTION "Efficient graph based phase-type distribution algorithms")
include(GNUInstallDirs)
include(FetchContent)

# NOTE: phasic_symbolic.c commented out due to missing function implementations
# Symbolic elimination is obsolete - use trace-based elimination instead
add_library(libphasic SHARED api/c/phasic.h src/c/phasic.c src/c/phasic_hash.c src/c/phasic_log.c src/c/phasic_log.h)

set_target_properties(libphasic PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER api/c/phasic.h)
configure_file(phasic.pc.in phasic.pc @ONLY)

target_include_directories(libphasic PRIVATE src/c)
install(TARGETS libphasic
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Install Python-facing header copy
install(FILES api/c/phasic.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(FILES ${CMAKE_BINARY_DIR}/phasic.pc
#         DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

        
# NOTE: phasic_symbolic.c commented out due to missing function implementations
add_library(libphasiccpp SHARED api/c/phasic.h src/c/phasic.c src/c/phasic_hash.c src/c/phasic_log.c src/c/phasic.h src/c/phasic_log.h src/cpp/phasiccpp.cpp api/cpp/phasiccpp.h api/cpp/scc_graph.cpp api/cpp/scc_graph.h src/cpp/phasiccpp.h)
set_target_properties(libphasiccpp PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER api/cpp/phasiccpp.h)
configure_file(phasiccpp.pc.in phasiccpp.pc @ONLY)
target_include_directories(libphasiccpp PRIVATE src/cpp)
install(TARGETS libphasiccpp
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Install Python-facing header copy
install(FILES api/cpp/phasiccpp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install(FILES ${CMAKE_BINARY_DIR}/phasiccpp.pc
#         DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)


# Python bindings ###################################################
project(phasic_pybind LANGUAGES CXX)

# Help CMake find conda/pixi packages by adding environment prefixes
# During conda-build, BUILD_PREFIX and PREFIX are set
# During pip install in pixi, CONDA_PREFIX is set
set(CONDA_PREFIXES "")

if(DEFINED ENV{BUILD_PREFIX})
    list(APPEND CONDA_PREFIXES "$ENV{BUILD_PREFIX}")
    message(STATUS "Found BUILD_PREFIX: $ENV{BUILD_PREFIX}")
endif()

if(DEFINED ENV{PREFIX})
    list(APPEND CONDA_PREFIXES "$ENV{PREFIX}")
    message(STATUS "Found PREFIX: $ENV{PREFIX}")
endif()

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CONDA_PREFIXES "$ENV{CONDA_PREFIX}")
    message(STATUS "Found CONDA_PREFIX: $ENV{CONDA_PREFIX}")
endif()

# Add all found prefixes to CMAKE_PREFIX_PATH
foreach(prefix ${CONDA_PREFIXES})
    list(APPEND CMAKE_PREFIX_PATH "${prefix}")
endforeach()

# Debug: Print CMAKE_PREFIX_PATH to help diagnose issues
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# When building with scikit-build-core, Python is already found
# For standalone builds, find Python if not already found
if(NOT Python_FOUND AND NOT SKBUILD)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Try to find Eigen3, fetch if not found
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitHub...")
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# Try to find nlohmann_json, fetch if not found
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, fetching from GitHub...")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(OpenMP)

# Get XLA FFI include directory from environment, jaxlib, or local install
message(STATUS "========================================")
message(STATUS "Detecting XLA FFI headers for JAX integration...")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
set(XLA_FFI_RESULT 1)

# First check environment variable (useful during pip install)
if(DEFINED ENV{XLA_FFI_INCLUDE_DIR})
    set(XLA_FFI_INCLUDE_DIR "$ENV{XLA_FFI_INCLUDE_DIR}")
    set(XLA_FFI_RESULT 0)
    message(STATUS "✓ Method 1/3: Found XLA FFI via environment variable")
    message(STATUS "  XLA_FFI_INCLUDE_DIR = ${XLA_FFI_INCLUDE_DIR}")
else()
    message(STATUS "✗ Method 1/3: XLA_FFI_INCLUDE_DIR not set")

    # Try to get from JAX
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from jax import ffi; print(ffi.include_dir())"
        OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE XLA_FFI_RESULT
        ERROR_QUIET
    )

    if(XLA_FFI_RESULT EQUAL 0)
        message(STATUS "✓ Method 2/3: Auto-detected from JAX Python package")
        message(STATUS "  XLA FFI directory = ${XLA_FFI_INCLUDE_DIR}")
    else()
        message(STATUS "✗ Method 2/3: JAX package not available or import failed")

        # Try to find headers in user's local include directory
        find_path(LOCAL_XLA_INCLUDE
            NAMES xla/ffi/api/ffi.h
            PATHS $ENV{HOME}/.local/include /usr/local/include
            NO_DEFAULT_PATH
        )

        if(LOCAL_XLA_INCLUDE)
            set(XLA_FFI_INCLUDE_DIR "${LOCAL_XLA_INCLUDE}")
            set(XLA_FFI_RESULT 0)
            message(STATUS "✓ Method 3/3: Found in local installation path")
            message(STATUS "  XLA FFI directory = ${XLA_FFI_INCLUDE_DIR}")
        else()
            message(STATUS "✗ Method 3/3: Not found in ~/.local/include or /usr/local/include")
            message(WARNING "")
            message(WARNING "XLA FFI headers NOT FOUND - FFI handlers will not be compiled")
            message(WARNING "This means JAX integration will be slower (Python callbacks instead of C++)")
            message(WARNING "")
            message(WARNING "To enable FFI support, install JAX before building:")
            message(WARNING "  pip install 'jax>=0.4.0'")
            message(WARNING "  pip install --no-build-isolation --force-reinstall --no-deps .")
            message(WARNING "")
            set(XLA_FFI_INCLUDE_DIR "")
        endif()
    endif()
endif()
message(STATUS "========================================")

# pybind11 method:
# NOTE: phasic_symbolic.c commented out due to missing function implementations
set(PYBIND_SOURCES
    src/cpp/phasic_pybind.cpp
    api/c/phasic.h
    api/c/phasic_hash.h
    src/c/phasic.c
    # src/c/phasic_symbolic.c  # DISABLED - use trace-based elimination instead
    src/c/phasic_hash.c
    src/c/phasic_log.c
    src/c/phasic.h
    src/c/phasic_log.h
    src/cpp/phasiccpp.cpp
    api/cpp/phasiccpp.h
    api/cpp/scc_graph.cpp
    api/cpp/scc_graph.h
    src/cpp/phasiccpp.h
    src/cpp/parameterized/graph_builder.cpp
    src/cpp/parameterized/graph_builder.hpp
)

# Add FFI handlers if XLA FFI is available
if(XLA_FFI_RESULT EQUAL 0)
    list(APPEND PYBIND_SOURCES
        src/cpp/parameterized/graph_builder_ffi.cpp
        src/cpp/parameterized/graph_builder_ffi.hpp
        src/cpp/parameterized/ffi_handlers.cpp
    )
    message(STATUS "")
    message(STATUS "✓✓✓ FFI handlers WILL be compiled (fast C++ JAX integration)")
    message(STATUS "")
endif()

pybind11_add_module(phasic_pybind ${PYBIND_SOURCES})

target_link_libraries(phasic_pybind PUBLIC Eigen3::Eigen pybind11::module nlohmann_json::nlohmann_json)

# Add XLA FFI include directory if available
if(XLA_FFI_RESULT EQUAL 0)
    target_include_directories(phasic_pybind PRIVATE ${XLA_FFI_INCLUDE_DIR})
    target_compile_definitions(phasic_pybind PRIVATE HAVE_XLA_FFI)
endif()

# Add OpenMP support for multi-core parallelization in FFI handlers
if(OpenMP_CXX_FOUND)
    target_link_libraries(phasic_pybind PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for multi-core parallelization")
else()
    message(WARNING "OpenMP not found - FFI will run sequentially")
endif()

install(TARGETS phasic_pybind DESTINATION phasic)

# Install C headers alongside Python package for users who want C API
install(FILES
    api/c/phasic.h
    api/c/phasic_hash.h
    DESTINATION phasic/include/c)

# Install C++ headers alongside Python package for users who want C++ API
install(FILES
    api/cpp/phasiccpp.h
    api/cpp/scc_graph.h
    api/cpp/user_model.h
    DESTINATION phasic/include/cpp)

# Install C++ implementation files needed by headers
install(FILES
    api/cpp/scc_graph.cpp
    DESTINATION phasic/include/cpp)


# # R bindings ###################################################
# add_subdirectory(./R)




### Pixi version ######
# find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED) 

# execute_process(
#   COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
#   OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
# ) 

# execute_process(
#     COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
#     OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# ) 

# find_package(nanobind CONFIG REQUIRED) 

# nanobind_add_module(${PROJECT_NAME} src/cpp/phasic_pybind.cpp) 

# install(
#     TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${BINDIR}
# )
####################

# # Test executables ###################################################
# # Phase 5 Week 2: Symbolic gradient tests
# add_executable(test_symbolic_gradient tests/test_symbolic_gradient.c)
# target_link_libraries(test_symbolic_gradient libphasic)
# target_include_directories(test_symbolic_gradient PRIVATE api/c)

# # Phase 5 Week 3: Forward algorithm gradient tests
# add_executable(test_pdf_gradient tests/test_pdf_gradient.c)
# target_link_libraries(test_pdf_gradient libphasic)
# target_include_directories(test_pdf_gradient PRIVATE api/c)
