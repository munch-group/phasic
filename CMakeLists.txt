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

add_library(libphasic SHARED api/c/phasic.h src/c/phasic.c src/c/phasic_symbolic.c src/c/phasic_hash.c)

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

        
add_library(libphasiccpp SHARED api/c/phasic.h src/c/phasic.c src/c/phasic_symbolic.c src/c/phasic_hash.c src/c/phasic.h src/cpp/phasiccpp.cpp api/cpp/phasiccpp.h src/cpp/phasiccpp.h)
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

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP)

# Get XLA FFI include directory from environment, jaxlib, or local install
set(XLA_FFI_RESULT 1)

# First check environment variable (useful during pip install)
if(DEFINED ENV{XLA_FFI_INCLUDE_DIR})
    set(XLA_FFI_INCLUDE_DIR "$ENV{XLA_FFI_INCLUDE_DIR}")
    set(XLA_FFI_RESULT 0)
    message(STATUS "Found XLA FFI headers from environment: ${XLA_FFI_INCLUDE_DIR}")
else()
    # Try to get from JAX
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from jax import ffi; print(ffi.include_dir())"
        OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE XLA_FFI_RESULT
    )

    if(XLA_FFI_RESULT EQUAL 0)
        message(STATUS "Found XLA FFI headers from JAX: ${XLA_FFI_INCLUDE_DIR}")
    else()
        # Try to find headers in user's local include directory
        find_path(LOCAL_XLA_INCLUDE
            NAMES xla/ffi/api/ffi.h
            PATHS $ENV{HOME}/.local/include /usr/local/include
            NO_DEFAULT_PATH
        )

        if(LOCAL_XLA_INCLUDE)
            set(XLA_FFI_INCLUDE_DIR "${LOCAL_XLA_INCLUDE}")
            set(XLA_FFI_RESULT 0)
            message(STATUS "Found XLA FFI headers in local install: ${XLA_FFI_INCLUDE_DIR}")
        else()
            message(WARNING "Could not find XLA FFI headers. FFI handlers will not be compiled.")
            set(XLA_FFI_INCLUDE_DIR "")
        endif()
    endif()
endif()

# pybind11 method:
set(PYBIND_SOURCES
    src/cpp/phasic_pybind.cpp
    api/c/phasic.h
    api/c/phasic_hash.h
    src/c/phasic.c
    src/c/phasic_symbolic.c
    src/c/phasic_hash.c
    src/c/phasic.h
    src/cpp/phasiccpp.cpp
    api/cpp/phasiccpp.h
    src/cpp/phasiccpp.h
    src/cpp/parameterized/graph_builder.cpp
    src/cpp/parameterized/graph_builder.hpp
)

# Add FFI handlers if XLA FFI is available
if(XLA_FFI_RESULT EQUAL 0)
    list(APPEND PYBIND_SOURCES
        src/cpp/parameterized/graph_builder_ffi.cpp
        src/cpp/parameterized/graph_builder_ffi.hpp
    )
    message(STATUS "FFI handlers will be compiled")
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
