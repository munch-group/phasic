cmake_minimum_required(VERSION 3.30)
project(PtDAlgorithms)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BINARY_DIR .)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_CXX_FLAGS_DEBUG "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(ptdalgorithms VERSION 0.19.81 DESCRIPTION "Efficient graph based phase-type distribution algorithms")
include(GNUInstallDirs)

add_library(libptdalgorithms SHARED api/c/ptdalgorithms.h src/c/ptdalgorithms.c)

set_target_properties(libptdalgorithms PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER api/c/ptdalgorithms.h)
configure_file(ptdalgorithms.pc.in ptdalgorithms.pc @ONLY)

target_include_directories(libptdalgorithms PRIVATE src/c)
install(TARGETS libptdalgorithms
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(FILES ${CMAKE_BINARY_DIR}/ptdalgorithms.pc
#         DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

        
add_library(libptdalgorithmscpp SHARED api/c/ptdalgorithms.h src/c/ptdalgorithms.c src/c/ptdalgorithms.h src/cpp/ptdalgorithmscpp.cpp api/cpp/ptdalgorithmscpp.h src/cpp/ptdalgorithmscpp.h)
set_target_properties(libptdalgorithmscpp PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER api/cpp/ptdalgorithmscpp.h)
configure_file(ptdalgorithmscpp.pc.in ptdalgorithmscpp.pc @ONLY)
target_include_directories(libptdalgorithmscpp PRIVATE src/cpp)
install(TARGETS libptdalgorithmscpp
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install(FILES ${CMAKE_BINARY_DIR}/ptdalgorithmscpp.pc
#         DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)


# Python bindings ###################################################
project(ptdalgorithmscpp_pybind LANGUAGES CXX)
# set(CMAKE_CXX_COMPILER_INIT g++)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
 
# pybind11 method:
pybind11_add_module(ptdalgorithmscpp_pybind src/cpp/ptdalgorithmscpp_pybind.cpp api/c/ptdalgorithms.h src/c/ptdalgorithms.c src/c/ptdalgorithms.h src/cpp/ptdalgorithmscpp.cpp api/cpp/ptdalgorithmscpp.h src/cpp/ptdalgorithmscpp.h)
target_link_libraries (ptdalgorithmscpp_pybind PUBLIC Eigen3::Eigen pybind11::module)
install(TARGETS ptdalgorithmscpp_pybind DESTINATION ptdalgorithms)


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

# nanobind_add_module(${PROJECT_NAME} src/cpp/ptdalgorithmscpp_pybind.cpp) 

# install( 
#     TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${BINDIR}
# )
####################
