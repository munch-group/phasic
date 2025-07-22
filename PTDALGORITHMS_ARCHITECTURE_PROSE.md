# PtDAlgorithms Project Architecture

## Overview

PtDAlgorithms represents a sophisticated multi-language scientific computing library designed to provide efficient algorithms for phase-type distributions through graph-based computational approaches. The project's architecture demonstrates a carefully orchestrated design that delivers unified functionality across Python, R, and C/C++ environments while maintaining the performance characteristics essential for scientific computing applications.

The library's foundation rests on a core C implementation that provides the fundamental graph algorithms and data structures, around which language-specific bindings have been constructed to offer idiomatic interfaces for each target language. This approach ensures that computational efficiency is maintained at the core while developers can access the library's capabilities through familiar programming paradigms in their language of choice.

## Project Structure and Organization

The PtDAlgorithms project follows a hierarchical organization that reflects its multi-layered architecture. At the root level, the `api/` directory houses the public interface definitions, with separate subdirectories for C and C++ APIs that establish the contract between the core implementation and its various language bindings. The `src/` directory contains the actual implementation files, organized to mirror the architectural layers, with core C code residing in `src/c/`, C++ wrappers and Python bindings in `src/cpp/`, and the Python package structure maintained within `src/ptdalgorithms/`.

The R integration follows standard R package conventions, with R-specific files located in the `R/` directory and compilation configuration managed through `src/Makevars` files. An experimental JAX extension exists in the `jax_extension/` directory, providing GPU-accelerated computation capabilities for specialized use cases. Documentation, examples, and test suites are organized in their respective directories, supporting the development workflow across all target languages.

## Component Architecture

### The Core C Foundation

The architectural foundation of PtDAlgorithms lies in its C implementation, which provides the essential graph algorithms and data structures that power phase-type distribution computations. This layer establishes the fundamental abstractions through structures like `ptd_graph`, `ptd_vertex`, and `ptd_edge`, which collectively represent the mathematical objects and their relationships within the computational framework.

The core C layer implements sophisticated algorithms for graph manipulation, including normalization procedures, reward transformation operations, and probability computations. A particularly important component is the AVL tree implementation (`ptd_avl_tree`) that enables efficient vertex lookup operations based on state vectors, ensuring that the library can handle large-scale problems with appropriate computational complexity guarantees.

The C implementation maintains state through distribution contexts that support iterative probability calculations, allowing for efficient computation of probability density functions, cumulative distribution functions, and related statistical measures. The complete C API, documented in `api/c/ptdalgorithms.h`, spans over 400 lines of carefully designed interface definitions that establish the contract for all higher-level language bindings.

Memory management within the C layer follows manual allocation and deallocation patterns typical of C programming, with careful attention to resource cleanup and error handling through a global error buffer mechanism. This approach ensures predictable performance characteristics while requiring careful coordination with higher-level language bindings to prevent memory leaks and ensure proper resource management.

### C++ Wrapper Layer Enhancement

The C++ wrapper layer transforms the foundational C implementation into a modern, object-oriented interface that leverages contemporary C++ programming practices. This layer introduces RAII (Resource Acquisition Is Initialization) patterns, STL integration, and comprehensive exception handling to create a more robust and user-friendly programming interface.

The central `Graph` class provides a reference-counted wrapper around the underlying C graph structure, ensuring that memory management occurs automatically and safely even when graph objects are copied or passed between functions. The class exposes methods that accept and return STL containers, particularly `std::vector` objects, making the interface natural for C++ programmers while maintaining efficiency through careful memory management.

Complementary `Vertex` and `Edge` classes provide type-safe wrappers around their C counterparts, ensuring that operations on graph components remain well-defined and that common programming errors are caught at compile time rather than runtime. The C++ layer also implements distribution context wrappers that provide iterator-like interfaces for probability calculations, enabling elegant integration with C++ algorithms and range-based operations.

The C++ implementation demonstrates sophisticated design patterns, including factory functions that support pybind11 integration, custom assignment operators that maintain reference counting semantics, and template-based programming techniques that enable efficient interoperation with various numeric libraries, particularly Eigen for linear algebra operations.

### Python Integration Through pybind11

The Python bindings represent a carefully crafted integration that leverages pybind11 to provide seamless interoperability between the C++ implementation and Python's scientific computing ecosystem. The binding layer handles automatic conversion between Python data structures and their C++ counterparts, enabling natural use of NumPy arrays, Python lists, and other familiar Python objects within the computational framework.

The Python interface extends beyond simple bindings by implementing additional functionality that takes advantage of Python's flexibility and rich ecosystem. The `Graph` class in Python supports callback-based construction patterns that allow for dynamic graph generation, enabling users to define complex mathematical models through Python functions rather than manual graph construction.

A particularly sophisticated feature is the `discretize` method, which transforms continuous phase-type distributions into their discrete counterparts by introducing auxiliary vertices and edges. This functionality demonstrates the power of the multi-layered architecture, where high-level Python code orchestrates complex transformations that ultimately execute through the efficient C core.

The Python integration also includes visualization capabilities through GraphViz integration, providing immediate visual feedback for graph structures and supporting exploratory data analysis workflows. The binding layer carefully manages memory transfer between Python and C++, ensuring that large numeric arrays can be processed efficiently without unnecessary copying while maintaining Python's garbage collection semantics.

### R Package Implementation

The R integration follows standard R package development practices, utilizing Rcpp to provide seamless interoperability between R and the underlying C++ implementation. The binding generation process relies on Rcpp's automatic code generation capabilities, where C++ functions annotated with special comments are automatically wrapped to provide R-native interfaces.

The R bindings handle conversion between R's data structures and the library's internal representations, supporting R vectors, lists, and data frames as natural input and output formats. The package structure includes comprehensive documentation following R's documentation standards, with man pages generated for each exported function and proper integration with R's help system.

Compilation configuration for the R package occurs through `src/Makevars` files that specify the necessary compiler flags and include paths to link against the C++ implementation. The build process integrates smoothly with R's package management system, allowing users to install the package through standard R mechanisms while ensuring that all necessary dependencies are properly resolved.

The R interface maintains consistency with the broader PtDAlgorithms API while respecting R's programming conventions and idioms. Functions accept and return R objects naturally, and error handling integrates with R's exception mechanism to provide meaningful feedback when operations fail or encounter invalid input.

### Experimental JAX Extension

The JAX extension represents an experimental effort to integrate PtDAlgorithms with JAX's automatic differentiation and GPU acceleration capabilities. This component demonstrates the library's extensibility by implementing custom JAX primitives that can execute phase-type distribution operations on GPU hardware while maintaining compatibility with JAX's transformation system.

The JAX integration includes HDF5 serialization capabilities that enable efficient storage and loading of large model configurations, supporting workflows that involve complex model parameters or large-scale computations. The extension compiles to a shared library that interfaces with JAX through its Foreign Function Interface (FFI), enabling high-performance execution while maintaining JAX's functional programming paradigms.

This experimental component showcases the potential for extending PtDAlgorithms into modern machine learning workflows where automatic differentiation and GPU acceleration are essential for practical applications. The design maintains separation from the core library while demonstrating how the fundamental algorithms can be adapted to contemporary computational frameworks.

## Build System Architecture

The build system architecture reflects the complexity of managing compilation across multiple languages while maintaining consistency and reliability. The core C/C++ components utilize CMake as the primary build orchestrator, which generates multiple targets including shared libraries for both C and C++ layers, as well as the Python extension module.

CMake manages dependencies on external libraries, particularly Eigen3 for linear algebra operations and pybind11 for Python integration. The configuration ensures that all components are built with consistent compiler flags and optimization settings while handling platform-specific compilation requirements.

The Python package build process leverages scikit-build-core, which provides a modern bridge between CMake and Python's packaging system. This approach ensures that Python wheels can be built reproducibly while maintaining the flexibility to incorporate the complex C++ compilation requirements. The configuration in `pyproject.toml` specifies both build-time and runtime dependencies, ensuring that the package installs correctly across different Python environments.

The R package follows standard R compilation practices, with `src/Makevars` providing the necessary configuration to link against the C++ components. The build process integrates with R's package management system while ensuring that the underlying C++ code is compiled with appropriate settings for R's requirements.

The JAX extension maintains its own build system through a custom Makefile that handles the specific requirements for creating shared libraries compatible with JAX's FFI mechanism. This build process includes linking against HDF5 libraries and managing the specific compiler flags required for proper symbol visibility in the shared library.

## Language Integration and Data Flow

The integration between languages demonstrates sophisticated engineering in handling data conversion, memory management, and error propagation across language boundaries. Each language binding implements appropriate conversion mechanisms that respect the semantics and performance characteristics of the target language while maintaining computational efficiency.

State representation varies across languages according to their natural conventions: C uses raw integer arrays, C++ employs STL vectors, Python supports both lists and NumPy arrays, and R utilizes integer vectors. The binding layers handle these conversions transparently while optimizing for performance-critical operations.

Memory management strategies differ significantly across the language boundaries, with C requiring manual management, C++ implementing reference counting with RAII, and both Python and R providing automatic memory management through their respective runtime systems. The binding layers coordinate these different approaches to ensure that resources are properly managed without imposing unnecessary overhead.

Error handling presents another area where language differences require careful coordination. The C layer utilizes error codes and a global error buffer, while C++ introduces exception-based error handling. The Python and R bindings translate these mechanisms into their respective error handling systems, ensuring that users receive appropriate feedback regardless of their chosen language interface.

The data flow through the system follows a consistent pattern where user input in any language is converted to appropriate internal representations, processed through the core C algorithms, and then translated back to the native data structures of the calling language. This approach ensures computational consistency while providing language-appropriate interfaces.

## Development and Maintenance Workflow

The development workflow accommodates the complexity of multi-language development through carefully structured testing, documentation, and dependency management practices. Testing occurs at multiple levels, with Python tests utilizing pytest, R tests employing the testthat framework, and basic C++ tests providing coverage of core functionality.

Documentation generation leverages Quarto for comprehensive API documentation that spans all language interfaces, with examples provided in multiple languages to demonstrate equivalent functionality across the different bindings. Jupyter notebooks provide interactive examples that showcase complex workflows and demonstrate the library's capabilities in realistic scientific computing scenarios.

Dependency management represents a significant challenge in multi-language projects, addressed through pixi for unified environment management that ensures consistent development environments across different platforms and languages. This approach simplifies the setup process for new developers while ensuring that all components build correctly with compatible dependency versions.

The project structure supports continuous integration and testing across all supported languages and platforms, with build configurations that validate functionality across the entire software stack. This comprehensive testing approach ensures that changes to core algorithms are properly reflected across all language bindings and that the user experience remains consistent regardless of the chosen interface.

## Conclusion

The PtDAlgorithms architecture demonstrates sophisticated engineering principles applied to create a robust, multi-language scientific computing library. The design successfully balances computational efficiency with developer accessibility, providing high-performance algorithms through a carefully optimized C core while offering idiomatic interfaces for multiple programming languages.

The layered architecture ensures that computational efficiency is preserved at the algorithmic level while enabling each language binding to leverage its ecosystem's strengths. Python users benefit from NumPy integration and visualization capabilities, R users access familiar data structures and documentation conventions, and C++ users enjoy modern programming practices with STL integration and RAII memory management.

This architectural approach provides a template for developing scientific computing libraries that must serve diverse user communities while maintaining the performance characteristics essential for computational research. The careful attention to build systems, testing practices, and documentation ensures that the library can evolve and expand while maintaining reliability and consistency across its supported platforms.