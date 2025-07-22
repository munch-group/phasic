# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PtDAlgorithms is a multi-language library for efficient phase-type distribution algorithms using graph-based approaches. The library provides APIs for Python, R, and C/C++, with the core implementation in C++ and bindings using pybind11 for Python and Rcpp for R.

## Build and Development Commands

### Python Development
- **Install in editable mode**: `pip install -e .`
- **Run Python tests**: `python -m pytest tests/`
- **Build Python package**: `python -m build`

### R Development  
- **Run R tests**: `Rscript tests/testthat.R`
- **Install R package**: Use R CMD INSTALL or devtools::install()

### C++ Development
- **Build C++ libraries**: `cmake . && make`
- **Build with CMake**: Uses CMakeLists.txt for building both C and C++ shared libraries
- **JAX extension**: Use `make` in jax_extension/ directory

### Environment Setup
- **Conda environment**: `conda env create -f binder/env-osx-arm64.yml` (or platform-specific file)
- **Pixi environment**: Project uses pixi for dependency management (pixi.toml)

### Version Management
- **Bump version**: `./scripts/bump_version.py --patch|--minor|--major`
- **Create release tag**: `./scripts/release-tag.py --patch|--minor|--major`

## Architecture and Code Structure

### Core Components
- **C API** (`api/c/`, `src/c/`): Core C implementation with graph algorithms
- **C++ API** (`api/cpp/`, `src/cpp/`): C++ wrapper with additional functionality
- **Python bindings** (`src/cpp/ptdalgorithmscpp_pybind.cpp`): pybind11-based Python interface
- **R bindings** (`R/`, `src/RcppExports.cpp`): Rcpp-based R interface

### Key Classes
- **Graph**: Main class for representing phase-type distributions
- **Vertex**: Represents states in the graph
- **Edge**: Represents transitions between states

### Language-Specific Entry Points
- **Python**: `src/ptdalgorithms/__init__.py` - Main Graph class with discretize() and plot() methods
- **R**: `R/package.r` - R wrapper functions
- **C++**: `api/cpp/ptdalgorithmscpp.h` - C++ API header

### Build System
- **Python**: Uses scikit-build-core with CMake backend
- **C/C++**: CMake-based build system generating shared libraries
- **R**: Standard R package structure with Makevars

### Documentation
- **Quarto docs**: `docs/` directory with API documentation and examples
- **Jupyter notebooks**: Extensive examples in `docs/examples/` and `docs/pages/`
- **API docs**: Auto-generated from source code

### Testing Structure
- **Python tests**: `tests/test_*.py` using pytest
- **R tests**: `tests/testthat/` directory using testthat framework
- **C++ tests**: `test/` directory with basic test files

### Dependencies
- **Core**: Eigen (linear algebra), pybind11 (Python bindings), Rcpp (R bindings)
- **Optional**: JAX (for jax_extension), HDF5, Boost
- **Development**: conda-build, pytest, testthat

## Development Workflow

### Debugging
- VSCode debugging configurations available for Python API, C++, and R API
- Requires conda environment setup and appropriate VSCode extensions
- Python debugging runs unit tests and code in `.vscode/debug.py`

### Code Organization
- Multi-language project with shared C/C++ core
- Language-specific wrappers maintain consistent API across languages
- Examples and documentation in multiple formats (notebooks, R scripts, C++ files)

### JAX Extension
- Experimental JAX integration in `jax_extension/` directory
- Provides JAX-compatible operations for phase-type distributions
- Separate build system using dedicated Makefile