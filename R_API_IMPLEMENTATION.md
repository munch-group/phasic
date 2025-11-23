# R API Implementation Summary

## Overview

Replaced the old Rcpp-based R API with a new reticulate-based API that wraps the Python implementation. This provides access to all Phase 1-5 features including trace elimination, exact PDF computation, and SVGD inference.

## Changes Made

### 1. Removed Old Files
- Deleted `R/RcppExports.R` (1200+ lines of auto-generated Rcpp code)
- Deleted `R/package.r` and `R/phasic-package.r`
- Deleted `DESCRIPTION` and `NAMESPACE` (old Rcpp-based)
- Deleted `tests/testthat/` (old tests for Rcpp API)

### 2. Created New Package Structure

**Package metadata:**
- `DESCRIPTION` - Updated to use reticulate instead of Rcpp
  - Dependencies: reticulate (>= 1.28), R6
  - Version: 0.22.0 (matches Python version)
  - Authors: Kasper Munch (maintainer), Tobias Røikjer, Asger Hobolth

**Core R files:**
- `R/zzz.R` - Package initialization, Python module import
- `R/phasic-package.R` - Package documentation
- `R/graph.R` - R6 Graph class wrapper (300 lines)
- `R/trace_elimination.R` - Trace recording/evaluation functions
- `R/svgd.R` - SVGD inference wrappers
- `NAMESPACE` - Exports for all public functions

**Documentation:**
- `vignettes/basic-usage.Rmd` - Getting started guide
- `vignettes/svgd-inference.Rmd` - Bayesian inference tutorial
- `src/R/README.md` - R-specific documentation

**Tests:**
- `tests/testthat.R` - Test configuration
- `tests/testthat/test-graph.R` - Graph construction tests
- `tests/testthat/test-trace.R` - Trace elimination tests
- `tests/testthat/test-svgd.R` - SVGD inference tests

## Key Features

### Graph Construction
```r
# Callback-based (recommended)
coalescent_callback <- function(state) {
  n <- state[1]
  if (n <= 1) return(list())
  rate <- n * (n - 1) / 2
  list(list(c(n - 1), 0.0, c(rate)))
}

graph <- create_graph(
  callback = coalescent_callback,
  parameterized = TRUE,
  state_length = 1,
  nr_samples = 5
)
```

### Trace-Based Elimination (Phase 1-2)
```r
# Record once (slow)
trace <- record_elimination_trace(graph, param_length = 1)

# Evaluate many times (fast)
concrete_graph <- instantiate_from_trace(trace, theta = c(2.0))
pdf_values <- concrete_graph$pdf(times)
```

### Exact PDF Computation (Phase 4)
```r
# Exact phase-type likelihood
log_lik <- trace_to_log_likelihood(
  trace,
  observed_times,
  granularity = 100
)
```

### SVGD Inference (Phase 3)
```r
results <- run_svgd(
  graph = graph,
  observed_data = observed_times,
  theta_dim = 1,
  n_particles = 100,
  n_iterations = 1000
)

print(results$theta_mean)
print(results$theta_std)
```

## API Design Principles

1. **R Conventions**:
   - Snake_case function names
   - R6 classes for OOP
   - Idiomatic R patterns (lists, vectors)

2. **Reticulate Integration**:
   - Automatic Python module import on load
   - Type conversion handled by reticulate
   - Error handling for missing Python dependencies

3. **Documentation**:
   - Roxygen2 for function docs
   - Vignettes for tutorials
   - Examples in all exported functions

4. **Testing**:
   - Testthat framework
   - Skip tests if Python/phasic unavailable
   - Coverage for core functionality

## Architecture

```
R Package (reticulate wrapper)
    ↓
Python phasic module
    ↓
C implementation (via pybind11)
```

**Benefits:**
- Full access to JAX features (jit, grad, vmap)
- All Phase 1-5 features available
- Simpler maintenance (single implementation)
- Automatic updates when Python API changes

**Trade-offs:**
- Requires Python environment setup
- Slightly more overhead than direct C bindings
- Dependency on reticulate package

## Installation

```r
# Install Python package first
# pip install phasic

# Install R dependencies
install.packages(c("reticulate", "R6"))

# Install R package
devtools::install(".")
```

## Testing

```r
# Run all tests
devtools::test()

# Run specific test file
testthat::test_file("tests/testthat/test-graph.R")

# Build and check package
devtools::check()
```

## Documentation Generation

```r
# Generate documentation from roxygen comments
devtools::document()

# Build vignettes
devtools::build_vignettes()

# View package help
?phasic
```

## Future Enhancements

1. **Additional wrappers**:
   - More Python API functions
   - Advanced JAX features
   - Multivariate models

2. **Performance**:
   - Caching of Python objects
   - Batch operations
   - Parallel SVGD

3. **Visualization**:
   - Plot methods for Graph objects
   - SVGD convergence diagnostics
   - Distribution comparison plots

4. **Integration**:
   - Stan/JAGS compatibility
   - tidyverse patterns
   - ggplot2 themes

## Files Summary

Total new R files: 9
Total lines of code: ~2000 (vs ~1200 auto-generated Rcpp)

**Breakdown:**
- Core implementation: 600 lines
- Documentation: 800 lines
- Tests: 300 lines
- Vignettes: 300 lines

**Maintainability**: Much simpler than Rcpp approach, follows R best practices.

## References

- [reticulate documentation](https://rstudio.github.io/reticulate/)
- [R6 classes](https://r6.r-lib.org/)
- [R packages book](https://r-pkgs.org/)
- [Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6)
