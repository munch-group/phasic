# CRAN Submission Comments

## Test environments

* local: macOS 26.1 (aarch64-apple-darwin20), R 4.4.0
* win-builder: (release and devel)
* R-hub: (ubuntu-latest, windows-latest, macos-latest)

## R CMD check results

0 errors ✓ | 0 warnings ✓ | 0 notes ✓

R CMD check --as-cran succeeded with no errors, warnings, or notes.

## Python Dependency

This package provides an R interface to a Python implementation via the 'reticulate' package.

### SystemRequirements

The package declares `SystemRequirements: Python (>= 3.8)` and uses the Config/reticulate field in DESCRIPTION to specify the Python package dependency.

### Handling of Python Dependency

1. **Package Loading**: The package uses `delay_load = TRUE` when importing the Python module in `.onLoad()`, allowing the R package to load successfully even when Python is not available. This ensures CRAN checks can proceed.

2. **User Notification**: When users load the package with `library(phasic)` and Python is not available, `.onAttach()` displays a helpful message with installation instructions via the `install_phasic()` helper function.

3. **Tests**: All tests use `skip_if(!reticulate::py_module_available("phasic"))` to skip when Python backend is unavailable, following reticulate best practices.

4. **Examples**: Examples that require Python are wrapped in `\dontrun{}` to prevent errors during CRAN checks.

5. **Vignettes**: Vignettes use `eval = FALSE` in code chunks to prevent execution when Python backend is unavailable. This follows the pattern used by other reticulate-based packages on CRAN. The vignettes serve as code examples and documentation, with full functionality available when users install the Python backend.

## Installation Helper

The package provides `install_phasic()` function to help users install the required Python package:

```r
library(phasic)
install_phasic()  # Installs Python backend
```

## Downstream Dependencies

This is a new submission with no existing reverse dependencies.

## Additional Notes

This package follows the patterns established by other CRAN packages that wrap Python code via reticulate (e.g., tensorflow, keras, spacyr). The Python dependency is clearly documented and gracefully handled to ensure the package works in CRAN's testing environment while providing full functionality when Python is available.
