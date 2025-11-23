# R Package Installation Check Enhancement

**Date:** 2025-11-23
**Version:** 0.22.0

## Summary

Enhanced the R package to automatically check for the Python backend on load and prompt users with installation instructions if it's missing.

## Problem

**User Question:** "Does devtools::install_github() also install the python package?"

**Answer:** No, `devtools::install_github()` only installs the R package. The Python backend must be installed separately.

## Solution

Updated `R/zzz.R` to add a Python module availability check in `.onLoad()`:

```r
.onLoad <- function(libname, pkgname) {
  # Import Python phasic module using reticulate
  phasic_py <<- reticulate::import("phasic", delay_load = TRUE)

  # Check if Python module is actually available
  if (!reticulate::py_module_available("phasic")) {
    packageStartupMessage(
      "\n",
      "Python phasic module not found.\n",
      "The R package requires the Python backend to be installed.\n",
      "\n",
      "Install with one of these methods:\n",
      "  1. reticulate::py_install('phasic', pip = TRUE)\n",
      "  2. From terminal: pip install phasic\n",
      "  3. From terminal: pip install -e /path/to/phasic  (development mode)\n",
      "\n"
    )
  }
}
```

## User Experience

### Before Enhancement

```r
# Install R package
devtools::install_github("munch-group/phasic")

# Load package - no warning
library(phasic)

# Try to use - confusing error
graph <- create_graph(state_length = 1)
# Error: Python phasic module not loaded. Please ensure phasic is installed in Python.
```

### After Enhancement

```r
# Install R package
devtools::install_github("munch-group/phasic")

# Load package - clear warning with instructions
library(phasic)
#
# Python phasic module not found.
# The R package requires the Python backend to be installed.
#
# Install with one of these methods:
#   1. reticulate::py_install('phasic', pip = TRUE)
#   2. From terminal: pip install phasic
#   3. From terminal: pip install -e /path/to/phasic  (development mode)
#

# User follows instructions
reticulate::py_install("phasic", pip = TRUE)

# Now everything works
library(phasic)
graph <- create_graph(state_length = 1)
```

## Installation Methods Comparison

| Method | Command | Installs Python | Installs R | User Action Required |
|--------|---------|-----------------|------------|----------------------|
| **devtools::install_github()** | R package only | ❌ No | ✅ Yes | User must install Python separately (prompted on load) |
| **conda install r-phasic** | Both automatically | ✅ Yes | ✅ Yes | None - fully automatic |
| **Manual two-step** | Both manually | ✅ Yes | ✅ Yes | Full control |

## Files Modified

1. **`R/zzz.R`**
   - Added `py_module_available()` check in `.onLoad()`
   - Added informative `packageStartupMessage()` with installation instructions

2. **`R_INSTALLATION_GUIDE.md`**
   - Updated Method 1 to show package load and prompt behavior
   - Added FAQ: "Does devtools::install_github() also install the Python package?"

3. **`R_PACKAGE_SUMMARY.md`**
   - Updated Method 4 to show package load behavior
   - Added FAQ about devtools installation behavior
   - Added note about automatic checking

## Benefits

1. **User-Friendly**: Clear, actionable error message immediately on package load
2. **Non-Intrusive**: Uses `packageStartupMessage()` which can be suppressed if needed
3. **No Breaking Changes**: Package still loads successfully, just warns user
4. **Multiple Options**: Shows 3 different ways to install Python backend
5. **Consistent with Best Practices**: Follows reticulate package patterns

## Testing

The enhancement can be tested by:

1. Install R package without Python backend:
```r
devtools::install("/Users/kmt/phasic")
library(phasic)  # Should show warning message
```

2. Install Python backend:
```r
reticulate::py_install("phasic", pip = TRUE)
```

3. Reload R package:
```r
library(phasic)  # Should load silently (no warning)
```

4. Verify functionality:
```r
graph <- create_graph(state_length = 1)
graph$vertices_length()  # Should work
```

## Alternative Approaches Considered

### 1. Auto-Install Python Package
```r
if (!reticulate::py_module_available("phasic")) {
  message("Installing Python phasic...")
  reticulate::py_install("phasic", pip = TRUE)
}
```
**Rejected**: Too aggressive, users may want control over Python environment

### 2. Fail on Load
```r
if (!reticulate::py_module_available("phasic")) {
  stop("Python phasic not found!")
}
```
**Rejected**: Prevents package from loading, makes troubleshooting harder

### 3. Silent Load (Current Behavior Before This Change)
```r
phasic_py <<- reticulate::import("phasic", delay_load = TRUE)
```
**Rejected**: Confusing error only appears when user tries to use functions

### 4. Chosen Approach: Warn on Load ✅
- Package loads successfully
- Clear message with instructions
- User retains control
- Non-intrusive (can be suppressed)

## Related Documentation

- `R_INSTALLATION_GUIDE.md` - Comprehensive installation guide
- `R_PACKAGE_SUMMARY.md` - Quick reference
- `CONDA_INSTALLATION.md` - Conda-specific installation
- `install_r_package.R` - Automated installer script

## Conclusion

The R package now provides a better user experience by immediately informing users when the Python backend is missing, rather than waiting until they try to use a function. This addresses the user's question about `devtools::install_github()` behavior and makes the installation process clearer.

For users who want fully automatic installation of both components, `conda install r-phasic` remains the recommended approach.
