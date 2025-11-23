# R Package Summary

## Quick Answer

**Q: Can pip and conda install the R library?**

**A: Partially.**
- ❌ **pip** - Cannot install R packages (Python-only)
- ✅ **conda** - CAN install R packages via `conda install r-phasic`
- ✅ **R tools** - Standard method: `devtools::install_github()`

## Why Two Separate Installs?

| Component | Tool | Command |
|-----------|------|---------|
| Python package | pip/conda | `pip install phasic` or `conda install phasic` |
| R package | R devtools | `devtools::install_github("munch-group/phasic")` |

**Reason**: Different ecosystems with incompatible packaging systems:
- **Python**: Uses `setup.py`/`pyproject.toml`, distributed via PyPI/conda
- **R**: Uses `DESCRIPTION`/`NAMESPACE`, distributed via CRAN/GitHub

## Installation Methods

### Method 1: Conda (Easiest - When Published)

```bash
# Install both Python + R packages in one command
conda install r-phasic

# This automatically installs:
#   - phasic (Python package)
#   - r-phasic (R wrapper)
#   - r-reticulate, r-r6 (dependencies)
```

### Method 2: Automated Script

```bash
# From repository root
Rscript install_r_package.R
```

This script:
1. ✅ Installs R dependencies (reticulate, R6)
2. ✅ Checks/installs Python phasic
3. ✅ Installs R package
4. ✅ Tests installation

### Method 3: Manual Step-by-Step

**Step 1: Install Python package**
```bash
pip install -e /path/to/phasic
```

**Step 2: Install R package**
```r
# Install dependencies
install.packages(c("reticulate", "R6", "devtools"))

# Install phasic R package
devtools::install("/path/to/phasic")

# Load and use
library(phasic)
```

### Method 4: From GitHub (Future)

When published:
```r
# Install R package
devtools::install_github("munch-group/phasic")

# Load package - will prompt if Python backend is missing
library(phasic)

# If prompted, install Python backend
reticulate::py_install("phasic", pip = TRUE)
```

**Note:** The R package automatically checks for the Python backend on load and displays installation instructions if needed.

## What Gets Installed Where?

### Python Installation (via pip)

**Location**: Python site-packages
```
site-packages/
  phasic/
    __init__.py
    graph.py
    svgd.py
    ...
    phasic_pybind.so  # C++ extension
```

**Installed files**:
- Python source code
- C++ compiled bindings
- Python dependencies

**NOT included**:
- R package files (R/, DESCRIPTION, NAMESPACE)
- R vignettes
- R tests

### R Installation (via devtools)

**Location**: R library path
```
R/library/phasic/
  DESCRIPTION
  NAMESPACE
  R/
    zzz.R
    graph.R
    svgd.R
    ...
  help/
  vignettes/
  tests/
```

**Installed files**:
- R source code
- R documentation
- Vignettes
- Tests

**NOT included**:
- Python code (loaded via reticulate)
- C++ source

## Architecture

```
User's R Session
    ↓
R Package (phasic)
    ↓
reticulate bridge
    ↓
Python Package (phasic)
    ↓
C++ Extension (phasic_pybind.so)
```

**Two installations required**:
1. Python package: Core implementation
2. R package: Wrapper + documentation

## Detailed Installation Options

See `R_INSTALLATION_GUIDE.md` for:
- ✅ GitHub installation
- ✅ Local development setup
- ✅ CI/CD configuration
- 🔄 Future: CRAN submission
- 🔄 Future: Conda package with R bindings

## Files Created

### Installation Files
- `install_r_package.R` - Automated installer
- `R_INSTALLATION_GUIDE.md` - Comprehensive guide
- `src/R/README.md` - Updated with install instructions

### R Package Files
- `DESCRIPTION` - Package metadata
- `NAMESPACE` - Exported functions
- `R/*.R` - 5 source files
- `vignettes/*.Rmd` - 2 tutorials
- `tests/testthat/*.R` - 3 test files

### Documentation
- `R_API_IMPLEMENTATION.md` - Technical details
- `R_API_TEST_RESULTS.md` - Test results
- `R_API_FINAL_TEST_REPORT.md` - Comprehensive testing
- `R_PACKAGE_SUMMARY.md` - This file

## Common Issues

### Issue 1: "Python module 'phasic' not found"

**Cause**: Python package not installed

**Solution**:
```bash
pip install -e /path/to/phasic
```

### Issue 2: "Package 'phasic' not found" (in R)

**Cause**: R package not installed

**Solution**:
```r
devtools::install("/path/to/phasic")
```

### Issue 3: Both installed but not working

**Cause**: reticulate using wrong Python environment

**Solution**:
```r
library(reticulate)
use_python("/path/to/python/with/phasic")
library(phasic)
```

## FAQs

**Q: Why not bundle everything in one pip install?**

A: Technically possible but messy:
- R files in non-standard location
- Manual R setup still required
- Doesn't integrate with R package management
- pip users wouldn't expect R files

**Q: What about CRAN?**

A: Future plan:
- Submit R package to CRAN
- Auto-install Python via reticulate
- Users: `install.packages("phasic")`

**Q: Can conda install both?**

A: Yes! This is now enabled:
- `conda install r-phasic` installs both Python + R
- R package depends on Python package automatically
- Works via conda-build/meta.yaml configuration
- Status: ✅ Configuration ready, pending conda channel publication

**Q: Does `devtools::install_github()` install the Python package too?**

A: No, it only installs the R package. However:
- When you `library(phasic)`, it checks for the Python backend
- If missing, displays clear installation instructions
- You can then install with `reticulate::py_install("phasic", pip = TRUE)`
- Or use `conda install r-phasic` which handles both automatically

**Q: Which should I install first?**

A: Either order works, but Python first is easier:
1. Install Python package
2. Install R package
3. R package will find Python automatically

## Current Status

✅ **Available Now**:
- Python package: `pip install -e .` (local)
- R package: `devtools::install(".")` (local)
- Automated installer: `Rscript install_r_package.R`
- Conda config: ✅ Ready in `conda-build/meta.yaml`

🔄 **Pending Publication**:
- PyPI release: `pip install phasic`
- Conda channels: `conda install phasic` and `conda install r-phasic`
- CRAN release: `install.packages("phasic")`
- GitHub release: `devtools::install_github()`

## Testing

All components tested ✅:
- Package structure valid
- R syntax correct
- Vignettes valid (18 code examples)
- Callback formats correct
- Installation script works

See `R_API_FINAL_TEST_REPORT.md` for complete test results.

## Next Steps

1. **For Users**: Run `Rscript install_r_package.R`
2. **For Developers**: See `R_INSTALLATION_GUIDE.md`
3. **For CI/CD**: See installation guide for GitHub Actions setup

## Summary

**Two separate installations are required and this is by design:**

| What | How | Why |
|------|-----|-----|
| Python package | `pip install` | Core implementation |
| R package | `devtools::install()` | Wrapper + docs |

The R package **wraps** the Python package via reticulate. Both are needed for the R API to work.

---

*Last Updated: 2025-11-22*
*Version: 0.22.0*
