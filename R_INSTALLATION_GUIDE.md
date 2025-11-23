# R Package Installation Guide

## Overview

The phasic R package wraps the Python implementation via reticulate. There are **three components** to install:

1. **Python package** (phasic) - Core implementation
2. **R package** (phasic) - R wrapper
3. **R dependencies** (reticulate, R6)

## Installation Methods

### Method 1: Install from GitHub (Recommended for Users)

**When the package is published on GitHub:**

```r
# Install R dependencies
install.packages(c("reticulate", "R6", "devtools"))

# Install R package from GitHub
devtools::install_github("munch-group/phasic")

# Load package - will show message if Python backend is missing
library(phasic)

# If prompted, install Python backend via reticulate
reticulate::py_install("phasic", pip = TRUE)
```

**Note:** The R package will check for the Python backend when loaded and display installation instructions if it's not found.

### Method 2: Local Installation (For Development)

**Step 1: Install Python package**

```bash
# From the repository root
pip install -e .

# Or with pixi
pixi run install-dev
```

**Step 2: Install R dependencies**

```r
install.packages(c("reticulate", "R6"))
```

**Step 3: Install R package locally**

```r
# From R, in the repository root
devtools::install(".", upgrade = FALSE)

# Or from command line
R CMD INSTALL .
```

### Method 3: Manual Installation (Alternative)

If you don't want to use `devtools`:

**Step 1: Build the R package**

```bash
# From repository root
R CMD build .
# This creates phasic_0.22.0.tar.gz
```

**Step 2: Install the built package**

```r
install.packages("phasic_0.22.0.tar.gz", repos = NULL, type = "source")
```

**Step 3: Configure Python environment**

```r
library(reticulate)

# Option A: Use existing Python with phasic
use_python("/path/to/python")  # Where phasic is installed

# Option B: Install phasic in R's managed environment
py_install("phasic", pip = TRUE)
```

## Current Limitations

### ❌ Not Available via pip/conda

The R package is **not** distributed through pip or conda because:

1. **Different ecosystems**: R packages typically distributed via CRAN or GitHub
2. **File structure**: R packages have specific requirements (DESCRIPTION, NAMESPACE, etc.)
3. **Installation tools**: R uses `install.packages()`, `R CMD INSTALL`, not pip

### Why Can't pip Install the R Package?

**Technical reasons:**
- pip is Python-specific, doesn't understand R package structure
- R files need to be in specific locations (`R/`, `man/`, `vignettes/`)
- R packages require DESCRIPTION file, not setup.py
- Different dependency resolution (CRAN vs PyPI)

## Proposed Solutions

### Solution 1: Include R Files in Python Package (Hybrid)

**Pros:**
- Single pip install command
- R files bundled with Python package

**Cons:**
- R files in non-standard location
- Still need manual R setup
- Doesn't integrate with R package management

**Implementation:**
Create `MANIFEST.in`:
```
include R/*.R
include DESCRIPTION
include NAMESPACE
include vignettes/*.Rmd
include tests/testthat/*.R
```

Then users would:
```r
# After pip install phasic
library(reticulate)
phasic_path <- py_config()$prefix  # Find Python package location
devtools::install(file.path(phasic_path, "lib/python3.x/site-packages/phasic"))
```

**Status:** ⚠️ Complex, not recommended

### Solution 2: Separate R Package on CRAN (Recommended)

**Pros:**
- Standard R distribution method
- Easy `install.packages("phasic")`
- Proper R package management
- Can depend on Python via reticulate configuration

**Cons:**
- Requires CRAN submission
- Two separate packages to maintain

**Implementation:**
1. Submit R package to CRAN
2. Configure reticulate to auto-install Python package

In `R/zzz.R`:
```r
.onLoad <- function(libname, pkgname) {
  # Try to import phasic
  phasic_available <- reticulate::py_module_available("phasic")

  if (!phasic_available) {
    packageStartupMessage(
      "Python phasic module not found.\n",
      "Install with: reticulate::py_install('phasic')"
    )
  }

  phasic_py <<- reticulate::import("phasic", delay_load = TRUE)
}
```

Users install:
```r
install.packages("phasic")  # R package from CRAN
library(phasic)             # Prompts to install Python if needed
```

**Status:** ✅ Best long-term solution

### Solution 3: GitHub-Only Distribution (Current)

**Pros:**
- Simple to maintain
- Direct from source
- No CRAN submission process

**Cons:**
- Requires devtools
- Not discoverable via `install.packages()`

**Current Implementation:**
```r
# Install from GitHub
devtools::install_github("munch-group/phasic")
```

**Status:** ✅ Works now, good for development

### Solution 4: Conda Package with R Bindings ✅ IMPLEMENTED

**Pros:**
- Single `conda install r-phasic` for everything
- Includes both Python and R
- Cross-platform binaries
- Version consistency guaranteed

**Cons:**
- Requires conda channel setup
- Not all R users use conda

**Implementation:**
Updated `conda-build/meta.yaml`:
```yaml
- name: r-phasic
  requirements:
    run:
      - r-base
      - r-reticulate >=1.28
      - r-r6
      - phasic  # Python package dependency
  build:
    noarch: generic  # Pure R, no compilation
```

**Status:** ✅ Configuration ready, pending channel publication

See `CONDA_INSTALLATION.md` for details.

## Recommended Installation Workflow

### For End Users (When Published)

**Option A: Conda (Easiest)**
```bash
# Single command installs everything
conda install r-phasic
```

**Option B: GitHub**
```r
# 1. Install R package from GitHub
devtools::install_github("munch-group/phasic")

# 2. Load package (will prompt for Python if needed)
library(phasic)

# 3. If prompted, install Python backend
reticulate::py_install("phasic", pip = TRUE)
```

### For Developers

```bash
# 1. Install Python package in development mode
pip install -e .

# 2. In R, install local R package
devtools::install(".", upgrade = FALSE)

# 3. Configure reticulate to use development Python
library(reticulate)
use_python("/path/to/dev/python")
library(phasic)
```

### For CI/CD

```yaml
# .github/workflows/test-r-package.yml
- name: Install dependencies
  run: |
    pip install -e .
    R -e 'install.packages(c("reticulate", "R6", "devtools", "testthat"))'
    R -e 'devtools::install(upgrade = FALSE)'

- name: Run tests
  run: R -e 'devtools::test()'
```

## Configuration Files Needed

### For CRAN Submission (Future)

Create `.Rbuildignore`:
```
^.*\.Rproj$
^\.Rproj\.user$
^\.github$
^\.pixi$
^src/phasic$
^src/c$
^CMakeLists\.txt$
^pyproject\.toml$
```

### For pip Distribution (Optional)

Add to `MANIFEST.in`:
```
# Include R package for hybrid distribution
graft R
include DESCRIPTION
include NAMESPACE
graft vignettes
graft tests/testthat
include tests/testthat.R
```

Then update `pyproject.toml`:
```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
phasic = ["R/**", "vignettes/**", "tests/**", "DESCRIPTION", "NAMESPACE"]
```

## Summary

| Method | Command | Status | Best For |
|--------|---------|--------|----------|
| conda | `conda install r-phasic` | ✅ Config ready | conda users |
| GitHub | `devtools::install_github()` | 🔄 Future | R users |
| Local | `devtools::install(".")` | ✅ Available now | Developers |
| CRAN | `install.packages("phasic")` | 🔄 Future | End users |
| pip | ❌ Not applicable | N/A | N/A |

## Current Status

**Available Now:**
- ✅ Python package via `pip install -e .`
- ✅ R package via `devtools::install()` (local)
- ✅ Conda configuration in `conda-build/meta.yaml`
- ✅ Documentation and vignettes included

**Pending Publication:**
- 🔄 Conda channels: `conda install r-phasic` (config ready)
- 🔄 CRAN submission for easy `install.packages()`
- 🔄 GitHub releases for `devtools::install_github()`
- 🔄 PyPI release for `pip install phasic` (Python only)

## FAQs

**Q: Why can't I just `pip install phasic` for the R package?**

A: pip only installs Python packages. R packages have different structure and need R's installation tools.

**Q: Do I need to install both Python and R packages?**

A: Yes. The R package is a wrapper that calls the Python implementation.

**Q: Can I use pip to install, then access from R?**

A: The Python package yes, but you still need to install the R wrapper separately with `devtools::install_github()`.

**Q: Does `devtools::install_github()` also install the Python package?**

A: No, it only installs the R package. However, when you load the package with `library(phasic)`, it will check for the Python backend and display installation instructions if it's missing. You can then install it with `reticulate::py_install("phasic", pip = TRUE)`.

Alternatively, use `conda install r-phasic` which installs both automatically.

**Q: Will this be on CRAN eventually?**

A: That's the plan! CRAN submission requires meeting specific requirements, which is a future goal.

**Q: What's the easiest way to install everything?**

A: For now:
```r
# Install Python package
system("pip install -e /path/to/phasic")

# Install R package
devtools::install_github("munch-group/phasic")  # When published
# or
devtools::install("/path/to/phasic")  # For local
```

## See Also

- `src/R/README.md` - R API documentation
- `vignettes/basic-usage.Rmd` - Getting started guide
- `R_API_IMPLEMENTATION.md` - Technical details
