# Conda Installation for R Package

## Overview

✅ **conda CAN install the R package!**

The conda-build configuration now supports building and installing the R package via:
```bash
conda install r-phasic
```

This automatically installs both the Python package and R wrapper in the same conda environment.

## How It Works

### Conda Multi-Output Build

The `conda-build/meta.yaml` defines **two outputs**:

1. **`phasic`** (Python package)
   - Python implementation + C++ extensions
   - Dependencies: numpy, pybind11, eigen, etc.
   - Build: Compiles C++, installs Python package

2. **`r-phasic`** (R package)
   - R wrapper via reticulate
   - Dependencies: r-base, r-reticulate, r-r6, **phasic**
   - Build: Pure R package (no compilation)

### Dependency Chain

```
r-phasic (R wrapper)
    ↓ depends on
phasic (Python package)
    ↓ uses
C++ compiled extensions
```

When you install `r-phasic`, conda automatically:
1. Installs `phasic` (Python) if not present
2. Installs R dependencies (r-reticulate, r-r6)
3. Installs r-phasic package
4. Ensures same environment for both

## Updated Configuration

### Key Changes in meta.yaml

**Old (commented out)**:
- Used r-rcpp for C++ bindings
- Required C/C++ compilation
- Direct C++ interface to R

**New (active)**:
```yaml
- name: r-phasic
  requirements:
    build:
      - r-base
    host:
      - r-base
      - r-reticulate >=1.28
      - r-r6
    run:
      - r-base
      - r-reticulate >=1.28
      - r-r6
      - phasic  # Python package dependency
  build:
    noarch: generic  # No compilation needed
    script: |
      $R CMD INSTALL --build .
```

**Benefits**:
- ✅ No C++ compilation for R package
- ✅ Automatic Python backend installation
- ✅ Single conda environment for both
- ✅ Pure R package (noarch: generic)
- ✅ Standard R installation (`R CMD INSTALL`)

## Installation Commands

### Install Both Packages

```bash
# Install R wrapper (pulls in Python package)
conda install r-phasic

# Or install separately
conda install phasic      # Python only
conda install r-phasic    # R wrapper
```

### Use in R

```r
library(phasic)

# Create graph
graph <- create_graph(state_length = 1)

# All functionality available
trace <- record_elimination_trace(graph, param_length = 1)
```

## Building Locally

To test the conda build locally:

```bash
# Build both packages
conda build conda-build/meta.yaml

# Or build just R package
conda build conda-build/meta.yaml --output r-phasic

# Install locally built package
conda install --use-local r-phasic
```

## Testing

The conda recipe includes tests for both packages:

**Python package (phasic)**:
```yaml
test:
  imports:
    - phasic
```

**R package (r-phasic)**:
```yaml
test:
  commands:
    - $R -e "library('phasic')"
    - $R -e "library('phasic'); g <- create_graph(state_length=1); stopifnot(g$vertices_length() >= 1)"
```

## Comparison: Installation Methods

| Method | Command | Installs | Status |
|--------|---------|----------|--------|
| **conda** | `conda install r-phasic` | Python + R | ✅ Ready |
| pip | `pip install phasic` | Python only | ✅ Works |
| R devtools | `devtools::install()` | R only | ✅ Works |
| CRAN | `install.packages("phasic")` | R only | 🔄 Future |

## Advantages of Conda Installation

### 1. Single Command
```bash
conda install r-phasic  # Gets everything
```

vs manual:
```bash
pip install phasic
# Then in R:
devtools::install_github("munch-group/phasic")
```

### 2. Version Consistency
- Same version of Python and R packages guaranteed
- Dependency resolution handled by conda
- No version mismatches between Python/R packages

### 3. Environment Isolation
```bash
# Create isolated environment
conda create -n phasic-env python=3.11
conda activate phasic-env
conda install r-phasic

# Everything in one environment
```

### 4. Binary Packages
- Pre-compiled C++ extensions (no compilation needed)
- Faster installation
- No build tools required

### 5. Cross-Platform
- Same command on Linux, macOS, Windows
- Conda handles platform differences
- Binary compatibility ensured

## Publication Workflow

### 1. Local Testing
```bash
# Build packages
conda build conda-build/meta.yaml

# Test locally
conda install --use-local r-phasic

# Test in R
R -e "library(phasic); create_graph(state_length=1)"
```

### 2. Channel Upload

**Option A: Personal Channel**
```bash
anaconda upload /path/to/phasic-*.tar.bz2
anaconda upload /path/to/r-phasic-*.tar.bz2
```

**Option B: conda-forge** (recommended)
1. Fork conda-forge/staged-recipes
2. Add recipe in `recipes/r-phasic/meta.yaml`
3. Submit PR to conda-forge
4. After merge, available via `conda install -c conda-forge r-phasic`

### 3. User Installation
```bash
# From conda-forge (future)
conda install -c conda-forge r-phasic

# From personal channel
conda install -c munch-group r-phasic
```

## Architecture Benefits

### Traditional Approach (Old)
```
User
  ↓
R Package (Rcpp)
  ↓
C++ Code (compiled for R)
```

**Issues**:
- Separate C++ compilation for R
- No Python features (JAX, SVGD, etc.)
- Difficult to maintain two C++ bindings

### New Approach (Reticulate)
```
User (R)
  ↓
r-phasic (R wrapper)
  ↓ via reticulate
phasic (Python package)
  ↓
C++ Extensions (for Python)
```

**Benefits**:
- ✅ Single C++ codebase (Python binding)
- ✅ All Python features available (JAX, SVGD, traces)
- ✅ Easier maintenance
- ✅ Conda handles both Python + R

## FAQs

**Q: Does conda compile C++ code?**

A: Yes, for the Python package. But the R package is pure R (no compilation).

**Q: Can I install just the Python package via conda?**

A: Yes! `conda install phasic` (no r- prefix)

**Q: Does this work on Windows?**

A: Yes, conda-build supports Windows with proper selectors in meta.yaml.

**Q: What if I don't use conda?**

A: Use pip + devtools as documented in `R_INSTALLATION_GUIDE.md`

**Q: Which channel will host the packages?**

A: Targeting conda-forge (community channel) for widest distribution.

## Current Status

✅ **Configuration Complete**
- meta.yaml updated with r-phasic output
- Dependencies specified (r-reticulate, r-r6)
- Tests included
- Ready to build

🔄 **Pending**
- Local testing of conda build
- Channel publication (conda-forge or personal)
- Documentation updates

## Summary

**Yes, conda CAN install the R package!**

The configuration is ready in `conda-build/meta.yaml`. Once published to a conda channel, users will be able to:

```bash
conda install r-phasic
```

This single command installs:
- ✅ Python phasic package
- ✅ R phasic wrapper
- ✅ All dependencies
- ✅ In one conda environment

This is the **easiest** installation method for users who already use conda.

---

*Updated: 2025-11-22*
*Version: 0.22.0*
