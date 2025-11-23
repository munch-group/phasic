# R API for phasic

The R API provides access to the Python implementation of phasic via the [reticulate](https://rstudio.github.io/reticulate/) package.

## Installation

### Quick Start

The R package **cannot** be installed via pip/conda. It must be installed using R tools.

**Option 1: From GitHub (when published)**
```r
# Install R package
devtools::install_github("munch-group/phasic")

# R will prompt to install Python backend if needed
library(phasic)
```

**Option 2: Local Development**
```bash
# 1. Install Python package first
pip install -e /path/to/phasic

# 2. Then install R package
```
```r
# From repository root in R
devtools::install(".", upgrade = FALSE)
library(phasic)
```

### Prerequisites

1. **R dependencies**:
   ```r
   install.packages(c("reticulate", "R6"))
   ```

2. **Python package** (installed automatically or manually):
   ```bash
   pip install phasic  # When published to PyPI
   # Or for development:
   pip install -e /path/to/phasic
   ```

### Why Not pip/conda?

**pip/conda only install Python packages.** The R package has a different structure (DESCRIPTION, NAMESPACE, R/ directory) and requires R's installation tools.

See [`R_INSTALLATION_GUIDE.md`](../../R_INSTALLATION_GUIDE.md) for detailed installation options.

## Quick Start

```r
library(phasic)

# Define a coalescent model
coalescent_callback <- function(state) {
  n <- state[1]
  if (n <= 1) return(list())
  rate <- n * (n - 1) / 2
  list(list(c(n - 1), 0.0, c(rate)))
}

# Build graph
graph <- create_graph(
  callback = coalescent_callback,
  parameterized = TRUE,
  state_length = 1,
  nr_samples = 5
)

# Record trace for efficient evaluation
trace <- record_elimination_trace(graph, param_length = 1)

# Instantiate with specific parameter
concrete_graph <- instantiate_from_trace(trace, theta = c(2.0))

# Compute PDF
times <- seq(0.1, 5.0, length.out = 50)
pdf_values <- concrete_graph$pdf(times)

# Plot
plot(times, pdf_values, type = "l",
     xlab = "Time", ylab = "Density",
     main = "Coalescent Distribution")
```

## Bayesian Inference with SVGD

```r
# Observed data
observed_times <- c(1.2, 2.3, 0.8, 1.9, 1.5)

# Run SVGD
results <- run_svgd(
  graph = graph,
  observed_data = observed_times,
  theta_dim = 1,
  n_particles = 100,
  n_iterations = 1000
)

cat("Posterior mean:", results$theta_mean, "\n")
cat("Posterior std:", results$theta_std, "\n")
```

## Documentation

- **Vignettes**:
  - `vignette("basic-usage")` - Getting started guide
  - `vignette("svgd-inference")` - Bayesian inference tutorial
- **Function help**: `?create_graph`, `?run_svgd`, etc.
- **Package overview**: `?phasic`

## Key Features

- **Graph construction**: Callback-based or manual building
- **PDF/PMF computation**: Forward algorithm with uniformization
- **Trace elimination**: Efficient repeated parameter evaluation
- **SVGD inference**: Bayesian posterior approximation
- **Multivariate models**: Reward transformation support

## Architecture

The R package wraps the Python implementation using reticulate:
- `R/zzz.R` - Package initialization and Python module import
- `R/graph.R` - R6 Graph class wrapper
- `R/trace_elimination.R` - Trace recording and evaluation
- `R/svgd.R` - SVGD inference functions

All core computations are performed by the Python backend, providing:
- Full access to JAX integration
- Automatic differentiation capabilities
- All Phase 1-5 features (trace elimination, exact PDF, SVGD)

## Troubleshooting

**Python module not found**:
```r
# Check if Python phasic is available
reticulate::py_module_available("phasic")

# Specify Python environment
reticulate::use_python("/path/to/python")
# or
reticulate::use_virtualenv("myenv")
```

**Version compatibility**:
Ensure Python phasic version matches R package requirements (>= 0.22.0).

## Contributing

See the main repository README for contribution guidelines.

## References

[Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6) - Statistics and Computing
