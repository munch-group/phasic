#!/usr/bin/env Rscript

# Simple test that uses the system Python with phasic already installed
cat("Testing phasic R API (using system Python)\n")
cat("===============================================\n\n")

# Load packages
suppressPackageStartupMessages({
  library(reticulate)
  library(R6)
})

# Point to pixi environment Python
pixi_python <- "/Users/kmt/phasic/.pixi/envs/default/bin/python"
if (file.exists(pixi_python)) {
  cat("Using pixi Python environment\n")
  use_python(pixi_python, required = TRUE)
} else {
  cat("Warning: pixi Python not found, using system Python\n")
}

# Check Python config
py_cfg <- py_config()
cat("Python:", py_cfg$python, "\n")
cat("Python version:", py_cfg$version, "\n\n")

# Check if phasic is available
has_phasic <- py_module_available("phasic")
cat("Phasic available:", has_phasic, "\n\n")

if (!has_phasic) {
  cat("❌ Python phasic module not available\n")
  cat("Please install: pip install -e .\n")
  quit(status = 1)
}

# Source R files
cat("Loading R package files...\n")
source("R/zzz.R")
source("R/phasic-package.R")
source("R/graph.R")
source("R/trace_elimination.R")
source("R/svgd.R")

# Initialize
.onLoad(NULL, NULL)

cat("✓ R package loaded\n\n")

# Test basic functionality
cat("Running tests:\n")
cat("==============\n\n")

# Test 1: Create graph
cat("1. Creating simple graph... ")
g <- create_graph(state_length = 1)
cat("✓ (", g$vertices_length(), " vertices)\n", sep = "")

# Test 2: Coalescent graph
cat("2. Building coalescent graph... ")
coalescent_callback <- function(state) {
  n <- state[1]
  if (n <= 1) return(list())
  rate <- n * (n - 1) / 2
  list(list(c(n - 1), 0.0, c(rate)))
}

coal_graph <- create_graph(
  callback = coalescent_callback,
  parameterized = TRUE,
  state_length = 1,
  nr_samples = 5
)
cat("✓ (", coal_graph$vertices_length(), " vertices)\n", sep = "")

# Test 3: Trace recording
cat("3. Recording elimination trace... ")
trace <- record_elimination_trace(coal_graph, param_length = 1)
cat("✓\n")

# Test 4: Instantiate from trace
cat("4. Instantiating from trace... ")
concrete_graph <- instantiate_from_trace(trace, theta = c(2.0))
cat("✓\n")

# Test 5: PDF computation
cat("5. Computing PDF... ")
pdf_val <- concrete_graph$pdf(c(1.0))
cat("✓ (pdf(1.0) = ", round(pdf_val, 6), ")\n", sep = "")

# Test 6: Vectorized PDF
cat("6. Computing vectorized PDF... ")
times <- c(0.5, 1.0, 1.5, 2.0)
pdf_values <- concrete_graph$pdf(times)
cat("✓ (", length(pdf_values), " values)\n", sep = "")

# Test 7: Log-likelihood
cat("7. Creating log-likelihood function... ")
observed_times <- c(1.0, 2.0, 1.5)
log_lik <- trace_to_log_likelihood(trace, observed_times)
cat("✓\n")

# Test 8: Serialization
cat("8. Serializing graph... ")
json_str <- concrete_graph$serialize()
cat("✓ (", nchar(json_str), " chars)\n", sep = "")

cat("\n✅ All tests passed!\n")
cat("\nThe R API is working correctly with the Python backend.\n")
