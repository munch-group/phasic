#!/usr/bin/env Rscript

# Test the new R API
cat("Testing phasic R API\n")
cat("====================\n\n")

# Test 1: Load required packages
cat("Test 1: Loading dependencies... ")
tryCatch({
  suppressPackageStartupMessages({
    if (!require("reticulate", quietly = TRUE)) {
      install.packages("reticulate", repos = "https://cloud.r-project.org")
      library(reticulate)
    }
    if (!require("R6", quietly = TRUE)) {
      install.packages("R6", repos = "https://cloud.r-project.org")
      library(R6)
    }
  })
  cat("✓\n")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 2: Check Python phasic availability
cat("Test 2: Checking Python phasic module... ")
tryCatch({
  has_phasic <- py_module_available("phasic")
  if (!has_phasic) {
    cat("✗ (Python phasic not installed)\n")
    cat("Please install: pip install phasic\n")
    quit(status = 0)  # Not a failure, just not set up
  }
  cat("✓\n")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 3: Source R package files
cat("Test 3: Loading R package files... ")
tryCatch({
  source("R/zzz.R")
  source("R/phasic-package.R")
  source("R/graph.R")
  source("R/trace_elimination.R")
  source("R/svgd.R")

  # Initialize the package
  .onLoad(NULL, NULL)

  cat("✓\n")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 4: Create a simple graph
cat("Test 4: Creating simple graph... ")
tryCatch({
  g <- create_graph(state_length = 1)
  n_vertices <- g$vertices_length()
  if (n_vertices < 1) {
    stop("Graph has no vertices")
  }
  cat("✓ (", n_vertices, " vertices)\n", sep = "")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 5: Callback-based graph construction
cat("Test 5: Building coalescent graph... ")
tryCatch({
  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) {
      return(list())
    }
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  coal_graph <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 5
  )

  n_vertices <- coal_graph$vertices_length()
  if (n_vertices < 5) {
    stop("Coalescent graph has too few vertices")
  }
  cat("✓ (", n_vertices, " vertices)\n", sep = "")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 6: Trace recording
cat("Test 6: Recording elimination trace... ")
tryCatch({
  trace <- record_elimination_trace(coal_graph, param_length = 1)
  if (is.null(trace)) {
    stop("Trace is NULL")
  }
  cat("✓\n")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 7: Instantiate from trace
cat("Test 7: Instantiating graph from trace... ")
tryCatch({
  concrete_graph <- instantiate_from_trace(trace, theta = c(2.0))
  n_vertices <- concrete_graph$vertices_length()
  cat("✓ (", n_vertices, " vertices)\n", sep = "")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 8: PDF computation
cat("Test 8: Computing PDF... ")
tryCatch({
  pdf_val <- concrete_graph$pdf(c(1.0))
  if (length(pdf_val) != 1) {
    stop("Expected 1 PDF value")
  }
  if (pdf_val < 0) {
    stop("PDF should be non-negative")
  }
  cat("✓ (pdf(1.0) = ", round(pdf_val, 6), ")\n", sep = "")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 9: Log-likelihood function
cat("Test 9: Creating log-likelihood function... ")
tryCatch({
  observed_times <- c(1.0, 2.0, 1.5)
  log_lik <- trace_to_log_likelihood(trace, observed_times)

  if (!is.function(log_lik)) {
    stop("log_lik is not a function")
  }
  cat("✓\n")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Test 10: Graph serialization
cat("Test 10: Serializing graph... ")
tryCatch({
  json_str <- concrete_graph$serialize()
  if (!is.character(json_str) || nchar(json_str) == 0) {
    stop("Invalid JSON output")
  }
  cat("✓ (", nchar(json_str), " characters)\n", sep = "")
}, error = function(e) {
  cat("✗\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

cat("\n")
cat("All tests passed! ✓\n")
cat("\nR API is working correctly.\n")
