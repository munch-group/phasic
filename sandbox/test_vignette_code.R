#!/usr/bin/env Rscript

# Test that code blocks in vignettes actually run without errors
cat("Testing Vignette Code Execution\n")
cat("=================================\n\n")

# Setup
suppressPackageStartupMessages({
  if (!require("knitr", quietly = TRUE)) {
    install.packages("knitr", repos = "https://cloud.r-project.org")
  }
  library(knitr)
})

# Load R package files (without Python backend for now)
cat("Loading R package files...\n")
source("R/zzz.R")
source("R/phasic-package.R")
source("R/graph.R")
source("R/trace_elimination.R")
source("R/svgd.R")
cat("✓ R files loaded\n\n")

# Extract code from vignettes
vignettes <- c(
  "vignettes/basic-usage.Rmd",
  "vignettes/svgd-inference.Rmd"
)

for (vignette in vignettes) {
  cat("========================================\n")
  cat("Testing:", basename(vignette), "\n")
  cat("========================================\n\n")

  # Extract R code
  temp_r <- tempfile(fileext = ".R")
  tryCatch({
    purl(vignette, output = temp_r, quiet = TRUE, documentation = 0)
  }, error = function(e) {
    cat("✗ Failed to extract code:", e$message, "\n")
    next
  })

  if (!file.exists(temp_r)) {
    cat("✗ No code extracted\n\n")
    next
  }

  # Read extracted code
  code_lines <- readLines(temp_r, warn = FALSE)
  unlink(temp_r)

  # Filter out setup chunks and library calls that need Python
  # We'll just check the R code syntax
  cat("Extracted", length(code_lines), "lines of code\n")

  # Parse the code to check for syntax errors
  cat("Checking R syntax... ")
  syntax_ok <- tryCatch({
    parse(text = code_lines)
    TRUE
  }, error = function(e) {
    cat("\n✗ Syntax error:", e$message, "\n")
    FALSE
  })

  if (syntax_ok) {
    cat("✓ No syntax errors\n")
  }

  # Analyze the code
  cat("\nCode analysis:\n")

  # Count function calls
  library_calls <- grep("library\\(", code_lines, value = TRUE)
  if (length(library_calls) > 0) {
    cat("  Library calls:", length(library_calls), "\n")
    cat("    -", paste(library_calls[1:min(3, length(library_calls))], collapse = "\n    - "), "\n")
  }

  # Check for phasic functions
  phasic_funcs <- c(
    "create_graph", "record_elimination_trace", "instantiate_from_trace",
    "run_svgd", "trace_to_log_likelihood", "create_pmf_model"
  )

  for (func in phasic_funcs) {
    pattern <- paste0(func, "\\(")
    if (any(grepl(pattern, code_lines))) {
      cat("  ✓ Uses", func, "\n")
    }
  }

  # Check for common issues
  cat("\nPotential issues:\n")
  issues <- 0

  # Check for undefined variables that might cause errors
  if (any(grepl("\\$theta_mean", code_lines))) {
    cat("  ⚠ Uses results$theta_mean (requires SVGD to run)\n")
    issues <- issues + 1
  }

  if (any(grepl("ggplot", code_lines))) {
    cat("  ⚠ Uses ggplot2 (optional dependency)\n")
    issues <- issues + 1
  }

  if (issues == 0) {
    cat("  ✓ No obvious issues detected\n")
  }

  cat("\n")
}

cat("========================================\n")
cat("Summary\n")
cat("========================================\n\n")

cat("Note: These tests check R syntax only.\n")
cat("Actual execution requires:\n")
cat("  1. Python environment with phasic installed\n")
cat("  2. Proper reticulate configuration\n")
cat("  3. JAX for SVGD examples\n\n")

cat("To test actual execution, run:\n")
cat("  library(phasic)  # with Python backend configured\n")
cat("  # Then manually run code from vignettes\n")
