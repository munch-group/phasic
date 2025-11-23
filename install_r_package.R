#!/usr/bin/env Rscript

# Installation script for phasic R package
# Run with: Rscript install_r_package.R

cat("phasic R Package Installer\n")
cat("===========================\n\n")

# Check if we're in the right directory
if (!file.exists("DESCRIPTION") || !file.exists("R")) {
  cat("✗ Error: Run this script from the phasic repository root\n")
  cat("  (Directory should contain DESCRIPTION and R/ folder)\n")
  quit(status = 1)
}

# Step 1: Check/install R dependencies
cat("Step 1: Installing R dependencies...\n")
required_packages <- c("reticulate", "R6", "devtools")

for (pkg in required_packages) {
  if (!require(pkg, quietly = TRUE, character.only = TRUE)) {
    cat("  Installing", pkg, "...\n")
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  } else {
    cat("  ✓", pkg, "already installed\n")
  }
}

cat("\n")

# Step 2: Check Python phasic availability
cat("Step 2: Checking Python phasic module...\n")
library(reticulate, quietly = TRUE)

has_phasic <- py_module_available("phasic")

if (has_phasic) {
  cat("  ✓ Python phasic module found\n")

  # Try to get version
  tryCatch({
    phasic_version <- py_run_string("import phasic; print(phasic.__version__)", convert = FALSE)
    cat("    Version:", phasic_version, "\n")
  }, error = function(e) {})

} else {
  cat("  ✗ Python phasic module not found\n")
  cat("\n")
  cat("  The Python backend must be installed first.\n")
  cat("  Choose an installation method:\n\n")

  cat("  Option A: Install from this repository (development mode)\n")
  cat("    Run from terminal:\n")
  cat("      pip install -e .\n\n")

  cat("  Option B: Install from PyPI (when available)\n")
  cat("    Run from terminal:\n")
  cat("      pip install phasic\n\n")

  cat("  Option C: Let reticulate install it\n")
  response <- readline("    Install Python phasic now via reticulate? (y/n): ")

  if (tolower(response) == "y") {
    cat("\n    Installing Python phasic via pip...\n")
    tryCatch({
      # For development, install from current directory
      if (file.exists("pyproject.toml")) {
        cat("    Installing from local repository...\n")
        system("pip install -e .")
      } else {
        # Otherwise try from PyPI
        py_install("phasic", pip = TRUE)
      }
      cat("    ✓ Python package installed\n")
    }, error = function(e) {
      cat("    ✗ Installation failed:", e$message, "\n")
      cat("    Please install manually with: pip install -e .\n")
      quit(status = 1)
    })
  } else {
    cat("\n  Please install Python phasic first, then re-run this script.\n")
    quit(status = 0)
  }
}

cat("\n")

# Step 3: Install R package
cat("Step 3: Installing R package...\n")

library(devtools, quietly = TRUE)

tryCatch({
  install(".", upgrade = FALSE, quiet = TRUE)
  cat("  ✓ R package installed successfully\n")
}, error = function(e) {
  cat("  ✗ Installation failed:", e$message, "\n")
  quit(status = 1)
})

cat("\n")

# Step 4: Test installation
cat("Step 4: Testing installation...\n")

tryCatch({
  library(phasic, quietly = TRUE)
  cat("  ✓ Package loads successfully\n")

  # Test basic functionality
  g <- create_graph(state_length = 1)
  n_verts <- g$vertices_length()
  cat("  ✓ Basic functions work (created graph with", n_verts, "vertices)\n")

}, error = function(e) {
  cat("  ✗ Test failed:", e$message, "\n")
  quit(status = 1)
})

cat("\n")
cat("========================================\n")
cat("✅ Installation complete!\n")
cat("========================================\n\n")

cat("You can now use phasic in R:\n\n")
cat("  library(phasic)\n")
cat("  graph <- create_graph(state_length = 1)\n")
cat("  # See vignettes for more examples\n\n")

cat("Documentation:\n")
cat("  ?phasic                     # Package help\n")
cat("  ?create_graph               # Function help\n")
cat("  vignette('basic-usage')     # Tutorial (if built)\n")
cat("  vignette('svgd-inference')  # SVGD tutorial (if built)\n\n")

cat("To build vignettes:\n")
cat("  devtools::build_vignettes()\n\n")
