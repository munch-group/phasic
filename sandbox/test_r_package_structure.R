#!/usr/bin/env Rscript

cat("Testing R package structure...\n")

# Check DESCRIPTION file
cat("\nChecking DESCRIPTION file:\n")
if (file.exists("DESCRIPTION")) {
  desc <- read.dcf("DESCRIPTION")
  cat("  Package:", desc[1, "Package"], "\n")
  cat("  Version:", desc[1, "Version"], "\n")
  cat("  ✓ DESCRIPTION exists\n")
} else {
  cat("  ✗ DESCRIPTION missing\n")
  quit(status = 1)
}

# Check NAMESPACE file
cat("\nChecking NAMESPACE file:\n")
if (file.exists("NAMESPACE")) {
  cat("  ✓ NAMESPACE exists\n")
} else {
  cat("  ✗ NAMESPACE missing\n")
  quit(status = 1)
}

# Check R/ directory
cat("\nChecking R/ directory:\n")
if (dir.exists("R")) {
  r_files <- list.files("R", pattern = "\\.R$")
  cat("  Files:", length(r_files), "\n")
  for (f in r_files) cat("    -", f, "\n")
  cat("  ✓ R/ directory exists\n")
} else {
  cat("  ✗ R/ directory missing\n")
  quit(status = 1)
}

# Check vignettes/ directory
cat("\nChecking vignettes/ directory:\n")
if (dir.exists("vignettes")) {
  rmd_files <- list.files("vignettes", pattern = "\\.Rmd$")
  cat("  Files:", length(rmd_files), "\n")
  for (f in rmd_files) cat("    -", f, "\n")
  cat("  ✓ vignettes/ directory exists\n")
} else {
  cat("  ⚠ vignettes/ directory missing (optional)\n")
}

# Check tests/ directory
cat("\nChecking tests/ directory:\n")
if (dir.exists("tests/testthat")) {
  test_files <- list.files("tests/testthat", pattern = "\\.R$")
  cat("  Files:", length(test_files), "\n")
  for (f in test_files) cat("    -", f, "\n")
  cat("  ✓ tests/testthat/ directory exists\n")
} else {
  cat("  ⚠ tests/ directory missing (optional)\n")
}

cat("\n✅ Package structure validation complete\n")
