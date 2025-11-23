#!/usr/bin/env Rscript

# Test that R package structure is valid (no Python needed)
cat("Testing R package structure\n")
cat("============================\n\n")

errors <- 0

# Test 1: Check R files exist
cat("1. Checking R files... ")
r_files <- c(
  "R/zzz.R",
  "R/phasic-package.R",
  "R/graph.R",
  "R/trace_elimination.R",
  "R/svgd.R"
)

for (f in r_files) {
  if (!file.exists(f)) {
    cat("\n  ❌ Missing:", f)
    errors <- errors + 1
  }
}
if (errors == 0) {
  cat("✓ (", length(r_files), " files)\n", sep = "")
}

# Test 2: Check package metadata
cat("2. Checking DESCRIPTION... ")
if (!file.exists("DESCRIPTION")) {
  cat("❌ Missing\n")
  errors <- errors + 1
} else {
  desc <- readLines("DESCRIPTION")
  has_reticulate <- any(grepl("reticulate", desc))
  has_r6 <- any(grepl("R6", desc))

  if (!has_reticulate || !has_r6) {
    cat("❌ Missing dependencies\n")
    errors <- errors + 1
  } else {
    cat("✓\n")
  }
}

# Test 3: Check NAMESPACE
cat("3. Checking NAMESPACE... ")
if (!file.exists("NAMESPACE")) {
  cat("❌ Missing\n")
  errors <- errors + 1
} else {
  ns <- readLines("NAMESPACE")
  exports <- sum(grepl("^export\\(", ns))

  if (exports < 5) {
    cat("❌ Too few exports (", exports, ")\n", sep = "")
    errors <- errors + 1
  } else {
    cat("✓ (", exports, " exports)\n", sep = "")
  }
}

# Test 4: Check vignettes
cat("4. Checking vignettes... ")
vignettes <- list.files("vignettes", pattern = "\\.Rmd$", full.names = TRUE)
if (length(vignettes) < 2) {
  cat("❌ Expected 2+ vignettes, found", length(vignettes), "\n")
  errors <- errors + 1
} else {
  cat("✓ (", length(vignettes), " files)\n", sep = "")
}

# Test 5: Check tests
cat("5. Checking tests... ")
test_files <- list.files("tests/testthat", pattern = "^test-.*\\.R$", full.names = TRUE)
if (length(test_files) < 3) {
  cat("❌ Expected 3+ test files, found", length(test_files), "\n")
  errors <- errors + 1
} else {
  cat("✓ (", length(test_files), " files)\n", sep = "")
}

# Test 6: Parse R files for syntax errors
cat("6. Checking R syntax... ")
for (f in r_files) {
  tryCatch({
    parse(f)
  }, error = function(e) {
    cat("\n  ❌ Syntax error in", f, ":", e$message)
    errors <<- errors + 1
  })
}
if (errors == 0) {
  cat("✓\n")
}

# Test 7: Check R6 class definitions
cat("7. Checking R6 classes... ")
suppressWarnings({
  source("R/zzz.R", local = TRUE)
  source("R/graph.R", local = TRUE)
  source("R/svgd.R", local = TRUE)
})

has_graph <- exists("Graph")
has_svgd <- exists("SVGD")

if (!has_graph || !has_svgd) {
  cat("❌ Missing R6 classes\n")
  errors <- errors + 1
} else {
  cat("✓ (Graph, SVGD)\n")
}

# Summary
cat("\n")
if (errors > 0) {
  cat("❌ Tests failed with", errors, "error(s)\n")
  quit(status = 1)
} else {
  cat("✅ All structure tests passed!\n")
  cat("\nR package structure is valid.\n")
  cat("Note: Runtime tests require Python phasic to be installed.\n")
}
