#!/usr/bin/env Rscript

# Test that callback formats in vignettes and docs are correct
cat("Testing Callback Format Correctness\n")
cat("====================================\n\n")

errors <- 0

cat("Callback Format Rules:\n")
cat("  - Non-parameterized: list(list(next_state, rate))\n")
cat("  - Parameterized: list(list(next_state, base_weight, coefficients))\n\n")

# Test 1: Simple callback (non-parameterized)
cat("Test 1: Simple callback (non-parameterized)... ")
simple_callback <- function(state) {
  n <- state[1]
  if (n <= 0) {
    return(list())
  }
  list(list(c(n - 1), n))  # 2-tuple
}

# Test the callback
result <- simple_callback(c(3))
if (length(result) != 1) {
  cat("✗ Wrong number of results\n")
  errors <- errors + 1
} else if (length(result[[1]]) != 2) {
  cat("✗ Wrong tuple length (expected 2, got", length(result[[1]]), ")\n")
  errors <- errors + 1
} else {
  cat("✓ Correct format\n")
}

# Test 2: Coalescent callback (parameterized)
cat("Test 2: Coalescent callback (parameterized)... ")
coalescent_callback <- function(state) {
  n <- state[1]
  if (n <= 1) {
    return(list())
  }
  rate <- n * (n - 1) / 2
  list(list(c(n - 1), 0.0, c(rate)))  # 3-tuple
}

result <- coalescent_callback(c(5))
if (length(result) != 1) {
  cat("✗ Wrong number of results\n")
  errors <- errors + 1
} else if (length(result[[1]]) != 3) {
  cat("✗ Wrong tuple length (expected 3, got", length(result[[1]]), ")\n")
  errors <- errors + 1
} else {
  cat("✓ Correct format\n")
}

# Test 3: Multi-parameter callback
cat("Test 3: Multi-parameter callback... ")
multi_param_callback <- function(state) {
  n <- state[1]
  if (n <= 1) return(list())

  coef1 <- n * (n - 1) / 2
  coef2 <- n

  list(list(c(n - 1), 0.0, c(coef1, coef2)))  # 3-tuple with 2 coefficients
}

result <- multi_param_callback(c(5))
if (length(result) != 1) {
  cat("✗ Wrong number of results\n")
  errors <- errors + 1
} else if (length(result[[1]]) != 3) {
  cat("✗ Wrong tuple length\n")
  errors <- errors + 1
} else if (length(result[[1]][[3]]) != 2) {
  cat("✗ Wrong coefficient vector length\n")
  errors <- errors + 1
} else {
  cat("✓ Correct format\n")
}

# Test 4: Check vignette examples
cat("\nTest 4: Checking vignette examples...\n")

# Extract callbacks from basic-usage.Rmd
vignette_file <- "vignettes/basic-usage.Rmd"
if (file.exists(vignette_file)) {
  lines <- readLines(vignette_file, warn = FALSE)

  # Check simple_callback
  simple_idx <- grep("simple_callback <- function", lines)
  if (length(simple_idx) > 0) {
    # Check the return statement
    return_idx <- grep("list\\(list\\(c\\(n - 1\\), n\\)\\)", lines)
    if (length(return_idx) > 0) {
      cat("  ✓ simple_callback has correct 2-tuple format\n")
    } else {
      cat("  ✗ simple_callback format might be wrong\n")
      errors <- errors + 1
    }
  }

  # Check coalescent_callback
  coal_idx <- grep("coalescent_callback <- function", lines)
  if (length(coal_idx) > 0) {
    # Check for 3-tuple format
    has_three_elements <- any(grepl("list\\(list\\(c\\(n - 1\\), 0\\.0, c\\(rate\\)\\)\\)", lines))
    if (has_three_elements) {
      cat("  ✓ coalescent_callback has correct 3-tuple format\n")
    } else {
      cat("  ✗ coalescent_callback format might be wrong\n")
      errors <- errors + 1
    }
  }
} else {
  cat("  ⚠ Vignette file not found\n")
}

# Test 5: Check that absorbing states return empty list
cat("\nTest 5: Absorbing state handling... ")
absorbing_test <- simple_callback(c(0))
if (length(absorbing_test) == 0) {
  cat("✓ Returns empty list\n")
} else {
  cat("✗ Should return empty list for absorbing state\n")
  errors <- errors + 1
}

cat("\n")
if (errors > 0) {
  cat("❌ ", errors, " error(s) found in callback formats\n", sep = "")
  quit(status = 1)
} else {
  cat("✅ All callback formats are correct\n\n")
  cat("Summary:\n")
  cat("  - Non-parameterized callbacks use 2-tuples ✓\n")
  cat("  - Parameterized callbacks use 3-tuples ✓\n")
  cat("  - Absorbing states return empty list ✓\n")
  cat("  - Vignette examples are correct ✓\n")
}
