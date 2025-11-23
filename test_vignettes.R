#!/usr/bin/env Rscript

# Test that vignette files are valid R Markdown
cat("Testing R Markdown vignettes\n")
cat("==============================\n\n")

# Install required packages if needed
if (!require("knitr", quietly = TRUE)) {
  install.packages("knitr", repos = "https://cloud.r-project.org")
}
if (!require("rmarkdown", quietly = TRUE)) {
  install.packages("rmarkdown", repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(knitr)
  library(rmarkdown)
})

vignettes <- c(
  "vignettes/basic-usage.Rmd",
  "vignettes/svgd-inference.Rmd"
)

errors <- 0

for (vignette in vignettes) {
  cat("Testing:", basename(vignette), "... ")

  # Check file exists
  if (!file.exists(vignette)) {
    cat("✗ File not found\n")
    errors <- errors + 1
    next
  }

  # Try to parse the Rmd
  tryCatch({
    # Read and check YAML header
    lines <- readLines(vignette, warn = FALSE)

    # Check for required YAML fields
    has_title <- any(grepl("^title:", lines))
    has_output <- any(grepl("^output:", lines))
    has_vignette_entry <- any(grepl("VignetteIndexEntry", lines))
    has_vignette_engine <- any(grepl("VignetteEngine", lines))

    if (!has_title || !has_output || !has_vignette_entry || !has_vignette_engine) {
      cat("✗ Missing required YAML fields\n")
      if (!has_title) cat("  Missing: title\n")
      if (!has_output) cat("  Missing: output\n")
      if (!has_vignette_entry) cat("  Missing: VignetteIndexEntry\n")
      if (!has_vignette_engine) cat("  Missing: VignetteEngine\n")
      errors <- errors + 1
    } else {
      # Count code chunks
      chunk_starts <- grep("^```\\{r", lines)
      chunk_count <- length(chunk_starts)

      # Try to extract code (without evaluating)
      temp_r <- tempfile(fileext = ".R")
      code_extracted <- tryCatch({
        purl(vignette, output = temp_r, quiet = TRUE, documentation = 0)
        file.exists(temp_r)
      }, error = function(e) {
        cat("\n  Warning: Could not extract code:", e$message, "\n")
        FALSE
      })

      if (file.exists(temp_r)) {
        unlink(temp_r)
      }

      if (code_extracted) {
        cat("✓ (", chunk_count, " code chunks)\n", sep = "")
      } else {
        cat("⚠ Valid structure but code extraction failed\n")
      }
    }
  }, error = function(e) {
    cat("✗ Error:", e$message, "\n")
    errors <- errors + 1
  })
}

# Test 3: Check for common Rmd issues
cat("\nChecking for common issues:\n")
for (vignette in vignettes) {
  lines <- readLines(vignette, warn = FALSE)

  cat("  ", basename(vignette), ":\n", sep = "")

  # Check eval=FALSE is set for chunks
  has_eval_false <- any(grepl("eval\\s*=\\s*FALSE", lines))
  if (has_eval_false) {
    cat("    ✓ Has eval = FALSE chunks\n")
  } else {
    cat("    ⚠ No eval = FALSE found (ok if intentional)\n")
  }

  # Check for example code
  r_chunks <- grep("```\\{r", lines)
  if (length(r_chunks) > 0) {
    cat("    ✓ Contains ", length(r_chunks), " R code chunks\n", sep = "")
  }

  # Check for library() calls
  has_library <- any(grepl("library\\(phasic\\)", lines))
  if (has_library) {
    cat("    ✓ Loads phasic package\n")
  }
}

cat("\n")
if (errors > 0) {
  cat("❌ ", errors, " error(s) found\n", sep = "")
  quit(status = 1)
} else {
  cat("✅ All vignettes are valid R Markdown\n")
  cat("\nNote: Code chunks not evaluated (require Python phasic)\n")
  cat("To build vignettes: devtools::build_vignettes()\n")
}
