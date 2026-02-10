test_that("Trace recording works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  # Build parameterized graph
  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) return(list())
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  g <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 3
  )

  # Record trace
  trace <- record_elimination_trace(g, param_length = 1)
  expect_true(!is.null(trace))
})

test_that("Trace evaluation works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) return(list())
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  g <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 3
  )

  trace <- record_elimination_trace(g, param_length = 1)

  # Evaluate with concrete parameter
  result <- evaluate_trace(trace, theta = c(2.0))

  expect_type(result, "list")
  # Should have vertex_rates, edge_probs, vertex_targets
  expect_true(!is.null(result))
})

test_that("Instantiate from trace works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) return(list())
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  g <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 3
  )

  trace <- record_elimination_trace(g, param_length = 1)
  concrete_g <- instantiate_from_trace(trace, theta = c(2.0))

  expect_s3_class(concrete_g, "Graph")
  expect_gte(concrete_g$vertices_length(), 1)
})

test_that("PDF computation from trace works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) return(list())
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  g <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 3
  )

  trace <- record_elimination_trace(g, param_length = 1)
  concrete_g <- instantiate_from_trace(trace, theta = c(2.0))

  # Compute PDF
  pdf_val <- concrete_g$pdf(c(1.0))

  expect_type(pdf_val, "double")
  expect_length(pdf_val, 1)
  expect_gte(pdf_val, 0)  # PDF should be non-negative
})

test_that("Log-likelihood function creation works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  coalescent_callback <- function(state) {
    n <- state[1]
    if (n <= 1) return(list())
    rate <- n * (n - 1) / 2
    list(list(c(n - 1), 0.0, c(rate)))
  }

  g <- create_graph(
    callback = coalescent_callback,
    parameterized = TRUE,
    state_length = 1,
    nr_samples = 3
  )

  trace <- record_elimination_trace(g, param_length = 1)
  observed <- c(1.0, 2.0, 1.5)

  log_lik <- trace_to_log_likelihood(trace, observed)

  # Should be a function
  expect_type(log_lik, "closure")
})
