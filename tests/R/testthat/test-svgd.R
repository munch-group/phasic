test_that("SVGD class creation works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  # Create a simple log-likelihood function
  log_lik <- function(theta) {
    # Simple Gaussian likelihood
    sum(dnorm(c(1, 2, 3), mean = theta[1], sd = 1, log = TRUE))
  }

  # This test may fail if JAX is not installed
  skip_if(!reticulate::py_module_available("jax"))

  svgd <- SVGD$new(
    model = log_lik,
    observed_data = c(1, 2, 3),
    theta_dim = 1,
    n_particles = 10,
    n_iterations = 10
  )

  expect_s3_class(svgd, "SVGD")
  expect_s3_class(svgd, "R6")
})

test_that("run_svgd with simple graph works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))
  skip_if(!reticulate::py_module_available("jax"))

  # Build small coalescent graph
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

  # Run SVGD with minimal iterations for testing
  results <- run_svgd(
    graph = g,
    observed_data = c(1.0, 1.5, 2.0),
    theta_dim = 1,
    n_particles = 10,
    n_iterations = 10,
    learning_rate = 0.01
  )

  expect_type(results, "list")
  expect_true(!is.null(results$theta_mean))
  expect_true(!is.null(results$theta_std))
})

test_that("create_pmf_model works", {
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

  model <- create_pmf_model(g, discrete = FALSE)

  # Model should be a function
  expect_type(model, "closure")
})

test_that("create_multivariate_model works", {
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

  model <- create_multivariate_model(g, nr_moments = 2, discrete = FALSE)

  # Model should be a function
  expect_type(model, "closure")
})
