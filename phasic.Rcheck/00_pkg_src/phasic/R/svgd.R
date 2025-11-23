#' Stein Variational Gradient Descent (SVGD)
#'
#' @description
#' R6 class for Bayesian inference using Stein Variational Gradient Descent.
#' Wraps the Python SVGD implementation for phase-type distributions.
#'
#' @export
SVGD <- R6::R6Class("SVGD",
  private = list(
    py_svgd = NULL
  ),

  public = list(
    #' @description
    #' Create a new SVGD object
    #' @param model Model function (from trace_to_log_likelihood or Graph$pmf_from_graph)
    #' @param observed_data Numeric vector or matrix of observed data
    #' @param theta_dim Integer dimension of parameter space
    #' @param n_particles Integer number of particles (default 100)
    #' @param n_iterations Integer number of iterations (default 1000)
    #' @param learning_rate Numeric learning rate (default 0.01)
    #' @param rewards Optional numeric vector or matrix of rewards
    #' @param ... Additional arguments passed to Python SVGD
    #' @return A new SVGD object
    initialize = function(model, observed_data, theta_dim,
                         n_particles = 100L, n_iterations = 1000L,
                         learning_rate = 0.01, rewards = NULL, ...) {
      phasic <- get_phasic()

      private$py_svgd <- phasic$SVGD(
        model = model,
        observed_data = as.numeric(observed_data),
        theta_dim = as.integer(theta_dim),
        n_particles = as.integer(n_particles),
        n_iterations = as.integer(n_iterations),
        learning_rate = as.numeric(learning_rate),
        rewards = if (!is.null(rewards)) as.numeric(rewards) else NULL,
        ...
      )
    },

    #' @description
    #' Run SVGD optimization
    #' @return List with theta_mean, theta_std, particles, and convergence info
    optimize = function() {
      private$py_svgd$optimize()
    },

    #' @description
    #' Get current particles
    #' @return Matrix of particle positions (n_particles x theta_dim)
    get_particles = function() {
      private$py_svgd$get_particles()
    }
  )
)

#' Run SVGD inference on a phase-type distribution
#'
#' @description
#' High-level convenience function for running SVGD inference on phase-type
#' distribution models.
#'
#' @param graph Graph object (parameterized)
#' @param observed_data Numeric vector or matrix of observed data
#' @param theta_dim Integer dimension of parameter space
#' @param n_particles Integer number of particles (default 100)
#' @param n_iterations Integer number of iterations (default 1000)
#' @param learning_rate Numeric learning rate (default 0.01)
#' @param discrete Logical, whether distribution is discrete (default FALSE)
#' @param rewards Optional numeric vector or matrix of rewards
#' @param ... Additional arguments passed to SVGD
#'
#' @return List with posterior mean, std, particles, and convergence info
#'
#' @details
#' This function wraps the complete SVGD workflow:
#' 1. Creates model from graph
#' 2. Initializes SVGD with observed data
#' 3. Runs optimization
#' 4. Returns posterior summaries
#'
#' @examples
#' \dontrun{
#' # Build parameterized coalescent model
#' coalescent_callback <- function(state) {
#'   n <- state[1]
#'   if (n <= 1) return(list())
#'   rate <- n * (n - 1) / 2
#'   list(list(c(n - 1), 0.0, c(rate)))
#' }
#' graph <- create_graph(callback = coalescent_callback, parameterized = TRUE, nr_samples = 5)
#'
#' # Simulate some data
#' observed_times <- c(1.5, 2.3, 0.8, 1.2, 2.1)
#'
#' # Run SVGD
#' results <- run_svgd(
#'   graph = graph,
#'   observed_data = observed_times,
#'   theta_dim = 1,
#'   n_particles = 100,
#'   n_iterations = 1000
#' )
#'
#' print(results$theta_mean)
#' print(results$theta_std)
#' }
#'
#' @export
run_svgd <- function(graph, observed_data, theta_dim,
                     n_particles = 100L, n_iterations = 1000L,
                     learning_rate = 0.01, discrete = FALSE,
                     rewards = NULL, ...) {
  py_graph <- graph$get_py_graph()

  # Use Python Graph.svgd method
  py_graph$svgd(
    observed_data = as.numeric(observed_data),
    theta_dim = as.integer(theta_dim),
    n_particles = as.integer(n_particles),
    n_iterations = as.integer(n_iterations),
    learning_rate = as.numeric(learning_rate),
    discrete = discrete,
    rewards = if (!is.null(rewards)) as.numeric(rewards) else NULL,
    ...
  )
}

#' Create PMF/PDF model from graph
#'
#' @description
#' Creates a probability mass/density function model from a parameterized graph
#' for use with SVGD or other inference methods.
#'
#' @param graph Graph object (parameterized)
#' @param discrete Logical, whether distribution is discrete (default FALSE)
#'
#' @return Model function that takes (theta, data) and returns log-probabilities
#'
#' @examples
#' \dontrun{
#' graph <- create_graph(callback = my_callback, parameterized = TRUE, nr_samples = 5)
#' model <- create_pmf_model(graph, discrete = FALSE)
#'
#' # Use with custom SVGD
#' svgd <- SVGD$new(
#'   model = model,
#'   observed_data = observed_times,
#'   theta_dim = 2,
#'   n_particles = 100
#' )
#' results <- svgd$optimize()
#' }
#'
#' @export
create_pmf_model <- function(graph, discrete = FALSE) {
  py_graph <- graph$get_py_graph()
  py_graph$pmf_from_graph(discrete = discrete)
}

#' Create multivariate PMF/PDF model from graph
#'
#' @description
#' Creates a multivariate probability model with moment regularization
#' for SVGD inference.
#'
#' @param graph Graph object (parameterized)
#' @param nr_moments Integer number of moments to compute
#' @param discrete Logical, whether distribution is discrete (default FALSE)
#'
#' @return Model function for multivariate inference
#'
#' @examples
#' \dontrun{
#' graph <- create_graph(callback = my_callback, parameterized = TRUE, nr_samples = 5)
#' model <- create_multivariate_model(graph, nr_moments = 2, discrete = FALSE)
#' }
#'
#' @export
create_multivariate_model <- function(graph, nr_moments, discrete = FALSE) {
  py_graph <- graph$get_py_graph()
  py_graph$pmf_and_moments_from_graph_multivariate(
    nr_moments = as.integer(nr_moments),
    discrete = discrete
  )
}
