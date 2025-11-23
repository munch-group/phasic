#' Record elimination trace for a parameterized graph
#'
#' @description
#' Records the graph elimination operations as a linear trace for efficient
#' repeated evaluation with different parameter values.
#'
#' @param graph Graph object (must be parameterized)
#' @param param_length Integer number of parameters
#'
#' @return Trace object (Python object, use with evaluate_trace)
#'
#' @details
#' This is a Phase 1-2 feature that records elimination operations once
#' (O(n³) complexity) to enable fast repeated evaluation (O(n) complexity).
#' Essential for SVGD and other inference methods that evaluate the model
#' many times with different parameters.
#'
#' @examples
#' \dontrun{
#' # Build parameterized graph
#' coalescent_callback <- function(state) {
#'   n <- state[1]
#'   if (n <= 1) return(list())
#'   rate <- n * (n - 1) / 2
#'   list(list(c(n - 1), 0.0, c(rate)))
#' }
#' graph <- create_graph(callback = coalescent_callback, parameterized = TRUE, nr_samples = 5)
#'
#' # Record trace once
#' trace <- record_elimination_trace(graph, param_length = 1)
#'
#' # Evaluate with different parameters (fast)
#' theta1 <- c(1.0)
#' result1 <- evaluate_trace(trace, theta1)
#' theta2 <- c(2.0)
#' result2 <- evaluate_trace(trace, theta2)
#' }
#'
#' @export
record_elimination_trace <- function(graph, param_length) {
  phasic <- get_phasic()
  trace_module <- reticulate::import("phasic.trace_elimination")

  py_graph <- graph$get_py_graph()
  trace_module$record_elimination_trace(py_graph, as.integer(param_length))
}

#' Evaluate a trace with concrete parameter values
#'
#' @description
#' Evaluates a previously recorded trace with specific parameter values.
#'
#' @param trace Trace object from record_elimination_trace()
#' @param theta Numeric vector of parameter values
#'
#' @return List with vertex_rates, edge_probs, vertex_targets
#'
#' @examples
#' \dontrun{
#' trace <- record_elimination_trace(graph, param_length = 1)
#' result <- evaluate_trace(trace, theta = c(2.0))
#' }
#'
#' @export
evaluate_trace <- function(trace, theta) {
  trace_module <- reticulate::import("phasic.trace_elimination")
  trace_module$evaluate_trace_jax(trace, as.numeric(theta))
}

#' Instantiate a concrete graph from a trace
#'
#' @description
#' Creates a concrete (non-parameterized) Graph object from a trace
#' evaluated at specific parameter values.
#'
#' @param trace Trace object from record_elimination_trace()
#' @param theta Numeric vector of parameter values
#'
#' @return Graph object with concrete edge weights
#'
#' @details
#' This is useful when you want to compute PDF/PMF or other operations
#' on a graph with specific parameter values.
#'
#' @examples
#' \dontrun{
#' trace <- record_elimination_trace(graph, param_length = 1)
#' concrete_graph <- instantiate_from_trace(trace, theta = c(2.0))
#' pdf_values <- concrete_graph$pdf(c(1.0, 2.0, 3.0))
#' }
#'
#' @export
instantiate_from_trace <- function(trace, theta) {
  trace_module <- reticulate::import("phasic.trace_elimination")
  py_graph <- trace_module$instantiate_from_trace(trace, as.numeric(theta))

  # Wrap in R6 Graph object
  graph <- Graph$new(state_length = 1L)  # Dummy init
  graph$set_py_graph(py_graph)
  return(graph)
}

#' Create log-likelihood function from trace
#'
#' @description
#' Creates a log-likelihood function for use with SVGD or other inference methods.
#' Uses exact phase-type PDF computation (Phase 4 feature).
#'
#' @param trace Trace object from record_elimination_trace()
#' @param observed_times Numeric vector of observed waiting times
#' @param reward_vector Optional numeric vector of rewards
#' @param granularity Integer granularity for PDF computation (default 0 = auto)
#'
#' @return Function that takes theta and returns log-likelihood
#'
#' @details
#' The returned function computes the exact phase-type log-likelihood:
#' sum(log(PDF(t_i | theta))) over all observed times.
#'
#' @examples
#' \dontrun{
#' trace <- record_elimination_trace(graph, param_length = 1)
#' observed <- c(1.5, 2.3, 0.8, 1.2)
#' log_lik <- trace_to_log_likelihood(trace, observed)
#'
#' # Use with optimization
#' theta <- c(2.0)
#' ll <- log_lik(theta)
#' }
#'
#' @export
trace_to_log_likelihood <- function(trace, observed_times, reward_vector = NULL, granularity = 0L) {
  trace_module <- reticulate::import("phasic.trace_elimination")

  trace_module$trace_to_log_likelihood(
    trace,
    as.numeric(observed_times),
    reward_vector = if (!is.null(reward_vector)) as.numeric(reward_vector) else NULL,
    granularity = as.integer(granularity)
  )
}
