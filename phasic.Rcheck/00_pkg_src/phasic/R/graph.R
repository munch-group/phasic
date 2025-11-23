#' Phase-Type Distribution Graph
#'
#' @description
#' R6 class wrapping the Python Graph class for phase-type distributions.
#' Provides methods for graph construction, PDF/PMF computation, reward
#' transformation, and moment computation.
#'
#' @details
#' This class provides an R interface to the Python phasic.Graph class.
#' Graphs represent phase-type distributions using vertices (states) and
#' weighted edges (transitions).
#'
#' @export
Graph <- R6::R6Class("Graph",
  private = list(
    py_graph = NULL
  ),

  public = list(
    #' @description
    #' Create a new Graph object
    #' @param state_length Integer length of state vector
    #' @param callback Optional R function for graph construction
    #' @param parameterized Logical, whether edges are parameterized (default FALSE)
    #' @param nr_samples Integer, number of samples for callback-based construction
    #' @return A new Graph object
    initialize = function(state_length = NULL, callback = NULL, parameterized = FALSE, nr_samples = NULL) {
      phasic <- get_phasic()

      if (!is.null(callback)) {
        # Wrap R callback for Python
        py_callback <- function(state) {
          r_result <- callback(state)
          # Convert R list to Python list format
          return(r_result)
        }

        private$py_graph <- phasic$Graph(
          state_length = as.integer(state_length),
          callback = py_callback,
          parameterized = parameterized,
          nr_samples = if (!is.null(nr_samples)) as.integer(nr_samples) else NULL
        )
      } else if (!is.null(state_length)) {
        private$py_graph <- phasic$Graph(state_length = as.integer(state_length))
      } else {
        stop("Either state_length or callback must be provided")
      }
    },

    #' @description
    #' Get the starting vertex of the graph
    #' @return Vertex object
    starting_vertex = function() {
      private$py_graph$starting_vertex()
    },

    #' @description
    #' Find or create a vertex with given state
    #' @param state Integer vector representing the state
    #' @return Vertex object
    find_or_create_vertex = function(state) {
      private$py_graph$find_or_create_vertex(as.integer(state))
    },

    #' @description
    #' Get number of vertices in graph
    #' @return Integer number of vertices
    vertices_length = function() {
      private$py_graph$vertices_length()
    },

    #' @description
    #' Compute PDF at given time(s)
    #' @param time Numeric vector of time points
    #' @param granularity Integer granularity for uniformization (default 0 = auto)
    #' @return Numeric vector of PDF values
    pdf = function(time, granularity = 0L) {
      private$py_graph$pdf(time, granularity = as.integer(granularity))
    },

    #' @description
    #' Compute PMF for discrete phase-type distribution
    #' @param jumps Integer vector of jump counts
    #' @return Numeric vector of PMF values
    dph_pmf = function(jumps) {
      private$py_graph$dph_pmf(as.integer(jumps))
    },

    #' @description
    #' Perform reward transformation
    #' @param rewards Numeric vector of rewards (one per vertex)
    #' @return New Graph object with reward transformation applied
    reward_transform = function(rewards) {
      new_py_graph <- private$py_graph$reward_transform(as.numeric(rewards))
      new_graph <- Graph$new(state_length = 1L)  # Dummy initialization
      new_graph$set_py_graph(new_py_graph)
      return(new_graph)
    },

    #' @description
    #' Compute moments of the distribution
    #' @param power Integer number of moments to compute
    #' @param rewards Optional numeric vector of rewards
    #' @return Numeric vector of moments
    moments = function(power, rewards = NULL) {
      if (!is.null(rewards)) {
        private$py_graph$moments(as.integer(power), as.numeric(rewards))
      } else {
        private$py_graph$moments(as.integer(power))
      }
    },

    #' @description
    #' Serialize graph to JSON
    #' @return JSON string representation
    serialize = function() {
      private$py_graph$serialize()
    },

    #' @description
    #' Set internal Python graph object (for internal use)
    #' @param py_graph Python Graph object
    set_py_graph = function(py_graph) {
      private$py_graph <- py_graph
    },

    #' @description
    #' Get internal Python graph object (for internal use)
    #' @return Python Graph object
    get_py_graph = function() {
      private$py_graph
    }
  )
)

#' Create a phase-type distribution graph
#'
#' @description
#' Convenience function to create a Graph object
#'
#' @param state_length Integer length of state vector
#' @param callback Optional R function for graph construction
#' @param parameterized Logical, whether edges are parameterized (default FALSE)
#' @param nr_samples Integer, number of samples for callback-based construction
#'
#' @return A Graph object
#'
#' @examples
#' \dontrun{
#' # Simple graph construction
#' g <- create_graph(state_length = 1)
#'
#' # Callback-based construction (coalescent model)
#' coalescent_callback <- function(state) {
#'   n <- state[1]
#'   if (n <= 1) return(list())
#'   rate <- n * (n - 1) / 2
#'   list(list(c(n - 1), 0.0, c(rate)))
#' }
#' g <- create_graph(callback = coalescent_callback, parameterized = TRUE, nr_samples = 5)
#' }
#'
#' @export
create_graph <- function(state_length = NULL, callback = NULL, parameterized = FALSE, nr_samples = NULL) {
  Graph$new(
    state_length = state_length,
    callback = callback,
    parameterized = parameterized,
    nr_samples = nr_samples
  )
}
