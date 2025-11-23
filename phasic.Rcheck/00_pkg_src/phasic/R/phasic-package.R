#' @keywords internal
"_PACKAGE"

#' phasic: Phase-Type Distribution Algorithms
#'
#' @description
#' High-performance library for computing with phase-type distributions using
#' graph-based algorithms. This R package provides an interface to the Python
#' implementation via reticulate.
#'
#' @details
#' Phase-type distributions model the time until absorption in continuous or
#' discrete-time Markov chains. This package uses graph-based algorithms that
#' are 10-100x faster than traditional matrix methods for sparse systems.
#'
#' Key features:
#' \itemize{
#'   \item Graph construction with callbacks or manual building
#'   \item PDF/PMF computation via forward algorithm
#'   \item Reward transformation for multivariate distributions
#'   \item Trace-based elimination for efficient parameter evaluation
#'   \item Bayesian inference via SVGD (Stein Variational Gradient Descent)
#'   \item JAX integration for automatic differentiation
#' }
#'
#' @section Main functions:
#' \itemize{
#'   \item \code{\link{create_graph}}: Create phase-type distribution graph
#'   \item \code{\link{record_elimination_trace}}: Record trace for efficient evaluation
#'   \item \code{\link{run_svgd}}: Bayesian inference via SVGD
#'   \item \code{\link{Graph}}: R6 class for graph objects
#' }
#'
#' @references
#' Røikjer, T., Hobolth, A., & Munch, K. (2022).
#' Phase-type distributions in population genetics.
#' \emph{Statistics and Computing}, 32(5), 1-21.
#' \doi{10.1007/s11222-022-10155-6}
#'
#' @import R6
#' @importFrom reticulate import py_to_r r_to_py
#'
#' @docType package
#' @name phasic-package
NULL
