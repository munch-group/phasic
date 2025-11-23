# Package-level variables to store Python module
phasic_py <- NULL

.onLoad <- function(libname, pkgname) {
  # Import Python phasic module using reticulate
  phasic_py <<- reticulate::import("phasic", delay_load = TRUE)
}

.onAttach <- function(libname, pkgname) {
  # Check if Python module is actually available
  if (!reticulate::py_module_available("phasic")) {
    packageStartupMessage(
      "\n",
      "Python phasic module not found.\n",
      "The R package requires the Python backend to be installed.\n",
      "\n",
      "Install with one of these methods:\n",
      "  1. install_phasic()  # R helper function\n",
      "  2. reticulate::py_install('phasic', pip = TRUE)\n",
      "  3. From terminal: pip install phasic\n",
      "\n"
    )
  }
}

#' @keywords internal
#' @noRd
get_phasic <- function() {
  if (is.null(phasic_py)) {
    stop("Python phasic module not loaded. Please ensure phasic is installed in Python.")
  }
  phasic_py
}

#' Install Python phasic backend
#'
#' Helper function to install the Python phasic package required by this R package.
#'
#' @param method Installation method to use. Default is "auto" which automatically
#'   finds a suitable Python installation. Other options include "virtualenv" and "conda".
#' @param conda Path to conda executable. Only used when method = "conda".
#' @param pip Logical, use pip for installation (default TRUE).
#' @param ... Additional arguments passed to \code{\link[reticulate]{py_install}}.
#'
#' @return NULL (invisibly). Called for side effect of installing Python package.
#'
#' @examples
#' \dontrun{
#' # Install phasic Python package
#' install_phasic()
#'
#' # Install in a specific conda environment
#' install_phasic(method = "conda", conda = "/path/to/conda")
#' }
#'
#' @export
install_phasic <- function(method = "auto", conda = "auto", pip = TRUE, ...) {
  reticulate::py_install("phasic", method = method, conda = conda, pip = pip, ...)
}
