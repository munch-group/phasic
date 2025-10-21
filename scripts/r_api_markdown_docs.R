


library(devtools)

#remove.packages("phasic")
# #devtools::install_github("kaspermunch/PtDAlgorithms")
# devtools::install_local('../PtDAlgorithms', quiet=FALSE)
#devtools::install_local()
devtools::load_all(path="./", quiet=FALSE, recompile=TRUE)

library(phasic)

Rcpp::compileAttributes() 

install.packages("roxygen2md")

library(roxygen2)
Rcpp::compileAttributes()           # this updates the Rcpp layer from C++ to R
roxygen2::roxygenize(roclets="rd")  # this updates the documentation based on roxygen comments


roxygen2md::roxygen2md(scope = "none")          # "none" only adds list(markdown = TRUE) to the Roxygen field in DESCRIPTION

roxygen2md::roxygen2md(scope = "simple")        # "simple" only converts elements like \code{} and \emph{}


roxygen2md::roxygen2md(scope = "full")          # "full" runs all conversions