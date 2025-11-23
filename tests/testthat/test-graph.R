test_that("Graph creation works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  # Test basic graph creation
  g <- create_graph(state_length = 1)
  expect_s3_class(g, "Graph")
  expect_s3_class(g, "R6")

  # Should have at least starting vertex
  expect_gte(g$vertices_length(), 1)
})

test_that("Callback-based graph construction works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  # Simple linear chain
  simple_callback <- function(state) {
    n <- state[1]
    if (n <= 0) return(list())
    list(list(c(n - 1), n, NULL))
  }

  g <- create_graph(
    callback = simple_callback,
    state_length = 1,
    nr_samples = 3
  )

  expect_s3_class(g, "Graph")
  # Should have created vertices for states 3, 2, 1, 0
  expect_gte(g$vertices_length(), 4)
})

test_that("Coalescent graph construction works", {
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
    nr_samples = 5
  )

  expect_s3_class(g, "Graph")
  # n=5: creates vertices for 5,4,3,2,1 plus absorbing
  expect_gte(g$vertices_length(), 5)
})

test_that("Graph serialization works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("phasic"))

  g <- create_graph(state_length = 1)
  json_str <- g$serialize()

  expect_type(json_str, "character")
  expect_gt(nchar(json_str), 0)
})
