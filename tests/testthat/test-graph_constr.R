


# Objects
# expect_equal() expect_identical()
# Does code return the expected value?
# expect_type() expect_s3_class() expect_s4_class()
# Does code return an object inheriting from the expected base type, S3 class, or S4 class?
# Vectors
# expect_length()
# Does code return a vector with the specified length?
# expect_lt() expect_lte() expect_gt() expect_gte()
# Does code return a number greater/less than the expected value?
# expect_named()
# Does code return a vector with (given) names?
# expect_setequal() expect_mapequal() expect_contains() expect_in()
# Does code return a vector containing the expected values?
# expect_true() expect_false()
# Does code return TRUE or FALSE?
# expect_vector()
# Does code return a vector with the expected size and/or prototype?

context("Graph construction")

# Build Kingman graph
n <- 4
graph <- create_graph(n)

test_that("there is a single vertex", {
  expect_equal(vertices_length(graph),  1)
})

starting_state <- c(n, rep(0, n-1))
start <- create_vertex(graph, starting_state)
add_edge(starting_vertex(graph), start, 1)

test_that("there are two vertices", {
  expect_equal(vertices_length(graph),  2)
})

index <- 2

while (index <= vertices_length(graph)) {
  vertex <- vertex_at(graph, index)

  test_that("vertex is not NULL", {
    expect_type(vertex, "list")
  })

  for (i in 1:n) {
    for (j in i:n) {
      state <- vertex$state

      test_that("state is not NULL", {
        expect_type(state, "list")
      })

      if (i == j) {
        if (state[i] < 2) {
          next
        }

        rate <- state[i] * (state[i] - 1) / 2
      } else {
        if (state[i] < 1 || state[j] < 1) {
          next
        }

        rate <- state[i] * state[j]
      }

      child_state <- state
      child_state[i] <- child_state[i] - 1
      child_state[j] <- child_state[j] - 1
      child_state[i + j] <- child_state[i + j] + 1

      child <- find_or_create_vertex(graph, child_state)

      test_that("state is not NULL", {
        expect_type(state, "list")
      })

      add_edge(vertex, child, rate)
    }
  }

  index <- index + 1
}

# Convert to matrix based
ptd <- graph_as_matrix(graph)


context("Expected singleton and doubleton length")

test_that("singleton length matches expectation", {
  expect_equal(as.integer(ptd$IPV
                          %*% solve(-ptd$SIM)
                          %*% diag(ptd$states[, 1])
                          %*% rep(1, length(ptd$IPV))),
               2)
})
test_that("doubleton length matches expectation", {
  expect_equal(as.integer(ptd$IPV
                          %*% solve(-ptd$SIM)
                          %*% diag(ptd$states[, 2])
                          %*% rep(1, length(ptd$IPV))),
               1)
})

# context("Graph based vs matrix")

# rw3 <- as.numeric(ptd$IPV %*% solve(-ptd$SIM) %*% diag(ptd$states[,3]) %*% rep(1, length(ptd$IPV)))
# rw3_o <- sum(expected_waiting_time(graph)*sapply(vertices(graph), function(v) {v$state[3]}))
# test_that("foo bar", {
#     expect_lt(abs(rw3-rw3_o), 0.0001) 
#   })

# # Graph based vs matrix
# stopifnot(abs(rw3-rw3_o) < 0.0001)

# # Reward transform graph
# reward_transform(graph, sapply(vertices(graph), function(v) {v$state[3]}))
# ptd2 <- graph_as_matrix(graph)

# # stopifnot(abs(as.numeric(ptd2$IPV %*% solve(-ptd2$SIM) %*% rep(1, length(ptd2$IPV)))-rw3) < 0.0001)


# # Make cyclic
# # stopifnot(graph_is_acyclic(graph))
# add_edge(vertex_at(graph, 5), vertex_at(graph, 3), 4)
# # stopifnot(!graph_is_acyclic(graph))

# # Must be able to compute expected waiting time
# ptd3 <- graph_as_matrix(graph)
# e <- as.numeric(ptd3$IPV %*% solve(-ptd3$SIM) %*% rep(1, length(ptd3$IPV)))
# e2 <- sum(expected_waiting_time(graph))
# # stopifnot(abs(e-e2) < 0.0001)

# # # Compute new rewards that are equivalent to 2nd moment when taken as
# # # the expected value
# # v <- as.numeric(ptd3$IPV %*% solve(-ptd3$SIM)%*% solve(-ptd3$SIM) %*% rep(1, length(ptd3$IPV)))
# # rw <- moment_rewards(graph, rep(1, vertices_length(graph)))
# # v2 <- sum(expected_waiting_time(graph)*rw)

# # stopifnot(abs(v-v2) < 0.0001)
# # reward_transform(graph, rw)

# # v3 <- sum(expected_waiting_time(graph))
# # stopifnot(abs(v-v3) < 0.0001)

