library(ptdalgorithms)

# function to construct graph
# params:
# edge_array: matrix, each row is a bidirectional edge between two nodes, shape (e, 2)
# m_array: symmetric migration weight of each edge, shape (e,)
# pop_size_array: population size of each node, shape (d,)
construct_graph <- function(edge_array, m_array, pop_size_array) {
  n_edges <- length(m_array)
  n_nodes <- length(pop_size_array)
  
  # represent states as coordinates of the two lineages: (i, j)
  # coalesced state: (0, 0)
  state_vector_length <- 2
  graph <- create_graph(state_vector_length)
  initial_state <- edge_array[1,]
  add_edge(
    starting_vertex(graph),
    find_or_create_vertex(graph, initial_state),
    1
  )
  
  # handle migration edges
  for (i in 1:n_edges) {
    print(i)
    edge <- edge_array[i,]
    u <- edge[1]
    v <- edge[2]
    # transition from states (u, *) to (v, *) and (v, *) to (u, *)
    for (j in 1:n_nodes) {
      state1 <- c(u, j)
      state2 <- c(v, j)
      add_edge(
        find_or_create_vertex(graph, state1),
        find_or_create_vertex(graph, state2),
        m_array[i]
      )
      add_edge(
        find_or_create_vertex(graph, state2),
        find_or_create_vertex(graph, state1),
        m_array[i]
      )
    }
    # transition from states (*, u) to (*, v) and (*, v) to (*, u)
    for (j in 1:n_nodes) {
      state1 <- c(j, u)
      state2 <- c(j, v)
      add_edge(
        find_or_create_vertex(graph, state1),
        find_or_create_vertex(graph, state2),
        m_array[i]
      )
      add_edge(
        find_or_create_vertex(graph, state2),
        find_or_create_vertex(graph, state1),
        m_array[i]
      )
    }
  }
  # handle coalescence
  for (u in 1:n_nodes) {
    state1 <- c(u, u)
    state2 <- c(0, 0)
    add_edge(
      find_or_create_vertex(graph, state1),
      find_or_create_vertex(graph, state2),
      1/pop_size_array[i]
    )
  }
  return(graph)
}

# test
edge_array <- as.matrix(read.table("lattice_20_nodes", quote="\"", comment.char=""))
n_edges <- dim(edge_array)[1]
d <- length(unique(as.vector(edge_array)))
m <- runif(n_edges)
popsize <- runif(d)
graph <- construct_graph(edge_array, m, popsize)
vertices_length(graph)  # equals d**2 + 2
expected_waiting_time(graph)
