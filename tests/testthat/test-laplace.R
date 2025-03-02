
standard_coalescent <- function(n) {
          
    state_vector_length <- n + 1
    graph <- create_graph(state_vector_length)
    starting_vertex <- vertex_at(graph, 1)
    initial_state <- c(rep(0, n), 0)
    initial_state[1] <- n
    
    add_edge(
      starting_vertex,
      create_vertex(graph, initial_state),
      1
    )
    index <- 2
    
    while (index <= vertices_length(graph)) {
      vertex <- vertex_at(graph, index)
      
      # loop over all classes of lineages
      for (i in 1:n) {
        for (j in i:n) {
          state <- vertex$state
          
          # if same class, there need to be at least two to coalesce
          if (i == j) {
            if (state[i] < 2) {
              next;
            }
            # coal rate
            rate <- state[i] * (state[i] - 1) / 2
          } else {
            # else at least one in each class to coalesce
            if (state[i] < 1 || state[j] < 1) {
              next;
            }
            # number of combinations
            rate <- state[i] * state[j]
          }
          
          # copy state
          child_state <- state
          # update child state
          child_state[i] <- child_state[i] - 1
          child_state[j] <- child_state[j] - 1
          child_state[i+j] <- child_state[i+j] + 1

          add_edge(
              vertex,
              find_or_create_vertex(graph, child_state),
              rate, c(rate)
            )
        }
      }
          
      index <- index + 1
    }
    return(graph)
}

# grid_graph <- function(n) {
          
#     state_vector_length <- 2
#     graph <- create_graph(state_vector_length)
#     starting_vertex <- vertex_at(graph, 1)
#     initial_state <- c(1, 1)
    
#     add_edge(
#       starting_vertex,
#         find_or_create_vertex(graph, initial_state), 1, c(1) ) 
#       # create_vertex(graph, initial_state), 1, c(1))  
#     index <- 2
    
#     while (index <= vertices_length(graph)) {
#       vertex <- vertex_at(graph, index)

#       state <- vertex$state

#       if (0 < state[1] && 0 < state[2]) {

#           if (state[1] == state[2]) {
#               child_state <- state
#               child_state[1] <- 0
#               child_state[2] <- 0       
#               add_edge(vertex, find_or_create_vertex(graph, child_state), 1, c(1) )                
#            }
            
#           child_state <- state
#           child_state[1] <- child_state[1] + 1
#           if (0 < child_state[1] && child_state[1] < n && 0 < child_state[2] && child_state[2] < n) {
#               add_edge(vertex, find_or_create_vertex(graph, child_state), 1, c(1))
#           }
#           child_state <- state
#           child_state[1] <- child_state[1] - 1
#           if (0 < child_state[1] && child_state[1] < n && 0 < child_state[2] && child_state[2] < n) {
#               add_edge(vertex, find_or_create_vertex(graph, child_state), 1, c(1))
#           }
#           child_state <- state
#           child_state[2] <- child_state[2] + 1
#           if (0 < child_state[1] && child_state[1] < n && 0 < child_state[2] && child_state[2] < n) {
#               add_edge(vertex, find_or_create_vertex(graph, child_state), 1, c(1))
#           }
#           child_state <- state
#           child_state[2] <- child_state[2] - 1
#           if (0 < child_state[1] && child_state[1] < n && 0 < child_state[2] && child_state[2] < n) {
#               add_edge(vertex, find_or_create_vertex(graph, child_state), 1, c(1))
#           }
#       }
#     index <- index + 1
#     }
#     return(graph)
# }

# graph <- grid_graph(1000)

# times <- c()
# for (i in 1:10) {
#     start <- proc.time()[3]
#     # expectation(graph)    
#     laplace_rewards = laplace_transform(graph, theta=3)
#     expectation(graph, laplace_rewards)    
#     end <- proc.time()[3]
#     times <- c(times, (end - start))
#     print((end - start))
# }
# print(mean(times))