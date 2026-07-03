// Standalone C++ program: the Kingman coalescent built with the callback API
// phasic::Graph::from_callback -- the native C++ mirror of the Python
//
//     def coalescent(state):
//         transitions = []
//         for i in range(state.size):
//             for j in range(i, state.size):
//                 same = int(i == j)
//                 if same and state[i] < 2: continue
//                 if not same and (state[i] < 1 or state[j] < 1): continue
//                 new = state.copy(); new[i]-=1; new[j]-=1; new[i+j+1]+=1
//                 transitions.append((new, state[i]*(state[j]-same)/(1+same)))
//         return transitions
//     graph = Graph(coalescent, ipv=[([nr_samples]+[0]*(nr_samples-1), 1)])
//     print(graph.expectation())
//
// ---------------------------------------------------------------------------
// Compile & run (from the repository root):
//
//   Simplest -- one clang++ command (compiles the C core as C via `-x c`;
//   relies on the compiler default C++ standard, which is >= C++14):
//
//     clang++ -O2 -I api/cpp -I api/c -I src/c \
//       docs/pages/examples/cpp/coalescent_from_callback.cpp \
//       src/cpp/phasiccpp.cpp api/cpp/scc_graph.cpp \
//       -x c src/c/phasic.c src/c/phasic_hash.c src/c/phasic_log.c \
//             src/c/scc_synthetic.c src/c/scc_compose.c \
//       -o coalescent && ./coalescent 4
//
//   Portable -- explicit standards (C core as C11, then link as C++17):
//
//     clang -O2 -std=c11 -I api/c -I src/c -c \
//       src/c/phasic.c src/c/phasic_hash.c src/c/phasic_log.c \
//       src/c/scc_synthetic.c src/c/scc_compose.c
//     clang++ -O2 -std=c++17 -I api/cpp -I api/c -I src/c \
//       docs/pages/examples/cpp/coalescent_from_callback.cpp \
//       src/cpp/phasiccpp.cpp api/cpp/scc_graph.cpp \
//       phasic.o phasic_hash.o phasic_log.o scc_synthetic.o scc_compose.o \
//       -o coalescent && ./coalescent 4
//
// (Use g++ instead of clang++/clang if you prefer; the same flags apply.
// No MPFR/GMP or OpenMP are required for this example.)
//
// The number of samples is an optional argument (default 4), e.g. `./coalescent 4`:
//     nr_samples: 4
//     vertices: 6  edges: 6
//     expectation (E[T]): 1.5
// ---------------------------------------------------------------------------

#include <phasiccpp.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>

// 1:1 port of the Python coalescent(state) callback. state[k] is the number of
// blocks of size (k+1); coalescing a size-(i+1) block with a size-(j+1) block
// yields a size-(i+j+2) block at index i+j+1. The coalescent invariant
// sum_k (k+1)*state[k] == nr_samples keeps i+j+1 in range for kept transitions.
phasic::Graph coalescent(int nr_samples) {
  std::vector<int> initial_state(nr_samples, 0);
  initial_state[0] = nr_samples;                                   // [nr_samples, 0, 0, ...]
  std::vector<std::pair<std::vector<int>, double>> ipv = {{initial_state, 1.0}};

  auto callback =
      [](const std::vector<int> &state) -> std::vector<phasic::Transition> {
    std::vector<phasic::Transition> transitions;
    int size = static_cast<int>(state.size());
    for (int i = 0; i < size; ++i) {
      for (int j = i; j < size; ++j) {
        int same = (i == j) ? 1 : 0;
        if (same && state[i] < 2) continue;
        if (!same && (state[i] < 1 || state[j] < 1)) continue;
        std::vector<int> next = state;                            // state.copy()
        next[i] -= 1;
        next[j] -= 1;
        next[i + j + 1] += 1;
        double rate = static_cast<double>(state[i]) * (state[j] - same) / (1 + same);
        transitions.emplace_back(next, rate);                     // constant edge
      }
    }
    return transitions;
  };

  return phasic::Graph::from_callback(nr_samples, ipv, callback);
}

int main(int argc, char **argv) {
  // Number of samples from the command line (default 4), e.g. `./coalescent 10`.
  int nr_samples = (argc > 1) ? std::atoi(argv[1]) : 4;
  if (nr_samples < 1) {
    std::fprintf(stderr, "usage: %s [nr_samples]   (positive integer, default 4)\n",
                 argv[0]);
    return 1;
  }

  phasic::Graph graph = coalescent(nr_samples);
  std::printf("nr_samples: %d\n", nr_samples);
  std::printf("vertices: %zu  edges: %zu\n",
              graph.vertices_length(), graph.edges_length());
  // graph.expectation() = E[time to absorption from the starting vertex] =
  // the distribution mean; same method name as Python's graph.expectation().
  // (Equivalent to the per-vertex graph.expected_waiting_time()[0].)
  std::printf("expectation (E[T]): %g\n", graph.expectation());
  return 0;
}
