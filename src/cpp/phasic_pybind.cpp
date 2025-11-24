#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

// FIXME: Had to:
// cd ~/miniconda3/envs/phasetype/include
// ln -s eigen3/Eigen
// #include <eigen3/Eigen/Core>
#include <Eigen/Core>

#include "phasiccpp.h"
#include "parameterized/graph_builder.hpp"

// Include C API for hash functions and logging
extern "C" {
#include "../../api/c/phasic_hash.h"
#include "../../src/c/phasic_log.h"
}

// Only include FFI headers if XLA FFI is available
#ifdef HAVE_XLA_FFI
#include "parameterized/graph_builder_ffi.hpp"

// No static symbols needed - we'll create handlers on-demand via functions
#endif

#include <deque>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstdlib>

// Platform-specific dynamic library loading headers
#ifdef _WIN32
    #include <windows.h>
    // Windows uses _popen and _pclose instead of popen and pclose
    #define popen _popen
    #define pclose _pclose
#else
    #include <dlfcn.h>
#endif

namespace py = pybind11;
using std::deque;
using std::vector;
using std::tuple;
using std::deque;
using std::endl;


///////////////////////////////////////////////////////
// Jax interface
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <cassert>
#include <iostream>
#include <iomanip>

// extern "C" {

//   // JAX custom call signature with scalar operands
//   __attribute__((visibility("default")))
//   void _pmf_jax_ffi_prim(void* out_ptr, void* in_ptrs);

//   // JAX custom call signature for jax_graph_method_pmf
//   void _pmf_jax_ffi_prim(void* out_ptr, void* in_ptrs) {

//       void** buffers = reinterpret_cast<void**>(in_ptrs);
//       phasic::Graph* graph = reinterpret_cast<phasic::Graph*>(buffers[0]);
//       int64_t* times = reinterpret_cast<int64_t*>(buffers[1]);
//       int64_t* n_ptr = reinterpret_cast<int64_t*>(buffers[2]);

//       double* output = reinterpret_cast<double*>(out_ptr);      
      
//       // Extract dimensions from scalar operands
//       int64_t n = *n_ptr;

//       for (int64_t idx = 0; idx < n; ++idx) {
//         int64_t k = times[idx];
//         output[idx] = graph->dph_pmf(k);
//       }
//   }

//   // XLA custom call registration
//   void register_jax_graph_method_pmf() {
//       // This would normally register with XLA, but for simplicity we'll rely on 
//       // the Python side custom call mechanism
//   }

// }

///////////////////////////////////////////////////////

using namespace pybind11::literals; // to bring in the `_a` literal

static void set_c_seed() {
  py::object random = py::module_::import("random");//.attr("randint");
  py::object obj = random.attr("randint")(0, 1000000);
  unsigned int i = (unsigned int) obj.cast<int>();
  srand(i);
  // py::print(i);
}


static int fac(int n) {
    if (n == 0) {
        return 1;
    }

    return n * fac(n - 1);
}


template<pybind11::return_value_policy Policy = pybind11::return_value_policy::reference_internal, typename Iterator, typename Sentinel, typename ValueType = typename pybind11::detail::iterator_access<Iterator>::result_type, typename ...Extra>
pybind11::typing::Iterator<ValueType> make_iterator(Iterator first, Sentinel last, Extra&&... extra);

/* Bind MatrixXd (or some other Eigen type) to Python */
// typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXd dMatrix;

typedef dMatrix::Scalar dScalar;
//  constexpr bool rowMajor = dMatrix::Flags & Eigen::RowMajorBit;

/* Bind MatrixXd (or some other Eigen type) to Python */
// typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXi iMatrix;

typedef iMatrix::Scalar iScalar;
constexpr bool rowMajor = iMatrix::Flags & Eigen::RowMajorBit;


struct matrix_representation {
    iMatrix states;
    dMatrix SIM;
    std::vector<double> IPV;
    std::vector<int> indices;
};

// matrix_representation* _graph_as_matrix(phasic::Graph graph) {



//     int nr_states = 0;
//     for (size_t i = 1; i < graph.vertices_length(); ++i) {
//       if (graph->vertices[i]->edge_length == 0) {
//         continue;
//       }
//       ++nr_states;
//     }

//     std::vector<int> indices(nr_states);
//     dMatrix SIM = dMatrix(nr_states, nr_states);
//     std::vector<double> IPV(nr_states);
//     iMatrix states = iMatrix(nr_states, graph.state_length());
    
//     for (int = 0; i < nr_states; ++i) {
//       IPV[idx] = 0;
//     }
//     phasic::Vertex starting_vertex = graph.vertices[0];
//     for (size_t i = 0; i < starting_vertex.edges_length(); ++i) {
//       phasic::Edge edge = starting_vertex->edges[i]->to()->index;

//       indices[i]

//       IPV[idx] = dist->initial_probability_vector[i];
//     }

    
//     int idx = 0;
//     for (size_t i = 1; i < graph.vertices_length(); ++i) {
//       if (graph->vertices[i]->edge_length == 0) {
//         continue;
//       }
//       indices[idx] = graph->vertices[i]->index + 1;
//     }



//     for (int = 0; i < nr_states; ++i) {
//       int vertex_idx = indices[i] - 1;

//     }


//     int idx = 0;
//     for (size_t i = 1; i < graph.vertices_length(); ++i) {
//       if (graph->vertices[i]->edge_length == 0) {
//         continue;
//       }
//       indices[idx] = graph->vertices[i]->index + 1;

//       for (size_t j = 0; j < dist->length; ++j) {
//           SIM(i, j) = dist->sub_intensity_matrix[i][j];
//       }
//         for (size_t j = 0; j < state_length; j++) {
//             states(i, j) = dist->vertices[i]->state[j];
//         }

//       ++idx;
//     }




//     }


//     for (size_t i = 0; i < dist->length; ++i) {
//         IPV[i] = dist->initial_probability_vector[i];

//         for (size_t j = 0; j < dist->length; ++j) {
//             SIM(i, j) = dist->sub_intensity_matrix[i][j];
//         }
//     }

//     size_t state_length = graph.state_length();

//     rows = dist->length;
//     cols = state_length;
//     iMatrix states = iMatrix(rows, cols);

//     for (size_t i = 0; i < dist->length; i++) {
//         for (size_t j = 0; j < state_length; j++) {
//             states(i, j) = dist->vertices[i]->state[j];
//         }
//     }

//     std::vector<int> indices(dist->length);
//     for (size_t i = 0; i < dist->length; i++) {
//         indices[i] = dist->vertices[i]->index + 1;
//     }

//     struct matrix_representation *matrix_rep = new matrix_representation();
//     matrix_rep->states = states;
//     matrix_rep->SIM = SIM;
//     matrix_rep->IPV = IPV;
//     matrix_rep->indices = indices;

//     ::ptd_phase_type_distribution_destroy(dist);
//     return matrix_rep;
// }

matrix_representation* _graph_as_matrix(phasic::Graph graph) {

    ::ptd_phase_type_distribution *dist = ::ptd_graph_as_phase_type_distribution(graph.c_graph());

    int nr_vertices = graph.vertices_length();

    int rows = dist->length;
    int cols = dist->length;
    dMatrix SIM = dMatrix(rows, cols);
    std::vector<double> IPV(dist->length);

    for (size_t i = 0; i < dist->length; ++i) {
        IPV[i] = dist->initial_probability_vector[i];

        for (size_t j = 0; j < dist->length; ++j) {
            SIM(i, j) = dist->sub_intensity_matrix[i][j];
        }
    }

    size_t state_length = graph.state_length();

    rows = dist->length;
    cols = state_length;
    iMatrix states = iMatrix(rows, cols);
    for (size_t i = 0; i < dist->length; i++) {      
        for (size_t j = 0; j < state_length; j++) {
            states(i, j) = dist->vertices[i]->state[j];
        }
    }

    std::vector<int> indices(dist->length);
    for (size_t i = 0; i < dist->length; i++) {
        indices[i] = dist->vertices[i]->index + 1;
    }

    struct matrix_representation *matrix_rep = new matrix_representation();
    matrix_rep->states = states;
    matrix_rep->SIM = SIM;
    matrix_rep->IPV = IPV;
    matrix_rep->indices = indices;

    ::ptd_phase_type_distribution_destroy(dist);
    return matrix_rep;
}


class MatrixRepresentation {
    private:

    public:
        iMatrix states;
        dMatrix sim;
        std::vector<double> ipv;
        std::vector<int> indices;

        MatrixRepresentation(phasic::Graph graph) {
            struct matrix_representation *rep = _graph_as_matrix(graph);
            this->states = rep->states;
            this->sim = rep->SIM;
            this->ipv = rep->IPV;
            this->indices = rep->indices;
            delete rep;  // Clean up the allocated memory
        }

        // // pybind11 factory function
        // static MatrixRepresentation init_factory(phasic::Graph graph) {
        //     return MatrixRepresentation(graph);
        // }

        ~MatrixRepresentation() {
        }
};


iMatrix _states(phasic::Graph &graph) {

      std::vector<phasic::Vertex> ver = graph.vertices();

      int rows = ver.size();
      int cols = graph.state_length();
      iMatrix states = iMatrix(rows, cols);

      for (size_t i = 0; i < ver.size(); i++) {
          for (size_t j = 0; j < graph.state_length(); j++) {
              states(i, j) = ver[i].state()[j];
          }
      }

      return states;
  }

  
  // std::vector<double> _sample(phasic::Graph graph, int n, std::vector<double> rewards) {

  //     if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
  //         char message[1024];

  //         snprintf(
  //                 message,
  //                 1024,
  //                 "Failed: Rewards must match the number of vertices. Expected %i, got %i",
  //                 (int) graph.c_graph()->vertices_length,
  //                 (int) rewards.size()
  //         );

  //         throw std::runtime_error(
  //                 message
  //         );
  //     }
  //     std::vector<double> res(n);

  //     set_c_seed();

  //     for (int i = 0; i < n; i++) {
  //         if (rewards.empty()) {
  //             res[i] = (double) (graph.random_sample());
  //         } else {
  //             res[i] = (double) (graph.random_sample(rewards));
  //         }
  //     }

  //     return res;

  //   }


// Utility function for use in both moments dph_expectation and dph_variance lambda functions
std::vector<double> _moments(phasic::Graph &graph, int power, const std::vector<double> &rewards = vector<double>()) {

      if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
          char message[1024];

          snprintf(
                  message,
                  1024,
                  "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                  (int) graph.c_graph()->vertices_length,
                  (int) rewards.size()
          );

          throw std::runtime_error(
                  message
          );
      }

      if (power <= 0) {
          char message[1024];

          snprintf(
                  message,
                  1024,
                  "Failed: power must be a strictly positive integer. Got %i",
                  power
          );

          throw std::runtime_error(
                  message
          );
      }

      std::vector<double> res(power);
      std::vector<double> rewards2 = graph.expected_waiting_time(rewards);
      std::vector<double> rewards3(rewards2.size());
      res[0] = rewards2[0];

      std::vector<double> rw = rewards;

      // if (!rewards.empty()) {
      //     rw = as<std::vector<double> >(rewards);
      // }

      for (int i = 1; i < power; i++) {
          if (!rewards.empty()) {
              for (int j = 0; j < (int) rewards2.size(); j++) {
                  rewards3[j] = rewards2[j] * rw[j];
              }
          } else {
              rewards3 = rewards2;
          }

          rewards2 = graph.expected_waiting_time(rewards3);
          res[i] = fac(i + 1) * rewards2[0];
      }

      return res;

  }
  
// // Vectorize this
    
// py::array_t<double> _expectation(phasic::Graph &graph, py::iterable_t<py::array_t<double> >() rewards) {


//     for (auto v : x)
//     std::cout << " " << v.to_string();
//   }


//   py::array_t<double> _expectation(phasic::Graph &graph, py::array_t<double> rewards) {

//     py::buffer_info reward_buf = rewards.request();
//     if (reward_buf.ndim != 1)
//       throw std::runtime_error("Number of dimensions must be one");

//     /* No pointer is passed, so NumPy will allocate the buffer */
//     auto result = py::array_t<double>(reward_buf.size);

//     py::buffer_info result_buf = result.request();

//     double *reward_ptr = static_cast<double *>(reward_buf.ptr);
//     double *result_ptr = static_cast<double *>(result_buf.ptr);

//     for (size_t idx = 0; idx < reward_buf.shape[0]; idx++) {

//       // std::vector<double> _vector(reward_ptr, reward_ptr + reward_buf[idx].shape[0]);
//       std::vector<double> _vector(reward_ptr, reward_ptr + reward_buf.shape[0]);

//       result_ptr[idx] = _moments(graph, 1, _vector)[0];
//     }

//     return result;
// }


double _expectation(
  phasic::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

  return _moments(graph, 1, rewards)[0];
}



double _variance(
  phasic::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

    std::vector<double> exp = graph.expected_waiting_time(rewards);
    std::vector<double> second;

    if (rewards.empty()) {
        second = graph.expected_waiting_time(exp);
    } else {
        std::vector<double> new_rewards(exp.size());
        std::vector<double> rw = rewards;

        for (int i = 0; i < (int) exp.size(); i++) {
            new_rewards[i] = exp[i] * rw[i];
        }

        second = graph.expected_waiting_time(new_rewards);
    }

    return (2 * second[0] - exp[0] * exp[0]);    

}

double _covariance(phasic::Graph &graph, 
  const std::vector<double> &rewards1 = vector<double>(),
  const std::vector<double> &rewards2 = vector<double>()) {

    std::vector<double> exp1 = graph.expected_waiting_time(rewards1);
    std::vector<double> exp2 = graph.expected_waiting_time(rewards2);


    std::vector<double> new_rewards(exp1.size());


    for (int i = 0; i < exp1.size(); i++) {
      new_rewards[i] = exp1[i] * rewards2[i];
    }

    std::vector<double> second1 = graph.expected_waiting_time(new_rewards);


    for (int i = 0; i < exp1.size(); i++) {
        new_rewards[i] = exp2[i] * rewards1[i];
    }

    std::vector<double> second2 = graph.expected_waiting_time(new_rewards);

    return (second1[0] + second2[0] - exp1[0] * exp2[0]);    

}

double _expectation_discrete(
  phasic::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {
    return _moments(graph, 1, rewards)[0];
}

double _variance_discrete(
  phasic::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

    if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
      char message[1024];

      snprintf(
              message,
              1024,
              "Failed: Rewards must match the number of vertices. Expected %i, got %i",
              (int) graph.c_graph()->vertices_length,
              (int) rewards.size()
      );

      throw std::runtime_error(
              message
      );
  }
  if (rewards.empty()) {
      std::vector<double> m = _moments(graph, 2);

      return m[1] - 2*m[0];
  } else {
      // std::vector<double> rw = as<std::vector<double> >(rewards);
      std::vector<double> sq_rewards(rewards.size());

      for (int i = 0; i < (int)rewards.size(); i++) {
          sq_rewards[i] = rewards[i] * rewards[i];
      }

      std::vector<double> momentsr = _moments(graph, 2, rewards);
      std::vector<double> momentsrr = _moments(graph, 1, sq_rewards);

      return momentsr[1] - momentsr[0] * momentsr[0] - momentsrr[0];
    }

}

double _covariance_discrete(phasic::Graph &graph, 
  const std::vector<double> &rewards1 = vector<double>(),
  const std::vector<double> &rewards2 = vector<double>()) {

    std::vector<double> rw1(rewards1);
    std::vector<double> rw2(rewards2);
    std::vector<double> sq_rewards(rw1.size());

    for (int i = 0; i < (int)rw1.size(); i++) {
        sq_rewards[i] = rw1[i] * rw2[i];
    }

    std::vector<double> rw1to2(rw1.size());
    std::vector<double> rw2to1(rw2.size());
    std::vector<double> exp1 = graph.expected_waiting_time(rewards1);
    std::vector<double> exp2 = graph.expected_waiting_time(rewards2);

    for (int i = 0; i < (int)rw1.size(); i++) {
        rw1to2[i] = exp1[i] * rw2[i];
        rw2to1[i] = exp2[i] * rw1[i];
    }

    return graph.expected_waiting_time(rw1to2)[0] +
              _moments(graph, 1, sq_rewards)[0] - 
              _moments(graph, 1, rewards1)[0] *
              _moments(graph, 1, rewards2)[0];

}


// phasic::Graph build_state_space_callback_dicts(
//   const std::function<std::vector<py::dict> (std::vector<int> &state)> &callback, std::vector<int> &initial_state) {

//       phasic::Graph *graph = new phasic::Graph(initial_state.size());

//       phasic::Vertex init = graph->find_or_create_vertex(initial_state);

//         graph->starting_vertex().add_edge(init, 1);

//         int index = 1;
//         while (index < graph->vertices_length()) {

//           phasic::Vertex this_vertex = graph->vertex_at(index);
//           std::vector<int> this_state = graph->vertex_at(index).state();

//           std::vector<py::dict> children = callback(this_state);
//               for (auto child : children) {
//                 std::vector<int> child_state = child["state"].cast<std::vector<int> >();
//                 long double weight = child["weight"].cast<long double>();
//                 phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
//                 if (child.size() == 3) {
//                   std::vector<double> edge_params = child["edge_params"].cast<std::vector<double> >();
//                   this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
//                 } else {
//                   this_vertex.add_edge(child_vertex, weight);
//                 }
//               }
//               ++index;
//             }
//       return *graph;
//   }

  phasic::Graph build_state_space_callback_tuples(    
      // const std::function< std::vector<const std::tuple<const py::array_t<int>, long double> > (const py::array_t<int> &state)> &callback) { 
      const std::function<std::vector<py::object> (const py::array_t<int> &state)> &callback) { 

      phasic::Graph *graph = nullptr;

      // IPV from callback with no state argument
      std::vector<py::object> children = callback(py::array_t<int>());

      for (const auto child : children) {

        std::tuple<py::array_t<int>, long double> tup = child.cast<std::tuple<py::array_t<int>, long double> >();

        py::array_t<int> a = std::get<0>(tup);
        py::buffer_info buf = a.request();
        if (buf.ndim != 1)
          throw std::runtime_error("Number of dimensions must be one");
        int *ptr = static_cast<int *>(buf.ptr);
        std::vector<int> child_state(ptr, ptr + buf.shape[0]);

        long double weight = std::get<1>(tup);

        if (!graph) {
          graph = new phasic::Graph(child_state.size());
        }
        phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
        graph->starting_vertex().add_edge(child_vertex, weight);
      }

        int index = 1;
        while (index < graph->vertices_length()) {

          phasic::Vertex this_vertex = graph->vertex_at(index);

          auto a = new std::vector<int>(graph->vertex_at(index).state());
          auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
          py::array_t<int> this_state = py::array(a->size(), a->data(), capsule);

          std::vector<py::object> children = callback(this_state);

          for (auto child : children) {

            std::tuple<py::array_t<int>, long double> tup = child.cast<std::tuple<py::array_t<int>, long double> >();
            py::array_t<int> a = std::get<0>(tup);
            py::buffer_info buf = a.request();
            if (buf.ndim != 1)
              throw std::runtime_error("Number of dimensions must be one");
            int *ptr = static_cast<int *>(buf.ptr);
            std::vector<int> child_state(ptr, ptr + buf.shape[0]);
            long double weight = std::get<1>(tup);

            phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
            // if (child.size() == 3) {
            //   std::vector<double> edge_params = child[2].cast<std::vector<double> >();
            //   this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
            // } else {
              this_vertex.add_edge(child_vertex, weight);
            // }
          }
          ++index;
        }
      return *graph;
  }

  // Parameterized version: callback returns tuples of (state, weight, edge_state)
  phasic::Graph build_state_space_callback_tuples_parameterized(
      const std::function<std::vector<py::object> (const py::array_t<int> &state)> &callback) {

      phasic::Graph *graph = nullptr;

      // IPV from callback with no state argument
      std::vector<py::object> children = callback(py::array_t<int>());

      for (const auto child : children) {

        std::tuple<py::array_t<int>, long double, std::vector<double>> tup =
            child.cast<std::tuple<py::array_t<int>, long double, std::vector<double>> >();

        py::array_t<int> a = std::get<0>(tup);
        py::buffer_info buf = a.request();
        if (buf.ndim != 1)
          throw std::runtime_error("Number of dimensions must be one");
        int *ptr = static_cast<int *>(buf.ptr);
        std::vector<int> child_state(ptr, ptr + buf.shape[0]);

        long double weight = std::get<1>(tup);
        std::vector<double> edge_state = std::get<2>(tup);

        if (!graph) {
          graph = new phasic::Graph(child_state.size());
        }
        phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);

        // Starting edges: use add_edge() if coefficients empty, add_edge_parameterized() otherwise
        if (edge_state.empty()) {
            graph->starting_vertex().add_edge(child_vertex, weight);
        } else {
            graph->starting_vertex().add_edge_parameterized(child_vertex, weight, edge_state);
        }
      }

        int index = 1;
        while (index < graph->vertices_length()) {

          phasic::Vertex this_vertex = graph->vertex_at(index);

          auto a = new std::vector<int>(graph->vertex_at(index).state());
          auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
          py::array_t<int> this_state = py::array(a->size(), a->data(), capsule);

          std::vector<py::object> children = callback(this_state);

          for (auto child : children) {

            std::tuple<py::array_t<int>, long double, std::vector<double>> tup =
                child.cast<std::tuple<py::array_t<int>, long double, std::vector<double>> >();

            py::array_t<int> a = std::get<0>(tup);
            py::buffer_info buf = a.request();
            if (buf.ndim != 1)
              throw std::runtime_error("Number of dimensions must be one");
            int *ptr = static_cast<int *>(buf.ptr);
            std::vector<int> child_state(ptr, ptr + buf.shape[0]);

            long double weight = std::get<1>(tup);
            std::vector<double> edge_state = std::get<2>(tup);

            phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);

            // Use add_edge() if coefficients empty, add_edge_parameterized() otherwise
            if (edge_state.empty()) {
                this_vertex.add_edge(child_vertex, weight);
            } else {
                this_vertex.add_edge_parameterized(child_vertex, weight, edge_state);
            }
          }
          ++index;
        }
      return *graph;
  }

  // phasic::Graph build_state_space_callback_tuples(
  //   // const std::function<std::vector<const py::tuple> (std::vector<int> &state)> &callback, std::vector<int> &initial_state) {
  //   const std::function<std::vector<const py::tuple> (py::array_t<int> &state)> &callback, std::vector<int> &initial_state) {

  //     phasic::Graph *graph = new phasic::Graph(initial_state.size());

  //     phasic::Vertex init = graph->find_or_create_vertex(initial_state);

  //       graph->starting_vertex().add_edge(init, 1);

  //       int index = 1;
  //       while (index < graph->vertices_length()) {

  //         phasic::Vertex this_vertex = graph->vertex_at(index);

          
  //         // std::vector<int> this_state = graph->vertex_at(index).state();

  //         auto a = new std::vector<int>(graph->vertex_at(index).state());
  //         auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
  //         py::array_t<int> this_state = py::array(a->size(), a->data(), capsule);


  //         std::vector<const py::tuple> children = callback(this_state);

  //             for (auto child : children) {
  //               std::vector<int> child_state = child[0].cast<std::vector<int> >();
  //               long double weight = child[1].cast<long double>();
  //               phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
  //               if (child.size() == 3) {
  //                 std::vector<double> edge_params = child[2].cast<std::vector<double> >();
  //                 this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
  //               } else {
  //                 this_vertex.add_edge(child_vertex, weight);
  //               }
  //             }
  //             ++index;
  //           }
  //     return *graph;
  // }
    
  

bool is_number(const py::object& obj) {
    static py::object np_number = py::module_::import("numpy").attr("number");
    return py::isinstance<py::float_>(obj) || 
           py::isinstance<py::int_>(obj) ||
           py::isinstance(obj, np_number);
}


PYBIND11_MODULE(phasic_pybind, m) {

  ///////////////////////////////////////////////////////
  // for jax interface
  m.def("jax_graph_method_pmf", []() {}); // No-op: only used for symbol registration

  ///////////////////////////////////////////////////////


  m.doc() = "These are the docs";

  py::class_<iMatrix>(m, "iMatrix", py::buffer_protocol())

    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<iScalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(iScalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(iScalar));

        auto map = Eigen::Map<iMatrix, 0, Strides>(
            static_cast<iScalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return iMatrix(map);
    }))

    .def_buffer([](iMatrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                /* Pointer to buffer */
        sizeof(iScalar),                          /* Size of one scalar */
        py::format_descriptor<iScalar>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { m.rows(), m.cols() },                  /* Buffer dimensions */
        { sizeof(iScalar) * (rowMajor ? m.cols() : 1),
          sizeof(iScalar) * (rowMajor ? 1 : m.rows()) }
                                                 /* Strides (in bytes) for each index */
    );
   })
  ;

  py::class_<dMatrix>(m, "dMatrix", py::buffer_protocol())

    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<dScalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(dScalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(dScalar));

        auto map = Eigen::Map<dMatrix, 0, Strides>(
            static_cast<dScalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return dMatrix(map);
    }))

    .def_buffer([](dMatrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                /* Pointer to buffer */
        sizeof(dScalar),                          /* Size of one scalar */
        py::format_descriptor<dScalar>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { m.rows(), m.cols() },                  /* Buffer dimensions */
        { sizeof(dScalar) * (rowMajor ? m.cols() : 1),
          sizeof(dScalar) * (rowMajor ? 1 : m.rows()) }
                                                 /* Strides (in bytes) for each index */
    );
   })
  ;

    
  py::class_<MatrixRepresentation>(m, "MatrixRepresentation", R"delim(
      Matrix representation of phase-type distribution
      )delim")
      
    // .def(py::init(&MatrixRepresentation::init_factory))
      
    .def(py::init<const MatrixRepresentation>(), py::arg("graph"), R"delim(
Construct MatrixRepresentation from a Graph object.

Parameters
----------
graph : Graph
    The phase-type graph to convert to matrix representation.
      )delim")

    .def_readwrite("states", &MatrixRepresentation::states, R"delim(
State matrix where each row represents a vertex state.

Returns
-------
ndarray
    Integer matrix of size (n_vertices, state_length).
      )delim")
      
    .def_readwrite("sim", &MatrixRepresentation::sim, R"delim(
Sub-intensity matrix of the phase-type distribution.

Returns
-------
ndarray
    Float matrix of size (n_vertices, n_vertices) representing transition rates.
      )delim")
    .def_readwrite("ipv", &MatrixRepresentation::ipv, R"delim(
Initial probability vector of the phase-type distribution.

Returns
-------
list of float
    Vector of length n_vertices with initial probabilities.
      )delim")
    .def_readwrite("indices", &MatrixRepresentation::indices, R"delim(
Vertex indices mapping matrix rows to graph vertices.

Returns
-------
list of int
    Vector of length n_vertices with 1-indexed vertex numbers.
      )delim")
  ;            

    
  py::class_<phasic::Graph>(m, "Graph")

    .def(py::init<int>(), py::arg("state_length"))


      // .def("__iter__",
      //   [](phasic::Graph &g) {
      //       return make_iterator(g.begin(), g.end());
      //   }, py::return_value_policy::reference_internal, R"delim(
  
      //   )delim")
  

    .def(py::init<struct ::ptd_graph* >(), py::arg("ptd_graph"))

    .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
      
    .def(py::init<const phasic::Graph>(), py::arg("o"))

    .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
    
    // .def(py::init(&build_state_space_callback_dicts), 
    //   py::arg("callback_dicts"), py::arg("initial_state"))

    // .def(py::init(&build_state_space_callback_tuples),
    //       py::arg("callback_tuples"), py::arg("initial_state"))



    ///////////////////////////////////////////////////////
    // for jax interface
    .def("pointer", [](phasic::Graph* self) -> uintptr_t {
        return reinterpret_cast<uintptr_t>(self);
    }, R"delim(
Get memory address of the Graph object as an integer.

Used internally for JAX FFI integration.

Returns
-------
int
    Memory address as unsigned integer.
      )delim")
    ///////////////////////////////////////////////////////


    .def(py::init(&build_state_space_callback_tuples),
      py::arg("callback_tuples"))

    // .def_static("from_callback", [](py::function callback, py::kwargs kwargs) {
    //     // Create a wrapper that applies kwargs to the callback
    //     auto wrapper = [callback, kwargs](const py::array_t<int> &state) -> std::vector<py::object> {
    //       return callback(state, **kwargs).cast<std::vector<py::object>>();
    //     };
    //     return build_state_space_callback_tuples(wrapper);
    //   }, py::arg("callback"), R"delim(
    //   Builds a graph from a callback function. The callback function must take a single argument, which is the current state as an integer array.
    //   The callback function must return a list of tuples, where each tuple contains:
    //   - An integer array representing the child state.
    //   - A float representing the weight of the edge to that child state.

    //   The first call to the callback function will be made with an empty array, and should return the initial states and their weights.

    //   Additional keyword arguments are passed to the callback function.

    //   Parameters
    //   ----------
    //   callback : function
    //       A function that takes an integer array and returns a list of tuples as described above.
    //   **kwargs :
    //       Additional keyword arguments to pass to the callback function.

    //   Returns
    //   -------
    //   Graph
    //       A graph object representing the state space defined by the callback function.

    //   Examples
    //   --------
    //   ```
    //   def callback(state, nr_samples=2):
    //       if len(state) == 0:
    //           return [(np.array([nr_samples, 0]), 1.0)]
    //       elif state[0] > 1:
    //           return [(np.array([state[0] - 1, state[1] + 1]), state[0])]
    //       else:
    //           return []

    //   graph = Graph.from_callback(callback, nr_samples=4)
    //   ```
    //   )delim")

    .def(py::init(&build_state_space_callback_tuples_parameterized),
      py::arg("callback_tuples_parameterized"))

    // .def_static("from_callback_parameterized", [](py::function callback, py::kwargs kwargs) {
    //     // Create a wrapper that applies kwargs to the callback
    //     auto wrapper = [callback, kwargs](const py::array_t<int> &state) -> std::vector<py::object> {
    //       return callback(state, **kwargs).cast<std::vector<py::object>>();
    //     };
    //     return build_state_space_callback_tuples_parameterized(wrapper);
    //   }, py::arg("callback"), R"delim(
    //   Builds a graph with parameterized edges from a callback function. The callback function must take a single argument,
    //   which is the current state as an integer array. The callback function must return a list of tuples, where each tuple contains:
    //   - An integer array representing the child state.
    //   - A float representing the weight of the edge to that child state.
    //   - A list of floats representing the edge state for parameterized edges (same length as the child state).

    //   The first call to the callback function will be made with an empty array, and should return the initial states and their weights.

    //   Additional keyword arguments are passed to the callback function.

    //   Parameters
    //   ----------
    //   callback : function
    //       A function that takes an integer array and returns a list of 3-tuples as described above.
    //   **kwargs :
    //       Additional keyword arguments to pass to the callback function.

    //   Returns
    //   -------
    //   Graph
    //       A graph object with parameterized edges representing the state space defined by the callback function.

    //   Examples
    //   --------
    //   ```
    //   def callback(state):
    //       if len(state) == 0:
    //           return [(np.array([0]), 1.0, [1.0])]
    //       elif state[0] < 2:
    //           # edge_state for parameterized edge (same length as child state)
    //           return [(np.array([state[0] + 1]), 0.0, [1.5])]
    //       else:
    //           return []

    //   graph = Graph.from_callback_parameterized(callback)
    //   ```
    //   )delim")


    .def("create_vertex", static_cast<phasic::Vertex (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      
      Warning: the function find_or_create_vertex() should be preferred. 
      This function will *not* update the lookup tree, so find_vertex() will *not* return it.
      Creates a vertex matching `state`. Creates the vertex and adds it to the graph object. 

      Parameters
      ----------
      state : list of int or ndarray
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The newly inserted vertex in the graph.
      )delim")


    .def("find_vertex", static_cast<phasic::Vertex (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::find_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Finds a vertex matching the `state` parameter.

      Parameters
      ----------
      state : list of int or ndarray
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The found vertex in the graph or None.
      )delim")
      
    .def("vertex_exists", static_cast<bool (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::vertex_exists), py::arg("state"),
      py::return_value_policy::reference_internal, R"delim(
Check if a vertex with the given state exists in the graph.

Parameters
----------
state : list of int
    Integer sequence defining the state to search for.

Returns
-------
bool
    True if vertex exists, False otherwise.
      )delim")
      
    .def("find_or_create_vertex", static_cast<phasic::Vertex (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Finds a vertex matching the `state` parameter. If no such vertex exists, it creates the vertex and adds it to the graph object instead.

      Parameters
      ----------
      state : list of int or ndarray
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The newly found or inserted vertex in the graph.

      Examples
      --------
      ```
      graph = Graph(4)
      graph.find_or_create_vertex([1,2,1,0])
      ````
      )delim")
      
    // .def("find_or_create_vertex",
    //   [](phasic::Graph &graph, std::vector<double> state) {
    //     std::vector<int> int_vec(state.begin(), state.end());
    //     return graph.find_or_create_vertex(int_vec);
    //   }, R"delim(

    //   )delim")

    .def("focv", static_cast<phasic::Vertex (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Alias for find_or_create_vertex
      )delim")      

    .def("starting_vertex", &phasic::Graph::starting_vertex, 
      py::return_value_policy::copy, R"delim(
      Returns the special starting vertex of the graph. The starting vertex is always added at graph creation and always has index 0.

      Returns
      -------
      Vertex
          The starting vertex.
      )delim")
      
    .def("vertices", &phasic::Graph::vertices_p,
      py::return_value_policy::reference_internal, R"delim(
      Returns all vertices that have been added to the graph from either calling `find_or_create_vertex` or `create_vertex`.
      The first vertex in the list is *always* the starting vertex.

      Returns
      -------
      List
          A list of references to all vertices in the graph.

      Examples
      --------
      graph.create_graph(4)
      vertex_a = find_or_create_vertex([1,2,1,0])
      vertex_b = find_or_create_vertex([2,0,1,0])
      graph.vertices()[0] == graph.starting_vertex()
      graph.vertices()[1] == graph.vertex_at(1)
      graph.vertices_length() == 3
      )delim")      

    .def("vertex_at", &phasic::Graph::vertex_at_p, py::arg("index"),
      py::return_value_policy::reference_internal, R"delim(
      Returns a vertex at a particular index. This method is much faster than `Graph.vertices()[i]`.

      Parameters
      ----------
      graph : Graph
          A reference to the graph created by create_graph().
      index : int
          The index of the vertex to find.

      Returns
      -------
      Vertex
          A reference to the vertex at index `index` in the graph.
        )delim")

    .def("vertex_at",[](phasic::Graph &graph, double index) {
      return graph.vertex_at_p((int) index);

    }, py::return_value_policy::reference_internal, R"delim(
Get vertex at given index (float overload).

Parameters
----------
index : float
    Vertex index (will be cast to int).

Returns
-------
Vertex
    The vertex at the given index.
      )delim")

    .def("vertices_length", &phasic::Graph::vertices_length,
      py::return_value_policy::reference_internal, R"delim(
      Returns the number of vertices in the graph. This method is much faster than `len(Graph.vertices())`.

      Returns
      -------
      int
          The number of vertices.
      )delim")

    .def("parameterized", &phasic::Graph::parameterized,
      py::return_value_policy::reference_internal, R"delim(
      Returns whether the graph is parameterized (has parameterized edges).

      Returns
      -------
      bool
          True if the graph has parameterized edges, False otherwise.
      )delim")

    .def("states", &_states,
      py::return_value_policy::copy, R"delim(
      Returns a matrix where each row is the state of the vertex at that index.

      Returns
      -------
      int
         A matrix of size vertices_length() where the rows match the state of the vertex at that index
      )delim")

    .def("__repr__",
      [](phasic::Graph &g) {
          return "<Graph (" + std::to_string(g.vertices_length()) + " vertices)>";
      }, py::return_value_policy::move,
      R"delim(
String representation of the Graph object.

Returns
-------
str
    String in format "<Graph (N vertices)>".
      )delim")

    .def("param_length",
      [](phasic::Graph &g) {
          return g.c_graph()->param_length;
      }, R"delim(
      Get the parameter length of the graph (number of coefficients per edge).

      Returns 0 if no edges have been added yet, otherwise returns the coefficient
      length set by the first add_edge() call.

      Returns
      -------
      int
          Number of parameters/coefficients per edge
      )delim")

    .def("is_parameterized",
      [](phasic::Graph &g) {
          return g.c_graph()->parameterized;
      }, R"delim(
      Check if this graph uses parameterized edges (param_length > 1).

      Returns
      -------
      bool
          True if param_length > 1, False otherwise
      )delim")

    .def("update_weights", &phasic::Graph::update_weights_parameterized,
      py::arg("params"),
      R"delim(
    Updates all edge weights using the provided parameter vector.

    For parameterized graphs (param_length > 1), computes new edge weights via:
        edge.weight = dot(edge.coefficients, params)

    For constant graphs (param_length = 1), params should be [1.0] or omitted.

    Parameters
    ----------
    params : list of int or ndarray
        Parameter vector matching graph.param_length()

    Examples
    --------
    # Parameterized graph
    graph = Graph(state_length=2)
    v1 = graph.find_or_create_vertex([1, 0])
    v2 = graph.find_or_create_vertex([0, 1])
    v1.add_edge(v2, [2.0, 3.0])  # weight = 2.0*theta[0] + 3.0*theta[1]

    graph.update_weights([1.0, 2.0])  # weight becomes 2.0*1.0 + 3.0*2.0 = 8.0
    print(v1.edges()[0].weight())  # => 8.0
      )delim")

    .def("update_parameterized_weights",
        [](phasic::Graph& self, std::vector<double> params) {
            // Issue deprecation warning
            py::module_ warnings = py::module_::import("warnings");
            py::object DeprecationWarning = py::module_::import("builtins").attr("DeprecationWarning");
            warnings.attr("warn")(
                "update_parameterized_weights() is deprecated. Use update_weights() instead.",
                DeprecationWarning
            );

            // Call the underlying method
            self.update_weights_parameterized(params);
        },
        py::arg("rewards"),
        R"delim(
    DEPRECATED: Use update_weights() instead.

    Updates all parameterized edges of the graph by given scalars. Given a vector of scalars,
    computes a new weight of the parameterized edges in the graph by a simple inner product of
    the edge state vector and the scalar vector.

    Parameters
    ----------
    scalars : list of int or ndarray
        A numeric vector of multiplies for the edge states.

    Examples
    --------
    graph = Graph(4)
    v1 = graph.find_or_create_vertex([1, 2, 1, 0])
    v2 = graph.find_or_create_vertex([2, 0, 1, 0])
    graph.starting_vertex().add_edge(v1, 5)
    v1.add_edge(v2, 0, [5,2])
    graph.starting_vertex().edges()[0].weight()5
    v1.edges()[0].weight() # => 0
    graph.update_weights_parameterized([9,7])
    graph.starting_vertex().edges()[0]].weight() # => 5
    v1.edges()[0].weight() # => 59
      )delim")

    // DISABLED: Symbolic elimination - missing function implementations
    // Use trace-based elimination (ptd_record_elimination_trace) instead
    // .def("_eliminate_to_dag_internal",
    //   [](phasic::Graph &graph) -> uintptr_t {
    //     // Call C function to perform symbolic elimination
    //     struct ptd_graph_symbolic *symbolic =
    //         ptd_graph_symbolic_elimination(graph.c_graph());
    //
    //     if (symbolic == NULL) {
    //       throw std::runtime_error("Symbolic elimination failed");
    //     }
    //
    //     // Return pointer as integer for Python to store
    //     return reinterpret_cast<uintptr_t>(symbolic);
    //   },
    //   R"delim(
    // Internal method: Performs symbolic graph elimination.
    //
    // Returns an opaque pointer (as integer) to the symbolic DAG structure.
    // This is used internally by the Python SymbolicDAG class.
    //
    // DO NOT call this directly from Python - use Graph.eliminate_to_dag() instead.
    //   )delim")

      
      // .def("moments", 
      //   [](phasic::Graph &graph, int power, std::vector<double> &rewards) {
        
          // return _moments(graph, power, rewards);
      
    // }, 
    .def("moments", &_moments,
      py::arg("power"), py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(
      Computes the first `power` moments of the phase-type distribution. This function invokes 
      `Graph.expected_waiting_times()` consecutively to find the first moments, given by the `power` argument.

      Parameters
      ----------
      power : int
          The number of moments to compute.
      rewards : list of int or ndarray, optional
          Rewards to apply to the phase-type distribution.

      Returns
      -------
      Array
          Array of the first `power` moments. The first entry is the first moment (mean).

      Examples
      --------
      >>> graph = Graph(4)
      >>> v1 = graph.create_vertex([1,2,3,4])
      >>> v2 = graph.create_vertex([4,0,3,3])
      >>> a = graph.create_vertex([0,0,0,0])
      >>> graph.starting_vertex().add_edge(v1, 1)
      >>> v1.add_edge(v2, 4)
      >>> v2.add_edge(a, 10)
      >>> graph.moments(3)
      (0.350000 0.097500 0.025375)
      >>> graph.moments( 3, [0,2,1,0])
      (0.600 0.160 0.041)
   )delim")
    

    .def("expectation", &_expectation,
      py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(

      Computes the expectation (mean) of the phase-type distribution.

    This function invokes `expected_waiting_times()` and takes the first entry (from starting vertex).

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The expectation of the distribution.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.expectation() # => 0.35
    >>> graph.expectation( [0,2,1,0]) # => 0.6
    >>> ph = MatrixRepresentation(graph)
    >>> # This is a much faster version of
      )delim")      

    .def("variance", &_variance,
        py::arg("rewards")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(
    Computes the variance of the phase-type distribution.

    This function invokes `expected_waiting_times()` twice to find the first and second moment.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The variance of the distribution.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.variance() # => 0.0725
    >>> graph.variance( [0,2,1,0]) # => 0.26
    >>> ph = MatrixRepresentation(graph)
    >>> # This is a much faster version of
      )delim")    

      .def("covariance", &_covariance,
        py::arg("rewards1")=std::vector<double>(), 
        py::arg("rewards2")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(
    Computes the covariance of the phase-type distribution.

    This function invokes `expected_waiting_times()` twice to find the first and second moments for two sets of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards1 : list of float or ndarray
        The first set of rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.
    rewards2 : list of float or ndarray
        The second set of rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The covariance of the distribution.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.covariance( [0,2,1,0], [1,0,2,1]) # => 0.15
    >>> ph = MatrixRepresentation(graph)
    >>> # This is a much faster version of
         )delim")   


    .def("covariance_discrete", &_covariance_discrete,
          py::arg("rewards1")=std::vector<double>(), 
          py::arg("rewards2")=std::vector<double>(), 
          py::return_value_policy::move, 
          R"delim(
    Computes the covariance of the discrete phase-type distribution.

    This function invokes `dph_expected_waiting_times()` twice to find the first and second moments for two sets of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards1 : list of float or ndarray
        The first set of rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.
    rewards2 : list of float or ndarray
        The second set of rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The covariance of the distribution.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.covariance_discrete( [0,2,1,0], [1,0,2,1]) # => 0.15
    >>> ph = MatrixRepresentation(graph)
    >>> # This is a much faster version of
         )delim")   
      


    .def("expected_waiting_time", &phasic::Graph::expected_waiting_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(
    Computes the expected waiting time of the phase-type distribution.

    This function computes the expected waiting time for the given rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    list of float or ndarray
        A numeric vector of the expected waiting times.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> expected_waiting_time(graph) # => [0.35, 0.1, 0.05]
    >>> graph.expected_waiting_time( [0,2,1,0]) # => [0.6, 0.2, 0.1]
      )delim")
      
    .def("expected_residence_time", &phasic::Graph::expected_residence_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(
Computes the expected residence time of the phase-type distribution.

    This function computes the expected residence time for the given rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    list of float or ndarray
        A numeric vector of the expected residence times.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> expected_residence_time(graph) # => [0.35, 0.1, 0.05]
    >>> graph.expected_residence_time( [0,2,1,0]) # => [0.6, 0.2, 0.1]
      )delim")
      
      

    .def("sample",
      [](phasic::Graph &graph, int n, std::vector<double>rewards) {


        if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
            char message[1024];

            snprintf(
                    message,
                    1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.size()
            );

            throw std::runtime_error(
                    message
            );
        }
        std::vector<double> res(n);

        set_c_seed();

        for (int i = 0; i < n; i++) {
            if (rewards.empty()) {
                res[i] = (double) (graph.random_sample());
            } else {
                res[i] = (double) (graph.random_sample(rewards));
            }
        }

        return res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the phase-type distribution.

    This function generates samples from the phase-type distribution, optionally using a set of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    list of float or ndarray
        A numeric vector of samples.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.sample( 5) # => [0.35, 0.1, 0.05, 0.2, 0.15]
    >>> graph.sample( 5, [0,2,1,0]) # => [0.6, 0.2, 0.1, 0.4, 0.3]
    )delim")


    .def("sample_discrete",
      [](phasic::Graph &graph, int n, std::vector<double>rewards) {


        if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
            char message[1024];

            snprintf(
                    message,
                    1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.size()
            );

            throw std::runtime_error(
                    message
            );
        }
        std::vector<double> res(n);

        set_c_seed();

        for (int i = 0; i < n; i++) {
            if (rewards.empty()) {
                res[i] = (double) (graph.dph_random_sample_c(NULL));
            } else {
                res[i] = (double) (graph.dph_random_sample_c(&rewards[0]));
            }
        }

        return res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the discrete phase-type distribution.

    This function generates samples from the discrete phase-type distribution, optionally using a set of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : list of float or ndarray, optional
        Optional rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    list of float or ndarray
        A numeric vector of samples.

    Examples
    --------
    >>> graph = Graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.sample_discrete( 5) # => [0.35, 0.1, 0.05, 0.2, 0.15]
    >>> graph.sample_discrete( 5, [0,2,1,0]) # => [0.6, 0.2, 0.1, 0.4, 0.3]
    )delim")

    ///////////////////////////////////////////

    .def("sample_multivariate",
      [](phasic::Graph &graph, int n, dMatrix rewards) -> dMatrix  {

        if ((int) rewards.rows() != (int) graph.c_graph()->vertices_length) {
            char message[1024];
    
            snprintf(
                    message,
                    1024,
                    "Failed: Rewards rows must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.rows()
            );
    
            throw std::runtime_error(
                    message
            );
        }
    
        double *vrewards = (double *) calloc(rewards.rows() * rewards.cols(), sizeof(double));
    
        size_t index = 0;
    
        for (int i = 0; i < rewards.rows(); i++) {
            for (int j = 0; j < rewards.cols(); j++) {
                vrewards[index] = rewards(i, j);
                index++;
            }
        }
    
        set_c_seed();

        dMatrix mat_res = dMatrix(rewards.cols(), n);
    
        for (int i = 0; i < n; i++) {
            long double *res = ptd_mph_random_sample(graph.c_graph(), vrewards, (size_t) rewards.cols());
    
            for (int j = 0; j < rewards.cols(); j++) {
                mat_res(j, i) = res[j];
            }
    
            free(res);
        }
    
        free(vrewards);
    
        return mat_res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the multivariate phase-type distribution.

    This function generates samples from the multivariate phase-type distribution, using a set of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : ndarray
        A matrix of rewards, which should be applied to the phase-type distribution. The number of rows must match the number of vertices.

    Returns
    -------
    ndarray
        A matrix of samples.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> rewards = matrix([1,2,3,4,5,6,7,8], nrow=4, ncol=2)
    >>> graph.sample_multivariate( 5, rewards)
    )delim")

    
    ///////////////////////////////////////////


    .def("sample_multivariate_discrete",
      [](phasic::Graph &graph, int n, dMatrix rewards) -> dMatrix  {

        if ((int) rewards.rows() != (int) graph.c_graph()->vertices_length) {
          char message[1024];
  
          snprintf(
                  message,
                  1024,
                  "Failed: Rewards rows must match the number of vertices. Expected %i, got %i",
                  (int) graph.c_graph()->vertices_length,
                  (int) rewards.rows()
          );
  
          throw std::runtime_error(
                  message
          );
      }

    
    
        double *vrewards = (double *) calloc(rewards.rows() * rewards.cols(), sizeof(double));
    
        size_t index = 0;
    
        for (int i = 0; i < rewards.rows(); i++) {
            for (int j = 0; j < rewards.cols(); j++) {
                vrewards[index] = rewards(i, j);
                index++;
            }
        }
    
        set_c_seed();
    
        dMatrix mat_res = dMatrix(rewards.cols(), n);
    
        for (int i = 0; i < n; i++) {
            long double *res = ptd_mdph_random_sample(graph.c_graph(), vrewards, (size_t) rewards.cols());
    
            for (int j = 0; j < rewards.cols(); j++) {
                mat_res(j, i) = res[j];
            }
    
            free(res);
        }
    
        free(vrewards);
    
        return mat_res;

    }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the multivariate discrete phase-type distribution.

    This function generates samples from the multivariate discrete phase-type distribution, using a set of rewards.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : ndarray
        A matrix of rewards, which should be applied to the discrete phase-type distribution. The number of rows must match the number of vertices.

    Returns
    -------
    ndarray
        A matrix of samples.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> rewards = matrix([1,2,3,4,5,6,7,8], nrow=4, ncol=2)
    >>> graph.sample_multivariate_discrete( 5, rewards)
    )delim")







    // .def("sample_multivariate", static_cast<std::vector<long double> (phasic::Graph::*)(std::vector<double>, size_t)>(&phasic::Graph::mph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"), 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")


    // .def("sample_multivariate_discrete", static_cast<std::vector<long double> (phasic::Graph::*)(std::vector<double>, size_t)>(&phasic::Graph::mdph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"), 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")

    
  //  .def("sample_multivariate",
  //     [](phasic::Graph &graph, int n, std::vector<double> rewards) {

  //       py::print(py::str("not implemented"));

  //     }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=dMatrix(), R"delim(

  //   )delim")

      

      
    .def("random_sample_stop_vertex", &phasic::Graph::random_sample_stop_vertex, py::arg("time"), 
      py::return_value_policy::copy, R"delim(
    Samples a stopping vertex from the phase-type distribution given a stopping time.

    This function generates a sample of the stopping vertex from the phase-type distribution given a stopping time.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    time : float
        The stopping time.

    Returns
    -------
    Vertex
        The stopping vertex.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.random_sample_stop_vertex( 0.5) # => Vertex at stopping time 0.5
      )delim")
      
    .def("random_sample_discrete_stop_vertex", &phasic::Graph::dph_random_sample_stop_vertex, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Samples a stopping vertex from the discrete phase-type distribution given a number of jumps.

    This function generates a sample of the stopping vertex from the discrete phase-type distribution given a number of jumps.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    jumps : int
        The number of jumps.

    Returns
    -------
    Vertex
        The stopping vertex.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> graph.random_sample_discrete_stop_vertex( 3) # => Vertex at 3 jumps
      )delim")
      
    .def("state_length", &phasic::Graph::state_length, 
      py::return_value_policy::copy, R"delim(
    Returns the length of the state vector used to represent and reference a state in the graph.

    This function returns the length of the integer vector used to represent and reference a state in the graph.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.

    Returns
    -------
    int
        The length of the state vector.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> state_length(graph) # => 4
      )delim")
      
    .def("is_acyclic", &phasic::Graph::is_acyclic,
      py::return_value_policy::copy, R"delim(
    Checks if the graph is acyclic.

    This function checks if the graph is acyclic, meaning it does not contain any cycles.

    Parameters
    ----------
    graph : Graph
        The phase-type graph object.

    Returns
    -------
    bool
        True if the graph is acyclic, False otherwise.

    Examples
    --------
    >>> graph = Graph(4)
    >>> is_acyclic(graph) # => True or False
      )delim")
      
    .def("validate", &phasic::Graph::validate, R"delim(
    Validates the graph structure.

    This function checks the integrity and consistency of the graph structure.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> graph = create_graph(4)
    >>> validate(graph)
      )delim")

    .def("scc_decomposition", &phasic::Graph::scc_decomposition, R"delim(
    Compute strongly connected component decomposition.

    Decomposes this graph into SCCs (strongly connected components).
    Returns a condensation graph where each vertex represents an SCC.

    Returns
    -------
    SCCGraph
        SCC decomposition (always a DAG)

    Examples
    --------
    >>> graph = Graph(5)
    >>> # ... build graph ...
    >>> scc_graph = graph.scc_decomposition()
    >>> print(f"Found {scc_graph.n_sccs()} SCCs")
    >>> for scc in scc_graph.sccs_in_topo_order():
    >>>     print(f"SCC {scc.index()}: {scc.size()} vertices")
      )delim")

    // .def("expectation_dag", static_cast<phasic::Graph (phasic::Graph::*)(std::vector<double>)>(&phasic::Graph::expectation_dag), py::arg("rewards"), 
    //   py::return_value_policy::reference_internal, R"delim(
    // Computes the expectation of the directed acyclic graph (DAG) representation of the phase-type distribution.

    // This function computes the expectation of the phase-type distribution when represented as a directed acyclic graph (DAG).

    // Parameters
    // ----------
    // rewards : list of float or ndarray
    //     A numeric vector of rewards to be applied to the phase-type distribution. Must have length equal to `vertices_length()`.

    // Returns
    // -------
    // Graph
    //     A graph object representing the expectation of the DAG.

    // Examples
    // --------
    // >>> graph = create_graph(4)
    // >>> rewards = [1.0, 2.0, 3.0, 4.0]
    // >>> dag_expectation = graph.expectation_dag(rewards)
    //   )delim")
      
    .def("reward_transform", static_cast<phasic::Graph (phasic::Graph::*)(std::vector<double>)>(&phasic::Graph::reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(
    Transforms the graph using the given rewards.

    This function transforms the graph by applying the given rewards to the edges.

    Parameters
    ----------
    rewards : list of float or ndarray
        A numeric vector of rewards to be applied to the edges of the graph. Must have length equal to `vertices_length()`.

    Returns
    -------
    Graph
        A graph object with the transformed rewards.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> rewards = [1.0, 2.0, 3.0, 4.0]
    >>> transformed_graph = graph.reward_transform(rewards)
      )delim")
      
    .def("reward_transform_discrete", static_cast<phasic::Graph (phasic::Graph::*)(std::vector<int>)>(&phasic::Graph::dph_reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(
    Transforms the discrete phase-type distribution graph using the given rewards.

    This function transforms the graph by applying the given rewards to the edges in the discrete phase-type distribution.

    Parameters
    ----------
    rewards : list of float or ndarray
        A numeric vector of rewards to be applied to the edges of the graph. Must have length equal to `vertices_length()`.

    Returns
    -------
    Graph
        A graph object with the transformed rewards.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> rewards = [1, 2, 3, 4]
    >>> transformed_graph = graph.reward_transform_discrete(rewards)
      )delim")
      
    .def("normalize", &phasic::Graph::normalize, 
      py::return_value_policy::reference_internal, R"delim(
    Normalizes the graph.

    This function normalizes the graph by ensuring that the sum of the weights of the outgoing edges from each vertex is equal to 1.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        The normalized graph.

    Examples
    --------
    >>> graph = Graph(4)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [1,2,3,4]), 0.5)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [4,0,3,3]), 0.5)
    >>> normalized_graph = normalize(graph)
      )delim")
      
    .def("normalize_discrete", &phasic::Graph::dph_normalize, 
      py::return_value_policy::reference_internal, R"delim(
    Normalizes the discrete phase-type distribution graph.

    This function normalizes the graph by ensuring that the sum of the weights of the outgoing edges from each vertex is equal to 1 in the discrete phase-type distribution.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        The normalized graph.

    Examples
    --------
    >>> graph = Graph(4)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [1,2,3,4]), 0.5)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [4,0,3,3]), 0.5)
    >>> normalized_graph = normalize_discrete(graph)
      )delim")
      
    .def("notify_change", &phasic::Graph::notify_change, R"delim(
    Notifies the graph of a change.

    This function should be called whenever the graph structure is modified to ensure internal consistency.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> graph = Graph(4)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [1,2,3,4]), 0.5)
    >>> notify_change(graph)
      )delim")
      
    .def("defect", &phasic::Graph::defect, 
      py::return_value_policy::copy, R"delim(
    Computes the defect of the graph.

    The defect is the probability that the process does not reach an absorbing state.

    Parameters
    ----------
    None

    Returns
    -------
    float
        The defect of the graph.

    Examples
    --------
    >>> graph = Graph(4)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [1,2,3,4]), 0.5)
    >>> graph.starting_vertex().add_edge(graph.create_vertex( [4,0,3,3]), 0.5)
    >>> defect_value = defect(graph)
      )delim")
      
    .def("clone", &phasic::Graph::clone,
      py::return_value_policy::reference_internal, R"delim(
    Creates a copy of the graph.

    This function creates a deep copy of the graph, including all vertices and edges.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        A copy of the graph.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> graph_copy = clone(graph)
      )delim")
      
    .def("distribution_context",
      [](phasic::Graph &graph, int granularity) {
        return new phasic::ProbabilityDistributionContext(graph, granularity);
      }, 
      
      py::arg("granularity")=0, py::return_value_policy::move, 
      
      R"delim(
    Creates a probability distribution context for the graph.

    This function creates a context for computing the probability distribution of the graph.

    Parameters
    ----------
    granularity : int, optional
        The granularity of the distribution context. Default is 0.

    Returns
    -------
    ProbabilityDistributionContext
        A context for computing the probability distribution of the graph.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> context = graph.distribution_context(10)
     )delim")      
      


     .def("distribution_context_discrete",
      [](phasic::Graph &graph) {
        return new phasic::DPHProbabilityDistributionContext(graph);
      }, 
      
      py::return_value_policy::move, 
      
      R"delim(
    Creates a discrete probability distribution context for the graph.

    This function creates a context for computing the discrete probability distribution of the graph.

    Parameters
    ----------
    None

    Returns
    -------
    DPHProbabilityDistributionContext
        A context for computing the discrete probability distribution of the graph.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> context = distribution_context_discrete(graph)
     )delim")      
      


    .def("pdf",
         py::vectorize(&phasic::Graph::pdf), py::arg("time"), py::arg("granularity") = 0,
         py::return_value_policy::copy, R"delim(
    Computes the probability density function (PDF) of the phase-type distribution at a given time.

    This function computes the PDF of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the PDF.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    float
        The value of the PDF at the specified time.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> graph.pdf( 1.0) # => PDF value at time 1.0
    >>> graph.pdf( 1.0, 10) # => PDF value at time 1.0 with granularity 10
      )delim")
      
    .def("cdf",
         py::vectorize(&phasic::Graph::cdf), py::arg("time"), py::arg("granularity") = 0,
         py::return_value_policy::copy, R"delim(
    Computes the cumulative distribution function (CDF) of the phase-type distribution at a given time.

    This function computes the CDF of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the CDF.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    float
        The value of the CDF at the specified time.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> graph.cdf( 1.0) # => CDF value at time 1.0
    >>> graph.cdf( 1.0, 10) # => CDF value at time 1.0 with granularity 10
      )delim")
      
    .def("pmf_discrete", py::vectorize(&phasic::Graph::dph_pmf), py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Probability mass function of the discrete phase-type distribution.

    Returns the density (probability mass function) at a specific number of jumps.

    Parameters
    ----------
    x : IntegerVector
        Vector of the number of jumps (discrete time).
    graph : Graph
        The phase-type graph object.

    Returns
    -------
    list of float or ndarray
        A numeric vector of the density.

    Examples
    --------
    >>> graph = Graph(4)
    >>> ddph([1, 2, 3], graph) # => density values at jumps 1, 2, and 3
      )delim")
      
    .def("cdf_discrete", py::vectorize(&phasic::Graph::dph_cdf), py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Cumulative distribution function of the discrete phase-type distribution.

    Returns the cumulative distribution function (CDF) at a specific number of jumps.

    Parameters
    ----------
    q : IntegerVector
        Vector of the quantiles (jumps, discrete time).
    graph : Graph
        The phase-type graph object.

    Returns
    -------
    list of float or ndarray
        A numeric vector of the distribution function.

    Examples
    --------
    >>> graph = Graph(4)
    >>> pdph([1, 2, 3], graph) # => CDF values at jumps 1, 2, and 3
      )delim")

    .def("stop_probability", &phasic::Graph::stop_probability, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(
    Computes the stopping probability of the phase-type distribution at a given time.

    This function computes the probability that the process has stopped (reached an absorbing state) by a specified time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the stopping probability.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    float
        The stopping probability at the specified time.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> graph.stop_probability( 1.0) # => Stopping probability at time 1.0
    >>> graph.stop_probability( 1.0, 10) # => Stopping probability at time 1.0 with granularity 10
      )delim")

    .def("accumulated_visiting_time", &phasic::Graph::accumulated_visiting_time, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(
    Computes the accumulated visiting time of the phase-type distribution at a given time.

    This function computes the accumulated visiting time of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the accumulated visiting time.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    float
        The accumulated visiting time at the specified time.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> graph.accumulated_visiting_time( 1.0) # => Accumulated visiting time at time 1.0
    >>> graph.accumulated_visiting_time( 1.0, 10) # => Accumulated visiting time at time 1.0 with granularity 10
      )delim")

    // .def("stop_probability", 
    //      py::vectorize(&phasic::Graph::stop_probability), py::arg("time"), py::arg("granularity") = 0,
    //      py::return_value_policy::copy, R"delim(


    //   )delim")

    // .def("accumulated_visiting_time", 
    //      py::vectorize(&phasic::Graph::accumulated_visiting_time), py::arg("time"), py::arg("granularity") = 0,
    //      py::return_value_policy::copy, R"delim(


    //   )delim")

    
    .def("expectation_discrete", &_expectation_discrete,
      py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(
    Computes the expectation (mean) of the discrete phase-type distribution.

    This function computes the expectation of the discrete phase-type distribution given a set of rewards.

    Parameters
    ----------
    rewards : list of float
        A vector of rewards to be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The expectation of the discrete phase-type distribution.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> rewards = [1.0, 2.0, 3.0, 4.0]
    >>> graph.expectation_discrete( rewards) # => Expectation value
    )delim")


      .def("variance_discrete", &_variance_discrete,
        py::arg("rewards")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(    
    Computes the variance of the discrete phase-type distribution.

    This function computes the variance of the discrete phase-type distribution given a set of rewards.

    Parameters
    ----------
    rewards : list of float
        A vector of rewards to be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.

    Returns
    -------
    float
        The variance of the discrete phase-type distribution.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> rewards = [1.0, 2.0, 3.0, 4.0]
    >>> graph.variance_discrete( rewards) # => Variance value
    )delim")


    .def("stop_probability_discrete", &phasic::Graph::dph_stop_probability, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the probability of the Markov Chain of the discrete phase-type distribution standing at each vertex after a given number of jumps.

    This function computes the probability of the Markov Chain of the discrete phase-type distribution standing at each vertex after a specified number of jumps.

    Parameters
    ----------
    jumps : int
        The number of jumps (discrete time).

    Returns
    -------
    list of float or ndarray
        A numeric vector of the stop probabilities for each vertex.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 0.5)
    >>> v1.add_edge(v2, 0.8)
    >>> v2.add_edge(a, 0.5)
    >>> graph.dph_stop_probability( 3) # => Stop probabilities after 3 jumps
      )delim")

    .def("accumulated_visits_discrete", &phasic::Graph::dph_accumulated_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the number of visits of the Markov Chain of the discrete phase-type distribution at each vertex after a given number of jumps.

    This function computes the number of visits of the Markov Chain of the discrete phase-type distribution at each vertex after a specified number of jumps.

    Parameters
    ----------
    jumps : int
        The number of jumps (discrete time).

    Returns
    -------
    list of float or ndarray
        A numeric vector of the accumulated visits for each vertex.

    Examples
    --------
    >>> graph = create_graph(4)
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> add_edge(starting_vertex(graph), v1, 0.5)
    >>> v1.add_edge(v2, 0.8)
    >>> v2.add_edge(a, 0.5)
    >>> graph.dph_accumulated_visits( 3) # => Accumulated visits after 3 jumps
      )delim")

    .def("expected_visits_discrete", &phasic::Graph::dph_expected_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the expected jumps (or accumulated rewards) until absorption.
    This function can be used to compute the moments of a discrete phase-type distribution very fast and without much
    memory usage compared with the traditional matrix-based equations.
    The function takes in non-integers as rewards, but to be a *strictly* valid rewarded discrete phase-type distribution these should be integers.
    Parameters
    ----------
    graph : Graph
        The phase-type graph object.
    rewards : Nullable<list of float or ndarray>, optional
        Optional rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `vertices_length()`.
    Returns
    -------
    list of float or ndarray
        A numeric vector where entry `i` is the expected rewarded jumps starting at vertex `i`.
    See Also
    --------
    moments
    expectation
    variance
    covariance
    Examples
    --------
    >>> graph = create_graph(4)
    >>> rewards = [1.0, 2.0, 3.0, 4.0]
    >>> graph.dph_expected_visits( rewards) # => Expected visits value
      )delim")
      
    .def("as_matrices",
      [](phasic::Graph &graph) -> py::dict {
              // Get the phase-type distribution representation directly
              ::ptd_phase_type_distribution *dist = ::ptd_graph_as_phase_type_distribution(graph.c_graph());

              // Create Python dictionary to return
              py::dict result;

              // Convert states to numpy array
              size_t state_length = graph.state_length();
              size_t n_states = dist->length;

              py::array_t<int> states_array({n_states, state_length});
              auto states_view = states_array.mutable_unchecked<2>();
              for(size_t i = 0; i < n_states; i++) {
                  for(size_t j = 0; j < state_length; j++) {
                      states_view(i, j) = dist->vertices[i]->state[j];
                  }
              }
              result["states"] = states_array;

              // Convert SIM matrix to numpy array
              py::array_t<double> sim_array({n_states, n_states});
              auto sim_view = sim_array.mutable_unchecked<2>();
              for(size_t i = 0; i < n_states; i++) {
                  for(size_t j = 0; j < n_states; j++) {
                      sim_view(i, j) = dist->sub_intensity_matrix[i][j];
                  }
              }
              result["sim"] = sim_array;

              // Convert IPV to numpy array
              py::array_t<double> ipv_array(n_states);
              auto ipv_view = ipv_array.mutable_unchecked<1>();
              for(size_t i = 0; i < n_states; i++) {
                  ipv_view(i) = dist->initial_probability_vector[i];
              }
              result["ipv"] = ipv_array;

              // Convert indices to numpy array
              py::array_t<int> indices_array(n_states);
              auto indices_view = indices_array.mutable_unchecked<1>();
              for(size_t i = 0; i < n_states; i++) {
                  indices_view(i) = dist->vertices[i]->index + 1;
              }
              result["indices"] = indices_array;

              // Clean up the C distribution structure
              ::ptd_phase_type_distribution_destroy(dist);

              return result;
      }, R"delim(
    Converts the graph-based phase-type distribution into a traditional sub-intensity matrix and initial probability vector.

    Used to convert to the traditional matrix-based formulation. Has three entries: `.SIM` the sub-intensity matrix, `.IPV` the initial probability vector, `.states` the state of each vertex. Does *not* have the same order as vertices(). The indices returned are 1-based, like the input to vertex_at().

    Parameters
    ----------
    graph : Graph
        A reference to the graph created by Graph().

    Returns
    -------
    List
        A list of the sub-intensity matrix, states, and initial probability vector, and graph indices matching the matrix (1-indexed).

    Examples
    --------
    >>> graph = Graph(4)
    >>> v2 = graph.create_vertex([4,0,3,3])
    >>> v1 = graph.create_vertex([1,2,3,4])
    >>> a = graph.create_vertex([0,0,0,0])
    >>> graph.starting_vertex().add_edge(v1, 1)
    >>> v1.add_edge(v2, 4)
    >>> v2.add_edge(a, 10)
    >>> MatrixRepresentation(graph)
    >>> # .`states`
    >>> #         [,1] [,2] [,3] [,4]
    >>> #   [1,]    1    2    3    4
    >>> #   [2,]    4    0    3    3
    >>> # .SIM
    >>> #         [,1]  [,2]
    >>> #   [1,]   -4     4
    >>> #   [2,]    0   -10
    >>> # .IPV
    >>> #   [1] 1 0
    >>> # .indices
    >>> #   [1] 3 2
      )delim")


    .def_static("from_matrices",
      [](py::array_t<double> IPV, py::array_t<double> SIM, py::object states) -> phasic::Graph {
              // Get array info
              auto ipv = IPV.unchecked<1>();
              auto sim = SIM.unchecked<2>();

              // Check dimensions
              size_t n = ipv.shape(0);

              if (sim.shape(0) != n || sim.shape(1) != n) {
                  throw std::runtime_error("SIM must be square and have same dimension as IPV length");
              }

              // Check if states provided
              size_t state_dim = 1;
              py::array_t<int> states_array;

              if (!states.is_none()) {
                  states_array = states.cast<py::array_t<int>>();
                  auto states_view = states_array.unchecked<2>();
                  if (states_view.shape(0) != n) {
                      throw std::runtime_error("states must have same number of rows as IPV length");
                  }
                  state_dim = states_view.shape(1);
              }

              // Create graph
              phasic::Graph graph(state_dim);

              // Create vertices
              std::vector<phasic::Vertex*> vertices;

              int s = 0;
              if (!states.is_none()) {
                  auto states_view = states_array.unchecked<2>();
                  for (size_t i = 0; i < n; i++) {
                      std::vector<int> state(state_dim);
                      for (size_t j = 0; j < state_dim; j++) {
                          state[j] = states_view(i, j);
                      }
                      vertices.push_back(graph.find_or_create_vertex_p(state));
                  }
              } else {
                  // Create default states [0], [1], [2], ...
                  for (s = 0; s < n; s++) {
                      std::vector<int> state = {static_cast<int>(s)};
                      vertices.push_back(graph.find_or_create_vertex_p(state));
                  }
                  
                }

              // Create absorbing vertex
              std::vector<int> absorbing_state(state_dim, static_cast<int>(s));
              auto* absorbing = graph.find_or_create_vertex_p(absorbing_state);
                
              // Add edges from starting vertex according to IPV
              auto* start = graph.starting_vertex_p();
              double sum_ipv = 0.0;

              for (size_t i = 0; i < n; i++) {
                  if (ipv(i) > 0) {
                      start->add_edge(*vertices[i], ipv(i));
                      sum_ipv += ipv(i);
                  }
              }

              // Add edge to absorbing if IPV doesn't sum to 1
              if (sum_ipv < 0.99999) {
                throw std::runtime_error(
                        "Initial probability vector does not sum to one\n"
                );
                 // start->add_edge(*absorbing, 1.0 - sum_ipv);
              }

              // Add edges according to SIM matrix
              for (size_t i = 0; i < n; i++) {
                  double row_sum = 0.0;

                  // Off-diagonal elements are transition rates
                  for (size_t j = 0; j < n; j++) {
                      if (i != j && sim(i, j) > 0) {
                          vertices[i]->add_edge(*vertices[j], sim(i, j));
                          row_sum += sim(i, j);
                      }
                  }

                  // Diagonal element represents negative total exit rate
                  double exit_rate = -(sim(i, i) + row_sum);
                  if (exit_rate > 0.000001) {
                      vertices[i]->add_edge(*absorbing, exit_rate);
                  }
              }

              return graph;
      },
      py::arg("ipv"),
      py::arg("sim"),
      py::arg("states") = py::none(), R"delim(
    Converts the matrix-based representation into a phase-type graph.

    Sometimes the user might want to use the fast graph algorithms, but have some state-space given as a matrix. Therefore we can construct a graph from a matrix. If desired, a discrete phase-type distribution should just have no self-loop given. Note that the function `graph_as_matrix` may reorder the vertices to make the graph represented as strongly connected components in an acyclic manner.

    Parameters
    ----------
    IPV : list of float or ndarray
        The initial probability vector (alpha).
    SIM : NumericMatrix
        The sub-intensity matrix (S).
    rewards : NumericMatrix, optional
        The state/rewards of each of the vertices.

    Returns
    -------
    SEXP
        A graph object.

    Examples
    --------
    >>> g = matrix_as_graph(
    >>>     [0.5,0.3, 0],
    >>>     matrix([-3, 0, 0, 2, -4, 1, 0, 1,-3], ncol=3),
    >>>     matrix([1,4,5,9,2,7], ncol=2)
    >>> )
    >>> graph_as_matrix(g)
      )delim")

    ;

  // =========================================================================
  // SCCVertex bindings
  // =========================================================================
  py::class_<phasic::SCCVertex>(m, "SCCVertex",
      "Strongly connected component vertex (one SCC in condensation graph)")
      .def("size", &phasic::SCCVertex::size,
           "Number of vertices in this SCC")
      .def("index", &phasic::SCCVertex::index,
           "Index of this SCC in parent graph")
      .def("as_graph", &phasic::SCCVertex::as_graph,
           "Extract this SCC as a standalone Graph object")
      .def("internal_vertex_indices", &phasic::SCCVertex::internal_vertex_indices,
           "Get indices of internal vertices in original graph")
      .def("hash", &phasic::SCCVertex::hash,
           "Compute content hash of this SCC subgraph")
      .def("outgoing_scc_edges", &phasic::SCCVertex::outgoing_scc_edges,
           "Get indices of target SCCs for outgoing edges")
      .def("__len__", &phasic::SCCVertex::size)
      .def("__repr__", [](const phasic::SCCVertex& v) {
          return "<SCCVertex index=" + std::to_string(v.index()) +
                 " size=" + std::to_string(v.size()) + ">";
      });

  // =========================================================================
  // SCCGraph bindings
  // =========================================================================
  py::class_<phasic::SCCGraph>(m, "SCCGraph",
      "SCC decomposition of a graph (condensation/quotient graph)")
      .def("n_sccs", &phasic::SCCGraph::n_sccs,
           "Number of strongly connected components")
      .def("scc_at", &phasic::SCCGraph::scc_at,
           py::return_value_policy::reference_internal,
           "Get SCC vertex by index")
      .def("sccs_in_topo_order", &phasic::SCCGraph::sccs_in_topo_order,
           "Get all SCCs in topological order")
      .def("scc_sizes", &phasic::SCCGraph::scc_sizes,
           "Get sizes (vertex counts) of all SCCs")
      .def("original_graph", &phasic::SCCGraph::original_graph,
           py::return_value_policy::reference_internal,
           "Get reference to original graph")
      .def("scc_hashes", &phasic::SCCGraph::scc_hashes,
           "Compute content hashes for all SCCs")
      .def("__len__", &phasic::SCCGraph::n_sccs)
      .def("__getitem__", &phasic::SCCGraph::scc_at,
           py::return_value_policy::reference_internal)
      .def("__repr__", [](const phasic::SCCGraph& g) {
          return "<SCCGraph n_sccs=" + std::to_string(g.n_sccs()) + ">";
      });

  // =========================================================================
  // Trace Cache Functions
  // =========================================================================

  m.def("_c_load_trace_from_cache",
      [](const std::string& hash_hex) -> uintptr_t {
          struct ptd_elimination_trace *trace = ptd_load_trace_from_cache(hash_hex.c_str());
          return reinterpret_cast<uintptr_t>(trace);
      },
      py::arg("hash_hex"),
      R"delim(
Load elimination trace from disk cache (internal).

Loads a trace from ~/.phasic_cache/traces/<hash>.json using C JSON deserializer.
Returns opaque pointer to C struct ptd_elimination_trace.

Parameters
----------
hash_hex : str
    64-character hexadecimal hash identifying the trace

Returns
-------
int
    Pointer to C trace struct (0 if not found or error)

Notes
-----
- Caller must call _c_elimination_trace_destroy to free memory
- Cache can be disabled via PHASIC_DISABLE_CACHE=1 environment variable
- This is an internal function, use trace_serialization.load_trace_from_cache instead
)delim");

  m.def("_c_save_trace_to_cache",
      [](const std::string& hash_hex, uintptr_t trace_ptr) -> bool {
          if (trace_ptr == 0) {
              return false;
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return ptd_save_trace_to_cache(hash_hex.c_str(), trace);
      },
      py::arg("hash_hex"),
      py::arg("trace_ptr"),
      R"delim(
Save elimination trace to disk cache (internal).

Saves a trace to ~/.phasic_cache/traces/<hash>.json using C JSON serializer.

Parameters
----------
hash_hex : str
    64-character hexadecimal hash identifying the trace
trace_ptr : int
    Pointer to C struct ptd_elimination_trace

Returns
-------
bool
    True on success, False on error or if cache disabled

Notes
-----
- Cache can be disabled via PHASIC_DISABLE_CACHE=1 environment variable
- This is an internal function, use trace_serialization.save_trace_to_cache instead
)delim");

  m.def("_c_elimination_trace_destroy",
      [](uintptr_t trace_ptr) {
          if (trace_ptr != 0) {
              struct ptd_elimination_trace *trace =
                  reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
              ptd_elimination_trace_destroy(trace);
          }
      },
      py::arg("trace_ptr"),
      R"delim(
Free C elimination trace memory (internal).

Destroys a C trace struct and frees all associated memory.

Parameters
----------
trace_ptr : int
    Pointer to C struct ptd_elimination_trace

Notes
-----
- Must be called for every trace loaded from cache
- Safe to call with 0 (NULL pointer)
- This is an internal function
)delim");

  // Accessor functions for C trace struct fields
  m.def("_c_trace_get_n_vertices",
      [](uintptr_t trace_ptr) -> size_t {
          if (trace_ptr == 0) return 0;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->n_vertices;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_state_length",
      [](uintptr_t trace_ptr) -> size_t {
          if (trace_ptr == 0) return 0;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->state_length;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_param_length",
      [](uintptr_t trace_ptr) -> size_t {
          if (trace_ptr == 0) return 0;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->param_length;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_starting_vertex_idx",
      [](uintptr_t trace_ptr) -> size_t {
          if (trace_ptr == 0) return 0;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->starting_vertex_idx;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_is_discrete",
      [](uintptr_t trace_ptr) -> bool {
          if (trace_ptr == 0) return false;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->is_discrete;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_operations_length",
      [](uintptr_t trace_ptr) -> size_t {
          if (trace_ptr == 0) return 0;
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);
          return trace->operations_length;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_states",
      [](uintptr_t trace_ptr) -> py::array_t<int> {
          if (trace_ptr == 0) {
              return py::array_t<int>();
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);

          // Create numpy array (n_vertices, state_length)
          auto result = py::array_t<int>({trace->n_vertices, trace->state_length});
          auto buf = result.mutable_unchecked<2>();

          for (size_t i = 0; i < trace->n_vertices; i++) {
              for (size_t j = 0; j < trace->state_length; j++) {
                  buf(i, j) = trace->states[i][j];
              }
          }
          return result;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_vertex_rates",
      [](uintptr_t trace_ptr) -> py::array_t<size_t> {
          if (trace_ptr == 0) {
              return py::array_t<size_t>();
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);

          auto result = py::array_t<size_t>(trace->n_vertices);
          auto buf = result.mutable_unchecked<1>();

          for (size_t i = 0; i < trace->n_vertices; i++) {
              buf(i) = trace->vertex_rates[i];
          }
          return result;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_edge_probs",
      [](uintptr_t trace_ptr) -> py::list {
          if (trace_ptr == 0) {
              return py::list();
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);

          py::list result;
          for (size_t i = 0; i < trace->n_vertices; i++) {
              py::list vertex_edges;
              for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
                  vertex_edges.append(trace->edge_probs[i][j]);
              }
              result.append(vertex_edges);
          }
          return result;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_vertex_targets",
      [](uintptr_t trace_ptr) -> py::list {
          if (trace_ptr == 0) {
              return py::list();
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);

          py::list result;
          for (size_t i = 0; i < trace->n_vertices; i++) {
              py::list targets;
              for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
                  targets.append(trace->vertex_targets[i][j]);
              }
              result.append(targets);
          }
          return result;
      },
      py::arg("trace_ptr"));

  m.def("_c_trace_get_operation",
      [](uintptr_t trace_ptr, size_t idx) -> py::dict {
          if (trace_ptr == 0) {
              return py::dict();
          }
          struct ptd_elimination_trace *trace =
              reinterpret_cast<struct ptd_elimination_trace*>(trace_ptr);

          if (idx >= trace->operations_length) {
              throw std::out_of_range("Operation index out of range");
          }

          struct ptd_trace_operation *op = &trace->operations[idx];
          py::dict result;

          result["op_type"] = static_cast<int>(op->op_type);
          result["const_value"] = op->const_value;
          result["param_idx"] = op->param_idx;

          // Coefficients
          if (op->coefficients_length > 0 && op->coefficients != NULL) {
              py::list coeffs;
              for (size_t i = 0; i < op->coefficients_length; i++) {
                  coeffs.append(op->coefficients[i]);
              }
              result["coefficients"] = coeffs;
          } else {
              result["coefficients"] = py::list();
          }

          // Operands
          if (op->operands_length > 0 && op->operands != NULL) {
              py::list operands;
              for (size_t i = 0; i < op->operands_length; i++) {
                  operands.append(op->operands[i]);
              }
              result["operands"] = operands;
          } else {
              result["operands"] = py::list();
          }

          return result;
      },
      py::arg("trace_ptr"),
      py::arg("idx"));

  // =========================================================================
  // Symbolic DAG Helper Functions - DISABLED
  // =========================================================================
  // These functions are disabled due to missing ptd_expr_* implementations
  // Use trace-based elimination (ptd_record_elimination_trace) instead

  /*
  m.def("_symbolic_dag_instantiate",
    [](uintptr_t symbolic_ptr, py::array_t<double> params) -> py::object {
      // Convert pointer back to struct
      struct ptd_graph_symbolic *symbolic =
          reinterpret_cast<struct ptd_graph_symbolic*>(symbolic_ptr);

      if (symbolic == NULL) {
        throw std::runtime_error("Invalid symbolic DAG pointer");
      }

      // Get parameter array
      py::buffer_info params_info = params.request();
      const double *params_data = static_cast<const double*>(params_info.ptr);
      size_t n_params = params_info.size;

      // Call C function to instantiate
      struct ptd_graph *concrete =
          ptd_graph_symbolic_instantiate(symbolic, params_data, n_params);

      if (concrete == NULL) {
        throw std::runtime_error("Symbolic instantiation failed");
      }

      // Wrap in C++ Graph object and return
      return py::cast(new phasic::Graph(concrete));
    },
    py::arg("symbolic_ptr"),
    py::arg("params"),
    R"delim(
  Internal helper: Instantiate symbolic DAG with concrete parameters.

  This evaluates all expression trees with the given parameter vector
  and returns a concrete Graph object. This is O(n) instead of O(n³)!

  Parameters
  ----------
  symbolic_ptr : int
      Opaque pointer to symbolic DAG structure
  params : np.ndarray
      Parameter vector, shape (n_params,)

  Returns
  -------
  Graph
      Instantiated graph with evaluated edge weights
    )delim");

  m.def("_symbolic_dag_destroy",
    [](uintptr_t symbolic_ptr) {
      // Convert pointer back to struct
      struct ptd_graph_symbolic *symbolic =
          reinterpret_cast<struct ptd_graph_symbolic*>(symbolic_ptr);

      if (symbolic != NULL) {
        ptd_graph_symbolic_destroy(symbolic);
      }
    },
    py::arg("symbolic_ptr"),
    R"delim(
  Internal helper: Destroy symbolic DAG and free memory.

  Parameters
  ----------
  symbolic_ptr : int
      Opaque pointer to symbolic DAG structure
    )delim");

  m.def("_symbolic_dag_get_info",
    [](uintptr_t symbolic_ptr) -> py::dict {
      // Convert pointer back to struct
      struct ptd_graph_symbolic *symbolic =
          reinterpret_cast<struct ptd_graph_symbolic*>(symbolic_ptr);

      if (symbolic == NULL) {
        throw std::runtime_error("Invalid symbolic DAG pointer");
      }

      py::dict info;
      info["vertices_length"] = symbolic->vertices_length;
      info["state_length"] = symbolic->state_length;
      info["param_length"] = symbolic->param_length;
      info["is_acyclic"] = symbolic->is_acyclic;
      info["is_discrete"] = symbolic->is_discrete;

      return info;
    },
    py::arg("symbolic_ptr"),
    R"delim(
  Internal helper: Get metadata from symbolic DAG.

  Parameters
  ----------
  symbolic_ptr : int
      Opaque pointer to symbolic DAG structure

  Returns
  -------
  dict
      Dictionary with metadata (vertices_length, param_length, etc.)
    )delim");
  */

  // =========================================================================

  py::class_<phasic::Vertex>(m, "Vertex", R"delim(
Represents a vertex (state) in a phase-type graph.

Each vertex has an integer state vector and can have outgoing edges to other vertices.
      )delim")

    .def(py::init(&phasic::Vertex::init_factory), R"delim(
Create a Vertex object (internal use).

Users should use Graph.find_or_create_vertex() instead.
      )delim")
      
    .def("add_edge", [](phasic::Vertex& self, phasic::Vertex& to, py::object weight_or_coeffs) {

        py::module_ np = py::module_::import("numpy");
        py::object np_number = np.attr("number");
        // py::object np_int32 = np.attr("int32");
        // py::object np_float64 = np.attr("float64");        

        if (is_number(weight_or_coeffs)) {
            // Scalar: constant edge
            double weight = weight_or_coeffs.cast<double>();
            self.add_edge(to, weight);

            // Check for errors (e.g., edge mode locking)
            if (ptd_err[0] != '\0') {
                std::string error_msg((const char*)ptd_err);
                ptd_err[0] = '\0';  // Clear error
                throw std::runtime_error(error_msg);
            }
        } else if (py::isinstance<py::list>(weight_or_coeffs) || py::isinstance<py::array>(weight_or_coeffs)) {
            // Array: parameterized edge
            std::vector<double> coeffs = weight_or_coeffs.cast<std::vector<double>>();
            if (coeffs.empty()) {
                throw std::invalid_argument("Edge coefficients cannot be empty");
            }
            self.add_edge_parameterized(to, 0.0, coeffs);

            // Check for errors (e.g., edge mode locking)
            if (ptd_err[0] != '\0') {
                std::string error_msg((const char*)ptd_err);
                ptd_err[0] = '\0';  // Clear error
                throw std::runtime_error(error_msg);
            }
        } else {
            throw std::invalid_argument(
                "add_edge() expects either a scalar (float/int) or array-like (list/ndarray) argument"
            );
        }
    }, py::arg("to"), py::arg("weight_or_coeffs"), R"delim(
    Add an edge to another vertex with constant or parameterized weight.

    Parameters
    ----------
    to : Vertex
        Target vertex
    weight_or_coeffs : float or array-like
        If scalar: constant edge weight (e.g., 3.0)
        If array: coefficient vector for parameterized edge (e.g., [2.0, 9.0])

    Returns
    -------
    Edge
        The created edge

    Notes
    -----
    All edges in a graph must use the same form (all scalar or all array).
    The first call to add_edge() sets the mode for the entire graph.

    Examples
    --------
    >>> # Constant edge
    >>> v.add_edge(target, 3.0)

    >>> # Parameterized edge: weight = 2.0*theta[0] + 9.0*theta[1]
    >>> v.add_edge(target, [2.0, 9.0])
      )delim")

    .def("ae", [](phasic::Vertex& self, phasic::Vertex& to, py::object weight_or_coeffs) {
        // Alias for add_edge
        if (is_number(weight_or_coeffs)) {
            double weight = weight_or_coeffs.cast<double>();
            self.add_edge(to, weight);
        } else {
            std::vector<double> coeffs = weight_or_coeffs.cast<std::vector<double>>();
            self.add_edge_parameterized(to, 0.0, coeffs);
        }
    }, py::arg("to"), py::arg("weight_or_coeffs"), R"delim(Alias for add_edge)delim")

    .def("add_aux_vertex", [](phasic::Vertex& self, py::object rate) -> phasic::Vertex {
        bool is_parameterized = self.c_vertex()->graph->parameterized;

        if (is_number(rate)) {
            // Scalar: constant rate
            if (is_parameterized) {
                throw std::invalid_argument(
                    "Graph is parameterized. add_aux_vertex() requires array of coefficients, not scalar. "
                    "Example: v.add_aux_vertex([2.0, 1.0])"
                );
            }
            double rate_val = rate.cast<double>();
            return self.add_aux_vertex(rate_val);

        } else if (py::isinstance<py::list>(rate) || py::isinstance<py::array>(rate)) {
            // Array: parameterized rate
            if (!is_parameterized) {
                throw std::invalid_argument(
                    "Graph is not parameterized. add_aux_vertex() requires scalar rate, not array. "
                    "Example: v.add_aux_vertex(3.0)"
                );
            }
            std::vector<double> rate_coeffs = rate.cast<std::vector<double>>();
            if (rate_coeffs.empty()) {
                throw std::invalid_argument("Rate coefficients cannot be empty");
            }
            return self.add_aux_vertex(rate_coeffs);

        } else {
            throw std::invalid_argument(
                "add_aux_vertex() expects either a scalar (float/int) or array-like (list/ndarray) argument"
            );
        }
    }, py::arg("rate"), py::return_value_policy::reference_internal, R"delim(
Add an auxiliary vertex with all-zero state for discrete graphs.

Creates a new vertex with state [0, 0, ..., 0] and adds two edges:
1. From aux vertex to this vertex with constant weight 1.0
2. From this vertex to aux vertex with the given rate

Auxiliary vertices are typically used in discrete graphs to model intermediate
states and should be skipped during parameter updates.

Parameters
----------
rate : float or array-like
    If graph.parameterized() is True: coefficient vector (e.g., [2.0, 1.0])
    If graph.parameterized() is False: constant rate (e.g., 3.0)

Returns
-------
Vertex
    The created auxiliary vertex (with all-zero state)

Raises
------
ValueError
    If rate type doesn't match graph parameterization mode

Examples
--------
>>> # Non-parameterized graph
>>> g = phasic.Graph(2)
>>> v = g.find_or_create_vertex([1, 0])
>>> aux = v.add_aux_vertex(3.0)
>>> print(aux.state())  # [0, 0]

>>> # Parameterized graph: rate = 2.0*theta[0] + 1.0*theta[1]
>>> g = phasic.Graph(2)
>>> v1 = g.find_or_create_vertex([1, 0])
>>> v2 = g.find_or_create_vertex([2, 0])
>>> v1.add_edge(v2, [1.0, 0.0])  # Lock to parameterized mode
>>> aux = v1.add_aux_vertex([2.0, 1.0])
>>> print(aux.state())  # [0, 0]

Notes
-----
- The edge from aux to parent is always constant (weight 1.0)
- The edge from parent to aux matches the graph's parameterization mode
- Auxiliary vertices can be identified by their all-zero state
)delim")

    .def("__repr__",
      [](phasic::Vertex &v) {

        std::ostringstream s;
        s << "(";
        std::vector<int> state = v.state();
        for (auto i(state.begin()); i != state.end(); i++) {
            if (state.begin() != i) s << ",";
            s << *i;
        }
        s << ")";
        return s.str();
      }, R"delim(
String representation of the vertex showing its state.

Returns
-------
str
    State as comma-separated integers in parentheses, e.g., "(1,2,0)".
      )delim")

    .def("index",
        [](phasic::Vertex &v) {
          int idx = v.vertex->index; // why is index not already an int?
          return  idx;
        }, py::return_value_policy::copy, R"delim(
Get the vertex index in the parent graph.

Returns
-------
int
    Zero-based index of this vertex in the graph.
        )delim")

    .def("add_edge_parameterized",
        [](phasic::Vertex& self, phasic::Vertex& to, double weight, std::vector<double> edge_state) {
            // Issue deprecation warning
            py::module_ warnings = py::module_::import("warnings");
            py::object DeprecationWarning = py::module_::import("builtins").attr("DeprecationWarning");
            warnings.attr("warn")(
                "add_edge_parameterized() is deprecated. Use add_edge(to, [coefficients]) instead.",
                DeprecationWarning
            );

            // Call the underlying method
            self.add_edge_parameterized(to, weight, edge_state);
        },
        py::arg("to"), py::arg("weight"), py::arg("edge_state"),
        R"delim(
      DEPRECATED: Use add_edge(to, [coefficients]) instead.

      Adds an edge between two vertices in the graph.
      The graph represents transitions between states as a weighted directed edge between two vertices.
      Parameters
      ----------
      phase_type_vertex_from : SEXP
          The vertex that transitions from.
      phase_type_vertex_to : SEXP
          The vertex that transitions to.
      weight : float
          The weight of the edge, i.e., the transition rate.
      parameterized_edge_state : list of float or ndarray, optional
          Associate a numeric vector to an edge, for faster computations of moments when weights are changed.
      See Also
      --------
      expected_waiting_time
      moments
      variance
      covariance
      graph_update_weights_parameterized
      Examples
      --------
      >>> graph = create_graph(4)
      >>> vertex_a = graph.find_or_create_vertex([1,2,1,0])
      >>> vertex_b = graph.find_or_create_vertex([2,0,1,0])
      >>> vertex_a.add_edge(vertex_b, 1.5)
      )delim")

    .def("state",
        [](phasic::Vertex &v) {
          // to make it return np.array instead of list without copying data
          auto a = new std::vector<int>(v.state());
          auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
          return py::array(a->size(), a->data(), capsule);
        }, py::return_value_policy::copy,
        R"delim(
Get the state vector of this vertex.

Returns
-------
ndarray
    Integer array representing the state (zero-copy).
      )delim")

    // .def("state", &phasic::Vertex::state, 
    //   py::return_value_policy::reference_internal, R"delim(

    //   )delim")
      
    .def("edges", &phasic::Vertex::edges,
      py::return_value_policy::reference_internal, R"delim(
    Returns the out-going edges of a vertex.

    Returns a list of edges added by add_edge().

    Parameters
    ----------
    phase_type_vertex : SEXP
        The vertex to find the edges for.

    Returns
    -------
    List
        A list of out-going edges.
      )delim")

    .def("parameterized_edges", &phasic::Vertex::parameterized_edges,
      py::return_value_policy::reference_internal, R"delim(
    Returns the out-going parameterized edges of a vertex.

    Returns a list of parameterized edges added by add_edge_parameterized().

    Parameters
    ----------
    phase_type_vertex : SEXP
        The vertex to find the parameterized edges for.

    Returns
    -------
    List
        A list of out-going parameterized edges.
      )delim")

    .def(py::self == py::self)
    .def("__assign__", [](phasic::Vertex &v, const phasic::Vertex &o) {
          return v = o;
    }, py::is_operator(),
    py::return_value_policy::move, R"delim(
Assignment operator for Vertex objects.

Parameters
----------
other : Vertex
    The vertex to assign from.

Returns
-------
Vertex
    Reference to this vertex.
      )delim")
      
    // .def("c_vertex", &phasic::Vertex::c_vertex, R"delim(

    //   )delim")
      
    .def("rate", &phasic::Vertex::rate,
      py::return_value_policy::reference_internal, R"delim(
Get the total exit rate from this vertex.

The sum of all outgoing edge weights.

Returns
-------
float
    Total rate of leaving this vertex.
      )delim")
      
    ;

  py::class_<phasic::Edge>(m, "Edge", R"delim(
Represents a directed edge between two vertices in a phase-type graph.

Each edge has a weight (transition rate) and points to a target vertex.
      )delim")

      .def("__repr__",
        [](phasic::Edge &e) {
          std::ostringstream s;
          s << "" << e.weight() << "-(";
          std::vector<int> state = e.to().state();
          for (auto i(state.begin()); i != state.end(); i++) {
              if (state.begin() != i) s << ",";
              s << *i;
          }
          s << ")";
          return s.str();
        }, R"delim(
String representation showing edge weight and target state.

Returns
-------
str
    Format: "weight-(state)", e.g., "3.5-(1,0,2)".
        )delim")

    .def(py::init(&phasic::Edge::init_factory), R"delim(
Create an Edge object (internal use).

Users should use Vertex.add_edge() instead.
      )delim")
      
    .def("to", &phasic::Edge::to,
      py::return_value_policy::reference_internal, R"delim(
Get the target vertex of this edge.

Returns
-------
Vertex
    The vertex this edge points to.
      )delim")

    .def("weight", &phasic::Edge::weight,
      py::return_value_policy::reference_internal, R"delim(
Get the weight (transition rate/probability) of this edge.

Returns
-------
float
    The edge weight.
      )delim")

    .def("update_to", &phasic::Edge::update_to, R"delim(
Update the target vertex of this edge.

Parameters
----------
to : Vertex
    New target vertex.
      )delim")

    .def("update_weight", &phasic::Edge::update_weight, R"delim(
Update the weight of this edge.

Parameters
----------
weight : float
    New edge weight.
      )delim")

    .def("__assign__", [](phasic::Edge &e, const phasic::Edge &o) {
          return e = o;
    }, py::is_operator(), R"delim(
Assignment operator for Edge objects.

Parameters
----------
other : Edge
    The edge to assign from.

Returns
-------
Edge
    Reference to this edge.
      )delim")
      
    ;

  py::class_<phasic::ParameterizedEdge>(m, "ParameterizedEdge", R"delim(
Represents a parameterized edge with coefficient vector.

The edge weight is computed as dot(coefficients, theta) where theta is the parameter vector.
      )delim")

    .def(py::init(&phasic::ParameterizedEdge::init_factory), R"delim(
Create a ParameterizedEdge object (internal use).

Users should use Vertex.add_edge(to, [coefficients]) instead.
      )delim")

    .def("to", &phasic::ParameterizedEdge::to,
      py::return_value_policy::reference_internal, R"delim(
Get the target vertex of this edge.

Returns
-------
Vertex
    The vertex this edge points to.
      )delim")

    .def("weight", &phasic::ParameterizedEdge::weight,
      py::return_value_policy::reference_internal, R"delim(
Get the current weight of this edge.

The weight is updated when Graph.update_weights() is called.

Returns
-------
float
    The current edge weight.
      )delim")

    // base_weight() method removed - starting edges are never parameterized

    // .def("edge_state", 
    //   [](phasic::ParameterizedEdge &edge) {
    //     auto a = edge.state;
    //     auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<double>*>(a); });
    //     return py::array(a->size(), a->data(), capsule);
    //   }, R"delim(

    // )delim")  
    .def("edge_state", &phasic::ParameterizedEdge::edge_state,
      py::return_value_policy::reference_internal, R"delim(
Get the coefficient vector for this parameterized edge.

Returns
-------
list of float
    Coefficient vector used to compute weight as dot(coefficients, theta).
      )delim")

    .def("__assign__", [](phasic::ParameterizedEdge &e, const phasic::ParameterizedEdge &o) {
          return e = o;
    }, py::is_operator(), py::return_value_policy::move, R"delim(
Assignment operator for ParameterizedEdge objects.

Parameters
----------
other : ParameterizedEdge
    The edge to assign from.

Returns
-------
ParameterizedEdge
    Reference to this edge.
      )delim")
      
    ;

  py::class_<phasic::PhaseTypeDistribution>(m, "PhaseTypeDistribution", R"delim(
Matrix representation of a phase-type distribution.

Contains the sub-intensity matrix, initial probability vector, and vertex states.
      )delim")

    .def(py::init(&phasic::PhaseTypeDistribution::init_factory), R"delim(
Create a PhaseTypeDistribution object.

Use Graph.as_matrices() to convert from graph representation.
      )delim")

    // .def("c_distribution", &phasic::PhaseTypeDistribution::c_distribution, R"delim(

    //   )delim")

    .def_readwrite("length", &phasic::PhaseTypeDistribution::length,
      py::return_value_policy::reference_internal, R"delim(
Number of transient states in the distribution.

Returns
-------
int
    Length of the initial probability vector.
      )delim")

    .def_readwrite("vertices", &phasic::PhaseTypeDistribution::vertices,
       py::return_value_policy::reference_internal, R"delim(
List of vertices in the distribution.

Returns
-------
list of Vertex
    Vertices corresponding to transient states.
      )delim")      
    ;


  py::class_<phasic::AnyProbabilityDistributionContext>(m, "AnyProbabilityDistributionContext", R"delim(
Base class for probability distribution contexts (continuous or discrete).

Provides methods for computing PDFs, CDFs, and stepping through the distribution.
      )delim")

    .def(py::init<>(), R"delim(
Create an empty probability distribution context.
      )delim")

    .def("is_discrete", &phasic::AnyProbabilityDistributionContext::is_discrete,
      py::return_value_policy::copy, R"delim(
Check if this is a discrete distribution context.

Returns
-------
bool
    True for discrete, False for continuous.
      )delim")

    .def("step", &phasic::AnyProbabilityDistributionContext::step,

      py::return_value_policy::copy, R"delim(
Perform one time step in the distribution.

Advances the internal state for iterative computation.
      )delim")

    .def("pmf", &phasic::AnyProbabilityDistributionContext::pmf,
      py::return_value_policy::copy, R"delim(
Get current probability mass function value (discrete distributions only).

Returns
-------
float
    PMF at current time/jump.
      )delim")

    .def("pdf", &phasic::AnyProbabilityDistributionContext::pdf,
      py::return_value_policy::copy, R"delim(
Get current probability density function value (continuous distributions only).

Returns
-------
float
    PDF at current time.
      )delim")

    .def("cdf", &phasic::AnyProbabilityDistributionContext::cdf,
      py::return_value_policy::copy, R"delim(
Get current cumulative distribution function value.

Returns
-------
float
    CDF at current time/jump.
      )delim")

    .def("time", &phasic::AnyProbabilityDistributionContext::time,
      py::return_value_policy::copy, R"delim(
Get current time (continuous distributions only).

Returns
-------
float
    Current time value.
      )delim")

    .def("jumps", &phasic::AnyProbabilityDistributionContext::jumps,
      py::return_value_policy::copy, R"delim(
Get current number of jumps (discrete distributions only).

Returns
-------
int
    Current jump count.
      )delim")

    .def("stop_probability", &phasic::AnyProbabilityDistributionContext::stop_probability,
      py::return_value_policy::copy, R"delim(
Get stopping probability at each vertex.

Returns
-------
list of float
    Probability of being at each vertex at current time/jump.
      )delim")

    .def("accumulated_visits", &phasic::AnyProbabilityDistributionContext::accumulated_visits,
      py::return_value_policy::copy, R"delim(
Get accumulated visit counts for each vertex.

Returns
-------
list of float
    Expected number of visits to each vertex.
      )delim")
      
    // .def("accumulated_visiting_time", &phasic::AnyProbabilityDistributionContext::accumulated_visiting_time, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")

    .def("accumulated_visiting_time",
      [](phasic::AnyProbabilityDistributionContext &context) {
        // to make it return np.array instead of list without copying data
        auto a = new std::vector<long double>(context.accumulated_visiting_time());
        auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<long double>*>(a); });
        return py::array(a->size(), a->data(), capsule);
      }, py::return_value_policy::copy,
      R"delim(
Get accumulated visiting time for each vertex.

Returns
-------
ndarray
    Expected time spent at each vertex (zero-copy).
      )delim")
  
    ;


  py::class_<phasic::ProbabilityDistributionContext>(m, "ProbabilityDistributionContext", R"delim(
Context for iterative computation of continuous phase-type distributions.

Maintains internal state for stepping through time and computing PDF/CDF values incrementally.
      )delim")

    .def(py::init(&phasic::ProbabilityDistributionContext::init_factory),
      py::return_value_policy::reference_internal, R"delim(
Create a ProbabilityDistributionContext for a graph.

Use Graph.distribution_context(granularity) instead.
        )delim")
        


      
    // .def("__enter__",
    //   [](phasic::ProbabilityDistributionContext &ctx) {

    //     // reset context

    //     return ctx;

    //   }, py::return_value_policy::move, R"delim(

    //   )delim")


    // .def("__exit__",
    //   [](phasic::ProbabilityDistributionContext &ctx) {


    //     // reset context

    //   }, R"delim(

    //   )delim")
      



    .def("step", &phasic::ProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(
      Performs one jump in the probability distribution context for the discrete phase-type distribution.
      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `distribution_context()`.
      See Also
      --------
      distribution_context
      )delim")
      
    .def("pdf", &phasic::ProbabilityDistributionContext::pdf, 
      py::return_value_policy::copy, R"delim(
      Returns the PDF for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `distribution_context()`.

      See Also
      --------
      distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.

      )delim")
      
    .def("cdf", &phasic::ProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(
      Returns the CDF for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `distribution_context()`.

      See Also
      --------
      distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.


      )delim")
      
    .def("time", &phasic::ProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(
      Returns the time for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `distribution_context()`.

      See Also
      --------
      distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.


      )delim");
      

//       distribution_context_stop_probability
// //' Returns the stop probability for the current probability distribution context for the phase-type distribution.
// //' 
// //' @description
// //' This allows the user to step through the distribution, computing e.g. the
// //' time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
// //' *mutates* the context
// //' 
// //' @seealso [phasic::distribution_context()]
// //' 
// //' @param probability_distribution_context The context created by [phasic::distribution_context()]
// //' 
// // [[Rcpp::export]]


// NumericVector distribution_context_accumulated_visiting_time
// //' Returns the accumulated visiting time (integral of stop probability) for the current probability distribution context for the phase-type distribution.
// //' 
// //' @description
// //' This allows the user to step through the distribution, computing e.g. the
// //' time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
// //' *mutates* the context
// //' 
// //' @seealso [phasic::distribution_context()]
// //' 
// //' @param probability_distribution_context The context created by [phasic::distribution_context()]
// //' 
// // [[Rcpp::export]]
      




    // .def("stop_probability", &phasic::ProbabilityDistributionContext::stop_probability, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")
      
    // .def("accumulated_visiting_time", &phasic::ProbabilityDistributionContext::accumulated_visiting_time, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")
    ;


  py::class_<phasic::DPHProbabilityDistributionContext>(m, "DPHProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init(&phasic::DPHProbabilityDistributionContext::init_factory), R"delim(

      )delim")
      
    .def("step", &phasic::DPHProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(
//' Performs one jump in a probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso dph_distribution_context()
//' 
//' @param probability_distribution_context The context created by dph_distribution_context()
//' 
// [[Rcpp::export]]
      )delim")
      
    .def("pmf", &phasic::DPHProbabilityDistributionContext::pmf, 
      py::return_value_policy::copy, R"delim(
      Returns the PMF for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `dph_distribution_context()`.

      See Also
      --------
      dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("cdf", &phasic::DPHProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(
      Returns the CDF for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `dph_distribution_context()`.

      See Also
      --------
      dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("jumps", &phasic::DPHProbabilityDistributionContext::jumps, 
      py::return_value_policy::copy, R"delim(
      Returns the jumps for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `dph_distribution_context()`.

      See Also
      --------
      dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("stop_probability", &phasic::DPHProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(
//' Returns the stop probability for the current probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso dph_distribution_context()
//' 
//' @param probability_distribution_context The context created by dph_distribution_context()
//' 
// [[Rcpp::export]]
      )delim")
      
    .def("accumulated_visits", &phasic::DPHProbabilityDistributionContext::accumulated_visits, 
      py::return_value_policy::copy, R"delim(
//' Returns the accumulated visits for the current probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso dph_distribution_context()
//' 
//' @param probability_distribution_context The context created by dph_distribution_context()
//' 
// [[Rcpp::export]]
      )delim")
    ;

  // ============================================================================
  // FFI Support for User-Defined C++ Models
  // ============================================================================
  // This allows users to load C++ model builders and get back Python Graph objects
  // which can be reused without rebuilding

  m.def("load_cpp_builder", [](const std::string& cpp_file) -> py::object {
      // Compile the C++ file to a shared library
      std::string pkg_dir = std::string(__FILE__);
      size_t pos = pkg_dir.rfind("/src/cpp/");
      if (pos != std::string::npos) {
          pkg_dir = pkg_dir.substr(0, pos);
      }

      // Read the source to generate a hash
      std::ifstream file(cpp_file);
      if (!file.is_open()) {
          throw std::runtime_error("Cannot open file: " + cpp_file);
      }
      std::stringstream buffer;
      buffer << file.rdbuf();
      std::string source_code = buffer.str();
      file.close();

      // Check if it implements build_model
      if (source_code.find("build_model") == std::string::npos) {
          throw std::runtime_error("C++ file must implement: phasic::Graph build_model(const double* theta, int n_params)");
      }

      // Create a hash for caching
      std::hash<std::string> hasher;
      size_t source_hash = hasher(source_code);
      std::string lib_file = "/tmp/ptd_builder_" + std::to_string(source_hash) + ".so";

      // Check if already compiled
      std::ifstream lib_check(lib_file);
      if (!lib_check.good()) {
          // Need to compile
          // Create wrapper that includes the user's code
          std::string wrapper_file = "/tmp/ptd_wrapper_" + std::to_string(source_hash) + ".cpp";
          std::ofstream wrapper(wrapper_file);
          wrapper << "#include \"phasiccpp.h\"\n";
          wrapper << "#include <vector>\n";
          wrapper << "#include \"" << cpp_file << "\"\n\n";
          wrapper << "extern \"C\" {\n";
          wrapper << "    void* build_graph_ffi(const double* theta, int n_params) {\n";
          wrapper << "        phasic::Graph* g = new phasic::Graph(build_model(theta, n_params));\n";
          wrapper << "        g->normalize();\n";
          wrapper << "        return static_cast<void*>(g);\n";
          wrapper << "    }\n";
          wrapper << "    void free_graph_ffi(void* graph) {\n";
          wrapper << "        delete static_cast<phasic::Graph*>(graph);\n";
          wrapper << "    }\n";
          wrapper << "}\n";
          wrapper.close();

          // Compile
          std::string compile_cmd = "g++ -O3 -fPIC -shared -std=c++14 ";
          compile_cmd += "-I" + pkg_dir + " ";
          compile_cmd += "-I" + pkg_dir + "/api/cpp ";
          compile_cmd += "-I" + pkg_dir + "/api/c ";
          compile_cmd += "-I" + pkg_dir + "/include ";
          compile_cmd += wrapper_file + " ";
          compile_cmd += pkg_dir + "/src/cpp/phasiccpp.cpp ";
          compile_cmd += pkg_dir + "/src/c/phasic.c ";
          compile_cmd += "-o " + lib_file + " 2>&1";

          FILE* pipe = popen(compile_cmd.c_str(), "r");
          if (!pipe) {
              std::remove(wrapper_file.c_str());
              throw std::runtime_error("Failed to compile C++ model");
          }

          char compile_buffer[256];
          std::string compile_output;
          while (fgets(compile_buffer, sizeof(compile_buffer), pipe) != nullptr) {
              compile_output += compile_buffer;
          }

          int compile_result = pclose(pipe);
          std::remove(wrapper_file.c_str());

          if (compile_result != 0) {
              throw std::runtime_error("Compilation failed: " + compile_output);
          }
      }

      // Load the library (platform-specific)
#ifdef _WIN32
      HMODULE lib_handle = LoadLibraryA(lib_file.c_str());
      if (!lib_handle) {
          throw std::runtime_error("Failed to load library (Windows error: " + std::to_string(GetLastError()) + ")");
      }

      // Get the build function
      typedef void* (*BuildFunc)(const double*, int);
      BuildFunc build_ffi = (BuildFunc)GetProcAddress(lib_handle, "build_graph_ffi");

      typedef void (*FreeFunc)(void*);
      FreeFunc free_ffi = (FreeFunc)GetProcAddress(lib_handle, "free_graph_ffi");

      if (!build_ffi || !free_ffi) {
          FreeLibrary(lib_handle);
          throw std::runtime_error("Failed to find functions in library");
      }
#else
      void* lib_handle = dlopen(lib_file.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!lib_handle) {
          throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
      }

      // Get the build function
      typedef void* (*BuildFunc)(const double*, int);
      BuildFunc build_ffi = (BuildFunc)dlsym(lib_handle, "build_graph_ffi");

      typedef void (*FreeFunc)(void*);
      FreeFunc free_ffi = (FreeFunc)dlsym(lib_handle, "free_graph_ffi");

      if (!build_ffi || !free_ffi) {
          dlclose(lib_handle);
          throw std::runtime_error("Failed to find functions in library");
      }
#endif

      // Return a Python function that builds and returns Graph objects
      return py::cpp_function([build_ffi, free_ffi](py::array_t<double> theta) -> phasic::Graph {
          auto buf = theta.request();
          double* theta_ptr = static_cast<double*>(buf.ptr);
          int n_params = buf.size;

          // Build the graph
          void* graph_ptr = build_ffi(theta_ptr, n_params);

          // Copy the graph (to avoid memory issues)
          phasic::Graph result = *static_cast<phasic::Graph*>(graph_ptr);

          // Free the original
          free_ffi(graph_ptr);

          return result;
      }, py::return_value_policy::copy);

  }, py::arg("cpp_file"), R"delim(
      Load a C++ model builder from a file and return a Python function that builds Graph objects.

      The C++ file should include "user_model.h" and implement:
      Graph build_model(const float* theta, int n_params);

      Parameters
      ----------
      cpp_file : str
          Path to the C++ file containing the build_model function

      Returns
      -------
      callable
          A Python function that takes parameters and returns a Graph object.
          The returned Graph can be used multiple times without rebuilding.

      Example
      -------
      >>> from phasic.phasic_pybind import load_cpp_builder
      >>> builder = load_cpp_builder("examples/user_models/simple_exponential.cpp")
      >>> graph = builder(np.array([1.0]))  # Build graph with rate=1.0
      >>> pdf1 = graph.pdf(0.5)  # Use graph
      >>> pdf2 = graph.pdf(1.0)  # Reuse same graph, no rebuild!
      >>> graph2 = builder(np.array([2.0]))  # Build new graph with different params
      )delim");

  // ============================================================================
  // Parameterized Graph Module for JAX FFI
  // ============================================================================

  py::module_ param_module = m.def_submodule("parameterized",
      R"delim(
      Parameterized graph utilities for efficient JAX FFI integration.

      This submodule provides the GraphBuilder class which enables efficient
      batch processing of parameterized phase-type distributions. The graph
      structure is parsed once, then rapidly rebuilt with different parameter
      values.
      )delim");

  py::class_<phasic::parameterized::GraphBuilder>(param_module, "GraphBuilder",
      R"delim(
      GraphBuilder: Efficient parameterized graph construction and computation.

      Separates graph structure (topology) from parameters (theta values) for
      efficient batch processing. The structure is parsed once from JSON, then
      graphs can be rapidly built with different theta values.

      Features:
      - Build graphs with different parameters without re-parsing structure
      - Compute PMF/PDF (continuous and discrete modes)
      - Compute distribution moments
      - Combined PMF+moments computation (efficient for SVGD)
      - Automatic GIL release during C++ computation (thread-safe)

      Thread-safety: Each GraphBuilder instance is NOT thread-safe. Create
      separate instances for concurrent access, or use external synchronization.

      Examples
      --------
      >>> import json
      >>> import numpy as np
      >>> from phasic import Graph
      >>> from phasic.phasic_pybind import parameterized
      >>>
      >>> # Build a parameterized graph in Python
      >>> graph = Graph(...)  # Your parameterized graph
      >>> structure_json = json.dumps(graph.serialize())
      >>>
      >>> # Create GraphBuilder
      >>> builder = parameterized.GraphBuilder(structure_json)
      >>>
      >>> # Compute moments for different theta values
      >>> theta1 = np.array([0.5, 0.8])
      >>> moments1 = builder.compute_moments(theta1, nr_moments=2)
      >>>
      >>> theta2 = np.array([0.7, 1.0])
      >>> moments2 = builder.compute_moments(theta2, nr_moments=2)
      >>>
      >>> # Compute PMF (continuous mode)
      >>> times = np.linspace(0.1, 5.0, 50)
      >>> pmf = builder.compute_pmf(theta1, times, discrete=False)
      >>>
      >>> # Combined computation (most efficient for SVGD)
      >>> pmf, moments = builder.compute_pmf_and_moments(
      ...     theta1, times, nr_moments=2, discrete=False
      ... )
      )delim")

      .def(py::init<const std::string&>(),
          py::arg("structure_json"),
          R"delim(
          Construct GraphBuilder from JSON-serialized graph structure.

          Parameters
          ----------
          structure_json : str
              JSON string from Graph.serialize() containing graph structure

          Raises
          ------
          RuntimeError
              If JSON is malformed or required fields are missing
          )delim")

      .def("compute_moments",
          &phasic::parameterized::GraphBuilder::compute_moments,
          py::arg("theta"),
          py::arg("nr_moments"),
          R"delim(
          Compute distribution moments: E[T^k] for k=1,2,...,nr_moments.

          Parameters
          ----------
          theta : numpy.ndarray
              Parameter array, shape (n_params,)
          nr_moments : int
              Number of moments to compute

          Returns
          -------
          numpy.ndarray
              Moments array, shape (nr_moments,)
              Contains [E[T], E[T^2], ..., E[T^nr_moments]]

          Notes
          -----
          GIL is released during C++ computation, enabling true parallelization
          when called from multiple Python threads.

          Examples
          --------
          >>> moments = builder.compute_moments(np.array([0.8]), nr_moments=3)
          >>> mean = moments[0]
          >>> variance = moments[1] - moments[0]**2
          )delim")

      .def("compute_pmf",
          &phasic::parameterized::GraphBuilder::compute_pmf,
          py::arg("theta"),
          py::arg("times"),
          py::arg("discrete") = false,
          py::arg("granularity") = 100,
          R"delim(
          Compute PMF (discrete) or PDF (continuous) values.

          Parameters
          ----------
          theta : numpy.ndarray
              Parameter array, shape (n_params,)
          times : numpy.ndarray
              Time points (continuous) or jump counts (discrete), shape (n_times,)
          discrete : bool, default=False
              If True, compute DPH (discrete phase-type)
              If False, compute PDF (continuous phase-type)
          granularity : int, default=100
              Discretization granularity for PDF computation (ignored for DPH)

          Returns
          -------
          numpy.ndarray
              PMF/PDF values, shape (n_times,)

          Notes
          -----
          GIL is released during computation for true parallelization.

          Examples
          --------
          >>> # Continuous PDF
          >>> times = np.linspace(0.1, 5.0, 50)
          >>> pdf = builder.compute_pmf(theta, times, discrete=False)
          >>>
          >>> # Discrete PMF
          >>> jumps = np.array([1, 2, 3, 4, 5])
          >>> pmf = builder.compute_pmf(theta, jumps, discrete=True)
          )delim")

      .def("compute_pmf_and_moments",
          &phasic::parameterized::GraphBuilder::compute_pmf_and_moments,
          py::arg("theta"),
          py::arg("times"),
          py::arg("nr_moments"),
          py::arg("discrete") = false,
          py::arg("granularity") = 100,
          py::arg("rewards") = py::none(),
          R"delim(
          Compute both PMF and moments efficiently in a single pass.

          More efficient than calling compute_pmf() and compute_moments()
          separately because the graph is built only once.

          Parameters
          ----------
          theta : numpy.ndarray
              Parameter array, shape (n_params,)
          times : numpy.ndarray
              Time points or jump counts, shape (n_times,)
          nr_moments : int
              Number of moments to compute
          discrete : bool, default=False
              If True, use DPH mode; if False, use PDF mode
          granularity : int, default=100
              Discretization granularity for PDF (ignored for DPH)
          rewards : numpy.ndarray or None, default=None
              Optional reward vector (one per vertex). If None, computes standard moments E[T^k].
              If provided, computes reward-transformed moments E[R·T^k].

          Returns
          -------
          tuple of (numpy.ndarray, numpy.ndarray)
              (pmf_values, moments)
              - pmf_values: shape (n_times,)
              - moments: shape (nr_moments,)

          Notes
          -----
          Primary use case: SVGD with moment-based regularization.
          GIL is released during computation.

          Examples
          --------
          >>> pmf, moments = builder.compute_pmf_and_moments(
          ...     theta, times, nr_moments=2, discrete=False
          ... )
          >>> # Use pmf for likelihood, moments for regularization
          )delim")

      .def_property_readonly("param_length",
          &phasic::parameterized::GraphBuilder::param_length,
          "Number of parameters (theta dimensions)")

      .def_property_readonly("vertices_length",
          &phasic::parameterized::GraphBuilder::vertices_length,
          "Number of vertices in graph")

      .def_property_readonly("state_length",
          &phasic::parameterized::GraphBuilder::state_length,
          "Dimension of state vectors");

  // ============================================================================
  // JAX FFI Handler Capsules (Phase 2)
  // ============================================================================

#ifdef HAVE_XLA_FFI
  param_module.def("get_compute_pmf_ffi_capsule", []() -> py::capsule {
      // Create handler on-demand (safe because JAX is already initialized)
      auto* handler = phasic::parameterized::CreateComputePmfHandler();
      return py::capsule(reinterpret_cast<void*>(handler), "xla._CUSTOM_CALL_TARGET");
  }, R"delim(
  Get PyCapsule for JAX FFI compute_pmf handler.

  This capsule can be registered with JAX using jax.ffi.register_ffi_target()
  AFTER JAX is fully initialized. This enables zero-copy, XLA-optimized PDF
  computation that can be parallelized by vmap/pmap for true multi-core usage.

  IMPORTANT: Must be called AFTER JAX initialization. The handler is created
  on-demand when this function is called, avoiding static initialization issues.

  Returns
  -------
  capsule
      PyCapsule containing pointer to XLA FFI handler

  Examples
  --------
  >>> import jax  # Initialize JAX FIRST
  >>> from phasic.phasic_pybind import parameterized
  >>> capsule = parameterized.get_compute_pmf_ffi_capsule()
  >>> jax.ffi.register_ffi_target("ptd_compute_pmf", capsule, platform="cpu")
  )delim");

  param_module.def("get_compute_moments_ffi_capsule", []() -> py::capsule {
      // Create handler on-demand (safe because JAX is already initialized)
      auto* handler = phasic::parameterized::CreateComputeMomentsHandler();
      return py::capsule(reinterpret_cast<void*>(handler), "xla._CUSTOM_CALL_TARGET");
  }, R"delim(
  Get PyCapsule for JAX FFI compute_moments handler.

  Returns
  -------
  capsule
      PyCapsule containing pointer to XLA FFI handler
  )delim");

  param_module.def("get_compute_pmf_and_moments_ffi_capsule", []() -> py::capsule {
      // Create handler on-demand (safe because JAX is already initialized)
      auto* handler = phasic::parameterized::CreateComputePmfAndMomentsHandler();
      return py::capsule(reinterpret_cast<void*>(handler), "xla._CUSTOM_CALL_TARGET");
  }, R"delim(
  Get PyCapsule for JAX FFI compute_pmf_and_moments handler.

  Returns
  -------
  capsule
      PyCapsule containing pointer to XLA FFI handler
  )delim");
#endif

  // ============================================================================
  // Graph Content Hashing for Symbolic DAG Cache
  // ============================================================================

  py::module_ hash_module = m.def_submodule("hash",
      R"delim(
      Graph content hashing utilities for symbolic DAG caching.

      Provides content-addressed hashing of phase-type distribution graphs
      to enable efficient cache lookup for previously computed symbolic DAGs.
      The hash is based on graph structure and parameterization pattern,
      independent of actual parameter values.
      )delim");

  // HashResult class - wraps C struct ptd_hash_result
  // Use shared_ptr as holder since compute_graph_hash returns shared_ptr
  py::class_<struct ptd_hash_result, std::shared_ptr<struct ptd_hash_result>>(hash_module, "HashResult",
      R"delim(
      Hash result containing multiple representations of graph content hash.

      Attributes
      ----------
      hash64 : int
          64-bit hash value (fast comparison)
      hash_hex : str
          Full 256-bit hash as hexadecimal string (collision-resistant)
      )delim")

      .def_property_readonly("hash64",
          [](const struct ptd_hash_result* h) { return h->hash64; },
          "64-bit hash value for fast comparison")

      .def_property_readonly("hash_hex",
          [](const struct ptd_hash_result* h) { return std::string(h->hash_hex); },
          "Full SHA-256 hash as hexadecimal string")

      .def("__eq__",
          [](const struct ptd_hash_result* self, const struct ptd_hash_result* other) {
              return ptd_hash_equal(self, other);
          },
          py::arg("other"),
          "Compare two hash results for equality")

      .def("__str__",
          [](const struct ptd_hash_result* h) { return std::string(h->hash_hex); },
          "String representation (hex hash)")

      .def("__repr__",
          [](const struct ptd_hash_result* h) {
              std::ostringstream oss;
              oss << "HashResult('" << h->hash_hex << "')";
              return oss.str();
          },
          "String representation with type")

      .def("__hash__",
          [](const struct ptd_hash_result* h) { return static_cast<py::ssize_t>(h->hash64); },
          "Python hash value for use in dict/set");

  // Graph content hashing function
  hash_module.def("compute_graph_hash",
      [](phasic::Graph& graph) {
          struct ptd_hash_result* hash = ptd_graph_content_hash(graph.c_graph());
          if (hash == NULL) {
              throw std::runtime_error("Failed to compute graph hash");
          }
          // Wrap in shared_ptr for automatic cleanup
          return std::shared_ptr<struct ptd_hash_result>(hash, ptd_hash_destroy);
      },
      py::arg("graph"),
      R"delim(
      Compute content hash of a graph structure.

      The hash is based solely on graph topology, state vectors, edge types,
      and parameterization patterns. It is independent of actual edge weights
      (parameter values), making it suitable for caching symbolic DAGs.

      Parameters
      ----------
      graph : Graph
          Phase-type distribution graph to hash

      Returns
      -------
      HashResult
          Hash result with multiple representations

      Examples
      --------
      >>> g1 = phasic.Graph(1)
      >>> # ... build graph ...
      >>> hash1 = phasic.hash.compute_graph_hash(g1)
      >>> print(hash1.hash_hex)
      'a3f2e9c8...'
      >>>
      >>> # Build identical structure with different weights
      >>> g2 = phasic.Graph(1)
      >>> # ... build same structure ...
      >>> hash2 = phasic.hash.compute_graph_hash(g2)
      >>> assert hash1 == hash2  # Same structure = same hash
      )delim");

  // Hash from hex string
  hash_module.def("hash_from_hex",
      [](const std::string& hex_str) {
          struct ptd_hash_result* hash = ptd_hash_from_hex(hex_str.c_str());
          if (hash == NULL) {
              throw std::invalid_argument("Invalid hex string (must be 64 characters)");
          }
          return std::shared_ptr<struct ptd_hash_result>(hash, ptd_hash_destroy);
      },
      py::arg("hex_str"),
      R"delim(
      Create hash result from hexadecimal string.

      Useful for loading cached symbolic DAGs by hash key.

      Parameters
      ----------
      hex_str : str
          64-character hexadecimal hash string

      Returns
      -------
      HashResult
          Hash result object

      Examples
      --------
      >>> hash = phasic.hash.hash_from_hex('a3f2e9c8...')
      >>> print(hash.hash64)
      )delim");

  // Hash parameterized edge (for testing)
  hash_module.def("hash_parameterized_edge",
      [](const phasic::ParameterizedEdge& edge, size_t state_length) {
          // Note: This is a simplified wrapper - actual implementation would need
          // access to underlying C edge structure
          // For now, return 0 as placeholder
          return static_cast<uint64_t>(0);
      },
      py::arg("edge"),
      py::arg("state_length"),
      "Hash individual parameterized edge (for testing)");

  // ============================================================================
  // C Logging Bridge
  // ============================================================================

  // Internal functions for C logging (not exposed to Python directly)
  m.def("_c_log_set_callback",
      [](py::function callback) {
          // Store callback in a way that survives but can be safely ignored during shutdown
          // We intentionally leak this memory to avoid shutdown crashes
          static py::function* py_callback = new py::function(callback);
          *py_callback = callback;  // Update existing callback

          // Set C callback that forwards to Python
          ptd_set_log_callback([](ptd_log_level_t level, const char* message) {
              // Check if Python is still alive before attempting callback
              if (Py_IsInitialized()) {
                  try {
                      // Acquire GIL for Python call
                      py::gil_scoped_acquire acquire;

                      // Map C level to Python level
                      int py_level;
                      switch (level) {
                          case PTD_LOG_DEBUG: py_level = 10; break;
                          case PTD_LOG_INFO: py_level = 20; break;
                          case PTD_LOG_WARNING: py_level = 30; break;
                          case PTD_LOG_ERROR: py_level = 40; break;
                          case PTD_LOG_CRITICAL: py_level = 50; break;
                          default: py_level = 30; break;  // Default to WARNING
                      }

                      // Call Python callback (dereference pointer)
                      (*py_callback)(py_level, message);
                  } catch (py::error_already_set &e) {
                      // Error in Python callback - print to stderr and continue
                      std::cerr << "Error in C logging callback: " << e.what() << std::endl;
                  } catch (...) {
                      // Catch any other exceptions
                      // Silently ignore to avoid crash
                  }
              }
          });
      },
      py::arg("callback"),
      R"delim(
      Set callback function for C logging (internal use only).

      This function is called automatically during module initialization
      to connect C logging to the Python logging system.
      )delim");

  m.def("_c_log_set_level",
      [](int level) {
          // Map Python level to C level
          ptd_log_level_t c_level;
          if (level <= 10) c_level = PTD_LOG_DEBUG;
          else if (level <= 20) c_level = PTD_LOG_INFO;
          else if (level <= 30) c_level = PTD_LOG_WARNING;
          else if (level <= 40) c_level = PTD_LOG_ERROR;
          else c_level = PTD_LOG_CRITICAL;

          ptd_set_log_level(c_level);
      },
      py::arg("level"),
      "Set C logging level (internal use only)");

  m.def("_c_log_get_level",
      []() {
          ptd_log_level_t c_level = ptd_get_log_level();
          return static_cast<int>(c_level);
      },
      "Get current C logging level (internal use only)");

}
