#include <chrono>
#include <cstdio>
#include <cstdlib>
extern "C" void* build_graph_ffi(const double*, int);
extern "C" void free_graph_ffi(void*);
int main(int argc, char** argv){
  double th[2] = {1.0, atof(argv[1])};
  for (int r=0;r<3;r++){
    auto t0=std::chrono::steady_clock::now();
    void* g = build_graph_ffi(th,2);
    auto t1=std::chrono::steady_clock::now();
    free_graph_ffi(g);
    auto t2=std::chrono::steady_clock::now();
    printf("N=%s build+normalize_s=%.4f free_s=%.4f\n", argv[1], std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t2-t1).count());
  }
}
