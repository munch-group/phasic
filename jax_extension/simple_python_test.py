"""
Simple test without JAX integration
"""

import ctypes
import subprocess
import tempfile
import os

print("Testing basic compilation and loading...")

# Test basic C++ compilation
simple_cpp = '''
#include "user_graph_api.h"

extern "C" {
Graph build_simple_graph(const double* theta, int theta_size, const UserConfig& config) {
    Graph graph;
    std::vector<int> state = {1};
    int v = graph.add_vertex(state);
    graph.set_absorption_rate(v, 1.0);
    return graph;
}

__attribute__((constructor))
void register_simple() {
    GraphBuilderRegistry::register_builder("simple", build_simple_graph);
}
}
'''

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
    f.write(simple_cpp)
    cpp_file = f.name

lib_path = "/tmp/simple_test.so"

try:
    # Compile
    cmd = [
        'g++', '-shared', '-fPIC', '-std=c++17',
        '-I', '/Users/kmt/PtDalgorithms/jax_extension',
        cpp_file, 'user_graph_api.o',
        '-o', lib_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True,
                          cwd='/Users/kmt/PtDalgorithms/jax_extension')
    
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        exit(1)
    
    print("✓ Compilation successful")
    
    # Load library
    lib = ctypes.CDLL(lib_path)
    print("✓ Library loaded")
    
    # Test if constructor ran (registry should have the builder)
    from ctypes import c_char_p, c_bool
    
    # We need to access the C++ registry somehow
    # For now, just check that loading doesn't crash
    print("✓ Constructor executed without crash")
    
finally:
    os.unlink(cpp_file)
    if os.path.exists(lib_path):
        os.unlink(lib_path)

print("Basic compilation test passed!")

# Now test the main separated_graph_pmf.so
print("\nTesting main library...")

try:
    main_lib = ctypes.CDLL("/Users/kmt/PtDalgorithms/jax_extension/separated_graph_pmf.so")
    print("✓ Main library loaded")
    
    func = main_lib.jax_separated_graph_pmf
    print("✓ Main function accessible")
    
except Exception as e:
    print(f"✗ Main library test failed: {e}")

print("All tests completed!")