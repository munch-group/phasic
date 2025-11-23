# Markov Cache Library - Quick Start Guide

## Download and Extract

```bash
# Extract the library
tar -xzf markov_cache.tar.gz
cd markov_cache
```

## Install Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y cmake gcc libzstd-dev libsqlite3-dev libssl-dev pkg-config
```

### macOS
```bash
brew install cmake zstd sqlite3 openssl pkg-config
```

## Build in 30 Seconds

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Run the Example

```bash
./examples/example_basic
```

You should see output demonstrating:
- ✓ Cache storage and retrieval
- ✓ SCC decomposition
- ✓ Graph hashing
- ✓ Compression with dictionary training
- ✓ Multiple graphs with cache reuse

## Run Again to See Cache Hits

```bash
./examples/example_basic
```

The second run will show 100% cache hit rate!

## File Structure

After building, you'll have:

```
markov_cache/
├── include/markov_cache/
│   └── markov_cache.h          # Public API
├── src/                        # Library source
├── examples/
│   ├── basic_example.c         # Comprehensive example
│   └── gaussian_elimination_stub.c  # Your implementation goes here
├── build/
│   ├── libmarkov_cache.so      # Shared library
│   └── examples/example_basic   # Compiled example
└── test_cache/                 # Created by example (persistent cache)
```

## Key Files to Modify for Your Use

1. **examples/gaussian_elimination_stub.c**
   - Replace stub with your actual phase-type Gaussian elimination
   - This is the only required customization!

2. **Your application code**
   - Include: `#include <markov_cache/markov_cache.h>`
   - Link: `-lmarkov_cache -lzstd -lsqlite3 -lcrypto -lm`

## Minimal Example

```c
#include <markov_cache/markov_cache.h>

// Implement your Gaussian elimination
DAGResult* gaussian_elimination(const MarkovGraph *graph) {
    // Your algorithm here
}

int main() {
    // Initialize cache (100MB hot cache)
    PersistentCache *cache = cache_init("./my_cache", 100*1024*1024);
    
    // Create graph
    MarkovGraph *g = markov_graph_create(100);
    markov_graph_add_edge(g, 0, 1, 1.5);
    // ... add more edges
    
    // Solve with caching
    DAGResult *result = solve_with_cache(cache, g, 0);
    
    // Use result
    printf("Nodes: %u, Edges: %u\n", result->num_nodes, result->num_edges);
    
    // Cleanup
    dag_result_free(result);
    markov_graph_free(g);
    cache_close(cache);
    return 0;
}
```

Compile:
```bash
gcc -o myapp myapp.c -I../include -L. -lmarkov_cache -lzstd -lsqlite3 -lcrypto -lm
```

## Next Steps

1. Read `README.md` for full API documentation
2. Implement your `gaussian_elimination()` function
3. Adapt the example to your specific phase-type distributions
4. Benchmark your cache hit rates
5. Train compression dictionary on your graphs for better compression

## Troubleshooting

**Missing dependencies?**
```bash
# Check what's available
pkg-config --modversion libzstd sqlite3
openssl version
```

**Build errors?**
```bash
# Try verbose build
make VERBOSE=1
```

**Segfault or crashes?**
- Make sure `gaussian_elimination()` returns a valid DAGResult
- Check that all graph node IDs are < num_nodes
- Verify edges don't reference invalid nodes

## Performance Tips

1. **Set appropriate hot cache size** (default: 10MB)
   - More = fewer disk reads
   - Adjust based on available RAM

2. **Train compression dictionary** on representative graphs
   - Improves compression 20-30%
   - See Example 6 in basic_example.c

3. **Chunk size tuning**
   - Smaller chunks = more reuse opportunity
   - Larger chunks = less overhead
   - Try different max_chunk_size values

4. **Monitor cache stats**
   ```c
   cache_print_stats(cache);
   ```

## Success Indicators

You're ready to use the library when:
- ✓ Example compiles and runs
- ✓ You see cache statistics printed
- ✓ `test_cache/` directory is created
- ✓ Second run shows cache hits
- ✓ Your gaussian_elimination() works on test graphs

Enjoy efficient Gaussian elimination with persistent caching!
