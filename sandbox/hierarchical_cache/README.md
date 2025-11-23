# Markov Cache Library

A high-performance persistent cache for Gaussian elimination results on Markov jump processes (phase-type distributions). The library provides hierarchical caching with automatic graph decomposition, content-addressable storage, and efficient compression.

## Features

- **Persistent Caching**: Cache Gaussian elimination results across program runs
- **Content-Addressable Storage**: Automatic deduplication of identical subgraphs
- **Hierarchical Decomposition**: Break large graphs into reusable chunks using SCC decomposition
- **Efficient Compression**: Zstandard compression with optional dictionary training
- **Space-Optimized Serialization**: CSR format with delta encoding
- **LRU Hot Cache**: In-memory cache for frequently accessed chunks
- **DAG Verification**: Ensures chunk dependencies form a DAG (no cycles)

## Dependencies

- **CMake** >= 3.15
- **C Compiler** with C11 support (GCC, Clang)
- **Zstandard** (libzstd) - Compression library
- **SQLite3** - Index database
- **OpenSSL** - SHA-256 hashing

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake gcc libzstd-dev libsqlite3-dev libssl-dev pkg-config
```

#### macOS (Homebrew)
```bash
brew install cmake zstd sqlite3 openssl pkg-config
```

#### Fedora/RHEL
```bash
sudo dnf install cmake gcc zstd-devel sqlite-devel openssl-devel pkgconfig
```

## Building

### Quick Build

```bash
# Clone or extract the library
cd markov_cache

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Build Options

```bash
# Build with examples (default: ON)
cmake -DBUILD_EXAMPLES=ON ..

# Build shared library (default: ON)
cmake -DBUILD_SHARED_LIBS=ON ..

# Build with tests (default: ON)
cmake -DBUILD_TESTS=ON ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Custom Installation Prefix

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

## Running the Example

After building:

```bash
# From the build directory
./examples/example_basic

# Or if installed:
example_basic
```

The example demonstrates:
1. Basic cache usage
2. SCC decomposition
3. Cache reuse across multiple graphs
4. CSR format conversion
5. Graph hashing
6. Dictionary training
7. Direct cache queries

## Library Usage

### Basic Example

```c
#include <markov_cache/markov_cache.h>

// 1. Initialize cache
PersistentCache *cache = cache_init("./cache_dir", 100 * 1024 * 1024); // 100MB

// 2. Create your Markov graph
MarkovGraph *graph = markov_graph_create(100);
markov_graph_add_edge(graph, 0, 1, 1.5);
// ... add more edges

// 3. Implement Gaussian elimination (required)
DAGResult* gaussian_elimination(const MarkovGraph *graph) {
    // Your implementation here
    // Must convert cyclic graph to DAG and compute moments
}

// 4. Solve with caching
DAGResult *result = solve_with_cache(cache, graph, 0);

// 5. Use results
printf("Computed %u nodes, %u edges\n", result->num_nodes, result->num_edges);

// 6. Cleanup
dag_result_free(result);
markov_graph_free(graph);
cache_close(cache);
```

### Implementing Gaussian Elimination

You **must** implement the `gaussian_elimination()` function for your specific phase-type distribution algorithm:

```c
DAGResult* gaussian_elimination(const MarkovGraph *graph) {
    // Your algorithm here:
    // 1. Identify and eliminate cycles
    // 2. Compute transition rates for DAG
    // 3. Calculate moments using dynamic programming
    
    DAGResult *dag = dag_result_create(graph->num_nodes);
    
    // Fill in DAG edges and weights
    // ...
    
    // Compute moments
    dag->num_moments = 2;  // First and second moments
    dag->moments = malloc(graph->num_nodes * dag->num_moments * sizeof(double));
    // ... compute moments
    
    return dag;
}
```

### Advanced Usage

#### Query Cache for Specific Graph

```c
CacheQueryResult *query = cache_query(cache, graph);

printf("Found %u cached chunks\n", query->num_cached);
printf("Need to compute %u chunks\n", query->num_uncached);

// Process uncached chunks
for (uint32_t i = 0; i < query->num_uncached; i++) {
    DAGResult *result = gaussian_elimination(query->uncached_subgraphs[i]);
    cache_store(cache, query->uncached_subgraphs[i], result, NULL);
    dag_result_free(result);
}

cache_query_result_free(query);
```

#### Train Compression Dictionary

```c
// Create sample graphs for training
MarkovGraph **samples = malloc(100 * sizeof(MarkovGraph*));
for (int i = 0; i < 100; i++) {
    samples[i] = create_sample_graph();
}

// Train dictionary (improves compression ratio)
cache_train_dictionary(cache, samples, 100);

// Cleanup samples
for (int i = 0; i < 100; i++) {
    markov_graph_free(samples[i]);
}
free(samples);
```

#### Check Cache Statistics

```c
uint64_t hits, misses, total_chunks;
cache_get_stats(cache, &hits, &misses, &total_chunks);

printf("Cache hit rate: %.2f%%\n", 
       (double)hits / (hits + misses) * 100.0);

// Or use convenience function
cache_print_stats(cache);
```

## API Reference

### Core Types

- **`MarkovGraph`**: Original cyclic Markov jump process
- **`DAGResult`**: Result after Gaussian elimination (acyclic)
- **`CSRGraph`**: Compressed Sparse Row format (internal)
- **`PersistentCache`**: Cache handle
- **`CacheQueryResult`**: Result of cache query

### Main Functions

#### Cache Management

```c
PersistentCache* cache_init(const char *cache_dir, size_t hot_cache_size);
void cache_close(PersistentCache *cache);
bool cache_store(PersistentCache *cache, const MarkovGraph *orig, 
                 const DAGResult *dag, const ChunkMetadata *metadata);
ChunkData* cache_load(PersistentCache *cache, const uint8_t *orig_hash, 
                      const uint8_t *dag_hash);
CacheQueryResult* cache_query(PersistentCache *cache, const MarkovGraph *graph);
void cache_get_stats(PersistentCache *cache, uint64_t *hits, uint64_t *misses, 
                     uint64_t *total_chunks);
bool cache_train_dictionary(PersistentCache *cache, MarkovGraph **samples, 
                            uint32_t num_samples);
```

#### High-Level API

```c
DAGResult* solve_with_cache(PersistentCache *cache, const MarkovGraph *graph, 
                            uint32_t max_chunk_size);
```

#### Graph Operations

```c
MarkovGraph* markov_graph_create(uint32_t num_nodes);
void markov_graph_add_edge(MarkovGraph *g, uint32_t src, uint32_t dst, double rate);
void markov_graph_free(MarkovGraph *g);
DAGResult* dag_result_create(uint32_t num_nodes);
void dag_result_free(DAGResult *g);
void compute_graph_hash(const MarkovGraph *g, uint8_t *hash);
bool is_dag(const MarkovGraph *g);
```

#### Graph Decomposition

```c
SCCDecomposition* find_sccs(const MarkovGraph *g);
void scc_free(SCCDecomposition *scc);
MarkovGraph* extract_subgraph(const MarkovGraph *g, const uint32_t *node_ids, 
                              uint32_t num_nodes);
```

## Performance

### Space Efficiency

The library achieves significant space savings through:

1. **CSR Format**: 50-70% reduction for sparse graphs
2. **Delta Encoding**: Additional 30-50% compression
3. **Zstandard Compression**: 60-80% compression ratio
4. **Dictionary Training**: Additional 20-30% improvement

**Overall**: Typically 10-20x compression vs. naive serialization

### Example

- Naive: 1M edges × 24 bytes = 24 MB
- Optimized: 1M edges × 6 bytes = 6 MB  
- Compressed: ~1-2 MB with Zstandard

### Speed

- **Cache Hit**: ~2-3x slower than in-memory (due to decompression)
- **Cache Miss**: Full Gaussian elimination required
- **Benefit**: Amortized savings grow with graph reuse

## Cache Directory Structure

```
cache_dir/
├── index.db          # SQLite index for fast lookup
├── dict.zstd         # Compression dictionary (optional)
└── chunks/
    ├── ab/
    │   └── cd/
    │       └── abcd1234...zst  # Compressed chunk data
    └── ...
```

## Advanced Topics

### Content-Addressable Storage

Chunks are stored by their SHA-256 hash, enabling:
- Automatic deduplication
- Integrity verification
- Content-based cache sharing

### Hierarchical Chunking

Large graphs are recursively decomposed:
1. Find strongly connected components (SCCs)
2. Create chunk for each SCC
3. Recursively decompose large SCCs
4. Cache all levels

### DAG Guarantee

The library ensures chunk dependencies form a DAG through:
- Topological level assignment
- Dependency validation
- SCC condensation

## Troubleshooting

### Build Errors

**Missing zstd**:
```bash
sudo apt-get install libzstd-dev
```

**Missing SQLite3**:
```bash
sudo apt-get install libsqlite3-dev
```

**Missing OpenSSL**:
```bash
sudo apt-get install libssl-dev
```

### Runtime Issues

**Cache directory permissions**:
```bash
chmod -R 755 ./cache_dir
```

**Disk space**:
Check available space in cache directory. The library will fail gracefully if disk is full.

## License

This library is provided as-is for research and educational purposes.

## Contributing

To extend the library:

1. Implement your Gaussian elimination algorithm
2. Add custom decomposition strategies
3. Optimize for your specific graph structures
4. Add benchmarks and tests

## Citation

If you use this library in your research, please cite appropriately.

## Support

For questions and issues, please refer to the example code and API documentation.
