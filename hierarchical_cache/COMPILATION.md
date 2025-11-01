# Compilation Instructions

## Step-by-Step Build Process

### 1. Extract the Archive

```bash
tar -xzf markov_cache.tar.gz
cd markov_cache
```

### 2. Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    gcc \
    libzstd-dev \
    libsqlite3-dev \
    libssl-dev \
    pkg-config
```

#### macOS (Homebrew)
```bash
brew install cmake zstd sqlite3 openssl pkg-config
```

#### Fedora/RHEL
```bash
sudo dnf install -y \
    cmake \
    gcc \
    zstd-devel \
    sqlite-devel \
    openssl-devel \
    pkgconfig
```

### 3. Build Library and Examples

#### Option A: Standard Build
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

#### Option B: Release Build (Optimized)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

#### Option C: Debug Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### 4. Run the Example

```bash
# From the build directory
./examples/example_basic

# Run again to see cache hits
./examples/example_basic
```

### 5. Install Library (Optional)

```bash
# System-wide installation
sudo make install

# Custom prefix
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
```

## Build Outputs

After building, you'll have:

```
build/
├── libmarkov_cache.so          # Shared library (Linux)
├── libmarkov_cache.dylib       # Shared library (macOS)
├── libmarkov_cache.a           # Static library (if built)
└── examples/
    └── example_basic           # Compiled example
```

## Using the Library in Your Project

### Method 1: With Installed Library

After `sudo make install`:

```bash
gcc -o myapp myapp.c -lmarkov_cache -lzstd -lsqlite3 -lcrypto -lm
```

### Method 2: From Build Directory

```bash
gcc -o myapp myapp.c \
    -I/path/to/markov_cache/include \
    -L/path/to/markov_cache/build \
    -lmarkov_cache \
    -lzstd -lsqlite3 -lcrypto -lm \
    -Wl,-rpath,/path/to/markov_cache/build
```

### Method 3: CMake Project

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyApp C)

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(SQLITE3 REQUIRED IMPORTED_TARGET sqlite3)
find_package(OpenSSL REQUIRED)

# If markov_cache is installed
find_library(MARKOV_CACHE_LIB markov_cache)
find_path(MARKOV_CACHE_INCLUDE markov_cache/markov_cache.h)

add_executable(myapp myapp.c gaussian_elimination.c)

target_include_directories(myapp PRIVATE ${MARKOV_CACHE_INCLUDE})

target_link_libraries(myapp
    ${MARKOV_CACHE_LIB}
    PkgConfig::ZSTD
    PkgConfig::SQLITE3
    OpenSSL::Crypto
    m
)
```

Build:
```bash
mkdir build && cd build
cmake ..
make
```

### Method 4: Add as Subdirectory

If markov_cache is part of your source tree:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyApp C)

# Add markov_cache as subdirectory
add_subdirectory(markov_cache)

# Your app
add_executable(myapp myapp.c gaussian_elimination.c)
target_link_libraries(myapp markov_cache)
```

## Compiler Flags Explained

### Required Flags

- `-lmarkov_cache`: Link the Markov cache library
- `-lzstd`: Link Zstandard compression library
- `-lsqlite3`: Link SQLite3 database library
- `-lcrypto`: Link OpenSSL crypto library (for SHA-256)
- `-lm`: Link math library (for some operations)

### Include Path

- `-I/path/to/include`: Tell compiler where to find `markov_cache.h`

### Library Path

- `-L/path/to/lib`: Tell linker where to find `libmarkov_cache.so`

### Runtime Path (Linux)

- `-Wl,-rpath,/path/to/lib`: Tell runtime where to find shared library

## Minimal Example Compilation

### Create `myapp.c`

```c
#include <markov_cache/markov_cache.h>
#include <stdio.h>

DAGResult* gaussian_elimination(const MarkovGraph *graph) {
    DAGResult *dag = dag_result_create(graph->num_nodes);
    dag->num_edges = 0;
    return dag;
}

int main() {
    PersistentCache *cache = cache_init("./cache", 10*1024*1024);
    
    MarkovGraph *g = markov_graph_create(10);
    markov_graph_add_edge(g, 0, 1, 1.0);
    markov_graph_add_edge(g, 1, 2, 2.0);
    
    DAGResult *result = solve_with_cache(cache, g, 0);
    
    printf("Result: %u nodes, %u edges\n", 
           result->num_nodes, result->num_edges);
    
    dag_result_free(result);
    markov_graph_free(g);
    cache_close(cache);
    
    return 0;
}
```

### Compile and Run

```bash
# If library is installed
gcc -o myapp myapp.c -lmarkov_cache -lzstd -lsqlite3 -lcrypto -lm
./myapp

# If using from build directory
gcc -o myapp myapp.c \
    -I../include \
    -L. \
    -lmarkov_cache \
    -lzstd -lsqlite3 -lcrypto -lm \
    -Wl,-rpath,.
./myapp
```

## Troubleshooting

### Problem: Cannot find -lzstd

**Solution:**
```bash
sudo apt-get install libzstd-dev
# or
brew install zstd
```

### Problem: Cannot find -lsqlite3

**Solution:**
```bash
sudo apt-get install libsqlite3-dev
# or
brew install sqlite3
```

### Problem: Cannot find -lcrypto

**Solution:**
```bash
sudo apt-get install libssl-dev
# or
brew install openssl
```

### Problem: error while loading shared libraries: libmarkov_cache.so

**Solution 1:** Install the library
```bash
cd build
sudo make install
sudo ldconfig  # Linux only
```

**Solution 2:** Use rpath
```bash
gcc -o myapp myapp.c ... -Wl,-rpath,/path/to/markov_cache/build
```

**Solution 3:** Set LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/path/to/markov_cache/build:$LD_LIBRARY_PATH
./myapp
```

### Problem: undefined reference to `gaussian_elimination`

**Solution:** Implement the function in your code
```c
DAGResult* gaussian_elimination(const MarkovGraph *graph) {
    // Your implementation
}
```

### Problem: CMake can't find dependencies

**Solution:** Install pkg-config
```bash
sudo apt-get install pkg-config
```

## Advanced Build Options

### Static Library

```bash
cmake -DBUILD_SHARED_LIBS=OFF ..
make
# Produces libmarkov_cache.a
```

### Without Examples

```bash
cmake -DBUILD_EXAMPLES=OFF ..
make
```

### Debug Symbols

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
# Run with gdb or valgrind
```

### Verbose Build

```bash
make VERBOSE=1
```

### Clean Build

```bash
cd build
rm -rf *
cmake ..
make
```

## Cross-Compilation

### For ARM (Raspberry Pi)

```bash
cmake -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=arm ..
make
```

## Build Time

Typical build times:
- Clean build: ~10 seconds
- Incremental: ~2 seconds
- With examples: +5 seconds

## Disk Space

- Source: 100 KB
- Build artifacts: 500 KB
- Installed: 200 KB

## Verification

After building, verify:

```bash
# Check library
ls -lh libmarkov_cache.so

# Check symbols
nm libmarkov_cache.so | grep cache_init

# Run example
./examples/example_basic

# Check example exit code
echo $?  # Should be 0
```

## Next Steps

1. ✓ Build successful
2. ✓ Example runs
3. → Implement your `gaussian_elimination()`
4. → Integrate into your application
5. → Run benchmarks
6. → Optimize cache parameters

Success! You're ready to use the Markov Cache Library.
