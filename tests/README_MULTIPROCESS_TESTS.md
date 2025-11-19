# Multi-Process and Multi-Node Testing

This directory contains test suites for verifying phasic's multi-process JAX compatibility, including both local multi-CPU and SLURM cluster multi-node scenarios.

## Test Files

### 1. `test_multi_process_ffi.py` - Local Multi-CPU Tests

**Purpose**: Verify FFI functions work correctly with multiple CPU devices on a single machine.

**What it tests**:
- Serialization safety (no closures in call paths)
- Thread-local caching works independently per process
- vmap batching across multiple CPU devices
- JIT + vmap combinations
- Numerical consistency across processes

**Run locally**:
```bash
# Uses 8 CPUs (auto-detected by phasic)
python -m pytest tests/test_multi_process_ffi.py -v

# Or specify CPU count
PTDALG_CPUS=4 python -m pytest tests/test_multi_process_ffi.py -v
```

**Expected output**:
```
tests/test_multi_process_ffi.py::TestComputePmfFfiMultiProcess::test_basic_pmf_serialization PASSED
tests/test_multi_process_ffi.py::TestComputePmfFfiMultiProcess::test_pmf_vmap_batching PASSED
...
============================== 11 passed in 3.53s ===============================
```

---

### 2. `test_slurm_multinode_ffi.py` - SLURM Multi-Node Tests

**Purpose**: Verify FFI functions work correctly across multiple nodes with distributed memory.

**What it tests**:
- SLURM environment detection
- JAX distributed initialization (jax.distributed.initialize())
- FFI serialization across node boundaries
- Independent thread-local caching per node
- Result consistency across all nodes
- Multi-node vmap and JIT combinations

**Run on SLURM cluster**:
```bash
# Submit batch job
sbatch tests/slurm_test_ffi.sh

# Monitor job
squeue -u $USER

# Check results
tail -f slurm_ffi_test_<JOBID>.log
```

**Manual run**:
```bash
# Run on 4 nodes with 8 CPUs each
srun --nodes=4 --ntasks=4 --cpus-per-task=8 \\
     python -m pytest tests/test_slurm_multinode_ffi.py -v -s
```

**Tests are SKIPPED if**:
- Not running in SLURM environment (missing `SLURM_PROCID`)
- This allows the test file to be imported locally without errors

---

### 3. `slurm_test_ffi.sh` - SLURM Batch Script

**Purpose**: Automated batch job for running multi-node tests.

**Configuration**:
```bash
#SBATCH --nodes=4           # Number of nodes
#SBATCH --ntasks=4          # One task per node
#SBATCH --cpus-per-task=8   # CPUs per node
#SBATCH --time=00:30:00     # 30 minute time limit
```

**Customize for your cluster**:
1. Edit `#SBATCH` directives for your cluster's requirements
2. Add `module load` commands if needed
3. Activate conda/virtual environment if needed
4. Adjust `XLA_FLAGS` for your CPU count

---

## Architecture & Design

### Multi-Process Safety

All FFI functions (`compute_pmf_ffi`, `compute_moments_ffi`, `compute_pmf_and_moments_ffi`) are designed for multi-process execution:

1. **Pure functions** - No closures, all data passed explicitly
2. **Thread-local caching** - C++ GraphBuilder cache per thread/process
3. **Serialization-safe** - JSON passed as static FFI attributes
4. **OpenMP parallelization** - Batch operations parallelized within each process

### JAX Distributed Pattern

```python
# Automatic initialization (via phasic auto_parallel.py)
from phasic import Graph  # Detects SLURM, calls jax.distributed.initialize()

# FFI functions work transparently
structure_json = graph.serialize()
moments = compute_moments_ffi(structure_json, theta, nr_moments)

# vmap distributes across all devices (local + remote)
moments_batch = jax.vmap(lambda t: compute_moments_ffi(...))(theta_batch)
```

### Thread-Local Caching

Each process maintains its own GraphBuilder cache:

```cpp
// C++: src/cpp/parameterized/graph_builder_ffi.cpp
thread_local std::unordered_map<std::string, std::shared_ptr<GraphBuilder>> builder_cache;
```

**Benefits**:
- No cross-process sharing → no locks needed
- Each node can cache independently
- Automatic cleanup when thread/process exits
- Safe for multi-controller JAX

---

## Expected Test Results

### Local Tests (test_multi_process_ffi.py)

```
JAX Device Info:
  Global devices: 8
  Local devices: 8
  Devices: [CpuDevice(id=0), CpuDevice(id=1), ...]

✅ 11/11 tests PASSED
```

### SLURM Tests (test_slurm_multinode_ffi.py)

**On 4 nodes with 8 CPUs each**:

```
Process 0/4:
  Node: node001
  CPUs per task: 8
  JAX Total processes: 4
  JAX Global devices: 32 (8 per node × 4 nodes)
  JAX Local devices: 8

Process 1/4:
  Node: node002
  ...

✅ All processes see consistent results
✅ 8/8 tests PASSED (per process)
```

---

## Troubleshooting

### Issue: "JAX must NOT be imported before phasic"

**Solution**: Always import phasic before JAX:
```python
from phasic import Graph  # FIRST
import jax                # AFTER
```

### Issue: Tests skip with "requires SLURM environment"

**Expected behavior**: SLURM tests only run in SLURM jobs. Run locally with:
```bash
python -m pytest tests/test_multi_process_ffi.py  # Local tests
```

### Issue: Different results across nodes

**Check**:
1. All nodes have same phasic version: `python -c "import phasic; print(phasic.__version__)"`
2. Same theta parameters passed to all processes
3. Check for numerical precision issues (use rtol=1e-10 for comparisons)

### Issue: JAX distributed not initializing

**Debug**:
```python
import jax
print(f"Process count: {jax.process_count()}")  # Should be > 1 on cluster
print(f"Process index: {jax.process_index()}")  # Unique per node
print(f"Devices: {jax.devices()}")              # Should see all nodes
```

**If process_count == 1**:
- Check SLURM environment variables are set
- Verify `jax.distributed.initialize()` was called
- See `src/phasic/auto_parallel.py` for auto-initialization logic

---

##Performance Expectations

**Local (8 CPUs)**:
- 11 tests complete in ~3-4 seconds
- Overhead: <1ms per FFI call
- vmap batching: ~linear speedup

**SLURM (4 nodes × 8 CPUs = 32 total)**:
- 8 tests complete in ~5-10 seconds per process
- Network overhead: minimal (embarrassingly parallel)
- Each process runs independently

---

## Integration with CI/CD

**GitHub Actions** (local tests):
```yaml
- name: Multi-process FFI tests
  run: |
    python -m pytest tests/test_multi_process_ffi.py -v
```

**Cluster CI** (if available):
```bash
# Add to cluster CI pipeline
sbatch --wait tests/slurm_test_ffi.sh
```

---

## Future Enhancements

Potential additions:

1. **test_distributed_init.py** - Unit tests for distributed utilities
   - Mock SLURM environment
   - Test `detect_slurm_environment()`
   - Test `get_coordinator_address()`

2. **test_multi_process_svgd.py** - SVGD multi-node integration
   - Verify particle sharding
   - Test gradient aggregation
   - Confirm convergence across nodes

3. **Benchmark suite**
   - Measure scaling efficiency
   - Compare single-node vs multi-node
   - Identify communication bottlenecks

---

## References

- **JAX Multi-Process**: https://docs.jax.dev/en/latest/multi_process.html
- **SLURM Documentation**: https://slurm.schedmd.com/
- **Phasic Distributed Guide**: `docs/pages/distributed/multi_node_svgd.md` (planned)

---

**Last Updated**: 2025-11-18
**Status**: Multi-process tests complete and verified
