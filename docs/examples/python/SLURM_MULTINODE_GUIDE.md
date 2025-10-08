# Multi-Node SLURM Guide for Distributed SVGD

This guide explains how to distribute SVGD particle computation across multiple nodes (machines) in a SLURM cluster.

**NEW:** We now have a modular system that reduces boilerplate from 200+ lines to ~20 lines! See [Modular Approach](#modular-approach-new) below.

## Table of Contents

1. [Modular Approach (NEW)](#modular-approach-new) ⭐ **Start here!**
2. [Overview](#overview)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration Guide](#configuration-guide)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)
9. [Legacy Approach](#legacy-approach)

---

## Modular Approach (NEW)

### Why This Matters

The original multi-node setup required 200+ lines of boilerplate code spread across shell scripts and Python files. With the new modular system, you write **20 lines of clean code** and let the modules handle the complexity.

### Three Simple Steps

**1. Choose a configuration (or create your own)**

```bash
# List available profiles
python examples/generate_slurm_script.py --list-profiles

# Available: debug, small, medium, large, production
# Or create custom YAML in examples/slurm_configs/
```

**2. Generate SLURM script**

```bash
# Generate script
python examples/generate_slurm_script.py \\
    --profile medium \\
    --script simple_multinode_example.py \\
    --output submit.sh

# Or pipe directly to sbatch
python examples/generate_slurm_script.py \\
    --profile small \\
    --script simple_multinode_example.py | sbatch
```

**3. Your Python script is trivial**

```python
from ptdalgorithms.distributed_utils import initialize_distributed

# That's it! All boilerplate handled
dist_info = initialize_distributed()

# Use dist_info in your code
run_my_svgd(dist_info)
```

### Complete Example

**Your entire script** (`my_svgd.py`):

```python
#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from ptdalgorithms.distributed_utils import initialize_distributed
from ptdalgorithms.ffi_wrappers import compute_pmf_ffi

# Initialize distributed computing (auto-detects SLURM)
dist_info = initialize_distributed()

print(f"Process {dist_info.process_id}/{dist_info.num_processes}")
print(f"Local devices: {dist_info.local_device_count}")
print(f"Global devices: {dist_info.global_device_count}")

# Your computation here
# pmap automatically distributes across all nodes!
```

**Generate and submit:**

```bash
# Generate submission script
python examples/generate_slurm_script.py \\
    --config slurm_configs/production.yaml \\
    --script my_svgd.py \\
    --output submit_production.sh

# Submit
sbatch submit_production.sh

# Or do both in one step
python examples/generate_slurm_script.py \\
    --profile medium \\
    --script my_svgd.py | sbatch
```

### Configuration Files

Create YAML configs in `examples/slurm_configs/`:

```yaml
# my_cluster.yaml
name: my_cluster
nodes: 4
cpus_per_node: 16
memory_per_cpu: "8G"
time_limit: "02:00:00"
partition: "compute"
coordinator_port: 12345
platform: "cpu"

# Environment variables
env_vars:
  JAX_ENABLE_X64: "1"
  NCCL_SOCKET_IFNAME: "ib0"

# Modules to load
modules_to_load:
  - "python/3.11"
  - "gcc/11.2.0"
```

### Benefits

✅ **80% less code** - 200+ lines → 20 lines
✅ **Reusable** - Import and use anywhere
✅ **Type-safe** - Full IDE support
✅ **Testable** - Mock SLURM environment for local testing
✅ **Maintainable** - One place to update
✅ **Flexible** - Easy to add new cluster configs

### Comparison

**Before (Legacy approach):**

```python
# 150+ lines of boilerplate
if 'SLURM_JOB_ID' not in os.environ:
    # ... 50 lines ...
coordinator_address = os.environ.get('SLURM_COORDINATOR_ADDRESS')
if coordinator_address is None:
    result = subprocess.run(['scontrol', 'show', 'hostnames', ...])
    # ... 30 more lines ...
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={cpus}"
# ... 70 more lines ...
jax.distributed.initialize(...)
```

**After (Modular approach):**

```python
# 3 lines total
from ptdalgorithms.distributed_utils import initialize_distributed

dist_info = initialize_distributed()  # Handles everything!
run_my_computation(dist_info)
```

### Quick Reference

```bash
# Available pre-defined profiles
debug       # 1 node × 4 CPUs = 4 devices (testing)
small       # 2 nodes × 8 CPUs = 16 devices
medium      # 4 nodes × 16 CPUs = 64 devices
large       # 8 nodes × 16 CPUs = 128 devices
production  # 8 nodes × 32 CPUs = 256 devices

# Generate from profile
python examples/generate_slurm_script.py --profile medium --script my_script.py

# Generate from custom config
python examples/generate_slurm_script.py --config my_config.yaml --script my_script.py

# List available options
python examples/generate_slurm_script.py --help
```

---

## Overview

### Single-Node vs Multi-Node Parallelization

**Single-Node (current `svgd.py` setup):**
- Uses `XLA_FLAGS` to create multiple CPU devices on one machine
- Limited by CPU count on single machine (typically 8-64 cores)
- Good for small-scale problems (<50 particles)

**Multi-Node (this guide):**
- Distributes computation across multiple machines
- Uses JAX's distributed initialization (`jax.distributed.initialize()`)
- Scales to hundreds of CPUs/GPUs across cluster
- Required for large-scale SVGD (>100 particles)

### Key Difference

```python
# Single-node: Just set XLA_FLAGS
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# 8 devices on 1 machine

# Multi-node: Initialize distributed JAX
jax.distributed.initialize(
    coordinator_address="node001:12345",
    num_processes=4,    # 4 nodes
    process_id=rank     # 0, 1, 2, 3
)
# 32 devices across 4 machines (8 per machine)
```

---

## Architecture

### Three-Level Parallelism

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Multi-Node Distribution (across machines)          │
│   - Node 1: Process 0 (rank 0)                              │
│   - Node 2: Process 1 (rank 1)                              │
│   - Node 3: Process 2 (rank 2)                              │
│   - Node 4: Process 3 (rank 3)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 2: Intra-Node Parallelism (within each machine)       │
│   - Each process has multiple CPU devices                   │
│   - 8 CPUs per node → 8 devices per process                 │
│   - Total: 4 nodes × 8 CPUs = 32 devices                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 3: Particle Distribution (across all devices)         │
│   - 128 particles distributed across 32 devices             │
│   - 4 particles per device                                  │
│   - pmap distributes across devices automatically           │
└─────────────────────────────────────────────────────────────┘
```

### JAX Distributed Communication

```
Coordinator (Node 1, Process 0)
      ↓
    [TCP/IP Communication on port 12345]
      ↓
Workers (Processes 1, 2, 3)
      ↓
All processes participate in pmap/pjit operations
```

---

## Quick Start

### 1. Basic Multi-Node Job (4 nodes × 8 CPUs = 32 devices)

```bash
# Submit job
sbatch examples/slurm_multinode.sh

# Check status
squeue -u $USER

# View output
tail -f svgd_multinode_<JOB_ID>.out
```

### 2. Advanced Multi-Node Job (8 nodes × 16 CPUs = 128 devices)

```bash
# Submit large-scale job
sbatch examples/slurm_multinode_advanced.sh

# Monitor progress
watch -n 5 'tail -20 logs/svgd_<JOB_ID>.out'
```

---

## Configuration Guide

### SLURM Parameters

```bash
#SBATCH --nodes=N                    # Number of nodes (machines)
#SBATCH --ntasks-per-node=1          # Processes per node (usually 1)
#SBATCH --cpus-per-task=C            # CPUs per process (devices per node)
```

**Total devices = N × C**

### Particle Count Calculation

```python
# Must satisfy: num_particles % total_devices == 0

total_devices = nodes × cpus_per_task
particles_per_device = 4  # Tune based on problem
num_particles = total_devices × particles_per_device
```

**Example configurations:**

| Nodes | CPUs/Node | Total Devices | Particles | Particles/Device |
|-------|-----------|---------------|-----------|------------------|
| 2     | 8         | 16            | 64        | 4                |
| 4     | 8         | 32            | 128       | 4                |
| 4     | 16        | 64            | 256       | 4                |
| 8     | 16        | 128           | 512       | 4                |
| 16    | 32        | 512           | 2048      | 4                |

### Coordinator Setup

The coordinator is the first node in the allocation:

```bash
# In SLURM script
COORDINATOR_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_COORDINATOR_ADDRESS=$COORDINATOR_NODE
export JAX_COORDINATOR_PORT=12345
```

### JAX Distributed Initialization

In your Python script:

```python
import jax

# Get SLURM environment
process_id = int(os.environ['SLURM_PROCID'])
num_processes = int(os.environ['SLURM_NTASKS'])
coordinator = f"{os.environ['SLURM_COORDINATOR_ADDRESS']}:12345"

# Initialize
jax.distributed.initialize(
    coordinator_address=coordinator,
    num_processes=num_processes,
    process_id=process_id
)

# Now pmap works across all nodes automatically!
```

---

## Performance Tuning

### 1. Optimal Particle Count

**Rule of thumb:** 2-8 particles per device

```python
# Too few particles per device (< 2)
# - Overhead dominates
# - Poor device utilization

# Optimal (2-8 particles per device)
# - Good balance between parallelism and overhead
# - Efficient device utilization

# Too many particles per device (> 16)
# - Memory pressure
# - Diminishing returns from adding more devices
```

### 2. Network Configuration

For **InfiniBand** clusters:
```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0
```

For **Ethernet** clusters:
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_NTHREADS=8
```

### 3. Memory Management

```bash
# Disable preallocation for CPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set memory limit per device (if needed)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### 4. XLA Optimization Flags

```bash
export XLA_FLAGS="
  --xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK
  --xla_cpu_multi_thread_eigen=true
  --xla_cpu_enable_fast_math=true
"
```

---

## Troubleshooting

### Common Issues

#### 1. "Coordinator connection timeout"

**Cause:** Firewall blocking port 12345 or wrong coordinator address

**Solution:**
```bash
# Test connectivity
ssh $COORDINATOR_NODE "nc -l 12345" &
nc -vz $COORDINATOR_NODE 12345

# Try different port
export JAX_COORDINATOR_PORT=54321

# Check firewall
sudo firewall-cmd --list-ports
```

#### 2. "Number of devices mismatch"

**Cause:** `XLA_FLAGS` not set correctly per process

**Solution:**
```bash
# In SLURM script, ensure:
export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# NOT a fixed number like:
# export XLA_FLAGS="--xla_force_host_platform_device_count=8"  # WRONG
```

#### 3. "Particles not divisible by device count"

**Cause:** `num_particles % (nodes × cpus_per_task) != 0`

**Solution:**
```python
# Calculate valid particle count
total_devices = nodes × cpus_per_task
particles_per_device = 4
num_particles = total_devices × particles_per_device
```

#### 4. "Slow inter-node communication"

**Cause:** Using wrong network interface or suboptimal NCCL settings

**Solution:**
```bash
# Identify fast network interface
ifconfig | grep -A 1 "ib0"  # InfiniBand
ifconfig | grep -A 1 "eth"  # Ethernet

# Set correct interface
export NCCL_SOCKET_IFNAME=ib0  # or eth0, eth1, etc.

# Enable debugging
export NCCL_DEBUG=INFO
```

#### 5. "Out of memory"

**Cause:** Too many particles per device or memory leak

**Solution:**
```bash
# Increase memory allocation
#SBATCH --mem-per-cpu=16G  # Increase from 8G

# Reduce particles per device
particles_per_device = 2  # Instead of 4

# Check memory usage
srun --nodes=$SLURM_JOB_NUM_NODES free -h
```

---

## Examples

### Example 1: Small-Scale Test (2 nodes)

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Total: 2 × 4 = 8 devices
# Particles: 8 × 4 = 32 particles

COORDINATOR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_COORDINATOR_ADDRESS=$COORDINATOR
export JAX_COORDINATOR_PORT=12345
export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

srun python examples/slurm_multinode_example.py
```

### Example 2: Medium-Scale SVGD (4 nodes)

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

# Total: 4 × 16 = 64 devices
# Particles: 64 × 4 = 256 particles

# ... (same setup as Example 1)
```

### Example 3: Large-Scale SVGD (16 nodes)

```bash
#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G

# Total: 16 × 32 = 512 devices
# Particles: 512 × 4 = 2048 particles

# ... (same setup as Example 1)
```

---

## Performance Scaling

### Expected Speedup

Assuming efficient network and well-balanced computation:

| Configuration | Devices | Particles | Est. Time | Speedup |
|---------------|---------|-----------|-----------|---------|
| Single-node   | 8       | 32        | 100s      | 1x      |
| 2 nodes       | 16      | 64        | 52s       | 1.9x    |
| 4 nodes       | 32      | 128       | 27s       | 3.7x    |
| 8 nodes       | 64      | 256       | 15s       | 6.7x    |
| 16 nodes      | 128     | 512       | 9s        | 11.1x   |

**Note:** Actual speedup depends on:
- Network bandwidth and latency
- Computation/communication ratio
- Problem size (larger = better scaling)

### Scaling Efficiency

```
Ideal scaling: Time(N nodes) = Time(1 node) / N
Actual scaling: ~0.8-0.9 of ideal (80-90% efficiency)
```

**Good scaling indicators:**
- Efficiency > 80% for up to 8 nodes
- Efficiency > 60% for up to 16 nodes
- Efficiency > 40% for up to 32 nodes

---

## Best Practices

1. **Always test on 2 nodes first** before scaling to many nodes
2. **Use even particle distribution**: `num_particles % total_devices == 0`
3. **Monitor network utilization** with `iftop` or `nload`
4. **Profile first iteration** to identify bottlenecks
5. **Use exclusive node access** (`#SBATCH --exclusive`) for consistent performance
6. **Save checkpoints** for long-running jobs
7. **Log per-process metrics** to identify slow nodes
8. **Validate results** by comparing single-node vs multi-node outputs

---

## Additional Resources

- [JAX Distributed Computing Guide](https://jax.readthedocs.io/en/latest/multi_process.html)
- [SLURM Multi-Node Documentation](https://slurm.schedmd.com/job_launch.html)
- [NCCL Performance Tuning](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

---

## Summary

**Single-Node:**
```python
# Set XLA_FLAGS → creates local devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
```

**Multi-Node:**
```python
# Initialize JAX distributed → enables global pmap
jax.distributed.initialize(coordinator_address, num_processes, process_id)
# Now pmap automatically distributes across all nodes!
```

**Key Takeaway:** With JAX distributed initialization, `pmap` works transparently across nodes. You write the same code, but it scales from 1 to 100+ machines!

---

*Generated by Claude Code - 2025-10-07*
