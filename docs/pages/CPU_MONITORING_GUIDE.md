# CPU Monitoring for PtDAlgorithms

## Overview

PtDAlgorithms now includes comprehensive CPU monitoring with beautiful visualizations for both terminal and Jupyter notebook environments. The monitor provides:

- **Per-core CPU monitoring** with Unicode bar charts
- **SLURM-aware** detection of allocated nodes and CPUs
- **Adaptive width** that fits terminal or notebook layout
- **Live updates** during computation
- **Summary statistics** after completion
- **Multiple interfaces**: cell magic, context manager, and decorator

## Installation

The CPU monitoring feature requires these dependencies (already added to `pyproject.toml`):

```bash
pip install psutil rich tqdm
```

Or if using the package:

```bash
pip install -e .
```

## Quick Start

### Jupyter Notebook (Cell Magic)

The easiest way to use CPU monitoring in Jupyter:

```python
import ptdalgorithms as pta  # Magic auto-registered

%%usage
# Your code here
result = heavy_computation()
```

**With options:**

```python
%%usage --width 120 --interval 0.5
# Your code here
```

### Python Script (Context Manager)

```python
import ptdalgorithms as pta

with pta.CPUMonitor():
    result = computation()
```

**With options:**

```python
with pta.CPUMonitor(width=100, update_interval=0.5):
    result = computation()
```

### Decorator

```python
@pta.monitor_cpu
def my_function():
    # Your computation
    pass

# Or with options
@pta.monitor_cpu(update_interval=1.0)
def my_function():
    pass
```

## Display Examples

### Single Node (Terminal)

```
┌─ macbookpro.lan (10 cores) ─┐
│ CPU 0 ▕█████░░░░░▏ 52%      │
│ CPU 1 ▕███████░░░▏ 67%      │
│ CPU 2 ▕████░░░░░░▏ 43%      │
│ CPU 3 ▕██████████▏ 95%      │
│ CPU 4 ▕████░░░░░░▏ 41%      │
│ CPU 5 ▕██████░░░░▏ 58%      │
│ CPU 6 ▕█████░░░░░▏ 49%      │
│ CPU 7 ▕███████░░░▏ 72%      │
│ CPU 8 ▕██░░░░░░░░▏ 23%      │
│ CPU 9 ▕█░░░░░░░░░▏ 15%      │
└──────────────────────────────┘
```

### Multi-Node SLURM (Columns)

```
┌─ compute-001 ─┐  ┌─ compute-002 ─┐  ┌─ compute-003 ─┐
│ CPU 0 [=====>] │  │ CPU 0 [=====>] │  │ CPU 0 [=====>] │
│ CPU 1 [====>] │  │ CPU 1 [=====>] │  │ CPU 1 [======] │
│ CPU 2 [======] │  │ CPU 2 [====>] │  │ CPU 2 [=====>] │
│ CPU 3 [====>] │  │ CPU 3 [======] │  │ CPU 3 [====>] │
│ CPU 4 [=====>] │  │ CPU 4 [=====>] │  │ CPU 4 [======] │
│ CPU 5 [====>] │  │ CPU 5 [====>] │  │ CPU 5 [=====>] │
│ CPU 6 [======] │  │ CPU 6 [======] │  │ CPU 6 [====>] │
│ CPU 7 [=====>] │  │ CPU 7 [=====>] │  │ CPU 7 [======] │
└────────────────┘  └────────────────┘  └────────────────┘
```

### Summary Statistics

After monitoring completes, you'll see:

```
============================================================
CPU Monitoring Summary
============================================================

Node: compute-001
  Duration: 45.3s
  Overall: mean=67.3%, max=89.1%, min=34.2%
  CPU-seconds: 243.5
  Per-core averages:
    CPU  0: mean= 65.2%, max= 87.3%, min= 32.1%
    CPU  1: mean= 71.8%, max= 91.2%, min= 45.6%
    ...
============================================================
```

## API Reference

### `CPUMonitor` Class

```python
CPUMonitor(
    width=None,              # Display width (auto-detect if None)
    update_interval=0.5,     # Update frequency in seconds
    per_node_layout=True,    # Group CPUs by node
    show_summary=True        # Show summary at end
)
```

**Methods:**
- `start()`: Start monitoring
- `stop()`: Stop monitoring
- `__enter__` / `__exit__`: Context manager support

### `monitor_cpu` Decorator

```python
@monitor_cpu(
    width=None,
    update_interval=0.5,
    show_summary=True
)
def your_function():
    pass
```

### Cell Magic `%%usage`

```python
%%usage [--width WIDTH] [--interval INTERVAL]
```

**Arguments:**
- `--width, -w`: Display width in characters
- `--interval, -i`: Update interval in seconds

### `detect_compute_nodes()` Function

Returns a list of `NodeInfo` objects representing available compute nodes:

```python
nodes = pta.detect_compute_nodes()
for node in nodes:
    print(f"{node.name}: {node.cpu_count} CPUs")
```

## SLURM Integration

The monitor automatically detects SLURM environments and adapts:

### Single-Node SLURM Job

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

python my_script.py  # Will monitor 16 allocated CPUs
```

### Multi-Node SLURM Job

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8

python my_script.py  # Will show 4 nodes × 8 CPUs each
```

## Environment Detection

The monitor automatically detects:

1. **Local execution**: Uses all system CPUs
2. **Single SLURM node**: Uses `SLURM_CPUS_PER_TASK`
3. **Multi-node SLURM**: Parses `SLURM_JOB_NODELIST`
4. **CPU affinity**: Respects `psutil.cpu_affinity()` when set

## Examples

### Example 1: Monitor NumPy Operations

```python
import numpy as np
import ptdalgorithms as pta

with pta.CPUMonitor():
    # Matrix operations
    A = np.random.randn(2000, 2000)
    B = np.random.randn(2000, 2000)
    C = np.dot(A, B)
    eigenvalues = np.linalg.eigvals(C[:500, :500])
```

### Example 2: Monitor PtDAlgorithms Computations

```python
import ptdalgorithms as pta
import numpy as np

# Initialize parallel configuration
config = pta.init_parallel(cpus=8)

# Create a phase-type distribution
g = pta.Graph(1)
# ... build graph ...

# Monitor PDF computation
with pta.CPUMonitor():
    times = np.linspace(0.1, 10.0, 100_000)
    pdf_values = g.pdf_batch(times)
```

### Example 3: Jupyter Notebook

```python
import ptdalgorithms as pta
import time

%%usage --interval 0.25
# Simulate computation
for i in range(10):
    x = sum(range(10_000_000))
    time.sleep(0.5)
```

### Example 4: Decorator

```python
@pta.monitor_cpu(update_interval=1.0)
def train_model(data):
    # Training code
    pass

model = train_model(my_data)
```

## Advanced Configuration

### Custom Width

```python
# Force specific width
with pta.CPUMonitor(width=80):
    computation()
```

### Disable Summary

```python
# No summary statistics
with pta.CPUMonitor(show_summary=False):
    computation()
```

### Faster Updates

```python
# Update 4 times per second
with pta.CPUMonitor(update_interval=0.25):
    computation()
```

## Troubleshooting

### "No module named 'rich'" or similar

Install dependencies:

```bash
pip install psutil rich tqdm
```

### Cell magic not working in Jupyter

The magic is auto-registered on import. Make sure you've imported ptdalgorithms:

```python
import ptdalgorithms as pta
```

### Not seeing all SLURM nodes

Make sure `scontrol` is available:

```bash
scontrol show hostnames $SLURM_JOB_NODELIST
```

### Display width issues

Explicitly set width:

```python
with pta.CPUMonitor(width=120):
    # ...
```

## Implementation Details

### Architecture

- **Background thread**: Samples CPU usage at specified interval
- **Rich library**: Handles terminal display with Unicode bars
- **tqdm**: Provides widget-based bars in Jupyter
- **psutil**: Collects per-core CPU percentages
- **Environment detection**: Automatically adapts to local/SLURM/Jupyter

### Files

- `src/ptdalgorithms/cpu_monitor.py`: Main implementation
- `src/ptdalgorithms/__init__.py`: Exports and magic registration
- `examples/cpu_monitoring_example.py`: Python script examples
- `examples/cpu_monitoring_notebook.ipynb`: Jupyter examples

### Key Classes

- `CPUMonitor`: Main monitoring class
- `NodeInfo`: Represents a compute node
- `CPUStats`: Collects and summarizes statistics
- `CPUMonitorMagics`: IPython magic implementation

## Performance Impact

The CPU monitor has minimal overhead:

- Background thread samples every 0.5s (configurable)
- Display updates are rate-limited
- Uses efficient psutil APIs
- Negligible impact on monitored computation

## Related Features

- `pta.init_parallel()`: Configure parallel computation
- `pta.detect_environment()`: Inspect execution environment
- `pta.parallel_config()`: Temporary parallel configuration
- `pta.disable_parallel()`: Disable parallelization

## Future Enhancements

Potential future additions:

- Memory monitoring
- GPU monitoring (if CUDA available)
- Network I/O monitoring
- Disk I/O monitoring
- Export to HTML/JSON
- Historical plots

## Support

For issues or questions:
- GitHub Issues: https://github.com/munch-group/ptdalgorithms/issues
- Examples: See `examples/` directory

## License

Same license as PtDAlgorithms (MIT).
