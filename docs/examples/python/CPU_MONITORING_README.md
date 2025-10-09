# CPU Monitoring Examples

This directory contains examples for using the CPU monitoring features in PtDAlgorithms.

## Files

### 📓 Jupyter Notebooks

**[cpu_monitoring_showcase.ipynb](cpu_monitoring_showcase.ipynb)** ⭐ **START HERE**
- **Complete showcase** with 8 sections covering all features
- Live demonstrations with real computations
- Examples for Jupyter, scripts, and decorators
- Tips, tricks, and best practices
- ~30 interactive examples
- **Recommended for all users**

**[cpu_monitoring_notebook.ipynb](cpu_monitoring_notebook.ipynb)**
- Alternative notebook with different examples
- Focus on specific use cases
- Good for quick reference

### 🐍 Python Scripts

**[cpu_monitoring_example.py](cpu_monitoring_example.py)**
- Standalone Python script examples
- No Jupyter required
- 7 different usage patterns
- Run with: `python cpu_monitoring_example.py`

## Quick Start

### For Jupyter Users

1. Open `cpu_monitoring_showcase.ipynb`
2. Run the first cell to import PtDAlgorithms
3. Try the `%%usage` cell magic:

```python
%%usage
# Your code here
result = computation()
```

### For Python Script Users

1. Look at `cpu_monitoring_example.py`
2. Use context manager:

```python
import ptdalgorithms as pta

with pta.CPUMonitor():
    result = computation()
```

### For Function Decoration

1. Use the `@monitor_cpu` decorator:

```python
@pta.monitor_cpu
def my_function():
    # Your computation
    pass
```

## What You'll Learn

### cpu_monitoring_showcase.ipynb Covers:

1. **Setup and Installation** - Getting started
2. **Cell Magic (`%%usage`)** - Easiest for Jupyter
3. **Context Manager** - For scripts and control
4. **Decorator** - For specific functions
5. **Real Computations** - Matrix ops, FFT, PtDAlgorithms graphs
6. **Advanced Configuration** - Custom settings
7. **SLURM Usage** - Multi-node monitoring
8. **Tips and Tricks** - Best practices

### cpu_monitoring_example.py Covers:

1. Basic context manager usage
2. Custom width and update intervals
3. Decorator usage (basic and custom)
4. Real NumPy computations
5. PtDAlgorithms graph operations
6. Multiple sequential operations
7. Summary statistics

## Features Demonstrated

- ✅ **Per-core CPU monitoring** with live bars
- ✅ **Auto-width detection** (terminal/notebook)
- ✅ **SLURM awareness** (single/multi-node)
- ✅ **Unicode bars** (terminal) and widgets (Jupyter)
- ✅ **Summary statistics** (mean/min/max, CPU-seconds)
- ✅ **Multiple interfaces** (magic/context/decorator)

## Running Examples

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open cpu_monitoring_showcase.ipynb
# Run cells interactively
```

### Python Scripts

```bash
# Run standalone example
python cpu_monitoring_example.py

# Or with pixi
pixi run python cpu_monitoring_example.py
```

### On SLURM

```bash
# Single node
sbatch --nodes=1 --cpus-per-task=16 my_job.sh

# Multi-node
sbatch --nodes=4 --cpus-per-task=8 my_job.sh
```

The monitor will automatically detect and display all allocated resources!

## Troubleshooting

### Dependencies Missing

```bash
pip install psutil rich tqdm
```

### Cell Magic Not Working

Make sure you imported ptdalgorithms:

```python
import ptdalgorithms as pta
```

The `%%usage` magic is auto-registered on import.

### Display Issues

Try specifying width explicitly:

```python
with pta.CPUMonitor(width=100):
    # ...
```

## More Information

- **Full Guide**: `../CPU_MONITORING_GUIDE.md`
- **Source Code**: `../src/ptdalgorithms/cpu_monitor.py`
- **Main Docs**: `../docs/`

## Comparison Table

| Feature | Showcase Notebook | Example Script | Example Notebook |
|---------|-------------------|----------------|------------------|
| Interactive | ✅ | ❌ | ✅ |
| Cell Magic | ✅ | ❌ | ✅ |
| Context Manager | ✅ | ✅ | ✅ |
| Decorator | ✅ | ✅ | ✅ |
| Real Computations | ✅ | ✅ | ❌ |
| SLURM Examples | ✅ | ❌ | ✅ |
| Tips & Tricks | ✅ | ❌ | ❌ |
| Beginner Friendly | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

**Recommendation**: Start with **cpu_monitoring_showcase.ipynb** for the complete experience!

## Example Output

### Terminal

```
┌─ macbookpro.lan (10 cores) ─┐
│ CPU 0 ▕█████░░░░░▏ 52%      │
│ CPU 1 ▕███████░░░▏ 67%      │
│ CPU 2 ▕████░░░░░░▏ 43%      │
...

============================================================
CPU Monitoring Summary
============================================================

Node: macbookpro.lan
  Duration: 2.0s
  Overall: mean=45.3%, max=78.2%, min=12.1%
  CPU-seconds: 9.1
  Per-core averages:
    CPU  0: mean= 52.1%, max= 78.2%, min= 34.5%
    ...
```

### Jupyter

Live widget-based progress bars that update in real-time!

## License

Same as PtDAlgorithms (MIT).
