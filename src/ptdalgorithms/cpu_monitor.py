"""
CPU Monitoring for PtDAlgorithms

This module provides live CPU usage monitoring with beautiful visualizations
for both terminal and Jupyter notebook environments. It supports:
- Per-core CPU monitoring
- SLURM multi-node awareness
- Adaptive width layouts
- Unicode bar charts in terminal
- HTML widgets in Jupyter
- Cell magic (%%usage) for easy Jupyter usage

Usage:
    >>> # Jupyter notebook
    >>> %%usage
    >>> result = heavy_computation()

    >>> # Python script or notebook
    >>> with pta.CPUMonitor():
    >>>     result = heavy_computation()

    >>> # Decorator
    >>> @pta.monitor_cpu
    >>> def my_function():
    >>>     pass

Author: PtDAlgorithms Team
Date: 2025-10-08
"""

import os
import sys
import time
import threading
import shutil
import functools
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict
import logging

# Core dependencies
import psutil

# Rich for terminal UI
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.text import Text

# tqdm for Jupyter
from tqdm import tqdm as std_tqdm
try:
    from tqdm.notebook import tqdm as notebook_tqdm
    HAS_NOTEBOOK_TQDM = True
except ImportError:
    HAS_NOTEBOOK_TQDM = False
    notebook_tqdm = std_tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NodeInfo:
    """Information about a compute node."""
    name: str
    cpu_count: int
    allocated_cpus: Optional[List[int]] = None
    process_id: int = 0

    def __post_init__(self):
        if self.allocated_cpus is None:
            self.allocated_cpus = list(range(self.cpu_count))


@dataclass
class CPUStats:
    """Statistics for CPU usage over time."""
    samples: List[List[float]] = field(default_factory=list)  # List of per-core samples
    memory_samples: List[float] = field(default_factory=list)  # Memory usage samples
    timestamps: List[float] = field(default_factory=list)

    def add_sample(self, per_core_usage: List[float], memory_percent: float = 0.0):
        """Add a CPU usage sample."""
        self.samples.append(per_core_usage)
        self.memory_samples.append(memory_percent)
        self.timestamps.append(time.time())

    def get_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.samples:
            return {}

        import numpy as np
        samples_array = np.array(self.samples)

        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0

        summary = {
            'duration': duration,
            'mean_per_core': np.mean(samples_array, axis=0).tolist(),
            'max_per_core': np.max(samples_array, axis=0).tolist(),
            'min_per_core': np.min(samples_array, axis=0).tolist(),
            'overall_mean': np.mean(samples_array),
            'overall_max': np.max(samples_array),
            'overall_min': np.min(samples_array),
            'cpu_seconds': np.sum(np.mean(samples_array, axis=0)) * duration / 100.0,
        }

        # Add memory statistics if available
        if self.memory_samples:
            memory_array = np.array(self.memory_samples)
            summary['memory_mean'] = np.mean(memory_array)
            summary['memory_max'] = np.max(memory_array)
            summary['memory_min'] = np.min(memory_array)

        return summary


# ============================================================================
# Environment Detection
# ============================================================================

def _clean_hostname(hostname: str) -> str:
    """
    Remove non-informative suffixes from hostnames.

    Strips common suffixes like .lan, .local, .localdomain, etc.
    """
    suffixes = ['.lan', '.local', '.localdomain', '.domain', '.home']
    for suffix in suffixes:
        if hostname.endswith(suffix):
            hostname = hostname[:-len(suffix)]
    return hostname

def detect_compute_nodes() -> List[NodeInfo]:
    """
    Detect compute nodes and allocated CPUs.

    Returns list of NodeInfo objects, one per node.
    For local execution, returns single node.
    For SLURM, returns all allocated nodes.
    """
    # Check for SLURM environment
    if 'SLURM_JOB_ID' not in os.environ:
        # Local execution
        cpu_count = os.cpu_count() or 1

        # Try to get CPU affinity if available
        try:
            allocated_cpus = psutil.Process().cpu_affinity()
            if allocated_cpus:
                cpu_count = len(allocated_cpus)
        except (AttributeError, OSError):
            allocated_cpus = list(range(cpu_count))

        import socket
        hostname = _clean_hostname(socket.gethostname())

        return [NodeInfo(
            name=hostname,
            cpu_count=cpu_count,
            allocated_cpus=allocated_cpus,
            process_id=0
        )]

    # SLURM environment
    from .distributed_utils import detect_slurm_environment

    slurm_env = detect_slurm_environment()

    if not slurm_env.get('is_slurm', False):
        # Fallback to local
        return detect_compute_nodes.__wrapped__()

    # Get node list
    nodelist = slurm_env.get('nodelist', '')
    node_count = slurm_env.get('node_count', 1)
    cpus_per_task = slurm_env.get('cpus_per_task', 1)
    process_id = slurm_env.get('process_id', 0)

    # Parse nodelist using scontrol
    nodes = []
    if nodelist:
        try:
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', nodelist],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            nodes = result.stdout.strip().split('\n')
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Could not parse SLURM nodelist: {e}")
            nodes = [f"node-{i}" for i in range(node_count)]
    else:
        nodes = [f"node-{i}" for i in range(node_count)]

    # Get CPU affinity for this process
    try:
        allocated_cpus = psutil.Process().cpu_affinity()
        if not allocated_cpus:
            allocated_cpus = list(range(cpus_per_task))
    except (AttributeError, OSError):
        allocated_cpus = list(range(cpus_per_task))

    # Create NodeInfo for each node
    # In multi-node SLURM, each process monitors its own node
    node_infos = []
    for i, node_name in enumerate(nodes):
        node_infos.append(NodeInfo(
            name=_clean_hostname(node_name),
            cpu_count=len(allocated_cpus) if i == process_id else cpus_per_task,
            allocated_cpus=allocated_cpus if i == process_id else list(range(cpus_per_task)),
            process_id=i
        ))

    return node_infos


def is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_type = str(type(shell))
        return 'ZMQInteractiveShell' in shell_type
    except (ImportError, NameError):
        return False


def is_vscode() -> bool:
    """Check if running in VSCode Jupyter."""
    try:
        import os
        # VSCode sets these environment variables
        return 'VSCODE_PID' in os.environ or 'VSCODE_CWD' in os.environ
    except:
        return False


# ============================================================================
# Module-level Cache
# ============================================================================

# Detect and cache compute nodes on module import
# This avoids redundant detection when creating multiple CPUMonitor instances
_CACHED_NODES = None

def get_cached_nodes() -> List[NodeInfo]:
    """
    Get cached compute nodes, detecting them if not already cached.

    Returns
    -------
    List[NodeInfo]
        List of detected compute nodes
    """
    global _CACHED_NODES
    if _CACHED_NODES is None:
        _CACHED_NODES = detect_compute_nodes()
    return _CACHED_NODES


# ============================================================================
# CPU Monitor Class
# ============================================================================

class CPUMonitor:
    """
    Live CPU monitoring with adaptive display.

    Monitors per-core CPU usage and displays it in a grid layout.
    Automatically detects SLURM nodes and Jupyter environment.

    Parameters
    ----------
    width : int, optional
        Display width in characters. If None, auto-detects terminal/notebook width.
    update_interval : float, default=0.5
        Time between updates in seconds
    per_node_layout : bool, default=True
        If True, group CPUs by node in column layout
    show_summary : bool, default=True
        Show summary statistics when monitoring ends
    persist : bool, default=False
        If True, keep display visible after completion. If False (default),
        the display disappears when monitoring stops.
    color : bool, default=False
        If True, use color coding (green < 50%, yellow 50-80%, red > 80%).
        If False (default), all bars are gray.
    summary_table : bool, default=False
        If True, display results as an HTML table with mean CPU usage per core
        instead of progress bars. Memory usage (mean/max) is shown next to the
        node name. The table will be shown after completion regardless of persist setting.

    Examples
    --------
    >>> with CPUMonitor():
    ...     result = computation()

    >>> with CPUMonitor(width=120, update_interval=1.0):
    ...     result = computation()

    >>> with CPUMonitor(persist=True):
    ...     result = computation()  # Display remains after completion

    >>> with CPUMonitor(color=True):
    ...     result = computation()  # Use color coding

    >>> with CPUMonitor(summary_table=True):
    ...     result = computation()  # Show table with CPU and memory stats
    """

    def __init__(
        self,
        width: Optional[int] = None,
        update_interval: float = 0.5,
        per_node_layout: bool = True,
        show_summary: bool = True,
        persist: bool = False,
        color: bool = False,
        summary_table: bool = False
    ):
        self.width = width
        self.update_interval = update_interval
        self.per_node_layout = per_node_layout
        self.show_summary = show_summary
        self.persist = persist
        self.color = color
        self.summary_table = summary_table

        # Detect environment
        self.is_jupyter = is_jupyter()
        self.is_vscode = is_vscode()
        self.nodes = get_cached_nodes()

        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._jupyter_update_thread = None
        self._stats = {node.name: CPUStats() for node in self.nodes}
        self._current_usage = {node.name: [] for node in self.nodes}
        self._current_memory = {node.name: 0.0 for node in self.nodes}

        # Display components
        self._console = Console()
        self._live = None
        self._tqdm_bars = []
        self._has_ipython_display = False
        self._use_html_display = False
        self._html_display = None

    def _get_display_width(self) -> int:
        """Get display width (auto-detect or user-specified)."""
        if self.width is not None:
            return self.width

        if self.is_jupyter:
            # Jupyter - use a reasonable default (will be responsive in HTML)
            return 120
        else:
            # Terminal - use terminal width
            return shutil.get_terminal_size().columns

    def _monitor_loop(self):
        """Background thread that samples CPU usage."""
        import socket
        current_hostname = _clean_hostname(socket.gethostname())

        # Find our node
        our_node = None
        for node in self.nodes:
            if node.name == current_hostname or node.process_id == 0:
                our_node = node
                break

        if our_node is None:
            logger.warning("Could not identify current node")
            return

        while self._monitoring:
            try:
                # Get per-core CPU usage
                # percpu=True returns list of percentages
                per_core = psutil.cpu_percent(interval=self.update_interval, percpu=True)

                # Get memory usage
                memory_percent = psutil.virtual_memory().percent

                # Filter to allocated CPUs if available
                if our_node.allocated_cpus:
                    try:
                        per_core = [per_core[i] for i in our_node.allocated_cpus if i < len(per_core)]
                    except IndexError:
                        pass  # Use all cores

                # Store current usage
                self._current_usage[our_node.name] = per_core
                self._current_memory[our_node.name] = memory_percent

                # Record statistics
                self._stats[our_node.name].add_sample(per_core, memory_percent)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _create_terminal_display(self) -> Table:
        """Create rich Table for terminal display."""
        display_width = self._get_display_width()

        # Calculate layout
        # Each node column needs: "CPU X [bar] XX.X%"
        # Estimate: label(6) + bar(10) + percentage(6) + padding(2) = 24 chars minimum
        min_col_width = 24
        max_nodes_per_row = max(1, display_width // min_col_width)

        # Create main table
        if len(self.nodes) == 1:
            # Single node - simple layout
            node = self.nodes[0]
            table = Table(title=f"CPU Monitor - {node.name}", show_header=False,
                         box=None, padding=(0, 1))

            usage = self._current_usage.get(node.name, [])
            for i, cpu_usage in enumerate(usage):
                # Create bar
                bar_width = 10
                filled = int((cpu_usage / 100.0) * bar_width)
                bar = '█' * filled + '░' * (bar_width - filled)

                # Color based on usage
                if cpu_usage < 50:
                    color = "green"
                elif cpu_usage < 80:
                    color = "yellow"
                else:
                    color = "red"

                table.add_row(
                    f"CPU {i:2d}",
                    Text(f"▕{bar}▏", style=color),
                    f"{cpu_usage:5.1f}%"
                )

            return table

        else:
            # Multi-node layout - create grid
            nodes_per_row = min(len(self.nodes), max_nodes_per_row)
            rows_needed = (len(self.nodes) + nodes_per_row - 1) // nodes_per_row

            # Create outer table for grid
            grid = Table(show_header=False, box=None, padding=(0, 2))
            for _ in range(nodes_per_row):
                grid.add_column()

            # Add nodes to grid
            for row_idx in range(rows_needed):
                row_nodes = []
                for col_idx in range(nodes_per_row):
                    node_idx = row_idx * nodes_per_row + col_idx
                    if node_idx >= len(self.nodes):
                        row_nodes.append("")
                        continue

                    node = self.nodes[node_idx]
                    usage = self._current_usage.get(node.name, [])

                    # Create node panel
                    node_table = Table(show_header=False, box=None, padding=(0, 0))
                    for i, cpu_usage in enumerate(usage):
                        bar_width = 8
                        filled = int((cpu_usage / 100.0) * bar_width)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        if cpu_usage < 50:
                            color = "green"
                        elif cpu_usage < 80:
                            color = "yellow"
                        else:
                            color = "red"

                        node_table.add_row(
                            f"CPU {i}",
                            Text(f"▕{bar}▏", style=color),
                            f"{cpu_usage:4.1f}%"
                        )

                    panel = Panel(node_table, title=node.name, border_style="blue")
                    row_nodes.append(panel)

                grid.add_row(*row_nodes)

            return grid

    def _create_jupyter_display(self):
        """Create tqdm bars for Jupyter display."""
        # VSCode has issues with tqdm.notebook - use HTML display instead
        if self.is_vscode:
            try:
                from IPython.display import display, HTML
                self._has_ipython_display = True
                self._use_html_display = True

                # Initialize HTML display container
                for node in self.nodes:
                    usage = self._current_usage.get(node.name, [])
                    if not usage:
                        usage = [0.0] * node.cpu_count
                        self._current_usage[node.name] = usage

                # Create initial HTML display
                html = self._generate_html_display()
                self._html_display = display(HTML(html), display_id=True)
                return
            except:
                # Fall through to tqdm if HTML fails
                pass

        if not HAS_NOTEBOOK_TQDM:
            # Fallback to terminal-style
            return self._create_terminal_display()

        # Force display update in Jupyter/VSCode
        try:
            from IPython.display import display, clear_output
            self._has_ipython_display = True
        except ImportError:
            self._has_ipython_display = False

        self._use_html_display = False

        # Create tqdm bars for each CPU on each node
        self._tqdm_bars = []

        for node in self.nodes:
            # Get current usage or initialize to zeros
            usage = self._current_usage.get(node.name, [])
            if not usage:
                usage = [0.0] * node.cpu_count
                self._current_usage[node.name] = usage

            # Node header
            print(f"\n{node.name} ({len(usage)} cores):")

            for i in range(len(usage)):
                bar = notebook_tqdm(
                    total=100,
                    desc=f"  CPU {i}",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
                    leave=True,
                    initial=usage[i] if i < len(usage) else 0
                )
                bar.n = usage[i] if i < len(usage) else 0
                bar.refresh()
                self._tqdm_bars.append(bar)

    def _generate_html_table(self):
        """Generate HTML table with CPU statistics."""
        html = '<div style="font-family: monospace; font-size: 11px; padding: 10px;">'

        for node in self.nodes:
            stats = self._stats[node.name]
            summary = stats.get_summary()
            if not summary:
                continue

            # Node name with memory percentage (mean/max)
            memory_mean = summary.get('memory_mean', 0.0)
            memory_max = summary.get('memory_max', 0.0)
            html += f'<div style="margin-bottom: 8px; font-size: 12px;"><strong>{node.name}</strong> {memory_mean:.0f}%/{memory_max:.0f}% mem (mean/max)</div>'

            mean_per_core = summary['mean_per_core']
            n_cpus = len(mean_per_core)

            # Create table with single row of CPU percentages
            html += '''<table style="border-collapse: collapse; width: 100%; margin-bottom: 16px;">'''

            # CPU usage row
            html += '<tr style="background-color: rgba(128, 128, 128, 0.1);">'
            html += '<td style="padding: 4px 8px; font-weight: bold;">CPU %</td>'
            for cpu_val in mean_per_core:
                html += f'<td style="padding: 4px 8px; text-align: center;">{cpu_val:.0f}</td>'
            html += '</tr>'

            html += '</table>'

        html += '</div>'
        return html

    def _generate_html_display(self, summary_mode=False):
        """Generate HTML for VSCode display."""
        # If summary_table mode is enabled and we're in summary, show table
        if self.summary_table and summary_mode:
            return self._generate_html_table()

        html = '<div style="font-family: monospace; font-size: 11px; padding: 10px;">'

        for node in self.nodes:
            if summary_mode:
                # Summary mode: show mean usage
                stats = self._stats[node.name]
                summary = stats.get_summary()
                if not summary:
                    continue

                # Node name with mean/max memory
                memory_mean = summary.get('memory_mean', 0.0)
                memory_max = summary.get('memory_max', 0.0)
                html += f'<div style="margin-bottom: 4px; font-size: 12px;"><strong>{node.name}</strong> {memory_mean:.0f}%/{memory_max:.0f}% mem (mean/max)</div>'

                n_cpus = len(summary['mean_per_core'])

                # Single row of bars - fill available width
                html += '<div style="display: flex; gap: 3px; margin-bottom: 8px; width: 100%;">'
                for i in range(n_cpus):
                    mean_val = summary['mean_per_core'][i]

                    # Color based on mode
                    summary_color = '#4CAF50' if self.color else '#666666'

                    # Show mean usage bar
                    html += f'''
                    <div style="flex: 1; min-width: 20px; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;">
                        <div style="width: {mean_val}%; height: 100%; background: {summary_color};"></div>
                    </div>
                    '''
                html += '</div>'

            else:
                # Live mode: show current usage
                usage = self._current_usage.get(node.name, [])
                if not usage:
                    usage = [0.0] * node.cpu_count

                # Node name with current memory
                current_memory = self._current_memory.get(node.name, 0.0)
                html += f'<div style="margin-bottom: 4px; font-size: 12px;"><strong>{node.name}</strong> {current_memory:.0f}% mem</div>'

                n_cpus = len(usage)

                # Single row of bars - fill available width
                html += '<div style="display: flex; gap: 3px; margin-bottom: 8px; width: 100%;">'
                for i in range(n_cpus):
                    cpu_usage = usage[i]

                    # Color based on usage
                    if self.color:
                        # Color mode: green/yellow/red
                        if cpu_usage < 50:
                            color = '#4CAF50'  # green
                        elif cpu_usage < 80:
                            color = '#FFC107'  # yellow
                        else:
                            color = '#F44336'  # red
                    else:
                        # Default: gray only
                        color = '#666666'  # gray

                    # Progress bar
                    width_pct = min(100, max(0, cpu_usage))
                    html += f'''
                    <div style="flex: 1; min-width: 20px; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;">
                        <div style="width: {width_pct}%; height: 100%; background: {color}; transition: width 0.3s;"></div>
                    </div>
                    '''
                html += '</div>'

        html += '</div>'  # Close main container
        return html

    def _update_jupyter_display(self):
        """Update tqdm bars in Jupyter."""
        # VSCode HTML display
        if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
            try:
                from IPython.display import HTML
                html = self._generate_html_display()
                if hasattr(self, '_html_display'):
                    self._html_display.update(HTML(html))
                return
            except:
                pass  # Fall through to tqdm

        if not self._tqdm_bars:
            return

        bar_idx = 0
        for node in self.nodes:
            usage = self._current_usage.get(node.name, [])

            for i, cpu_usage in enumerate(usage):
                if bar_idx < len(self._tqdm_bars):
                    bar = self._tqdm_bars[bar_idx]
                    # Update value and percentage
                    bar.n = cpu_usage
                    bar.set_postfix_str(f"{cpu_usage:.1f}%")
                    bar.refresh()
                    bar_idx += 1

        # Force display update for VSCode
        if self._has_ipython_display:
            try:
                from IPython.display import display
                # This helps VSCode update the display
                pass
            except:
                pass

    def _print_summary(self):
        """Print summary statistics."""
        if not self.show_summary:
            return

        # Don't print text summary if using HTML display in Jupyter/VSCode
        if self.is_jupyter and self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
            # HTML summary will be shown instead
            return

        print("\n" + "=" * 60)
        print("CPU Monitoring Summary")
        print("=" * 60)

        for node in self.nodes:
            stats = self._stats[node.name]
            summary = stats.get_summary()

            if not summary:
                continue

            print(f"\nNode: {node.name}")
            print(f"  Duration: {summary['duration']:.1f}s")
            print(f"  Overall: mean={summary['overall_mean']:.1f}%, "
                  f"max={summary['overall_max']:.1f}%, "
                  f"min={summary['overall_min']:.1f}%")
            print(f"  CPU-seconds: {summary['cpu_seconds']:.1f}")

            # Memory stats if available
            if 'memory_mean' in summary:
                print(f"  Memory: mean={summary['memory_mean']:.1f}%, "
                      f"max={summary['memory_max']:.1f}%, "
                      f"min={summary['memory_min']:.1f}%")

            # Per-core stats
            print(f"  Per-core averages:")
            mean_per_core = summary['mean_per_core']
            for i, mean_usage in enumerate(mean_per_core):
                max_usage = summary['max_per_core'][i]
                min_usage = summary['min_per_core'][i]
                print(f"    CPU {i:2d}: mean={mean_usage:5.1f}%, "
                      f"max={max_usage:5.1f}%, min={min_usage:5.1f}%")

        print("=" * 60)

    def start(self):
        """Start monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True

        # Get initial CPU sample before starting display
        try:
            import socket
            current_hostname = _clean_hostname(socket.gethostname())
            our_node = None
            for node in self.nodes:
                if node.name == current_hostname or node.process_id == 0:
                    our_node = node
                    break

            if our_node:
                # Get initial CPU reading
                initial_usage = psutil.cpu_percent(interval=0.1, percpu=True)
                if our_node.allocated_cpus:
                    try:
                        initial_usage = [initial_usage[i] for i in our_node.allocated_cpus
                                        if i < len(initial_usage)]
                    except IndexError:
                        pass
                self._current_usage[our_node.name] = initial_usage

                # Get initial memory reading
                initial_memory = psutil.virtual_memory().percent
                self._current_memory[our_node.name] = initial_memory
        except Exception as e:
            logger.debug(f"Could not get initial CPU sample: {e}")

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start display
        if self.is_jupyter:
            self._create_jupyter_display()

            # Start update loop in background
            def jupyter_update_loop():
                while self._monitoring:
                    self._update_jupyter_display()
                    time.sleep(self.update_interval)

            self._jupyter_update_thread = threading.Thread(target=jupyter_update_loop, daemon=True)
            self._jupyter_update_thread.start()
        else:
            # Terminal - use rich Live
            self._live = Live(self._create_terminal_display(),
                            console=self._console,
                            refresh_per_second=1.0/self.update_interval)
            self._live.start()

            # Update loop
            def terminal_update_loop():
                while self._monitoring:
                    if self._live:
                        self._live.update(self._create_terminal_display())
                    time.sleep(self.update_interval)

            self._terminal_update_thread = threading.Thread(target=terminal_update_loop, daemon=True)
            self._terminal_update_thread.start()

    def stop(self):
        """Stop monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        # Wait for monitor thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Wait for jupyter update thread
        if self._jupyter_update_thread:
            self._jupyter_update_thread.join(timeout=2.0)

        # Stop display
        if self._live:
            self._live.stop()
            self._live = None

        # Close tqdm bars
        for bar in self._tqdm_bars:
            bar.close()
        self._tqdm_bars = []

        # Handle display based on persist and summary_table settings
        show_final_display = self.persist or self.summary_table

        if show_final_display:
            # Show summary (bars or table)
            if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
                if hasattr(self, '_html_display'):
                    from IPython.display import HTML
                    summary_html = self._generate_html_display(summary_mode=True)
                    self._html_display.update(HTML(summary_html))
            # Print summary (will be skipped for VSCode HTML)
            self._print_summary()
        else:
            # Clear display if not persisting
            if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
                if hasattr(self, '_html_display'):
                    try:
                        from IPython.display import clear_output
                        clear_output(wait=False)
                    except:
                        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# ============================================================================
# Decorator
# ============================================================================

def monitor_cpu(func: Callable = None, **monitor_kwargs):
    """
    Decorator for CPU monitoring.

    Parameters
    ----------
    func : callable
        Function to monitor
    **monitor_kwargs
        Arguments passed to CPUMonitor

    Examples
    --------
    >>> @monitor_cpu
    >>> def my_function():
    ...     # computation
    ...     pass

    >>> @monitor_cpu(width=100, update_interval=1.0)
    >>> def my_function():
    ...     # computation
    ...     pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with CPUMonitor(**monitor_kwargs):
                return f(*args, **kwargs)
        return wrapper

    if func is None:
        # Called with arguments: @monitor_cpu(...)
        return decorator
    else:
        # Called without arguments: @monitor_cpu
        return decorator(func)


# ============================================================================
# IPython Magic
# ============================================================================

try:
    from IPython.core.magic import Magics, magics_class, cell_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

    @magics_class
    class CPUMonitorMagics(Magics):
        """IPython magics for CPU monitoring."""

        @cell_magic
        @magic_arguments()
        @argument('--width', '-w', type=int, default=None,
                 help='Display width in characters')
        @argument('--interval', '-i', type=float, default=0.5,
                 help='Update interval in seconds')
        @argument('--persist', '-p', action='store_true',
                 help='Keep display visible after completion')
        @argument('--color', '-c', action='store_true',
                 help='Use color coding (green/yellow/red). Default is gray only.')
        @argument('--summary', '-s', action='store_true',
                 help='Show summary table with mean CPU usage per core (mean/max memory shown next to node name)')
        def usage(self, line, cell):
            """
            Monitor CPU usage during cell execution.

            Usage:
                %%usage
                # your code here

                %%usage --width 100 --interval 1.0
                # your code here

                %%usage --persist
                # display remains after completion

                %%usage --color
                # use color coding (green/yellow/red)

                %%usage --summary
                # show table with CPU and memory statistics
            """
            args = parse_argstring(self.usage, line)

            with CPUMonitor(width=args.width, update_interval=args.interval,
                          persist=args.persist, color=args.color, summary_table=args.summary):
                # Execute cell
                self.shell.run_cell(cell)

    HAS_IPYTHON_MAGIC = True

except ImportError:
    HAS_IPYTHON_MAGIC = False
    CPUMonitorMagics = None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CPUMonitor',
    'monitor_cpu',
    'CPUMonitorMagics',
    'detect_compute_nodes',
    'get_cached_nodes',
    'is_jupyter',
    'is_vscode',
]
