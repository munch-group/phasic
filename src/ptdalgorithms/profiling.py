"""
Profiling utilities for analyzing SVGD performance.

Functions for analyzing pstats.Stats objects from Graph.svgd() calls
to understand computation vs JAX overhead distribution.
"""

import pstats
from typing import Dict, List, Tuple, Optional
import re


def analyze_svgd_profile(stats: pstats.Stats,
                         top_n: int = 20,
                         print_report: bool = True) -> Dict:
    """
    Analyze profiling data from Graph.svgd() to show computation vs JAX overhead.

    This function categorizes function calls into:
    - Core computation (model evaluation, gradient computation)
    - JAX overhead (JIT compilation, dispatch, tracing)
    - SVGD algorithm (kernel, updates, particles)
    - Other overhead

    Args:
        stats: pstats.Stats object from profiling a Graph.svgd() call
        top_n: Number of top functions to show in each category
        print_report: If True, print a formatted report to stdout

    Returns:
        Dictionary with keys:
            - 'total_time': Total cumulative time
            - 'computation_pct': Percentage spent on core computation
            - 'jax_overhead_pct': Percentage spent on JAX operations
            - 'svgd_pct': Percentage spent on SVGD algorithm
            - 'other_pct': Percentage spent on other operations
            - 'categories': Dict mapping category name to list of (function, time, pct)
            - 'jax_breakdown': Detailed breakdown of JAX overhead types

    Example:
        >>> import cProfile
        >>> import pstats
        >>> from ptdalgorithms import Graph
        >>>
        >>> # Profile SVGD
        >>> profiler = cProfile.Profile()
        >>> profiler.enable()
        >>> results = Graph.svgd(model, data, theta_dim=2, n_iterations=100)
        >>> profiler.disable()
        >>>
        >>> # Analyze
        >>> stats = pstats.Stats(profiler)
        >>> analysis = analyze_svgd_profile(stats)
    """

    # Extract function statistics
    func_stats = []
    total_time = 0.0

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        # func is (filename, line, function_name)
        filename, line, func_name = func

        # Use cumulative time as the metric
        time_spent = ct
        total_time += time_spent

        func_stats.append({
            'filename': filename,
            'line': line,
            'name': func_name,
            'cumtime': ct,
            'tottime': tt,
            'ncalls': nc,
        })

    # Sort by cumulative time
    func_stats.sort(key=lambda x: x['cumtime'], reverse=True)

    # Categorize functions
    categories = {
        'computation': [],      # Model evaluation, PDF/PMF computation
        'jax_jit': [],         # JIT compilation
        'jax_dispatch': [],    # JAX dispatch and execution
        'jax_tracing': [],     # JAX tracing and transformation
        'jax_vmap': [],        # vmap/pmap vectorization overhead
        'jax_grad': [],        # Gradient computation overhead
        'jax_array': [],       # Array creation and conversion
        'jax_primitive': [],   # Primitive operations
        'jax_callback': [],    # pure_callback and FFI overhead
        'jax_other': [],       # Other JAX overhead
        'svgd_kernel': [],     # SVGD kernel computation
        'svgd_update': [],     # SVGD particle updates
        'svgd_other': [],      # Other SVGD operations
        'numpy': [],           # NumPy operations
        'other': [],           # Everything else
    }

    for func in func_stats:
        name = func['name']
        filename = func['filename']
        cumtime = func['cumtime']

        # Skip trivial functions (< 0.1% of total time)
        if total_time > 0 and cumtime / total_time < 0.001:
            continue

        categorized = False

        # Core computation patterns
        if any(pattern in name.lower() for pattern in [
            'pdf', 'pmf', 'compute_pmf', 'forward_algorithm',
            'build_model', 'evaluate', 'likelihood'
        ]):
            categories['computation'].append(func)
            categorized = True

        # JAX JIT compilation
        elif any(pattern in name for pattern in [
            'compile', 'jit_compile', '_compile_', 'lower',
            'xla_compile', 'backend_compile'
        ]) or 'jax' in filename and 'compile' in name.lower():
            categories['jax_jit'].append(func)
            categorized = True

        # JAX dispatch
        elif any(pattern in name for pattern in [
            'dispatch', '_dispatch_', 'bind', 'apply_primitive',
            'call_wrapped', 'core.call'
        ]) or ('jax' in filename and 'dispatch' in name.lower()):
            categories['jax_dispatch'].append(func)
            categorized = True

        # JAX tracing
        elif any(pattern in name for pattern in [
            'trace', 'tracer', 'make_jaxpr', 'trace_to_jaxpr',
            'dynamic_trace', 'eval_jaxpr'
        ]) or ('jax' in filename and 'trace' in name.lower()):
            categories['jax_tracing'].append(func)
            categorized = True

        # JAX vmap/pmap
        elif any(pattern in name for pattern in [
            'vmap', 'pmap', 'vectorize', 'batching', '_batch',
            'batch_jaxpr', 'batch_subtrace'
        ]) or ('jax' in filename and any(p in name.lower() for p in ['vmap', 'pmap', 'batch'])):
            categories['jax_vmap'].append(func)
            categorized = True

        # JAX gradient
        elif any(pattern in name for pattern in [
            'grad', 'vjp', 'jvp', 'custom_vjp', 'defvjp',
            'backward', 'tangent', 'cotangent', 'autodiff'
        ]) or ('jax' in filename and any(p in name.lower() for p in ['grad', 'vjp', 'jvp'])):
            categories['jax_grad'].append(func)
            categorized = True

        # JAX array operations
        elif any(pattern in name for pattern in [
            'Array', 'ArrayImpl', 'make_array', 'device_put',
            'device_get', '_convert', 'asarray', 'array_copy',
            'copy_to_device', '_array_from_'
        ]) or ('jax' in filename and any(p in name.lower() for p in ['array', 'device'])):
            categories['jax_array'].append(func)
            categorized = True

        # JAX primitive operations
        elif any(pattern in name for pattern in [
            'standard_primitive', 'call_primitive', 'custom_primitive',
            'Primitive', 'def_impl', 'def_abstract_eval',
            'concatenate_p', 'dot_general_p', 'reduce_sum_p'
        ]) or ('jax' in filename and 'primitive' in name.lower()):
            categories['jax_primitive'].append(func)
            categorized = True

        # JAX callback/FFI
        elif any(pattern in name for pattern in [
            'pure_callback', 'io_callback', 'callback',
            'host_callback', 'call_host', '_callback_',
            'ffi', 'foreign'
        ]) or ('jax' in filename and any(p in name.lower() for p in ['callback', 'ffi'])):
            categories['jax_callback'].append(func)
            categorized = True

        # Other JAX
        elif 'jax' in filename or any(pattern in name for pattern in [
            'jax.', '_jax_', 'xla_', 'pjit'
        ]):
            categories['jax_other'].append(func)
            categorized = True

        # SVGD kernel
        elif any(pattern in name.lower() for pattern in [
            'kernel', 'rbf', 'bandwidth', 'gram'
        ]):
            categories['svgd_kernel'].append(func)
            categorized = True

        # SVGD updates
        elif any(pattern in name.lower() for pattern in [
            'update_particles', 'svgd_step', 'phi'
        ]):
            categories['svgd_update'].append(func)
            categorized = True

        # SVGD other
        elif 'svgd' in name.lower() or 'svgd.py' in filename:
            categories['svgd_other'].append(func)
            categorized = True

        # NumPy
        elif 'numpy' in filename or name.startswith('np.'):
            categories['numpy'].append(func)
            categorized = True

        # Other
        if not categorized:
            categories['other'].append(func)

    # Calculate percentages
    def sum_category(cat_list):
        return sum(f['cumtime'] for f in cat_list)

    computation_time = sum_category(categories['computation'])
    jax_time = (sum_category(categories['jax_jit']) +
                sum_category(categories['jax_dispatch']) +
                sum_category(categories['jax_tracing']) +
                sum_category(categories['jax_vmap']) +
                sum_category(categories['jax_grad']) +
                sum_category(categories['jax_array']) +
                sum_category(categories['jax_primitive']) +
                sum_category(categories['jax_callback']) +
                sum_category(categories['jax_other']))
    svgd_time = (sum_category(categories['svgd_kernel']) +
                 sum_category(categories['svgd_update']) +
                 sum_category(categories['svgd_other']))
    numpy_time = sum_category(categories['numpy'])
    other_time = sum_category(categories['other'])

    if total_time == 0:
        total_time = 1.0  # Avoid division by zero

    # JAX overhead breakdown
    jax_breakdown = {
        'JIT compilation': sum_category(categories['jax_jit']),
        'Dispatch': sum_category(categories['jax_dispatch']),
        'Tracing': sum_category(categories['jax_tracing']),
        'vmap/pmap': sum_category(categories['jax_vmap']),
        'Gradient (AD)': sum_category(categories['jax_grad']),
        'Array ops': sum_category(categories['jax_array']),
        'Primitives': sum_category(categories['jax_primitive']),
        'Callbacks/FFI': sum_category(categories['jax_callback']),
        'Other JAX': sum_category(categories['jax_other']),
    }

    # Prepare result
    result = {
        'total_time': total_time,
        'computation_time': computation_time,
        'computation_pct': 100 * computation_time / total_time,
        'jax_overhead_time': jax_time,
        'jax_overhead_pct': 100 * jax_time / total_time,
        'svgd_time': svgd_time,
        'svgd_pct': 100 * svgd_time / total_time,
        'numpy_time': numpy_time,
        'numpy_pct': 100 * numpy_time / total_time,
        'other_time': other_time,
        'other_pct': 100 * other_time / total_time,
        'categories': categories,
        'jax_breakdown': jax_breakdown,
    }

    # Print report
    if print_report:
        print("=" * 80)
        print("SVGD Performance Profile Analysis")
        print("=" * 80)
        print()
        print(f"Total time: {total_time:.3f}s")
        print()

        print("Time Distribution:")
        print("-" * 80)
        print(f"  Core Computation:    {computation_time:8.3f}s  ({result['computation_pct']:5.1f}%)")
        print(f"  JAX Overhead:        {jax_time:8.3f}s  ({result['jax_overhead_pct']:5.1f}%)")
        print(f"  SVGD Algorithm:      {svgd_time:8.3f}s  ({result['svgd_pct']:5.1f}%)")
        print(f"  NumPy Operations:    {numpy_time:8.3f}s  ({result['numpy_pct']:5.1f}%)")
        print(f"  Other:               {other_time:8.3f}s  ({result['other_pct']:5.1f}%)")
        print()

        if jax_time > 0:
            print("JAX Overhead Breakdown:")
            print("-" * 80)
            for name, time in sorted(jax_breakdown.items(),
                                    key=lambda x: x[1], reverse=True):
                if time > 0:
                    pct = 100 * time / jax_time
                    pct_total = 100 * time / total_time
                    print(f"  {name:20s} {time:8.3f}s  "
                          f"({pct:5.1f}% of JAX, {pct_total:5.1f}% total)")
            print()

        # Top functions in each major category
        def print_top_functions(title, func_list, n=top_n):
            if not func_list:
                return

            print(f"{title}:")
            print("-" * 80)

            for i, func in enumerate(func_list[:n]):
                pct = 100 * func['cumtime'] / total_time
                ncalls = func['ncalls']

                # Format function name
                name = func['name']
                if len(name) > 50:
                    name = name[:47] + "..."

                print(f"  {i+1:2d}. {name:50s} {func['cumtime']:8.3f}s  "
                      f"({pct:5.1f}%)  [{ncalls:>8d} calls]")
            print()

        print_top_functions("Top Computation Functions",
                          categories['computation'], n=min(top_n, 10))

        if categories['jax_jit']:
            print_top_functions("Top JAX JIT Functions",
                              categories['jax_jit'], n=min(top_n, 5))

        if categories['jax_dispatch']:
            print_top_functions("Top JAX Dispatch Functions",
                              categories['jax_dispatch'], n=min(top_n, 5))

        if categories['jax_vmap']:
            print_top_functions("Top JAX vmap/pmap Functions",
                              categories['jax_vmap'], n=min(top_n, 5))

        if categories['jax_grad']:
            print_top_functions("Top JAX Gradient Functions",
                              categories['jax_grad'], n=min(top_n, 5))

        if categories['jax_array']:
            print_top_functions("Top JAX Array Functions",
                              categories['jax_array'], n=min(top_n, 5))

        if categories['jax_primitive']:
            print_top_functions("Top JAX Primitive Functions",
                              categories['jax_primitive'], n=min(top_n, 5))

        if categories['jax_callback']:
            print_top_functions("Top JAX Callback/FFI Functions",
                              categories['jax_callback'], n=min(top_n, 5))

        print_top_functions("Top SVGD Functions",
                          categories['svgd_kernel'] +
                          categories['svgd_update'] +
                          categories['svgd_other'],
                          n=min(top_n, 10))

        # Add recommendations based on breakdown
        print("Optimization Recommendations:")
        print("-" * 80)

        if jax_breakdown['Callbacks/FFI'] / total_time > 0.15:
            print("  • HIGH FFI/Callback overhead ({:.1f}%):".format(
                100 * jax_breakdown['Callbacks/FFI'] / total_time))
            print("    - Consider batching pure_callback operations")
            print("    - Check if C++ model building can be cached")
            print("    - Profile C++ code separately to find bottlenecks")
            print()

        if jax_breakdown['Array ops'] / total_time > 0.10:
            print("  • HIGH array conversion overhead ({:.1f}%):".format(
                100 * jax_breakdown['Array ops'] / total_time))
            print("    - Avoid unnecessary device transfers")
            print("    - Use jax.numpy arrays consistently")
            print("    - Check for numpy/jax array conversions in hot loops")
            print()

        if jax_breakdown['vmap/pmap'] / total_time > 0.10:
            print("  • HIGH vmap/pmap overhead ({:.1f}%):".format(
                100 * jax_breakdown['vmap/pmap'] / total_time))
            print("    - vmap overhead is normal for vectorized operations")
            print("    - Consider using explicit loops if vmap is over small batches")
            print("    - Check if batch size can be increased")
            print()

        if jax_breakdown['Gradient (AD)'] / total_time > 0.20:
            print("  • HIGH gradient computation overhead ({:.1f}%):".format(
                100 * jax_breakdown['Gradient (AD)'] / total_time))
            print("    - Consider using forward-mode AD for high-dimensional outputs")
            print("    - Check if custom_vjp can simplify gradient computation")
            print("    - Profile gradient function separately")
            print()

        if jax_breakdown['Dispatch'] / total_time > 0.15:
            print("  • HIGH dispatch overhead ({:.1f}%):".format(
                100 * jax_breakdown['Dispatch'] / total_time))
            print("    - Try using larger batch sizes to amortize dispatch")
            print("    - Consider fusing operations with jax.jit")
            print("    - Check for excessive function calls in hot loops")
            print()

        if not any([
            jax_breakdown['Callbacks/FFI'] / total_time > 0.15,
            jax_breakdown['Array ops'] / total_time > 0.10,
            jax_breakdown['vmap/pmap'] / total_time > 0.10,
            jax_breakdown['Gradient (AD)'] / total_time > 0.20,
            jax_breakdown['Dispatch'] / total_time > 0.15,
        ]):
            print("  ✓ No major JAX overhead bottlenecks detected")
            print("    Profile looks good for this workload size.")
            print()

        print("=" * 80)

    return result


def profile_svgd(model, observed_data, **svgd_kwargs):
    """
    Convenience function to profile a Graph.svgd() call.

    Args:
        model: JAX-compatible model function
        observed_data: Observed data for inference
        **svgd_kwargs: Keyword arguments to pass to Graph.svgd()

    Returns:
        Tuple of (svgd_results, profile_analysis)

    Example:
        >>> from ptdalgorithms import Graph
        >>> from ptdalgorithms.profiling import profile_svgd
        >>>
        >>> # Build model
        >>> graph = build_my_model()
        >>> model = Graph.pmf_from_graph(graph, discrete=False)
        >>>
        >>> # Profile SVGD
        >>> results, analysis = profile_svgd(
        ...     model,
        ...     observed_data,
        ...     theta_dim=2,
        ...     n_particles=100,
        ...     n_iterations=200
        ... )
        >>>
        >>> print(f"Computation: {analysis['computation_pct']:.1f}%")
        >>> print(f"JAX overhead: {analysis['jax_overhead_pct']:.1f}%")
    """
    import cProfile
    from ..import Graph

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        results = Graph.svgd(model, observed_data, **svgd_kwargs)
    finally:
        profiler.disable()

    stats = pstats.Stats(profiler)
    analysis = analyze_svgd_profile(stats)

    return results, analysis
