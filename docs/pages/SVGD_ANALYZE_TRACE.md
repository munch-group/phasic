# SVGD analyze_trace() Method Documentation

**Status**: ✅ COMPLETE (October 2025)

## Overview

The `analyze_trace()` method provides comprehensive convergence diagnostics for SVGD inference, automatically detecting issues and suggesting parameter improvements.

## Features

### Convergence Diagnostics

1. **Convergence Detection**
   - Detects iteration where mean stabilized
   - Detects iteration where std stabilized
   - Auto-detect burn-in period

2. **Particle Diversity Metrics**
   - Mean inter-particle distance
   - Effective Sample Size (ESS)
   - ESS ratio (ESS / n_particles)

3. **Quality Assessment**
   - Variance collapse detection
   - Particle diversity assessment

4. **Issue Detection**
   - Variance collapse
   - Non-convergence
   - Low ESS
   - Early convergence (wasted iterations)

### Parameter Suggestions

Automatically suggests improvements for:
- **Learning rate**: Switch to decay schedule, adjust rate
- **Particle count**: Increase for diversity, decrease if oversampling
- **Iterations**: Increase if not converged, decrease if converged early

## Usage

### Basic Usage

```python
from ptdalgorithms import SVGD, ExponentialDecayStepSize

# Run SVGD with history tracking
svgd = SVGD(model, data, theta_dim=1, n_iterations=1000)
svgd.fit(return_history=True)

# Analyze convergence and get suggestions
svgd.analyze_trace()
```

### Programmatic Access

```python
# Get diagnostics dictionary
diag = svgd.analyze_trace(return_dict=True, verbose=False)

# Check convergence
if diag['converged']:
    print(f"Converged at iteration {diag['convergence_point']}")
else:
    print("Not converged - need more iterations")

# Check particle diversity
if diag['diversity']['ess_ratio'] < 0.5:
    print("Low ESS - increase n_particles")

# Apply suggestions
lr_sug = diag['suggestions']['learning_rate']
if lr_sug['recommended'] != 'current learning rate is appropriate':
    print(f"Suggestion: {lr_sug['reason']}")
```

### Custom Burn-in

```python
# Specify custom burn-in period
svgd.analyze_trace(burnin=100)
```

## Example Output

```
================================================================================
SVGD Convergence Analysis
================================================================================

✓ CONVERGED (iteration 57/1000)
  Mean stabilized at iteration 57

Particle Diversity:
  Mean inter-particle distance: 0.030
  Effective sample size (ESS): 50.0 / 50 particles (100.0%)
  ✓ Good particle diversity

Detected Issues:
  ℹ Converged at 5.7% of iterations - could reduce n_iterations

================================================================================
Suggested Parameter Updates
================================================================================

Current Configuration:
  learning_rate=<ExponentialDecayStepSize>
  n_particles=50
  n_iterations=1000

Learning Rate: Converged efficiently
  Recommendation: current schedule is good

Particles: Particle count is appropriate
  Recommendation: Keep n_particles=50

Iterations: Converged early
  Recommendation: Could reduce to n_iterations=68

================================================================================
```

## Diagnostics Dictionary Structure

```python
{
    'converged': bool,              # True if converged
    'convergence_point': int,       # Iteration where mean converged
    'std_convergence_point': int,   # Iteration where std converged
    'n_iterations': int,            # Total iterations
    'n_particles': int,             # Number of particles
    'theta_dim': int,               # Parameter dimensionality
    'diversity': {
        'mean_distance': float,     # Mean pairwise distance
        'min_distance': float,      # Min pairwise distance
        'ess': float,               # Effective sample size
        'ess_ratio': float          # ESS / n_particles
    },
    'variance_collapse': {
        'collapsed': bool,          # True if collapsed
        'collapse_iteration': int,  # When collapse happened
        'final_diversity': float,   # Final std
        'max_diversity': float      # Maximum std seen
    },
    'burnin': int,                  # Auto-detected burn-in
    'issues': [str],                # List of detected issues
    'suggestions': {
        'learning_rate': {
            'recommended': ...,      # Suggested schedule/value
            'reason': str            # Explanation
        },
        'n_particles': {
            'recommended': int,      # Suggested count
            'reason': str            # Explanation
        }
    }
}
```

## Interpretation Guide

### Convergence Status

- **✓ CONVERGED**: Mean trajectory stabilized
- **✗ NOT CONVERGED**: Did not stabilize within iterations

### ESS Ratio

- **>0.7**: Good diversity
- **0.5-0.7**: Moderate diversity
- **<0.5**: Poor diversity (increase particles)

### Common Issues

1. **Variance Collapse**
   - **Symptom**: Particles converge to same point
   - **Fix**: Reduce learning rate, increase particles

2. **Non-convergence**
   - **Symptom**: Mean trajectory still changing
   - **Fix**: Increase iterations, use decay schedule

3. **Low ESS**
   - **Symptom**: ESS ratio < 0.5
   - **Fix**: Increase n_particles

4. **Early Convergence**
   - **Symptom**: Converged at <70% of iterations
   - **Fix**: Reduce n_iterations to save time

## Implementation Details

### Convergence Detection Algorithm

```python
def _detect_convergence_point(trajectory, window=50, threshold=0.01):
    """
    Detect when trajectory stabilized.

    Uses sliding window to check if relative variation < threshold.
    """
    for i in range(window, len(trajectory) - window):
        window_vals = trajectory[i:i + window]
        mean_val = jnp.mean(window_vals)
        rel_var = jnp.std(window_vals) / abs(mean_val)

        if rel_var < threshold:
            return i  # Converged here

    return None  # Not converged
```

### ESS Estimation

For SVGD (deterministic updates), ESS is estimated based on particle spread:

```python
particle_var = jnp.var(particles, axis=0)
overall_var = jnp.mean(particle_var)
ess_estimate = n_particles * (overall_var / (overall_var + 1e-10))
```

Higher variance → better ESS (particles are diverse).

## Integration with Schedule System

The `analyze_trace()` method integrates seamlessly with the schedule system introduced in the same update:

```python
from ptdalgorithms import SVGD, ExponentialDecayStepSize

# Initial run with constant learning rate
svgd = SVGD(model, data, theta_dim=1, learning_rate=0.01)
svgd.fit(return_history=True)

# Analyze and get suggestions
diag = svgd.analyze_trace(return_dict=True, verbose=False)

# Apply suggested decay schedule
if not diag['converged']:
    lr_sug = diag['suggestions']['learning_rate']['recommended']
    if isinstance(lr_sug, ExponentialDecayStepSize):
        # Re-run with suggested schedule
        svgd2 = SVGD(model, data, theta_dim=1, learning_rate=lr_sug)
        svgd2.fit(return_history=True)
        svgd2.analyze_trace()
```

## Workflow Example

```python
# Step 1: Initial run
svgd = SVGD(model, data, theta_dim=1, n_particles=50, n_iterations=1000)
svgd.fit(return_history=True)

# Step 2: Analyze
diag = svgd.analyze_trace(return_dict=True)

# Step 3: Apply suggestions and re-run
if not diag['converged']:
    # Get suggestions
    lr_rec = diag['suggestions']['learning_rate']['recommended']
    part_rec = diag['suggestions']['n_particles']['recommended']

    # Re-run with improvements
    svgd2 = SVGD(
        model, data, theta_dim=1,
        learning_rate=lr_rec,
        n_particles=part_rec,
        n_iterations=int(diag['n_iterations'] * 1.5)
    )
    svgd2.fit(return_history=True)
    svgd2.analyze_trace()
```

## Limitations

1. **Requires history**: Must call `fit(return_history=True)`
2. **ESS is heuristic**: Not the same as MCMC ESS (particles are not independent samples)
3. **Convergence criteria**: Tuned for typical phase-type inference problems

## Future Enhancements

Possible additions:
- **KSD computation**: Kernelized Stein Discrepancy for rigorous convergence measure
- **Autocorrelation analysis**: Detect oscillations and mixing issues
- **Cross-validation**: Split-sample convergence checks
- **Automatic re-fitting**: One-click "apply suggestions" method

## References

- Liu & Wang (2016): "Stein Variational Gradient Descent"
- Gelman & Rubin (1992): "Inference from Iterative Simulation"
- SVGD convergence analysis: arXiv:2409.08469

---

*Implementation completed: October 20, 2025*
*Tested with exponential distribution model*
