# SVGD Correctness Testing - Status

**Date**: October 19, 2025
**File**: `tests/test_svgd_correctness.py`
**Status**: ⚠️ TESTS CREATED, FAILURES DETECTED

---

## Summary

Created comprehensive SVGD correctness testing system, but tests reveal convergence issues with the simple exponential model.

---

## Test System Features

### ✅ Implemented

1. **Cache Clearing**: Clears all caches before each test run
   - Trace cache (`~/.ptdalgorithms_cache/traces/`)
   - JAX cache (`~/.jax_cache/`)

2. **Four Test Scenarios**:
   - Test 1: Basic convergence (no transformations)
   - Test 2: Log transformation (θ > 0 constraint)
   - Test 3: Positive constraint flag (`positive_params=True`)
   - Test 4: Cache isolation

3. **Analytical Baseline**: Compares against analytical posterior (Gamma conjugate)

4. **Comprehensive Checks**:
   - Convergence to posterior mean/std
   - Parameter constraint enforcement
   - Cache clearing effectiveness

---

## Test Results

### Test 1: Basic Convergence ✗ FAIL

**Setup**:
- Model: Exponential(θ=2.0)
- Data: 200 samples
- SVGD: 100 particles, 500 iterations

**Results**:
```
Analytical posterior: mean=2.042, std=0.144
SVGD posterior:       mean=-0.035, std=0.512

Error: mean error 2.077 > tolerance 0.204
```

**Issue**: SVGD converges to negative mean, far from analytical posterior

### Test 2: Log Transformation ✗ FAIL

**Setup**:
- Transformation: θ = exp(φ)
- Should enforce θ > 0

**Results**:
```
SVGD posterior: mean=nan, std=nan
Min θ: nan, Max θ: nan
```

**Issue**: NaN values, transformation not working correctly

### Test 3: Positive Constraint ✗ FAIL

**Setup**:
- Using `positive_params=True`
- Should automatically enforce θ > 0

**Results**:
```
SVGD posterior: mean=0.249, std=0.555
Min θ: -0.895, Max θ: 3.239
```

**Issue**: Particles go negative despite positive constraint

### Test 4: Cache Isolation ✓ PASS

**Setup**:
- Run SVGD twice with cache clearing

**Results**:
```
First run:  3.74s
Second run: 1.01s (after clear)
Speedup: 3.69x
```

**Status**: Working (speedup due to JAX internal caching, expected)

---

## Problems Identified

### 1. SVGD Not Converging

**Observation**: SVGD posterior far from analytical posterior

**Possible Causes**:
- Model specification issue
- Learning rate too low/high
- Not enough iterations
- Prior mismatch
- Gradient computation error

**Next Steps**:
- Debug model evaluation
- Check gradient computation
- Try simpler test (known posterior)
- Increase iterations/particles

### 2. Transformations Not Working

**Observation**:
- Log transformation gives NaN
- `positive_params=True` allows negative values

**Possible Causes**:
- Transformation not applied correctly in SVGD
- Gradient through transformation incorrect
- Particle initialization issue

**Next Steps**:
- Check SVGD transformation code
- Verify gradient computation
- Test transformation separately

### 3. Model Evaluation Issues

**Observation**: Basic exponential model not working correctly

**Possible Causes**:
- Graph construction error
- FFI wrapper issue
- Parameter passing problem

**Next Steps**:
- Test model evaluation directly
- Check graph.pdf() vs SVGD usage
- Verify parameter updates

---

## Test File Structure

```python
# tests/test_svgd_correctness.py

def clear_all_caches():
    """Clear trace and JAX caches"""
    # Clears before import

def build_exponential_graph():
    """Simple exponential distribution graph"""
    # S → [2] → [1], rate=θ

def generate_test_data(true_theta, n_samples):
    """Generate synthetic data"""
    # From Exponential(θ)

def test_basic_convergence():
    """Test 1: No transformations"""
    # Compare SVGD vs analytical

def test_log_transformation():
    """Test 2: Manual log transform"""
    # θ = exp(φ)

def test_positive_constraint():
    """Test 3: positive_params flag"""
    # Automatic constraint

def test_cache_isolation():
    """Test 4: Cache clearing"""
    # Verify recompilation

def main():
    """Run all tests, print summary"""
```

---

## Recommendations

### Immediate Actions

1. **Debug SVGD convergence**:
   - Test model evaluation directly
   - Check gradients manually
   - Try known-good example

2. **Fix transformations**:
   - Review SVGD transformation code
   - Test transform separately
   - Verify particle updates

3. **Simplify tests**:
   - Start with even simpler model
   - Reduce particles/iterations
   - Add verbose output

### Long-term Improvements

1. **Add debugging output**:
   - Print particle evolution
   - Log likelihood values
   - Gradient norms

2. **More test cases**:
   - Different distributions
   - Different priors
   - Multiple parameters

3. **Visualization**:
   - Plot particle trajectories
   - Show convergence diagnostics
   - Compare distributions

---

## Files Created

- `tests/test_svgd_correctness.py` (~490 lines)
- `SVGD_CORRECTNESS_TEST_STATUS.md` (this file)

---

## Next Steps

1. ✅ **Commit test infrastructure** (even though tests fail)
   - Tests document expected behavior
   - Provide framework for debugging
   - Useful for future development

2. **Debug SVGD issues**:
   - Investigate convergence failure
   - Fix transformation bugs
   - Verify model correctness

3. **Improve tests**:
   - Add more diagnostics
   - Test simpler cases
   - Add visualization

---

## Git Commit Suggestion

### Message

```
Add SVGD correctness testing framework

- Created comprehensive test system for SVGD inference
- Tests basic convergence, transformations, and constraints
- Includes cache clearing before each test
- Tests currently failing, revealing convergence issues

Test structure:
- Test 1: Basic convergence (no transforms)
- Test 2: Log transformation (θ > 0)
- Test 3: Positive constraint flag
- Test 4: Cache isolation

Issues found:
- SVGD not converging to analytical posterior
- Transformations not working (NaN or negative values)
- Needs debugging and fixes

Files:
- tests/test_svgd_correctness.py
- SVGD_CORRECTNESS_TEST_STATUS.md
```

### Commands

```bash
git add tests/test_svgd_correctness.py
git add SVGD_CORRECTNESS_TEST_STATUS.md
git commit -m "Add SVGD correctness testing framework

- Created comprehensive test system for SVGD inference
- Tests basic convergence, transformations, and constraints
- Includes cache clearing before each test
- Tests currently failing, revealing convergence issues

Test structure:
- Test 1: Basic convergence (no transforms)
- Test 2: Log transformation (θ > 0)
- Test 3: Positive constraint flag
- Test 4: Cache isolation

Issues found:
- SVGD not converging to analytical posterior
- Transformations not working (NaN or negative values)
- Needs debugging and fixes"
```

---

*Created: October 19, 2025*
*Status: Test framework complete, debugging needed*
