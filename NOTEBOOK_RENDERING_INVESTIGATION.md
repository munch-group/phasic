# Notebook Rendering Issue Investigation

**Date:** 2025-11-08
**Status:** 🔍 Investigating - Likely NOT a logging issue

## Problem

The `simple_example.ipynb` notebook displays a **gray background** and fails to show outputs. The issue appears **"as soon as I scroll in the notebook"**, not during cell execution.

## Investigation Steps Taken

### 1. ✅ Cleared Old Error Outputs
- Found KeyboardInterrupt errors from previous runs (8985 bytes)
- Cleared all cell outputs
- **Result:** Problem persists

### 2. ✅ Disabled Colored Logging
- Changed default from auto-detect to `colors=False`
- Prevents ANSI escape codes in output
- **Result:** Problem persists

### 3. ✅ Set Logging to NONE Level
- Added `NONE` level (higher than CRITICAL)
- Completely disables all logging output (Python + C)
- **Result:** Problem persists

### 4. ✅ Skipped Logging Setup Entirely
- Added `PHASIC_SKIP_LOGGING_SETUP` environment variable
- Prevents any handler/formatter creation
- **Result:** Awaiting test...

## Current Notebook Configuration

```python
import os
os.environ['PHASIC_SKIP_LOGGING_SETUP'] = '1'

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np
```

This configuration:
- ✅ Skips all logging setup
- ✅ No handlers created
- ✅ No formatters instantiated
- ✅ No C logging callback registered
- ✅ Zero overhead from logging infrastructure

## Key Observation

**The gray background appears "as soon as I scroll", not during cell execution.**

This suggests:
- ❌ NOT a code execution issue
- ❌ NOT a logging output issue
- ❌ NOT a stderr/stdout problem
- ✅ Likely an **IDE rendering/display issue**

## Possible Non-Logging Causes

### 1. IDE Rendering Bug
**Symptoms:**
- Gray background on scroll
- Outputs don't display
- Happens after cells run successfully

**Solutions:**
```bash
# Restart VS Code entirely
killall "Visual Studio Code"

# Or use Command Palette
Cmd+Shift+P → "Reload Window"
```

### 2. Jupyter Extension Issue
**Solutions:**
```bash
# Update VS Code Jupyter extension
# Go to Extensions → Jupyter → Update

# Or try alternative notebook interface
jupyter lab docs/pages/tutorials/simple_example.ipynb
```

### 3. Corrupted Notebook Metadata
**Solutions:**
```bash
# Strip all metadata
jupyter nbconvert --clear-output --inplace docs/pages/tutorials/simple_example.ipynb

# Or use nbstripout
pip install nbstripout
nbstripout docs/pages/tutorials/simple_example.ipynb
```

### 4. Kernel State Issue
**Solutions:**
```python
# In notebook, run:
%reset -f  # Reset all variables

# Or restart kernel:
# Kernel → Restart Kernel
```

### 5. Display/Graphics Driver
**Symptoms:**
- Visual artifacts
- Rendering glitches
- Gray overlays

**Solutions:**
```bash
# macOS: Reset display preferences
defaults delete com.apple.dock; killall Dock

# Update macOS and VS Code
```

### 6. Cell Output Limits
**Check:**
```bash
# VS Code settings.json
{
  "notebook.output.textLineLimit": 30000,
  "notebook.output.wordWrap": true
}
```

### 7. Python/IPython Matplotlib Backend
**Check if matplotlib is interfering:**
```python
# Add to notebook
%matplotlib inline
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## Diagnostic Tests

### Test 1: Does the issue occur in different viewers?

```bash
# Try Jupyter Lab
jupyter lab docs/pages/tutorials/simple_example.ipynb

# Try classic Jupyter
jupyter notebook docs/pages/tutorials/simple_example.ipynb

# Try VS Code web version
```

**If issue only in VS Code:** VS Code/extension bug
**If issue everywhere:** Notebook file corruption

### Test 2: Does a minimal notebook work?

Create `test_minimal.ipynb`:
```python
# Cell 1
print("Hello")

# Cell 2
import phasic
print(f"Phasic version: {phasic.__version__}")

# Cell 3
import numpy as np
for i in range(100):
    print(i)
```

**If minimal notebook works:** Issue specific to `simple_example.ipynb`
**If minimal notebook breaks:** System-wide rendering issue

### Test 3: Check browser console (for Jupyter Lab)

```bash
# Open Jupyter Lab
jupyter lab

# Open browser dev tools: F12
# Check Console for JavaScript errors
# Check Network tab for failed requests
```

### Test 4: Check VS Code logs

```
# View → Output → Select "Jupyter" or "Python"
# Look for errors when scrolling
```

### Test 5: Create fresh notebook

```bash
# Copy cells to new file
cp simple_example.ipynb simple_example_backup.ipynb

# Create new notebook in IDE
# Manually copy cell contents (not outputs)
# Run cells one by one
```

## Code-Level Checks

### No Side Effects in Logging Code

✅ **Checked:** All logging calls use parameterized format strings
✅ **Checked:** C logging has early exit before callback
✅ **Checked:** No print() statements in critical paths
✅ **Checked:** No expensive computations before level checks

### Potential Side Effects (Ruled Out)

- ❌ String formatting (proper parameterized logging)
- ❌ C callback overhead (level checked in C before callback)
- ❌ Handler processing (skipped with PHASIC_SKIP_LOGGING_SETUP)
- ❌ Formatter instantiation (skipped with PHASIC_SKIP_LOGGING_SETUP)

## Next Steps

### If SKIP_LOGGING_SETUP Fixes It
Then it was a **handler/formatter side effect**, and we need to investigate:
- StreamHandler interaction with Jupyter
- Formatter string processing
- Logger hierarchy setup

### If SKIP_LOGGING_SETUP Doesn't Fix It
Then it's **NOT a phasic code issue**, and you should:
1. Try diagnostic tests above
2. Check IDE/extension versions
3. Report to VS Code Jupyter extension: https://github.com/microsoft/vscode-jupyter/issues
4. Try alternative notebook interface (Jupyter Lab)

## Files Modified

1. `src/phasic/__init__.py` - Added PHASIC_SKIP_LOGGING_SETUP check
2. `src/phasic/logging_config.py` - Added NONE level
3. `docs/pages/tutorials/simple_example.ipynb` - Set SKIP_LOGGING_SETUP

## Recommendation

**Try the notebook now** with `PHASIC_SKIP_LOGGING_SETUP=1`:

1. Close and reopen notebook
2. Restart kernel
3. Run all cells
4. Scroll through notebook
5. Report if gray background still appears

If issue persists, **it's definitively not a phasic logging issue** - it's an IDE/rendering/system problem that requires different debugging approaches.

---

## Alternative: Use Plain Python Script

If notebook continues to have issues:

```bash
# Convert notebook to script
jupyter nbconvert --to script simple_example.ipynb

# Edit simple_example.py to remove magic commands

# Run as script
python simple_example.py

# Or use IPython
ipython simple_example.py
```

This avoids notebook rendering entirely while preserving code functionality.
