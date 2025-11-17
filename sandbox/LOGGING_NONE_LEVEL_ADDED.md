# NONE Logging Level Implementation

**Date:** 2025-11-08
**Status:** ✅ Complete

## Summary

Added `NONE` logging level that completely disables all logging output (Python and C) to help diagnose notebook rendering issues.

## Problem

The `simple_example.ipynb` notebook was displaying a gray background and failing to show outputs after scrolling. This happened despite:
- Default WARNING logging level producing no output
- C/C++ logging properly filtered
- No ANSI color codes in output
- Old error outputs cleared

## Solution

Added a `NONE` logging level that sets the log level higher than `CRITICAL`, ensuring absolutely no logging output is produced from any source.

## Implementation

### 1. Added NONE Level Constant (`src/phasic/logging_config.py`)

```python
# Define NONE level - higher than CRITICAL to disable all logging
NONE = logging.CRITICAL + 10
logging.addLevelName(NONE, 'NONE')
```

### 2. Updated Level Parsing

Both `setup_logging()` and `set_log_level()` now handle `'NONE'` as a special case:

```python
if level.upper() == 'NONE':
    numeric_level = NONE
else:
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
```

### 3. Updated `disable_logging()` Function

Simplified to use the new NONE level:

```python
def disable_logging() -> None:
    """
    Disable all phasic logging.

    Useful for testing or when you want complete silence.
    Equivalent to set_log_level('NONE').
    """
    set_log_level('NONE')
```

### 4. Updated Documentation

- Docstrings updated to mention NONE level
- Environment variable docs updated
- Examples added for disabling logging

### 5. Updated Notebook

Modified `docs/pages/tutorials/simple_example.ipynb` to use NONE logging:

```python
import phasic
from phasic.state_indexing import Property, StateSpace
from phasic.logging_config import set_log_level
set_log_level('NONE')

import numpy as np
```

## Usage

### Method 1: set_log_level()
```python
from phasic.logging_config import set_log_level
set_log_level('NONE')
```

### Method 2: disable_logging()
```python
from phasic.logging_config import disable_logging
disable_logging()
```

### Method 3: Environment Variable
```bash
export PHASIC_LOG_LEVEL=NONE
python my_script.py
```

### Method 4: In Notebook
```python
import phasic
from phasic.logging_config import set_log_level
set_log_level('NONE')
```

## Testing

All tests passed:

✅ **Test 1:** Default WARNING level - 0 bytes output
✅ **Test 2:** NONE via set_log_level() - 0 bytes output
✅ **Test 3:** NONE via environment variable - 0 bytes stderr
✅ **Test 4:** disable_logging() function - 0 bytes output

## How It Works

### Python Logging

When NONE level is set:
1. Python logger level is set to `CRITICAL + 10`
2. All log messages (DEBUG, INFO, WARNING, ERROR, CRITICAL) are below this threshold
3. Python's logging module filters them all out before any output

### C Logging

When NONE level is set:
1. `set_log_level()` calls `phasic_pybind._c_log_set_level(NONE)`
2. C code checks: `if (level < g_log_level) return;` (line 39 in phasic_log.c)
3. All C log messages are filtered before formatting or callback

### Complete Silence

With NONE level:
- **No stderr output** (Python or C)
- **No stdout pollution**
- **No performance overhead** (early exit in filter)
- **Both Python and C** logging disabled simultaneously

## Available Logging Levels

From most to least verbose:

1. `DEBUG` - Detailed diagnostic information
2. `INFO` - General informational messages
3. `WARNING` - Warnings (default level)
4. `ERROR` - Error messages
5. `CRITICAL` - Critical failures
6. `NONE` - No output (complete silence)

## Files Modified

1. `src/phasic/logging_config.py` - Added NONE level and updated functions
2. `docs/pages/tutorials/simple_example.ipynb` - Set NONE logging at start

## Next Steps

1. **Test notebook rendering** - Reload notebook in IDE and verify gray background issue is resolved
2. **If issue persists** - Problem is unrelated to logging (IDE, kernel, or other system issue)
3. **If issue resolved** - Consider whether default should be NONE for notebooks (currently WARNING)

## Recommendation

If the notebook still has gray background issues after this change, the problem is **not related to logging** and is likely:
- IDE rendering bug
- Jupyter kernel issue
- Memory/buffer issue in notebook server
- Display driver or graphics issue

In that case, recommend:
- Restart VS Code / Jupyter Lab entirely
- Clear notebook metadata: `jupyter nbconvert --clear-output --inplace notebook.ipynb`
- Check for IDE/extension updates
- Try different notebook viewer (VS Code vs Jupyter Lab vs classic Jupyter)

---

**Test in Notebook:**

1. Close and reopen `simple_example.ipynb`
2. Restart kernel
3. Run cells one by one
4. Check for gray background after scrolling
5. Report results

If gray background persists with NONE logging, we can definitively rule out logging as the cause.
