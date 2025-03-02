
# Debugging 

Debugging in vscode offers three debugging modes: "Python API (python)", "Python API (c++)", and "R API".

## Requirements

You need to set up a conda environment with the conda packages specified in the platform specific conda specs in binder/. E.g.:

```txt
    conda env create -f binder/env-osx-arm64.yml
```

If you cannot find you platform try `env-from-history.yml`.

### Linux

- gcc compiler
- gdb debugger

The VScode extensions:

- C/C++
- C/C++ Extensions

### Mac

- XCode command line tool

The VScode extensions:

- C/C++
- C/C++ Extensions
- CodeLLDB

## Debugging

### Python API

Debugging requires that the module compiled and installed in editable mode:

    pip install -e .

> Tip: if compiling an editable install fails, it is often easier to identifying the problem by compiling/installing with regular `pip install .`. Just remember to uninstall (`pip uninstall ptdalgorithms`) and do the `pip install -e .` before you continue debugging.
> 
Debugging in vscode offers the debugging modes "Python API (python)" and "Python API (c++)". The Python debugger runs the unit tests in tests/ and then any code in `.vscode/debug.py`.

The C++ debugger runs ...

### R API

Debugging in vscode offers a "R API" debugging mode. You need to have an open R file from tests/ when launching the debugger.

