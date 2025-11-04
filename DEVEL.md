
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

   pip install -e ".[jax]" --force-reinstall

> Tip: if compiling an editable install fails, it is often easier to identifying the problem by compiling/installing with regular `pip install .`. Just remember to uninstall (`pip uninstall phasic`) and do the `pip install -e .` before you continue debugging.
> 
Debugging in vscode offers the debugging modes "Python API (python)" and "Python API (c++)". The Python debugger runs the unit tests in tests/ and then any code in `.vscode/debug.py`.

The C++ debugger runs ...

### R API

Debugging in vscode offers a "R API" debugging mode. You need to have an open R file from tests/ when launching the debugger.

# Conda package

    pixi run bump-version
    pixi run github-release




# TODO

- clean up obsolete code
- inline comments in all source code
- all new doc strings
- pybind11 docs
  
- new documentation pages

Now we need to clean up code made obsolete by the new unified edge handling an by the new unified
  elimination / trace / caching approach 


I intend to implement an approach that integrates SCC computation, caching and trace recording. 

# phase 1
For a safer migration, I would like to fist change the existing c codebase so that:

- Code involved in producing a trace from a graph is isolated in the trace module (src/c/trace folder) that this only interface with the remaining codebase like this "trace = trace_from_graph(graph)". All SCC computation, elimination, serialization 
-  different backends fit the intended interface to trace computation. 

That way I can test that first, develop new approach, drop it in and iteratively test old vs new approach. I think it would make the interface be a function called trace_from_graph that takes an original graph and returns a non-rewarded trace. reward transformation should then be computed using the returned trace. The implementation of trace_from_graph and all SCC and caching related code should be in src/c/trace. To drop in a new trace backend, I would put the new implementation in another folder beside src/c/trace and just call trace_from_graph_new_version instead of trace_from_graph. Please make a plan for how to refactor the current code to accommodate this interface




Make a plan for how to put all the     



Would it be possible to also cache SCCs so that it could be it could be queried for SCCs in the cases
 where the full graph is not cached. If queried with a list of SCCs rather than a single graph, the 
cache could return a corresponding list of hits and missed (NULLs and traces). The workflow could then 
continue computing (and caching) traces for the missing SCCs and merge them to a single trace for the 
full graph. It would be nice if all this could be hidden behind a trace/cache interface that 
transparently produces a trace this way. See the hierachial_cache folder for a Claude-generated 
suggestion for a similar idea.

