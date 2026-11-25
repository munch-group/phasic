#!/bin/bash
# usage: envrun.sh <python-binary> <script> [args]   (isolated HOME/cache, memory-capped)
S=/private/tmp/claude-501/-Users-kmt-phasic/c3f361f0-c1bf-476e-a943-8c5641e4a00a/scratchpad/impl5review
PY="$1"; shift
exec env -u PHASIC_COMPILATION_CACHE_DIR -u JAX_COMPILATION_CACHE_DIR HOME=$S/home PHASIC_CACHE_DIR=$S/cache MPLCONFIGDIR=$S/mpl \
  python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 6000 --timeout 1800 -- "$PY" "$@"
