#!/bin/bash
# usage: spy_run.sh <test file> ; runs it not-slow on unmodified code with the hit spy
R=/Users/kmt/phasic-bug1/experiments/bug1_compiled_cache/review3
n=$(echo "$1" | tr '/' '_')
cd /Users/kmt/phasic-bug1
HITSPY_LOG=$R/spy/$n.hits PYTHONPATH=$R python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 600 -- \
  pixi run python -m pytest "$1" -m "not slow" -p hitspy -p no:cacheprovider -q -rfEX --durations=5 > $R/spy/$n.log 2>&1
echo "$1 :: $(grep -aE ' passed| failed| skipped|KILLED' $R/spy/$n.log | tail -2 | tr '\n' ' ') :: hits: $(sort $R/spy/$n.hits 2>/dev/null | cut -f1 | uniq -c | tr '\n' ' ')"
