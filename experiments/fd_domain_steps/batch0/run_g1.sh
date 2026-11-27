#!/bin/bash
# one process per (kind, mode), memory-capped
cd "$(dirname "$0")"
: > g1_refusal_grid.log
for kind in cont dph wasdph; do
  for mode in linear log formula_lin formula_div formula_sqrt callback_lin callback_sqrt; do
    python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 5000 --timeout 1500 -- \
      pixi run python g1_refusal_grid.py $kind $mode >> g1_refusal_grid.log 2>&1 || echo "KILLED/FAILED $kind $mode" >> g1_refusal_grid.log
  done
done
for kind in cont dph; do
  python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 5000 --timeout 1500 -- \
    pixi run python g1_refusal_grid.py $kind cpp >> g1_refusal_grid.log 2>&1 || echo "KILLED/FAILED $kind cpp" >> g1_refusal_grid.log
done
echo ALLDONE >> g1_refusal_grid.log
