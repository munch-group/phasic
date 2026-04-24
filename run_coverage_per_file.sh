#!/bin/bash
# Run pytest with coverage per file; use append mode so crashes only lose one file's data.
# Skip known segfaulting/crashing tests.
set +e

SKIP_FILES=(
  "test_nan_observations_correctness.py"
  "test_notebook_multivar_reproduction.py"
  "test_sample_with_rewards.py"
  "test_scc_api.py"
  "test_mcmc_accuracy.py"
  "test_mcmc.py"
)

cd /Users/kmt/phasic
rm -f .coverage .coverage.json

for f in tests/pytest/test_*.py; do
  base=$(basename "$f")
  skip=0
  for s in "${SKIP_FILES[@]}"; do
    if [ "$base" = "$s" ]; then skip=1; break; fi
  done
  if [ $skip -eq 1 ]; then
    echo "=== SKIPPING $base ==="
    continue
  fi
  echo "=== RUNNING $base ==="
  pixi run -- coverage run --data-file=.coverage -a --source=phasic -m pytest "$f" --tb=line -q 2>&1 | tail -3
  echo "exit=$?"
done

echo "=== GENERATING JSON ==="
pixi run -- coverage json --data-file=.coverage -o .coverage.json 2>&1 | tail -3
ls -la .coverage.json
