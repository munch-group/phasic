#!/bin/bash
R=/Users/kmt/phasic-bug1/experiments/bug1_compiled_cache/review3
cd /Users/kmt/phasic-bug1
while pgrep -f "hitspy" >/dev/null; do /bin/sleep 5; done
python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 600 -- pixi run python -u $R/r5_nonnested_fixed.py > $R/r5.log 2>&1
for f in tests/pytest/inference/test_svgd_correctness.py tests/pytest/inference/test_svgd_api_parity.py tests/pytest/inference/test_nan_observations_correctness.py tests/pytest/inference/test_multivariate.py tests/pytest/test_svgd_assumptions.py tests/pytest/inference/test_svgd_exposure.py tests/pytest/inference/test_preconditioning.py; do
  $R/spy_run.sh $f >> $R/spy_summary.txt 2>&1
done
echo QUEUE-DONE >> $R/spy_summary.txt
