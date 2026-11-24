#!/bin/bash
# Run discrete-related test files against the tolerance-patched copy (1.0001 -> 1+1e-9, raise only).
cd /Users/kmt/phasic-bug4
R=experiments/bug4_discrete_rowsum/review3
PRIO="tests/pytest/test_omp_exception_safety.py tests/pytest/test_discrete_moments_and_reward.py tests/pytest/test_svgd_divergence_robustness.py tests/pytest/test_notebook_model_recovery.py tests/pytest/inference/test_exact_grad_discrete.py tests/pytest/test_graph_discretize.py tests/pytest/test_gate_discretize_pins.py tests/pytest/test_gate_discretize_cpp.py tests/pytest/test_reward_transform_discrete_flag.py tests/pytest/test_reward_transform_idempotent.py"
REST=$(grep -rlE "discrete=True|is_discrete|discretize\(|dph_|pdf_discrete|reward_transform_discrete|joint_prob_graph\(|sample_discrete|discrete" tests/pytest --include='test_*.py' | sort)
for f in $PRIO $REST; do
  n=$(echo $f | tr '/' '_')
  [ -f $R/pt/$n.patched.log ] && continue
  PYTHONPATH=$PWD/$R/patched python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 600 -- pixi run python -m pytest $f -q -m "not slow" -p no:cacheprovider --timeout=300 > $R/pt/$n.patched.log 2>&1
  echo "$f :: $(grep -E 'passed|failed|error|no tests ran|runcap' $R/pt/$n.patched.log | tail -2 | tr '\n' ' ')" >> $R/pt/SUMMARY.txt
done
echo DONE >> $R/pt/SUMMARY.txt
