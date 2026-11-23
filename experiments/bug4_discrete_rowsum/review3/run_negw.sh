#!/bin/bash
cd /Users/kmt/phasic-bug4
R=experiments/bug4_discrete_rowsum/review3
mkdir -p $R/nw; export NEGW_LOG=$PWD/$R/nw/negw.jsonl; : > $NEGW_LOG; rm -f $R/nw/SUMMARY.txt
for f in $(grep -v "test_model_selection\|test_jax_integration\|test_exact_grad_callback_mode\|test_exact_grad_formula_mode" $R/pt/native_list.txt) tests/pytest/test_graph.py tests/pytest/test_reward_transform_idempotent.py tests/pytest/test_graph_discretize.py tests/pytest/test_notebook_model_recovery.py tests/pytest/test_sampling_conditioned.py tests/pytest/test_update_weights_invalidation.py tests/pytest/test_joint_prob_tot_cap.py tests/pytest/test_weight_formula_kwarg.py; do
  n=$(echo $f | tr '/' '_')
  if [ "$f" = "tests/pytest/test_svgd_divergence_robustness.py" ]; then
    PYTHONPATH=$PWD/$R python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 420 -- pixi run python -m pytest $f -q -m "not slow" -p negw_plugin -p no:cacheprovider --timeout=200 -k "non_rate_runtime_error or is_rate_blowup" > $R/nw/$n.log 2>&1
  else
    PYTHONPATH=$PWD/$R python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 420 -- pixi run python -m pytest $f -q -m "not slow" -p negw_plugin -p no:cacheprovider --timeout=200 > $R/nw/$n.log 2>&1
  fi
  echo "$f :: $(grep -E 'passed|failed|error|no tests ran|runcap' $R/nw/$n.log | tail -2 | tr '\n' ' ')" >> $R/nw/SUMMARY.txt
done
echo DONE >> $R/nw/SUMMARY.txt
