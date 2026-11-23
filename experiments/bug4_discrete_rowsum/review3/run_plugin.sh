#!/bin/bash
# Unmodified library + instrumentation plugin: which existing tests call elimination read-outs /
# discrete models on a discrete graph with a row sum > 1+1e-9 (would raise under plan item 3)?
cd /Users/kmt/phasic-bug4
R=experiments/bug4_discrete_rowsum/review3
export ROWSUM_LOG=$PWD/$R/pl/rowsum.jsonl
: > $ROWSUM_LOG
for f in $(grep -v "test_model_selection\|test_jax_integration\|test_gate_trace_ffi" $R/pt/native_list.txt) tests/pytest/test_graph.py tests/pytest/test_reward_transform_idempotent.py tests/pytest/test_graph_discretize.py tests/pytest/test_method_of_moments.py; do
  n=$(echo $f | tr '/' '_')
  K=""; [ "$f" = "tests/pytest/test_svgd_divergence_robustness.py" ] && K="-k non_rate_runtime_error or is_rate_blowup"
  PYTHONPATH=$PWD/$R python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 420 -- pixi run python -m pytest $f -q -m "not slow" -p rowsum_plugin -p no:cacheprovider --timeout=200 ${K:+-k "non_rate_runtime_error or is_rate_blowup"} > $R/pl/$n.log 2>&1
  echo "$f :: $(grep -E 'passed|failed|error|no tests ran|runcap' $R/pl/$n.log | tail -2 | tr '\n' ' ')" >> $R/pl/SUMMARY.txt
done
# tight-rule (patched) extras
PYTHONPATH=$PWD/$R/patched python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 420 -- pixi run python -m pytest tests/pytest/test_svgd_divergence_robustness.py -q -p no:cacheprovider --timeout=200 -k "non_rate_runtime_error or is_rate_blowup" > $R/pl/patched_divergence_k.log 2>&1
echo "PATCHED divergence -k :: $(grep -E 'passed|failed|runcap' $R/pl/patched_divergence_k.log | tail -2 | tr '\n' ' ')" >> $R/pl/SUMMARY.txt
PYTHONPATH=$PWD/$R/patched python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 590 -- pixi run python -m pytest tests/pytest/test_notebook_model_recovery.py -q -p no:cacheprovider --timeout=500 -k "joint_prob_svgd and tens" > $R/pl/patched_joint_svgd_25v.log 2>&1
echo "PATCHED joint svgd 25v (slow tier) :: $(grep -E 'passed|failed|runcap' $R/pl/patched_joint_svgd_25v.log | tail -2 | tr '\n' ' ')" >> $R/pl/SUMMARY.txt
echo DONE >> $R/pl/SUMMARY.txt
