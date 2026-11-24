#!/bin/bash
# Narrow sweep: files that use native discrete graphs / discrete read-outs, against the tolerance-patched copy.
cd /Users/kmt/phasic-bug4
R=experiments/bug4_discrete_rowsum/review3
for f in $(cat $R/pt/native_list.txt); do
  n=$(echo $f | tr '/' '_')
  [ -f $R/pt/$n.patched.log ] && continue
  PYTHONPATH=$PWD/$R/patched python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 2500 --timeout 420 -- pixi run python -m pytest $f -q -m "not slow" -p no:cacheprovider --timeout=200 > $R/pt/$n.patched.log 2>&1
  echo "$f :: $(grep -E 'passed|failed|error|no tests ran|runcap' $R/pt/$n.patched.log | tail -2 | tr '\n' ' ')" >> $R/pt/SUMMARY.txt
done
echo DONE >> $R/pt/SUMMARY.txt
