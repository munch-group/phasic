#!/bin/bash
cd "$(dirname "$0")"; : > g2_refusal_joint.log; rm -f g2_refusal_joint.jsonl
for k in cont disc; do
  python3 /Users/kmt/phasic/.git/claude-scratch/runcap.py --mb 5000 --timeout 2400 -- \
    pixi run python g2_refusal_joint.py $k >> g2_refusal_joint.log 2>&1 || echo "KILLED/FAILED $k" >> g2_refusal_joint.log
done
echo ALLDONE >> g2_refusal_joint.log
