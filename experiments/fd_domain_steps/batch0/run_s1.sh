#!/bin/bash
# Step-size scans, one model per memory-capped process, sequential.
cd "$(dirname "$0")"
RC=/Users/kmt/phasic/.git/claude-scratch/runcap.py
: > s1.log
for m in elim_moments_coal elim_moments_island elim_moments_split elim_jix \
         dph_island wasdph_coal euler_coal euler_island euler_reward euler_split \
         epoch_gran epoch; do
  echo "### $m" >> s1.log
  python3 $RC --mb 6000 --timeout 3600 -- pixi run python s1_step_sizes.py $m >> s1.log 2>&1 \
    || echo "KILLED/FAILED $m" >> s1.log
done
echo ALLDONE >> s1.log
