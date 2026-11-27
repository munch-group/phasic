#!/bin/bash
# Capture the bit-identity goldens on master code, one memory-capped process per cell.
cd "$(dirname "$0")"
RC=${RUNCAP:-/Users/kmt/phasic/.git/claude-scratch/runcap.py}
rm -rf golden_parts; : > capture_golden.log
N=$(cd ../../../tests/pytest && pixi run python -c "import _fd_golden_cells as C; print(len(C.CELLS))" 2>/dev/null | tail -1)
for ((k=0; k<N; k++)); do
  python3 "$RC" --mb 5000 --timeout 1200 -- pixi run python capture_golden.py $k >> capture_golden.log 2>&1 \
    || echo "FAILED cell $k" >> capture_golden.log
done
pixi run python capture_golden.py merge >> capture_golden.log 2>&1
echo ALLDONE >> capture_golden.log
