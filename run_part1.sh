#!/bin/bash
SCRIPT="pso_pyomo.py"
OUTFILE="results_part1.txt"
ITER=50
REL_VALUES=(0.05 0.10 0.15)
DAR_VALUES=(1.0 1.2)
PPA_VALUES=(60 80)

echo "Battery-only runs starting..." > "$OUTFILE"

for MODE in rbdis optdis; do
  for REL in "${REL_VALUES[@]}"; do
    for DAR in "${DAR_VALUES[@]}"; do
      echo -e "\n### Battery-$MODE | rel=$REL | DAR=$DAR ###" >> "$OUTFILE"
      python3 $SCRIPT --battery --$MODE --rel $REL --iter $ITER --dar $DAR >> "$OUTFILE" 2>&1
    done
    for PPA in "${PPA_VALUES[@]}"; do
      echo -e "\n### Battery-$MODE | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
      python3 $SCRIPT --battery --$MODE --rel $REL --iter $ITER --ppa $PPA >> "$OUTFILE" 2>&1
    done
  done
done
