#!/bin/bash
SCRIPT="pso_pyomo.py"
OUTFILE="results_part4.txt"
ITER=50
H2_PRICE=8
REL_VALUES=(0.05 0.10 0.15)
DAR_VALUES=(1.0 1.2)
PPA_VALUES=(60 80)

echo "Hydrogen+Battery optdis h2=8..." > "$OUTFILE"

for REL in "${REL_VALUES[@]}"; do
  for DAR in "${DAR_VALUES[@]}"; do
    echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | DAR=$DAR ###" >> "$OUTFILE"
    python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --dar $DAR --h2_price $H2_PRICE >> "$OUTFILE" 2>&1
  done
  for PPA in "${PPA_VALUES[@]}"; do
    echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
    python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --ppa $PPA --h2_price $H2_PRICE >> "$OUTFILE" 2>&1
  done
done
