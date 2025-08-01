#!/bin/bash

# Define your parameter options
H2_PRICES=(5 7.5 10)
REL_VALUES=(0.05 0.10 0.20)
DAR_VALUES=(1.0 1.2)
PPA_VALUES=(60 80)

ITER=30
SCRIPT="pso_pyomo.py"
OUTFILE="sensitivity_results.txt"

echo "Starting all runs..." > "$OUTFILE"

# -----------------------------------------
# BATTERY-ONLY CONFIG
# -----------------------------------------
for MODE in optdis rbdis; do
  for REL in "${REL_VALUES[@]}"; do
    for DAR in "${DAR_VALUES[@]}"; do
      echo -e "\n\n### Battery-only | $MODE | rel=$REL | DAR=$DAR ###" >> "$OUTFILE"
      python3 $SCRIPT --battery --$MODE --rel $REL --iter $ITER --dar $DAR >> "$OUTFILE" 2>&1
    done
    for PPA in "${PPA_VALUES[@]}"; do
      echo -e "\n\n### Battery-only | $MODE | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
      python3 $SCRIPT --battery --$MODE --rel $REL --iter $ITER --ppa $PPA >> "$OUTFILE" 2>&1
    done
  done
done

# -----------------------------------------
# HYDROGEN + BATTERY CONFIG
# -----------------------------------------
for MODE in optdis rbdis; do
  for H2 in "${H2_PRICES[@]}"; do
    for REL in "${REL_VALUES[@]}"; do
      for DAR in "${DAR_VALUES[@]}"; do
        echo -e "\n\n### Hydrogen+Battery | $MODE | h2_price=$H2 | rel=$REL | DAR=$DAR ###" >> "$OUTFILE"
        python3 $SCRIPT --hydrogen --$MODE --rel $REL --iter $ITER --dar $DAR --h2_price $H2 >> "$OUTFILE" 2>&1
      done
      for PPA in "${PPA_VALUES[@]}"; do
        echo -e "\n\n### Hydrogen+Battery | $MODE | h2_price=$H2 | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
        python3 $SCRIPT --hydrogen --$MODE --rel $REL --iter $ITER --ppa $PPA --h2_price $H2 >> "$OUTFILE" 2>&1
      done
    done
  done
done

echo -e "\nAll runs completed." >> "$OUTFILE"
