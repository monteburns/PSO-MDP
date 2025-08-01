#!/bin/bash
SCRIPT="pso_pyomo.py"
OUTFILE="results_part4.txt"
ITER=50
H2_PRICE=8

echo "Eksik kombinasyonlar tamamlanıyor..." >> "$OUTFILE"

# rel=0.10 eksikleri
REL=0.10

# DAR=1.2
echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | DAR=1.2 ###" >> "$OUTFILE"
python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --dar 1.2 --h2_price $H2_PRICE >> "$OUTFILE" 2>&1

# PPA=60, 80
for PPA in 60 80; do
  echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
  python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --ppa $PPA --h2_price $H2_PRICE >> "$OUTFILE" 2>&1
done

# rel=0.15 tüm kombinasyonlar
REL=0.15

for DAR in 1.0 1.2; do
  echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | DAR=$DAR ###" >> "$OUTFILE"
  python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --dar $DAR --h2_price $H2_PRICE >> "$OUTFILE" 2>&1
done

for PPA in 60 80; do
  echo -e "\n### H2-optdis | h2=$H2_PRICE | rel=$REL | PPA=$PPA ###" >> "$OUTFILE"
  python3 $SCRIPT --hydrogen --optdis --rel $REL --iter $ITER --ppa $PPA --h2_price $H2_PRICE >> "$OUTFILE" 2>&1
done
