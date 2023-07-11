#!/usr/bin/bash
#SBATCH --job-name=fluxonium_10
#SBATCH --output=fluxonium_10.%j.out
#SBATCH --error=fluxonium_10.%j.err
#SBATCH --time=1:00:00
#SBATCH -p normal
#SBATCH -c 10
#SBATCH --mem=8GB

CIRCUIT_TYPES="JJ JL"

for CIRCUIT_CODE in $CIRCUIT_TYPES
do
  for n in {0..9}
  do
     python3 optimize.py "$CIRCUIT_CODE" "$n"
  done
done
