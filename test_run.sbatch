#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH --output=full_run.%j.out
#SBATCH --error=full_run.%j.err
#SBATCH --time=0:10:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8GB

CIRCUIT_TYPES="JJ JL"

for CIRCUIT_CODE in $CIRCUIT_TYPES
do
  for n in {0..10}
  do
     python3 optimize.py "$CIRCUIT_CODE" "$n"
  done
done
