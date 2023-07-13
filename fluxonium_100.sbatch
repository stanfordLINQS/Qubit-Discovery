#!/usr/bin/bash
#SBATCH --job-name=fluxonium_100
#SBATCH --output=/home/groups/safavi/sqcircuit/Qubit-Discovery/logs/fluxonium_100.%j.out
#SBATCH --error=/home/groups/safavi/sqcircuit/Qubit-Discovery/logs/fluxonium_100.%j.err
#SBATCH --array=1-100
#SBATCH --time=0:30:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

CIRCUIT_TYPES="JL"

for CIRCUIT_CODE in $CIRCUIT_TYPES
do
  srun python3 optimize.py "$CIRCUIT_CODE" "$(($SLURM_ARRAY_TASK_ID-1))"
done
