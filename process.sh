#!/usr/bin/bash
#SBATCH --job-name=circuit_optimize
#SBATCH --output=circuit_optimize.%j.out
#SBATCH --error=circuit_optimize.%j.err
#SBATCH --time=1:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

# Launch optimization for some number of circuits

conda activate sqc

NUM_CIRCUITS=10
CIRCUIT_TYPES="JJ JL"

for CIRCUIT_CODE in $CIRCUIT_TYPES
do
	for ((INDEX=1; INDEX<=NUM_CIRCUITS; INDEX++))
	do
		echo $CIRCUIT_CODE $INDEX
		# srun python3 optimize.py $CIRCUIT_CODE $INDEX
	done
done
