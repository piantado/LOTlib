#!/bin/bash

#SBATCH -A colala --qos colala
#SBATCH -p colala

#SBATCH -J NumberGameMakeHypothesis
#SBATCH -e slurm-err_%j
#SBATCH -o slurm-out_%j
#SBATCH --mem-per-cpu=1000
#SBATCH -t 500:15:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=24
#SBATCH --signal=2@600

module load numpy
module load python/2.7.10/b1
module load openmpi/1.6.5/b1

# This will run them sequentially
srun time python MakeHypotheses.py

exit
