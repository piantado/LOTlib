#!/bin/bash


## An example slurm script for running Search on Rochester BlueHive

# Choose the colala cluster
#SBATCH -A colala --qos colala
#SBATCH -p colala

#SBATCH -J LOTlibSearch

# Where to put output, etc. 
#SBATCH -e output/err_%j
#SBATCH -o output/out_%j
#SBATCH --mem-per-cpu=1000
#SBATCH -t 100:15:00

# Number of nodes
#SBATCH -N 4
#SBATCH --ntasks-per-node=24

#SBATCH --signal=2@600

module load numpy
module load python/2.7.6
module load openmpi/1.6.5/b1

srun python Search.py --model=Number --data=300 --steps=10000

exit
